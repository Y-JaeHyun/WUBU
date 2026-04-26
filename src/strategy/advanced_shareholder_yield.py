"""고도화 주주환원 전략 모듈.

배당과 자사주 소각(treasury share cancellation)을 결합한
고도화 주주환원 스코어를 기반으로 포트폴리오를 구성하는 전략이다.

자사주 소각은 단순 자사주 매입보다 강한 신호로, Value-Up 정책하에서
구조적 재평가를 유발한다.

데이터 소스:
- pykrx get_market_fundamental_by_ticker에서 DIV(배당수익률)
- DART 전자공시에서 자사주 소각(treasury share retirement) 데이터
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedShareholderYieldStrategy(Strategy):
    """고도화 주주환원 전략.

    배당수익률 + 자사주 소각 비율을 결합하여 주주환원 스코어를 계산한다.
    자사주 소각은 단순 자사주 매입보다 높은 가중치를 받는다.

    Args:
        num_stocks: 포트폴리오 종목 수 (기본 15)
        min_market_cap: 최소 시가총액 (기본 1000억원)
        min_volume: 최소 일 거래대금 (기본 1억원)
        div_weight: 배당수익률 가중치 (기본 0.4)
        cancellation_weight: 자사주 소각 가중치 (기본 0.6, 강력한 신호)
        min_div_yield: 최소 배당수익률 (기본 0.0)
    """

    def __init__(
        self,
        num_stocks: int = 15,
        min_market_cap: int = 100_000_000_000,
        min_volume: int = 100_000_000,
        div_weight: float = 0.4,
        cancellation_weight: float = 0.6,
        min_div_yield: float = 0.0,
    ):
        self.num_stocks = num_stocks
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.div_weight = div_weight
        self.cancellation_weight = cancellation_weight
        self.min_div_yield = min_div_yield

        logger.info(
            f"AdvancedShareholderYieldStrategy 초기화: num_stocks={num_stocks}, "
            f"div_weight={div_weight}, cancellation_weight={cancellation_weight}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"AdvancedShareholderYield(top{self.num_stocks})"

    def _filter_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """유니버스 필터링: 시가총액, 거래대금, 최소 배당수익률 적용."""
        if df.empty:
            return df

        original_count = len(df)

        if "market_cap" in df.columns:
            df = df[df["market_cap"] >= self.min_market_cap]

        if "volume" in df.columns and "close" in df.columns:
            trade_value = df["volume"] * df["close"]
            df = df[trade_value >= self.min_volume]

        if "div_yield" in df.columns and self.min_div_yield > 0:
            df = df[df["div_yield"] >= self.min_div_yield]

        logger.debug(
            f"고도화 주주환원 유니버스 필터링: {original_count} -> {len(df)}개 종목"
        )
        return df

    def _calculate_advanced_shareholder_yield_score(
        self,
        fundamentals: pd.DataFrame,
        cancellation_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """고도화 주주환원 스코어를 계산한다.

        배당수익률과 자사주 소각을 결합한 스코어.
        자사주 소각은 단순 자사주 매입보다 강한 신호로 취급.

        Args:
            fundamentals: 전 종목 기본 지표 DataFrame (div_yield 포함)
            cancellation_data: 자사주 소각 데이터 (optional)
                DataFrame with columns: ['ticker', 'cancellation_ratio']

        Returns:
            pd.Series (index=ticker, values=score)
        """
        if fundamentals.empty or "ticker" not in fundamentals.columns:
            return pd.Series(dtype=float)

        df = fundamentals.copy()

        if "div_yield" not in df.columns:
            logger.warning("div_yield 컬럼 없음, 빈 스코어 반환")
            return pd.Series(dtype=float)

        valid = df[df["div_yield"] >= 0].copy()
        if valid.empty:
            return pd.Series(dtype=float)

        # 배당수익률 백분위 순위
        div_score = valid.set_index("ticker")["div_yield"].rank(pct=True)

        # 자사주 소각 데이터 결합
        if (
            cancellation_data is not None
            and not cancellation_data.empty
            and "ticker" in cancellation_data.columns
            and "cancellation_ratio" in cancellation_data.columns
        ):
            cancellation_score = cancellation_data.set_index("ticker")[
                "cancellation_ratio"
            ].rank(pct=True)

            # 공통 종목에 대해 가중 결합
            common = div_score.index.intersection(cancellation_score.index)
            if not common.empty:
                combined = (
                    div_score.loc[common] * self.div_weight
                    + cancellation_score.loc[common] * self.cancellation_weight
                )
                # 자사주 소각 데이터 없는 종목은 배당만으로
                div_only = div_score.index.difference(common)
                if not div_only.empty:
                    combined = pd.concat([combined, div_score.loc[div_only]])
                return combined

        # 소각 데이터 없으면 배당수익률 단독
        return div_score

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {
                'fundamentals': DataFrame (div_yield 포함),
                'cancellation': DataFrame (자사주 소각 데이터, optional),
            }

        Returns:
            종목코드: 비중 딕셔너리
        """
        fundamentals = data.get("fundamentals", pd.DataFrame())

        if fundamentals.empty:
            logger.warning(f"펀더멘탈 데이터 없음 ({date}), 빈 시그널 반환")
            return {}

        # 유니버스 필터링
        filtered = self._filter_universe(fundamentals.copy())
        if filtered.empty:
            logger.warning(f"필터링 후 종목 없음 ({date})")
            return {}

        # 자사주 소각 데이터
        cancellation_data = data.get("cancellation", None)

        # 스코어 계산
        scores = self._calculate_advanced_shareholder_yield_score(
            filtered, cancellation_data
        )

        if scores.empty:
            logger.warning(f"고도화 주주환원 스코어 계산 실패 ({date})")
            return {}

        # 상위 N개 선택
        top = scores.sort_values(ascending=False).head(self.num_stocks)

        if top.empty:
            return {}

        # 동일 비중 할당
        weight = 1.0 / len(top)
        signals = {ticker: weight for ticker in top.index}

        logger.info(
            f"고도화 주주환원 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"개별 비중={weight:.4f}"
        )

        return signals
