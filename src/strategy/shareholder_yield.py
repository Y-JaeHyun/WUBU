"""주주환원 전략 모듈.

배당수익률과 자사주매입을 결합한 주주환원 스코어를 기반으로
주주환원 우수 종목을 선별하여 포트폴리오를 구성하는 전략이다.

데이터 소스:
- pykrx get_market_fundamental_by_ticker에서 DIV(배당수익률) 활용
- DART 자사주 데이터 (가능한 경우) — 없으면 배당수익률 단독 사용
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShareholderYieldStrategy(Strategy):
    """주주환원 전략.

    배당수익률 + 자사주매입 비율을 결합하여 주주환원 스코어 상위 종목을 선별한다.
    자사주 데이터가 없으면 배당수익률 단독으로 스코어링한다.

    Args:
        num_stocks: 포트폴리오 종목 수 (기본 10)
        min_market_cap: 최소 시가총액 (기본 1000억원)
        min_volume: 최소 일 거래대금 (기본 1억원)
        div_weight: 배당수익률 가중치 (기본 0.6)
        buyback_weight: 자사주매입 가중치 (기본 0.4)
        min_div_yield: 최소 배당수익률 (기본 0.0, 무배당 제외용)
    """

    def __init__(
        self,
        num_stocks: int = 10,
        min_market_cap: int = 100_000_000_000,
        min_volume: int = 100_000_000,
        div_weight: float = 0.6,
        buyback_weight: float = 0.4,
        min_div_yield: float = 0.0,
    ):
        self.num_stocks = num_stocks
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.div_weight = div_weight
        self.buyback_weight = buyback_weight
        self.min_div_yield = min_div_yield

        logger.info(
            f"ShareholderYieldStrategy 초기화: num_stocks={num_stocks}, "
            f"div_weight={div_weight}, buyback_weight={buyback_weight}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"ShareholderYield(top{self.num_stocks})"

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

        # 최소 배당수익률 필터
        if "div_yield" in df.columns and self.min_div_yield > 0:
            df = df[df["div_yield"] >= self.min_div_yield]

        logger.info(
            f"주주환원 유니버스 필터링: {original_count} -> {len(df)}개 종목"
        )
        return df

    def _calculate_shareholder_yield_score(
        self,
        fundamentals: pd.DataFrame,
        buyback_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """주주환원 스코어를 계산한다.

        배당수익률의 백분위 순위에 가중치를 적용하고,
        자사주매입 데이터가 있으면 결합한다.

        Args:
            fundamentals: 전 종목 기본 지표 DataFrame (div_yield 포함)
            buyback_data: 자사주매입 데이터 (optional)
                DataFrame with columns: ['ticker', 'buyback_ratio']

        Returns:
            pd.Series (index=ticker, values=shareholder_yield_score)
        """
        if fundamentals.empty or "ticker" not in fundamentals.columns:
            return pd.Series(dtype=float)

        df = fundamentals.copy()

        # 배당수익률 스코어
        if "div_yield" not in df.columns:
            logger.warning("div_yield 컬럼 없음, 빈 스코어 반환")
            return pd.Series(dtype=float)

        # 배당수익률이 0 이상인 종목만
        valid = df[df["div_yield"] >= 0].copy()
        if valid.empty:
            return pd.Series(dtype=float)

        # 배당수익률 백분위 순위
        div_score = valid.set_index("ticker")["div_yield"].rank(pct=True)

        # 자사주매입 데이터 결합
        if (
            buyback_data is not None
            and not buyback_data.empty
            and "ticker" in buyback_data.columns
            and "buyback_ratio" in buyback_data.columns
        ):
            buyback_score = buyback_data.set_index("ticker")["buyback_ratio"].rank(
                pct=True
            )
            # 공통 종목에 대해 가중 결합
            common = div_score.index.intersection(buyback_score.index)
            if not common.empty:
                combined = (
                    div_score.loc[common] * self.div_weight
                    + buyback_score.loc[common] * self.buyback_weight
                )
                # 자사주 데이터 없는 종목은 배당만으로
                div_only = div_score.index.difference(common)
                if not div_only.empty:
                    combined = pd.concat(
                        [combined, div_score.loc[div_only]]
                    )
                return combined

        # 자사주 데이터 없으면 배당수익률 단독
        return div_score

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        주주환원 스코어 상위 N개 종목을 동일 비중으로 선택한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {
                'fundamentals': DataFrame (div_yield 포함),
                'buyback': DataFrame (자사주매입 데이터, optional),
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

        # 자사주매입 데이터
        buyback_data = data.get("buyback", None)

        # 주주환원 스코어 계산
        scores = self._calculate_shareholder_yield_score(filtered, buyback_data)

        if scores.empty:
            logger.warning(f"주주환원 스코어 계산 실패 ({date})")
            return {}

        # 상위 N개 선택
        top = scores.sort_values(ascending=False).head(self.num_stocks)

        if top.empty:
            return {}

        # 동일 비중 할당
        weight = 1.0 / len(top)
        signals = {ticker: weight for ticker in top.index}

        logger.info(
            f"주주환원 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"개별 비중={weight:.4f}"
        )

        return signals
