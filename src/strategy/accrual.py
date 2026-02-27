"""발생액 역추세 전략 모듈.

발생액(Accruals)이 낮은 종목, 즉 이익의 질이 높은 종목을 선호하는 전략이다.
발생액 = (순이익 - 영업CF) / 총자산으로 계산하며, 낮을수록 현금 기반 이익 비중이 높다.

데이터 소스:
- DART 재무제표 (dart_collector.get_financial_statements) — accruals 필드 활용
- pykrx 기본 데이터 fallback (제한적)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AccrualStrategy(Strategy):
    """발생액 기반 전략.

    발생액(Accruals)이 낮은 종목(이익의 질이 높은 종목)을 선별하여
    동일 비중 포트폴리오를 구성한다.

    발생액 = (순이익 - 영업활동현금흐름) / 총자산
    - 음수: 현금 이익 > 회계 이익 (좋음)
    - 양수: 회계 이익 > 현금 이익 (나쁨)

    Args:
        num_stocks: 포트폴리오 종목 수 (기본 10)
        min_market_cap: 최소 시가총액 (기본 1000억원)
        min_volume: 최소 일 거래대금 (기본 1억원)
        exclude_negative_earnings: 적자 기업 제외 여부 (기본 True)
    """

    def __init__(
        self,
        num_stocks: int = 10,
        min_market_cap: int = 100_000_000_000,
        min_volume: int = 100_000_000,
        exclude_negative_earnings: bool = True,
    ):
        self.num_stocks = num_stocks
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.exclude_negative_earnings = exclude_negative_earnings

        logger.info(
            f"AccrualStrategy 초기화: num_stocks={num_stocks}, "
            f"min_market_cap={min_market_cap:,}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"Accrual(top{self.num_stocks})"

    def _filter_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """유니버스 필터링: 시가총액, 거래대금 하한 적용."""
        if df.empty:
            return df

        original_count = len(df)

        if "market_cap" in df.columns:
            df = df[df["market_cap"] >= self.min_market_cap]

        if "volume" in df.columns and "close" in df.columns:
            trade_value = df["volume"] * df["close"]
            df = df[trade_value >= self.min_volume]

        # 적자 기업 제외 (EPS < 0)
        if self.exclude_negative_earnings and "eps" in df.columns:
            df = df[df["eps"] > 0]

        logger.info(
            f"발생액 유니버스 필터링: {original_count} -> {len(df)}개 종목"
        )
        return df

    def _get_accrual_scores(
        self,
        fundamentals: pd.DataFrame,
        quality_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """발생액 스코어를 계산한다.

        발생액이 낮을수록 높은 스코어를 부여한다.

        Args:
            fundamentals: 전 종목 기본 지표 DataFrame
            quality_data: DART 퀄리티 데이터 (accruals 포함, optional)

        Returns:
            pd.Series (index=ticker, values=accrual_score)
            높을수록 좋음 (저발생액)
        """
        if fundamentals.empty or "ticker" not in fundamentals.columns:
            return pd.Series(dtype=float)

        # DART 퀄리티 데이터에서 accruals 필드 우선 사용
        if (
            quality_data is not None
            and not quality_data.empty
            and "ticker" in quality_data.columns
            and "accruals" in quality_data.columns
        ):
            valid = quality_data[quality_data["accruals"].notna()].copy()
            if not valid.empty:
                # 발생액이 낮을수록 좋으므로 역순위
                accruals = pd.Series(
                    valid["accruals"].values,
                    index=valid["ticker"].values,
                )
                # 유효 종목만 필터
                fund_tickers = set(fundamentals["ticker"].values)
                accruals = accruals[accruals.index.isin(fund_tickers)]

                if not accruals.empty:
                    # 낮을수록 좋으므로 1 - rank(pct=True)
                    score = 1 - accruals.rank(pct=True)
                    return score

        # DART 데이터 없으면 fundamentals의 accruals 필드 확인
        if "accruals" in fundamentals.columns:
            valid = fundamentals[fundamentals["accruals"].notna()].copy()
            if not valid.empty:
                accruals = pd.Series(
                    valid["accruals"].values,
                    index=valid["ticker"].values,
                )
                if not accruals.empty:
                    score = 1 - accruals.rank(pct=True)
                    return score

        # 발생액 데이터 없으면 EPS/BPS 기반 근사
        # 이익의 질 proxy: EPS 안정성 (PER의 역수, 높을수록 수익성 대비 저평가)
        if "per" in fundamentals.columns and "eps" in fundamentals.columns:
            valid = fundamentals[
                (fundamentals["per"] > 0) & (fundamentals["eps"] > 0)
            ].copy()
            if not valid.empty:
                # PER 역수를 이익의 질 proxy로 사용
                per_inv = 1.0 / valid["per"]
                score = pd.Series(
                    per_inv.values,
                    index=valid["ticker"].values,
                )
                score = score.rank(pct=True)
                return score

        return pd.Series(dtype=float)

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        저발생액(이익의 질 높은) 상위 N개 종목을 동일 비중으로 선택한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {
                'fundamentals': DataFrame,
                'quality': DataFrame (DART 퀄리티 데이터, optional),
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

        # 퀄리티 데이터
        quality_data = data.get("quality", None)

        # 발생액 스코어 계산
        scores = self._get_accrual_scores(filtered, quality_data)

        if scores.empty:
            logger.warning(f"발생액 스코어 계산 실패 ({date})")
            return {}

        # 필터된 종목만
        filtered_tickers = set(filtered["ticker"].values)
        scores = scores[scores.index.isin(filtered_tickers)]

        if scores.empty:
            return {}

        # 상위 N개 선택 (스코어 높은 순 = 저발생액)
        top = scores.sort_values(ascending=False).head(self.num_stocks)

        if top.empty:
            return {}

        # 동일 비중 할당
        weight = 1.0 / len(top)
        signals = {ticker: weight for ticker in top.index}

        logger.info(
            f"발생액 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"개별 비중={weight:.4f}"
        )

        return signals
