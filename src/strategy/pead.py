"""PEAD (Post-Earnings Announcement Drift) 전략 모듈.

실적발표 후 드리프트 현상을 활용하여 실적 서프라이즈가 큰 종목에 투자하는 전략이다.
전분기 대비 영업이익/순이익 변화율로 서프라이즈를 측정하고,
서프라이즈 상위 종목을 동일 비중으로 보유한다.

데이터 소스:
- DART 재무제표 (dart_collector.get_financial_statements) 기반 실적 데이터
- pykrx 기본 데이터 (EPS 변화율) fallback
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PEADStrategy(Strategy):
    """실적발표 후 드리프트 (PEAD) 전략.

    실적 서프라이즈(전분기 대비 영업이익/순이익 변화율)가 높은 종목을 선별하여
    동일 비중 포트폴리오를 구성한다.

    Args:
        surprise_threshold: 실적 서프라이즈 임계값 (기본 0.1 = 10%)
        holding_days: 보유 기간 (기본 40거래일, 백테스트 리밸런싱 주기와 연동)
        num_stocks: 포트폴리오 종목 수 (기본 10)
        min_market_cap: 최소 시가총액 (기본 1000억원)
        min_volume: 최소 일 거래대금 (기본 1억원)
        surprise_metric: 서프라이즈 지표 ('operating_income', 'net_income', 'eps')
    """

    def __init__(
        self,
        surprise_threshold: float = 0.1,
        holding_days: int = 40,
        num_stocks: int = 10,
        min_market_cap: int = 100_000_000_000,
        min_volume: int = 100_000_000,
        surprise_metric: str = "operating_income",
    ):
        self.surprise_threshold = surprise_threshold
        self.holding_days = holding_days
        self.num_stocks = num_stocks
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.surprise_metric = surprise_metric

        if surprise_metric not in ("operating_income", "net_income", "eps"):
            raise ValueError(
                f"지원하지 않는 서프라이즈 지표: {surprise_metric}. "
                "'operating_income', 'net_income', 'eps' 중 선택하세요."
            )

        logger.info(
            f"PEADStrategy 초기화: surprise_threshold={surprise_threshold}, "
            f"holding_days={holding_days}, num_stocks={num_stocks}, "
            f"metric={surprise_metric}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"PEAD({self.surprise_metric}, top{self.num_stocks})"

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

        logger.info(f"PEAD 유니버스 필터링: {original_count} -> {len(df)}개 종목")
        return df

    def _calculate_surprise_from_earnings(self, earnings: pd.DataFrame) -> pd.Series:
        """DART 실적 데이터에서 서프라이즈를 계산한다.

        Args:
            earnings: DataFrame with columns:
                - ticker: 종목코드
                - operating_income_current: 당기 영업이익
                - operating_income_prev: 전분기 영업이익
                - net_income_current: 당기 순이익
                - net_income_prev: 전분기 순이익

        Returns:
            pd.Series (index=ticker, values=surprise_ratio)
        """
        if earnings.empty or "ticker" not in earnings.columns:
            return pd.Series(dtype=float)

        df = earnings.copy()

        if self.surprise_metric == "operating_income":
            current_col = "operating_income_current"
            prev_col = "operating_income_prev"
        elif self.surprise_metric == "net_income":
            current_col = "net_income_current"
            prev_col = "net_income_prev"
        else:
            return pd.Series(dtype=float)

        if current_col not in df.columns or prev_col not in df.columns:
            return pd.Series(dtype=float)

        # 전분기 값이 0인 경우 제외
        valid = df[df[prev_col].abs() > 0].copy()
        if valid.empty:
            return pd.Series(dtype=float)

        surprise = (valid[current_col] - valid[prev_col]) / valid[prev_col].abs()

        return pd.Series(
            surprise.values,
            index=valid["ticker"].values,
            name="surprise",
        )

    def _calculate_surprise_from_eps(self, fundamentals: pd.DataFrame) -> pd.Series:
        """pykrx EPS 데이터로 서프라이즈를 근사한다 (fallback).

        EPS 자체를 시계열로 비교할 수 없으므로, EPS/close 비율(E/P)을 스코어로 사용한다.
        E/P가 높을수록 실적이 주가 대비 양호하다고 판단한다.

        Args:
            fundamentals: pykrx 전 종목 기본 지표 DataFrame

        Returns:
            pd.Series (index=ticker, values=eps_score)
        """
        if fundamentals.empty or "ticker" not in fundamentals.columns:
            return pd.Series(dtype=float)

        df = fundamentals.copy()

        if "eps" not in df.columns or "close" not in df.columns:
            return pd.Series(dtype=float)

        # EPS > 0인 종목만 (흑자 기업)
        valid = df[(df["eps"] > 0) & (df["close"] > 0)].copy()
        if valid.empty:
            return pd.Series(dtype=float)

        # E/P ratio를 서프라이즈 대용으로 사용
        ep_ratio = valid["eps"] / valid["close"]

        return pd.Series(
            ep_ratio.values,
            index=valid["ticker"].values,
            name="surprise",
        )

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        실적 서프라이즈 상위 종목을 동일 비중으로 선택한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {
                'fundamentals': DataFrame (pykrx 기본 지표),
                'earnings': DataFrame (DART 실적 데이터, optional),
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

        filtered_tickers = set(filtered["ticker"].values)

        # 서프라이즈 계산
        earnings = data.get("earnings", pd.DataFrame())
        if not earnings.empty and self.surprise_metric != "eps":
            surprise = self._calculate_surprise_from_earnings(earnings)
        else:
            surprise = self._calculate_surprise_from_eps(filtered)

        if surprise.empty:
            logger.warning(f"서프라이즈 계산 실패 ({date})")
            return {}

        # 필터된 유니버스 종목만 선택
        surprise = surprise[surprise.index.isin(filtered_tickers)]

        if surprise.empty:
            logger.warning(f"필터 후 서프라이즈 데이터 없음 ({date})")
            return {}

        # 서프라이즈 임계값 적용 (양의 서프라이즈만)
        above_threshold = surprise[surprise >= self.surprise_threshold]

        if above_threshold.empty:
            logger.info(
                f"서프라이즈 임계값({self.surprise_threshold}) 이상 종목 없음 ({date})"
            )
            return {}

        # 서프라이즈 상위 종목 선정
        top = above_threshold.sort_values(ascending=False).head(self.num_stocks)

        # 동일 비중 할당
        weight = 1.0 / len(top)
        signals = {ticker: weight for ticker in top.index}

        logger.info(
            f"PEAD 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"개별 비중={weight:.4f}"
        )

        return signals
