"""저변동성(Low Volatility) 팩터 전략 모듈.

실현 변동성이 낮은 종목을 선별하여 포트폴리오를 구성하는 전략을 제공한다.
저변동성 이상현상(Low Volatility Anomaly)에 기반하여, 변동성이 낮은 종목이
장기적으로 위험 대비 우수한 수익률을 기록하는 현상을 활용한다.

핵심 지표:
- 실현 변동성 (Realized Volatility): 일별 수익률의 표준편차를 연율화한 값
- 스코어: 1 / 변동성 (변동성이 낮을수록 높은 스코어)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LowVolatilityStrategy(Strategy):
    """저변동성 팩터 전략.

    실현 변동성이 낮은 종목에 투자하는 저변동성 전략이다.
    과거 일정 기간의 일별 수익률로부터 연율화 변동성을 계산하고,
    변동성이 낮은 상위 종목을 선별한다.

    Args:
        vol_period: 변동성 계산 기간 (거래일 수, 기본 60)
        num_stocks: 포트폴리오 종목 수 (기본 20)
        min_market_cap: 최소 시가총액 (기본 1000억원)
        min_trading_days: 최소 거래일 수 (기본 120)
        weighting: 가중 방식 - "equal" (기본 "equal")
    """

    def __init__(
        self,
        vol_period: int = 60,
        num_stocks: int = 20,
        min_market_cap: int = 100_000_000_000,
        min_trading_days: int = 120,
        weighting: str = "equal",
    ):
        self.vol_period = vol_period
        self.num_stocks = num_stocks
        self.min_market_cap = min_market_cap
        self.min_trading_days = min_trading_days
        self.weighting = weighting.lower()

        logger.info(
            f"LowVolatilityStrategy 초기화: vol_period={self.vol_period}, "
            f"num_stocks={num_stocks}, min_market_cap={min_market_cap:,}, "
            f"min_trading_days={min_trading_days}, weighting={self.weighting}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"LowVol({self.vol_period}d, top{self.num_stocks})"

    def _filter_universe(
        self,
        fundamentals: pd.DataFrame,
        prices: dict[str, pd.DataFrame],
    ) -> list[str]:
        """유니버스 필터링을 수행한다.

        조건:
        - 시가총액 >= min_market_cap
        - 가격 데이터 >= min_trading_days

        Args:
            fundamentals: 전 종목 기본 지표 DataFrame
            prices: {ticker: DataFrame(OHLCV)} 가격 캐시

        Returns:
            필터링을 통과한 종목 코드 리스트
        """
        if fundamentals.empty:
            return []

        df = fundamentals.copy()

        # 시가총액 필터
        if "market_cap" in df.columns:
            df = df[df["market_cap"] >= self.min_market_cap]

        tickers = df["ticker"].tolist() if "ticker" in df.columns else []

        # 최소 거래일 수 필터
        filtered = []
        for ticker in tickers:
            if ticker in prices and len(prices[ticker]) >= self.min_trading_days:
                filtered.append(ticker)

        logger.info(
            f"유니버스 필터링: 전체 {len(fundamentals)} -> "
            f"기본필터 {len(tickers)} -> 거래일필터 {len(filtered)}개"
        )
        return filtered

    def _compute_volatility(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        date: str = "",
    ) -> Optional[float]:
        """단일 종목의 연율화 실현 변동성을 계산한다.

        일별 수익률의 표준편차를 sqrt(252)로 연율화한다.

        Args:
            ticker: 종목 코드
            price_df: OHLCV DataFrame (DatetimeIndex)
            date: 기준 날짜 (YYYYMMDD). Look-Ahead Bias 방지용.

        Returns:
            연율화 변동성 (float). 데이터 부족 또는 변동성 0이면 None 반환.
        """
        if price_df.empty or "close" not in price_df.columns:
            return None

        # C3: date 기준으로 미래 데이터 제거 (Look-Ahead Bias 방지)
        if date:
            target = pd.Timestamp(date)
            price_df = price_df[price_df.index <= target]
            if price_df.empty:
                return None

        daily_returns = price_df["close"].pct_change().dropna()

        if len(daily_returns) < self.vol_period:
            return None

        # 최근 vol_period 기간의 변동성
        recent_returns = daily_returns.iloc[-self.vol_period :]
        vol = recent_returns.std() * np.sqrt(252)

        # 변동성이 사실상 0인 경우 (거래정지 등) 제외
        if vol <= 1e-8:
            return None

        return float(vol)

    def _compute_scores(
        self,
        tickers: list[str],
        prices: dict[str, pd.DataFrame],
        date: str = "",
    ) -> pd.Series:
        """종목별 저변동성 스코어를 계산한다.

        스코어 = 1 / 변동성 (변동성이 낮을수록 높은 스코어)
        상하위 1% Winsorizing을 적용한다.

        Args:
            tickers: 대상 종목 리스트
            prices: {ticker: DataFrame(OHLCV)} 가격 캐시
            date: 기준 날짜 (YYYYMMDD). Look-Ahead Bias 방지용.

        Returns:
            pd.Series (index=ticker, values=low_vol_score)
        """
        scores: dict[str, float] = {}

        for ticker in tickers:
            if ticker not in prices or prices[ticker].empty:
                continue

            vol = self._compute_volatility(ticker, prices[ticker], date)
            if vol is not None:
                scores[ticker] = 1.0 / vol

        score_series = pd.Series(scores, name="low_vol_score")

        if score_series.empty:
            return score_series

        # Winsorizing: 상/하위 1% 클리핑
        lower = score_series.quantile(0.01)
        upper = score_series.quantile(0.99)
        score_series = score_series.clip(lower=lower, upper=upper)

        return score_series

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        저변동성 스코어 상위 N개 종목을 선별하고 동일 비중을 부여한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {'fundamentals': DataFrame, 'prices': dict[ticker, DataFrame]} 형태

        Returns:
            종목코드: 비중 딕셔너리
        """
        fundamentals = data.get("fundamentals", pd.DataFrame())
        prices = data.get("prices", {})

        if fundamentals.empty:
            logger.warning(f"펀더멘탈 데이터 없음 ({date}), 빈 시그널 반환")
            return {}

        # 유니버스 필터링
        universe = self._filter_universe(fundamentals, prices)

        if not universe:
            logger.warning(f"유니버스가 비어 있음 ({date})")
            return {}

        # 저변동성 스코어 계산
        scores = self._compute_scores(universe, prices, date)

        if scores.empty:
            logger.warning(f"저변동성 스코어 계산 실패 ({date})")
            return {}

        # 상위 N개 선정 (내림차순 = 저변동성 종목)
        top_scores = scores.sort_values(ascending=False).head(self.num_stocks)

        if top_scores.empty:
            return {}

        # 동일 비중 할당
        weight = 1.0 / len(top_scores)
        signals = {ticker: weight for ticker in top_scores.index}

        logger.info(
            f"저변동성 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"개별 비중={weight:.4f}, "
            f"스코어 범위=[{top_scores.min():.4f}, {top_scores.max():.4f}]"
        )

        return signals

    def get_scores(self, data: dict, date: str = "") -> pd.Series:
        """외부에서 저변동성 스코어에 접근할 수 있도록 제공한다.

        ThreeFactorStrategy, MultiFactorStrategy 등에서 팩터 결합 시 사용한다.

        Args:
            data: {'fundamentals': DataFrame, 'prices': dict[ticker, DataFrame]} 형태
            date: 기준 날짜 (YYYYMMDD). Look-Ahead Bias 방지용.

        Returns:
            pd.Series (index=ticker, values=low_vol_score)
        """
        fundamentals = data.get("fundamentals", pd.DataFrame())
        prices = data.get("prices", {})

        universe = self._filter_universe(fundamentals, prices)
        if not universe:
            return pd.Series(dtype=float)

        return self._compute_scores(universe, prices, date)
