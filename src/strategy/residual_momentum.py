"""잔차 모멘텀 전략 모듈.

전통적인 가격 모멘텀에서 시장 베타를 제거하여
개별 종목의 진정한 강도(idiosyncratic strength)를 캡처하는 전략이다.

시장 모멘텀 크래시(MDD가 큰 경기 변동기)에 더 견고하다는 특징이 있다.

데이터 소스:
- pykrx daily price data
- KOSPI index (시장 팩터)
- 섹터 인덱스 (선택적 베타 제거)
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_residual_momentum(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    lookback: int = 252,
    skip: int = 21,
) -> float:
    """개별 종목의 잔차 모멘텀을 계산한다.

    시장 수익률 대비 베타를 회귀분석으로 추정하고,
    잔차 수익률의 모멘텀을 계산한다.

    Args:
        stock_returns: 종목 수익률 시계열 (거래일 기준)
        market_returns: 시장 수익률 시계열 (같은 길이)
        lookback: 룩백 기간 (거래일 수, 기본 252=1년)
        skip: 최근 스킵 기간 (기본 21=1개월)

    Returns:
        잔차 모멘텀 (float). 계산 불가 시 NaN 반환.
    """
    if len(stock_returns) < lookback or len(market_returns) < lookback:
        return float("nan")

    if skip >= lookback:
        return float("nan")

    # 룩백 구간 추출 (최근 skip일 제외)
    end_idx = len(stock_returns) - skip if skip > 0 else len(stock_returns)
    start_idx = end_idx - lookback

    if start_idx < 0 or end_idx <= start_idx:
        return float("nan")

    stock_r = stock_returns.iloc[start_idx:end_idx].values
    market_r = market_returns.iloc[start_idx:end_idx].values

    # 데이터 검증
    if np.isnan(stock_r).any() or np.isnan(market_r).any():
        return float("nan")

    # 베타 추정 (회귀분석)
    try:
        # 상수항과 시장 수익률로 선형 회귀
        X = np.column_stack([np.ones(len(market_r)), market_r])
        try:
            # numpy 선형대수를 사용한 추정
            coeffs = np.linalg.lstsq(X, stock_r, rcond=None)[0]
            alpha, beta = coeffs[0], coeffs[1]
        except np.linalg.LinAlgError:
            # 특이 행렬인 경우 fallback
            slope, intercept, _, _, _ = stats.linregress(market_r, stock_r)
            beta, alpha = slope, intercept

        # 잔차 = 실제 수익률 - 베타 × 시장 수익률
        predicted = alpha + beta * market_r
        residuals = stock_r - predicted

        # 잔차의 모멘텀: 마지막 시점의 누적 잔차 수익률
        residual_momentum = np.sum(residuals) / len(residuals) if len(residuals) > 0 else float("nan")

        return float(residual_momentum)
    except Exception as e:
        logger.debug(f"잔차 모멘텀 계산 실패: {e}")
        return float("nan")


class ResidualMomentumStrategy(Strategy):
    """잔차 모멘텀 전략.

    전통적인 가격 모멘텀에서 시장 베타를 제거하여
    개별 종목의 진정한 모멘텀을 포착한다.

    Args:
        lookback_days: 회귀분석 기간 (거래일, 기본 252=1년)
        skip_days: 최근 스킵 기간 (거래일, 기본 21=1개월)
        num_stocks: 포트폴리오 종목 수 (기본 20)
        min_market_cap: 최소 시가총액 (기본 1000억원)
        min_volume: 최소 일 거래대금 (기본 1억원)
        weighting: 가중 방식 - "equal" 또는 "score" (기본 "equal")
    """

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        num_stocks: int = 20,
        min_market_cap: int = 100_000_000_000,
        min_volume: int = 100_000_000,
        weighting: str = "equal",
    ):
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.num_stocks = num_stocks
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.weighting = weighting.lower()

        if self.weighting not in ("equal", "score"):
            raise ValueError(f"weighting은 'equal' 또는 'score'이어야 함: {weighting}")

        logger.info(
            f"ResidualMomentumStrategy 초기화: num_stocks={num_stocks}, "
            f"lookback_days={lookback_days}, skip_days={skip_days}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"ResidualMomentum(top{self.num_stocks})"

    def _filter_universe(
        self, fundamentals: pd.DataFrame, prices: dict[str, pd.DataFrame]
    ) -> set[str]:
        """유니버스 필터링.

        시가총액, 거래대금, 가격 데이터 가용성을 확인한다.
        """
        if fundamentals.empty:
            return set()

        # 시가총액 필터
        valid = fundamentals[fundamentals.get("market_cap", 0) >= self.min_market_cap]

        # 거래대금 필터
        if "volume" in valid.columns and "close" in valid.columns:
            trade_value = valid["volume"] * valid["close"]
            valid = valid[trade_value >= self.min_volume]

        # 가격 데이터 가용성 확인
        tickers = set()
        for ticker in valid.get("ticker", []):
            if ticker in prices and not prices[ticker].empty:
                tickers.add(ticker)

        logger.debug(
            f"잔차 모멘텀 유니버스 필터링: "
            f"{len(fundamentals)} -> {len(valid)} -> {len(tickers)}개 종목"
        )
        return tickers

    def _compute_scores(
        self,
        tickers: set[str],
        prices: dict[str, pd.DataFrame],
        index_prices: pd.Series,
    ) -> pd.Series:
        """잔차 모멘텀 스코어를 계산한다.

        Args:
            tickers: 대상 종목 코드 집합
            prices: {ticker: DataFrame} 가격 데이터
            index_prices: KOSPI 인덱스 가격 시계열

        Returns:
            pd.Series (index=ticker, values=residual_momentum_score)
        """
        if not tickers or index_prices.empty:
            return pd.Series(dtype=float)

        # 시장 수익률 계산
        if len(index_prices) < 2:
            logger.warning("인덱스 데이터 부족, 빈 스코어 반환")
            return pd.Series(dtype=float)

        market_returns = index_prices.pct_change()

        scores: dict[str, float] = {}

        for ticker in tickers:
            if ticker not in prices:
                continue

            price_df = prices[ticker]
            if price_df.empty or "close" not in price_df.columns:
                continue

            prices_series = price_df["close"]
            if len(prices_series) < 2:
                continue

            # 수익률 계산
            stock_returns = prices_series.pct_change()

            # 인덱스와 길이 맞추기
            common_idx = stock_returns.index.intersection(market_returns.index)
            if len(common_idx) < self.lookback_days:
                continue

            stock_r_aligned = stock_returns.loc[common_idx]
            market_r_aligned = market_returns.loc[common_idx]

            # 잔차 모멘텀 계산
            score = calculate_residual_momentum(
                stock_r_aligned,
                market_r_aligned,
                lookback=self.lookback_days,
                skip=self.skip_days,
            )

            if not np.isnan(score):
                scores[ticker] = score

        if not scores:
            logger.warning("잔차 모멘텀 스코어 계산 실패 (모든 종목)")
            return pd.Series(dtype=float)

        return pd.Series(scores)

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {
                'fundamentals': DataFrame,
                'prices': dict[ticker, DataFrame],
                'index_prices': pd.Series (KOSPI),
            }

        Returns:
            종목코드: 비중 딕셔너리
        """
        fundamentals = data.get("fundamentals", pd.DataFrame())
        prices = data.get("prices", {})
        index_prices = data.get("index_prices", pd.Series(dtype=float))

        if fundamentals.empty:
            logger.warning(f"펀더멘탈 데이터 없음 ({date}), 빈 시그널 반환")
            return {}

        # 유니버스 필터링
        universe = self._filter_universe(fundamentals, prices)
        if not universe:
            logger.warning(f"유니버스가 비어 있음 ({date})")
            return {}

        # 잔차 모멘텀 스코어 계산
        scores = self._compute_scores(universe, prices, index_prices)

        if scores.empty:
            logger.warning(f"잔차 모멘텀 스코어 계산 실패 ({date})")
            return {}

        # 상위 N개 선정
        top_scores = scores.sort_values(ascending=False).head(self.num_stocks)

        if top_scores.empty:
            return {}

        # 가중치 부여
        if self.weighting == "equal":
            weight = 1.0 / len(top_scores)
            signals = {ticker: weight for ticker in top_scores.index}
        else:  # score
            # 스코어 비례 가중 (음수는 0으로 처리)
            positive_scores = top_scores.clip(lower=0)
            if positive_scores.sum() <= 0:
                weight = 1.0 / len(top_scores)
                signals = {ticker: weight for ticker in top_scores.index}
            else:
                total = positive_scores.sum()
                signals = {
                    ticker: positive_scores[ticker] / total
                    for ticker in top_scores.index
                }

        logger.info(
            f"잔차 모멘텀 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"평균 점수={scores.loc[list(signals.keys())].mean():.4f}"
        )

        return signals
