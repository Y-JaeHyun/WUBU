"""모멘텀 팩터 전략 모듈.

과거 수익률(모멘텀)을 기반으로 상위 종목을 선별하여 포트폴리오를 구성하는 전략을 제공한다.
복합 모멘텀(여러 룩백 기간)과 직전 1개월 스킵(단기 반전 회피)을 지원한다.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_momentum_score(
    prices: pd.Series,
    lookback: int,
    skip: int,
) -> float:
    """단일 종목의 모멘텀 스코어를 계산한다.

    모멘텀 = (P[t - skip] / P[t - lookback]) - 1

    Args:
        prices: 종가 시계열 (DatetimeIndex, 거래일 기준)
        lookback: 전체 룩백 기간 (거래일 수)
        skip: 최근 스킵 기간 (거래일 수, 단기 반전 회피용)

    Returns:
        모멘텀 스코어 (float). 데이터 부족 시 NaN 반환.
    """
    if len(prices) < lookback:
        return float("nan")

    if skip >= lookback:
        return float("nan")

    # 가장 최근 날짜 기준으로 lookback, skip 지점의 가격 사용
    price_end = prices.iloc[-1 - skip] if skip > 0 else prices.iloc[-1]
    price_start = prices.iloc[-lookback]

    if price_start <= 0 or np.isnan(price_start) or np.isnan(price_end):
        return float("nan")

    return float(price_end / price_start - 1)


class MomentumStrategy(Strategy):
    """모멘텀 팩터 전략.

    과거 수익률이 높은 종목에 투자하는 모멘텀 전략이다.
    복수의 룩백 기간을 지원하며, 단기 반전 효과를 회피하기 위해
    직전 1개월을 스킵할 수 있다.

    Args:
        lookback_months: 룩백 기간 리스트 (월 단위, 기본 [12])
        skip_month: 직전 1개월 스킵 여부 (기본 True)
        skip_days: 스킵할 거래일 수 (기본 21)
        num_stocks: 포트폴리오 종목 수 (기본 20)
        weighting: 가중 방식 - "equal" 또는 "score" (기본 "equal")
        min_market_cap: 최소 시가총액 (기본 1000억원)
        min_volume: 최소 일 거래대금 (기본 1억원)
        residual: 잔차 모멘텀 사용 여부 (기본 False).
            활성화 시 KOSPI 수익률 대비 베타를 회귀분석으로 제거한
            잔차 수익률을 모멘텀으로 사용. data에 'index_prices' 필요.
        high_52w_factor: 52주 고점 거리 팩터 결합 여부 (기본 False).
            현재가/52주고가 비율을 모멘텀 스코어에 결합.
        momentum_cap: 모멘텀 절대 상한 (기본 3.0 = 300%).
            극단적 모멘텀이 z-score를 왜곡하거나 리버설 위험을 높이는 것을 방지.
            0 이하이면 캡 비활성화.
    """

    def __init__(
        self,
        lookback_months: Optional[list[int]] = None,
        skip_month: bool = True,
        skip_days: int = 21,
        num_stocks: int = 20,
        weighting: str = "equal",
        min_market_cap: int = 100_000_000_000,
        min_volume: int = 100_000_000,
        residual: bool = False,
        high_52w_factor: bool = False,
        momentum_cap: float = 3.0,
    ):
        self.lookback_months = lookback_months or [12]
        self.skip_month = skip_month
        self.skip_days = skip_days if skip_month else 0
        self.num_stocks = num_stocks
        self.weighting = weighting.lower()
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.residual = residual
        self.high_52w_factor = high_52w_factor
        self.momentum_cap = momentum_cap

        if self.weighting not in ("equal", "score"):
            raise ValueError(
                f"지원하지 않는 가중 방식: {weighting}. 'equal' 또는 'score' 중 선택하세요."
            )

        logger.info(
            f"MomentumStrategy 초기화: lookback_months={self.lookback_months}, "
            f"skip_month={skip_month}, num_stocks={num_stocks}, "
            f"weighting={self.weighting}, residual={residual}, "
            f"high_52w_factor={high_52w_factor}, momentum_cap={momentum_cap}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        lb_str = "/".join(str(m) for m in self.lookback_months)
        skip_str = "skip1m" if self.skip_month else "noskip"
        return f"Momentum({lb_str}M, {skip_str}, top{self.num_stocks})"

    def _filter_universe(
        self,
        fundamentals: pd.DataFrame,
        prices: dict[str, pd.DataFrame],
    ) -> list[str]:
        """유니버스 필터링을 수행한다.

        조건:
        - 시가총액 >= min_market_cap
        - 일 거래대금 >= min_volume
        - 상장 12개월 이상 (가격 데이터 252일 이상)

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

        # 거래대금 필터
        if "volume" in df.columns and "close" in df.columns:
            trade_value = df["volume"] * df["close"]
            df = df[trade_value >= self.min_volume]

        tickers = df["ticker"].tolist() if "ticker" in df.columns else []

        # 상장 12개월(약 252거래일) 이상 필터
        min_trading_days = 252
        filtered = []
        for ticker in tickers:
            if ticker in prices and len(prices[ticker]) >= min_trading_days:
                filtered.append(ticker)

        logger.info(
            f"유니버스 필터링: 전체 {len(fundamentals)} -> "
            f"기본필터 {len(tickers)} -> 상장기간필터 {len(filtered)}개"
        )
        return filtered

    def _compute_residual_momentum(
        self,
        ticker: str,
        prices: dict[str, pd.DataFrame],
        index_prices: pd.Series,
    ) -> float:
        """잔차 모멘텀을 계산한다.

        종목 수익률에서 시장 베타를 회귀분석으로 제거한 잔차 수익률의
        누적값을 모멘텀으로 사용한다.

        Args:
            ticker: 종목코드
            prices: 가격 데이터 딕셔너리
            index_prices: 시장 지수 종가 시계열

        Returns:
            잔차 모멘텀 스코어 (float). 계산 불가 시 NaN.
        """
        if ticker not in prices or prices[ticker].empty:
            return float("nan")
        if index_prices is None or index_prices.empty:
            return float("nan")

        close = prices[ticker]["close"]
        lookback_days = max(m * 21 for m in self.lookback_months)

        if len(close) < lookback_days or len(index_prices) < lookback_days:
            return float("nan")

        # 최근 lookback_days 기간의 수익률
        stock_ret = close.iloc[-lookback_days:].pct_change().dropna()
        idx_ret = index_prices.iloc[-lookback_days:].pct_change().dropna()

        # 공통 날짜만 사용
        common_idx = stock_ret.index.intersection(idx_ret.index)
        if len(common_idx) < 30:
            return float("nan")

        y = stock_ret.loc[common_idx].values
        x = idx_ret.loc[common_idx].values

        # 선형 회귀: stock_ret = alpha + beta * index_ret + residual
        coeffs = np.polyfit(x, y, 1)
        residuals = y - (coeffs[0] * x + coeffs[1])

        # 잔차 수익률의 누적합 = 잔차 모멘텀
        return float(np.sum(residuals))

    def _compute_52w_high_ratio(
        self,
        ticker: str,
        prices: dict[str, pd.DataFrame],
    ) -> float:
        """현재가 / 52주 고가 비율을 계산한다.

        Args:
            ticker: 종목코드
            prices: 가격 데이터 딕셔너리

        Returns:
            비율 (0~1, 1에 가까울수록 52주 고점 근접). 계산 불가 시 NaN.
        """
        if ticker not in prices or prices[ticker].empty:
            return float("nan")

        close = prices[ticker]["close"]
        if len(close) < 252:
            return float("nan")

        high_52w = close.iloc[-252:].max()
        current = close.iloc[-1]

        if high_52w <= 0 or np.isnan(high_52w):
            return float("nan")

        return float(current / high_52w)

    def _compute_scores(
        self,
        tickers: list[str],
        prices: dict[str, pd.DataFrame],
        index_prices: Optional[pd.Series] = None,
    ) -> pd.Series:
        """종목별 모멘텀 스코어를 계산한다.

        복수의 룩백 기간이 있을 경우 각 기간별 스코어를 동일 가중 합산한다.
        residual=True 시 잔차 모멘텀을, high_52w_factor=True 시 52주 고점 거리를 결합한다.

        Args:
            tickers: 대상 종목 리스트
            prices: {ticker: DataFrame(OHLCV)} 가격 캐시
            index_prices: 시장 지수 종가 시계열 (잔차 모멘텀용)

        Returns:
            pd.Series (index=ticker, values=momentum_score)
        """
        scores: dict[str, float] = {}

        for ticker in tickers:
            if ticker not in prices or prices[ticker].empty:
                continue

            close_prices = prices[ticker]["close"]
            if close_prices.empty:
                continue

            # 잔차 모멘텀 모드
            if self.residual and index_prices is not None and not index_prices.empty:
                residual_score = self._compute_residual_momentum(
                    ticker, prices, index_prices
                )
                if not np.isnan(residual_score):
                    scores[ticker] = residual_score
                continue

            period_scores: list[float] = []
            for months in self.lookback_months:
                lookback_days = months * 21  # 월 → 거래일 환산
                score = calculate_momentum_score(
                    close_prices,
                    lookback=lookback_days,
                    skip=self.skip_days,
                )
                if not np.isnan(score):
                    period_scores.append(score)

            if period_scores:
                # 복합 모멘텀: 동일 가중 평균
                scores[ticker] = float(np.mean(period_scores))

        score_series = pd.Series(scores, name="momentum_score")

        if score_series.empty:
            return score_series

        # 절대 모멘텀 캡: 극단적 상승 모멘텀 제한 (리버설 위험 방지)
        if self.momentum_cap > 0:
            capped_mask = score_series > self.momentum_cap
            if capped_mask.any():
                capped_count = int(capped_mask.sum())
                logger.info(
                    f"모멘텀 캡 적용: {capped_count}개 종목 "
                    f"(>{self.momentum_cap:.0%}) → {self.momentum_cap:.0%}로 제한"
                )
                score_series = score_series.clip(upper=self.momentum_cap)

        # 52주 고점 거리 팩터 결합
        if self.high_52w_factor:
            high_ratios: dict[str, float] = {}
            for ticker in score_series.index:
                ratio = self._compute_52w_high_ratio(ticker, prices)
                if not np.isnan(ratio):
                    high_ratios[ticker] = ratio

            if high_ratios:
                ratio_series = pd.Series(high_ratios)
                # 공통 종목만 대상
                common = score_series.index.intersection(ratio_series.index)
                if not common.empty:
                    # Z-Score 정규화 후 결합 (모멘텀 70% + 52주고점 30%)
                    mom_z = self._zscore(score_series.loc[common])
                    ratio_z = self._zscore(ratio_series.loc[common])
                    score_series = mom_z * 0.7 + ratio_z * 0.3

        # Winsorizing: 상/하위 1% 클리핑
        lower = score_series.quantile(0.01)
        upper = score_series.quantile(0.99)
        score_series = score_series.clip(lower=lower, upper=upper)

        return score_series

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """시리즈를 Z-Score로 표준화한다."""
        std = series.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        모멘텀 스코어 상위 N개 종목을 선별하고 가중치를 부여한다.

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

        # 모멘텀 스코어 계산 (잔차 모멘텀용 index_prices 전달)
        index_prices = data.get("index_prices", pd.Series(dtype=float))
        scores = self._compute_scores(universe, prices, index_prices)

        if scores.empty:
            logger.warning(f"모멘텀 스코어 계산 실패 ({date})")
            return {}

        # 상위 N개 선정 (내림차순)
        top_scores = scores.sort_values(ascending=False).head(self.num_stocks)

        if top_scores.empty:
            return {}

        # 가중치 부여
        if self.weighting == "equal":
            weight = 1.0 / len(top_scores)
            signals = {ticker: weight for ticker in top_scores.index}
        elif self.weighting == "score":
            # 스코어 비례 가중 (음수 스코어는 제외)
            positive_scores = top_scores[top_scores > 0]
            if positive_scores.empty:
                logger.warning(f"양수 모멘텀 종목 없음 ({date})")
                return {}
            total_score = positive_scores.sum()
            if total_score <= 0:
                return {}
            signals = {
                ticker: float(score / total_score)
                for ticker, score in positive_scores.items()
            }
        else:
            signals = {}

        logger.info(
            f"모멘텀 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"스코어 범위=[{top_scores.min():.4f}, {top_scores.max():.4f}]"
        )

        return signals

    def get_scores(self, data: dict) -> pd.Series:
        """외부에서 모멘텀 스코어에 접근할 수 있도록 제공한다.

        MultiFactorStrategy 등에서 팩터 결합 시 사용한다.

        Args:
            data: {'fundamentals': DataFrame, 'prices': dict[ticker, DataFrame]} 형태

        Returns:
            pd.Series (index=ticker, values=momentum_score)
        """
        fundamentals = data.get("fundamentals", pd.DataFrame())
        prices = data.get("prices", {})

        universe = self._filter_universe(fundamentals, prices)
        if not universe:
            return pd.Series(dtype=float)

        index_prices = data.get("index_prices", pd.Series(dtype=float))
        return self._compute_scores(universe, prices, index_prices)
