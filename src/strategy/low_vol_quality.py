"""저변동성+품질 결합 전략 모듈.

저변동성 이상현상(Low Volatility Anomaly)과 퀄리티 팩터를 결합하여
안정적이면서도 재무적으로 우량한 종목을 선별하는 전략이다.

1단계: 일별 수익률 표준편차 기준 변동성 하위 종목 필터
2단계: ROE + GP/A 기반 품질 스코어 상위 종목 선택
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LowVolQualityStrategy(Strategy):
    """저변동성 + 품질 결합 전략.

    변동성이 낮으면서 ROE, GP/A가 높은 종목을 선별한다.

    Args:
        vol_period: 변동성 계산 기간 (기본 60거래일)
        vol_percentile: 변동성 하위 N% 선택 (기본 30)
        num_stocks: 최종 포트폴리오 종목 수 (기본 10)
        min_market_cap: 최소 시가총액 (기본 1000억원)
        min_volume: 최소 일 거래대금 (기본 1억원)
        roe_weight: ROE 가중치 (기본 0.5)
        gpa_weight: GP/A 가중치 (기본 0.5)
    """

    def __init__(
        self,
        vol_period: int = 60,
        vol_percentile: int = 30,
        num_stocks: int = 10,
        min_market_cap: int = 100_000_000_000,
        min_volume: int = 100_000_000,
        roe_weight: float = 0.5,
        gpa_weight: float = 0.5,
    ):
        self.vol_period = vol_period
        self.vol_percentile = vol_percentile
        self.num_stocks = num_stocks
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.roe_weight = roe_weight
        self.gpa_weight = gpa_weight

        logger.info(
            f"LowVolQualityStrategy 초기화: vol_period={vol_period}, "
            f"vol_percentile={vol_percentile}, num_stocks={num_stocks}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"LowVolQuality(vol{self.vol_percentile}pct, top{self.num_stocks})"

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

        logger.info(
            f"저변동성 유니버스 필터링: {original_count} -> {len(df)}개 종목"
        )
        return df

    def _calculate_volatility(
        self, prices: dict[str, pd.DataFrame], tickers: set[str]
    ) -> pd.Series:
        """종목별 변동성(일별 수익률 표준편차)을 계산한다.

        Args:
            prices: {ticker: OHLCV DataFrame} 딕셔너리
            tickers: 대상 종목코드 집합

        Returns:
            pd.Series (index=ticker, values=volatility)
        """
        volatilities = {}

        for ticker in tickers:
            if ticker not in prices:
                continue

            price_df = prices[ticker]
            if price_df.empty or "close" not in price_df.columns:
                continue

            closes = price_df["close"].tail(self.vol_period)
            if len(closes) < self.vol_period // 2:
                continue

            returns = closes.pct_change().dropna()
            if len(returns) < 10:
                continue

            volatilities[ticker] = float(returns.std())

        if not volatilities:
            return pd.Series(dtype=float)

        return pd.Series(volatilities, name="volatility")

    def _calculate_quality_score(self, fundamentals: pd.DataFrame) -> pd.Series:
        """ROE + GP/A 기반 품질 스코어를 계산한다.

        Args:
            fundamentals: 종목 기본 지표 DataFrame

        Returns:
            pd.Series (index=ticker, values=quality_score)
        """
        if fundamentals.empty or "ticker" not in fundamentals.columns:
            return pd.Series(dtype=float)

        df = fundamentals.copy()
        scores = pd.Series(0.0, index=df["ticker"].values)
        total_weight = 0.0

        # ROE 스코어 (높을수록 좋음)
        if "roe" in df.columns:
            roe = pd.Series(df["roe"].values, index=df["ticker"].values)
            valid_roe = roe.dropna()
            if len(valid_roe) > 0:
                roe_pctrank = valid_roe.rank(pct=True)
                scores.loc[roe_pctrank.index] += roe_pctrank * self.roe_weight
                total_weight += self.roe_weight
        elif "eps" in df.columns and "bps" in df.columns:
            # ROE 추정: EPS / BPS * 100
            bps = df["bps"].replace(0, np.nan)
            estimated_roe = df["eps"] / bps * 100
            roe = pd.Series(estimated_roe.values, index=df["ticker"].values)
            valid_roe = roe.dropna()
            if len(valid_roe) > 0:
                roe_pctrank = valid_roe.rank(pct=True)
                scores.loc[roe_pctrank.index] += roe_pctrank * self.roe_weight
                total_weight += self.roe_weight

        # GP/A 스코어 (높을수록 좋음)
        gpa_col = None
        if "gp_over_assets" in df.columns:
            gpa_col = "gp_over_assets"
        elif "gpa" in df.columns:
            gpa_col = "gpa"

        if gpa_col:
            gpa = pd.Series(df[gpa_col].values, index=df["ticker"].values)
            valid_gpa = gpa.dropna()
            if len(valid_gpa) > 0:
                gpa_pctrank = valid_gpa.rank(pct=True)
                scores.loc[gpa_pctrank.index] += gpa_pctrank * self.gpa_weight
                total_weight += self.gpa_weight

        if total_weight == 0:
            return pd.Series(dtype=float)

        scores = scores / total_weight
        return scores.dropna()

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        1. 변동성 하위 vol_percentile% 종목 필터
        2. 그 중 품질 스코어 상위 num_stocks개 선택

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {
                'fundamentals': DataFrame,
                'prices': dict[ticker, DataFrame],
                'quality': DataFrame (DART 퀄리티 데이터, optional),
            }

        Returns:
            종목코드: 비중 딕셔너리
        """
        fundamentals = data.get("fundamentals", pd.DataFrame())
        prices = data.get("prices", {})

        if fundamentals.empty:
            logger.warning(f"펀더멘탈 데이터 없음 ({date}), 빈 시그널 반환")
            return {}

        # 유니버스 필터링
        filtered = self._filter_universe(fundamentals.copy())
        if filtered.empty:
            logger.warning(f"필터링 후 종목 없음 ({date})")
            return {}

        # DART 퀄리티 데이터 merge
        quality_data = data.get("quality", pd.DataFrame())
        if not quality_data.empty and "ticker" in quality_data.columns:
            quality_cols = ["ticker", "roe", "gp_over_assets"]
            available = [c for c in quality_cols if c in quality_data.columns]
            if len(available) > 1:
                filtered = filtered.merge(
                    quality_data[available],
                    on="ticker",
                    how="left",
                    suffixes=("", "_dart"),
                )

        filtered_tickers = set(filtered["ticker"].values)

        # 변동성 계산
        if not prices:
            logger.warning(f"가격 데이터 없음 ({date}), 품질만으로 선정")
            # 가격 데이터 없으면 품질 스코어만으로 선정
            quality_scores = self._calculate_quality_score(filtered)
            if quality_scores.empty:
                return {}
            top = quality_scores.sort_values(ascending=False).head(self.num_stocks)
            weight = 1.0 / len(top)
            return {ticker: weight for ticker in top.index}

        volatility = self._calculate_volatility(prices, filtered_tickers)

        if volatility.empty:
            logger.warning(f"변동성 계산 실패 ({date})")
            return {}

        # 변동성 하위 vol_percentile% 종목 필터
        vol_threshold = volatility.quantile(self.vol_percentile / 100.0)
        low_vol_tickers = set(volatility[volatility <= vol_threshold].index)

        if not low_vol_tickers:
            logger.warning(f"저변동성 종목 없음 ({date})")
            return {}

        # 저변동성 종목 중 품질 스코어 계산
        low_vol_fundamentals = filtered[
            filtered["ticker"].isin(low_vol_tickers)
        ].copy()

        if low_vol_fundamentals.empty:
            logger.warning(f"저변동성 종목의 펀더멘탈 데이터 없음 ({date})")
            return {}

        quality_scores = self._calculate_quality_score(low_vol_fundamentals)

        if quality_scores.empty:
            logger.warning(f"품질 스코어 계산 실패 ({date})")
            return {}

        # 품질 상위 N개 선택
        top = quality_scores.sort_values(ascending=False).head(self.num_stocks)

        if top.empty:
            return {}

        # 동일 비중 할당
        weight = 1.0 / len(top)
        signals = {ticker: weight for ticker in top.index}

        logger.info(
            f"저변동성+품질 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"저변동성 풀={len(low_vol_tickers)}개, 개별 비중={weight:.4f}"
        )

        return signals
