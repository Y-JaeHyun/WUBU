"""Enhanced ETF Rotation 전략 모듈.

기존 ETF 로테이션 전략을 개선한 버전으로, 다음을 추가한다:
1. 복합 모멘텀 스코어 (단기+중기+장기 가중 합산)
2. 변동성 가중 (inverse volatility weighting)
3. 시장 레짐 필터 (KOSPI 200일 이평선 기반 현금 보호)
4. 추세 확인 필터 (단기 MA > 장기 MA 확인)

데이터 소스:
- pykrx ETF 가격 데이터 (etf_collector.get_etf_price)
- KOSPI 지수 (index_collector or etf_prices['069500'] as proxy)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.strategy.etf_rotation import (
    DEFAULT_ETF_UNIVERSE,
    DEFAULT_SAFE_ASSET,
    ETF_SECTOR_MAP,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedETFRotationStrategy(Strategy):
    """Enhanced ETF 로테이션 전략.

    기존 ETFRotationStrategy 대비 개선 사항:
    - 복합 모멘텀: 1M/3M/6M/12M 모멘텀의 가중 평균 스코어
    - 역변동성 가중: 변동성 낮은 ETF에 더 많은 비중
    - 마켓 레짐 필터: KOSPI가 200MA 아래이면 채권 비중 증가
    - 추세 확인: 20MA > 60MA인 ETF만 선정 (추세 전환 확인)
    - 최대 하락 필터: 최근 고점 대비 N% 이상 하락 ETF 제외

    Args:
        num_etfs: 선정할 ETF 수 (기본 3)
        safe_asset: 안전자산 ETF 종목코드
        etf_universe: ETF 유니버스 딕셔너리
        momentum_weights: 모멘텀 기간별 가중치 {거래일수: 가중치}
        use_vol_weight: 역변동성 가중 사용 여부 (기본 True)
        use_market_filter: 마켓 레짐 필터 사용 여부 (기본 True)
        use_trend_filter: 추세 확인 필터 사용 여부 (기본 True)
        market_proxy: 시장 대표 ETF 종목코드 (기본 '069500' KODEX 200)
        market_ma_period: 시장 레짐 판단용 MA 기간 (기본 200일)
        trend_short_ma: 추세 확인 단기 MA (기본 20일)
        trend_long_ma: 추세 확인 장기 MA (기본 60일)
        max_drawdown_filter: 최대 하락 필터 임계값 (기본 0.15 = 15%)
        vol_lookback: 변동성 계산 기간 (기본 60일)
        max_same_sector: 같은 섹터 최대 수 (기본 1)
        abs_momentum: 절대모멘텀 적용 여부 (기본 True)
        cash_ratio_risk_off: RISK_OFF 시 현금/채권 비율 (기본 0.5)
        momentum_cap: 모멘텀 캡 (기본 3.0)
    """

    def __init__(
        self,
        num_etfs: int = 3,
        safe_asset: str = DEFAULT_SAFE_ASSET,
        etf_universe: Optional[dict[str, str]] = None,
        momentum_weights: Optional[dict[int, float]] = None,
        use_vol_weight: bool = True,
        use_market_filter: bool = True,
        use_trend_filter: bool = True,
        market_proxy: str = "069500",
        market_ma_period: int = 200,
        trend_short_ma: int = 20,
        trend_long_ma: int = 60,
        max_drawdown_filter: float = 0.15,
        vol_lookback: int = 60,
        max_same_sector: int = 1,
        abs_momentum: bool = True,
        cash_ratio_risk_off: float = 0.5,
        momentum_cap: float = 3.0,
    ):
        self.num_etfs = num_etfs
        self.safe_asset = safe_asset
        self.etf_universe = etf_universe or DEFAULT_ETF_UNIVERSE.copy()
        self.use_vol_weight = use_vol_weight
        self.use_market_filter = use_market_filter
        self.use_trend_filter = use_trend_filter
        self.market_proxy = market_proxy
        self.market_ma_period = market_ma_period
        self.trend_short_ma = trend_short_ma
        self.trend_long_ma = trend_long_ma
        self.max_drawdown_filter = max_drawdown_filter
        self.vol_lookback = vol_lookback
        self.max_same_sector = max_same_sector
        self.abs_momentum = abs_momentum
        self.cash_ratio_risk_off = cash_ratio_risk_off
        self.momentum_cap = momentum_cap

        # 복합 모멘텀 가중치 (기본: 1M=20%, 3M=30%, 6M=30%, 12M=20%)
        self.momentum_weights = momentum_weights or {
            21: 0.20,   # 1개월
            63: 0.30,   # 3개월
            126: 0.30,  # 6개월
            252: 0.20,  # 12개월
        }

        # 진단 정보
        self.last_diagnostics: dict = {}

        logger.info(
            f"EnhancedETFRotationStrategy 초기화: "
            f"num_etfs={num_etfs}, vol_weight={use_vol_weight}, "
            f"market_filter={use_market_filter}, trend_filter={use_trend_filter}, "
            f"momentum_weights={self.momentum_weights}, "
            f"universe={len(self.etf_universe)}개"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        features = []
        if self.use_vol_weight:
            features.append("vol")
        if self.use_market_filter:
            features.append("mkt")
        if self.use_trend_filter:
            features.append("trd")
        feat_str = "+".join(features) if features else "base"
        return f"EnhancedETF(top{self.num_etfs},{feat_str})"

    def _calculate_composite_momentum(
        self,
        etf_prices: dict[str, pd.DataFrame],
    ) -> pd.Series:
        """복합 모멘텀 스코어를 계산한다.

        여러 기간의 모멘텀을 가중 평균하여 하나의 스코어로 합산한다.
        개별 모멘텀은 캡 적용 후 z-score 정규화하여 합산한다.

        Args:
            etf_prices: {ticker: OHLCV DataFrame} 딕셔너리

        Returns:
            pd.Series (index=ticker, values=composite_momentum_score)
        """
        all_momentums: dict[int, dict[str, float]] = {}

        for lookback, weight in self.momentum_weights.items():
            momentums = {}
            for ticker in self.etf_universe:
                if ticker == self.safe_asset:
                    continue
                if ticker not in etf_prices:
                    continue

                price_df = etf_prices[ticker]
                if price_df.empty or "close" not in price_df.columns:
                    continue

                closes = price_df["close"]
                if len(closes) < lookback:
                    continue

                price_current = float(closes.iloc[-1])
                price_past = float(closes.iloc[-lookback])

                if price_past <= 0 or np.isnan(price_past) or np.isnan(price_current):
                    continue

                mom = price_current / price_past - 1

                # 모멘텀 캡
                if self.momentum_cap > 0 and mom > self.momentum_cap:
                    mom = self.momentum_cap

                momentums[ticker] = mom

            all_momentums[lookback] = momentums

        # 각 기간별 z-score 정규화 후 가중 합산
        composite_scores: dict[str, float] = {}

        # 모든 기간에서 한 번이라도 등장한 종목 목록
        all_tickers = set()
        for moms in all_momentums.values():
            all_tickers.update(moms.keys())

        for ticker in all_tickers:
            score = 0.0
            total_weight = 0.0

            for lookback, weight in self.momentum_weights.items():
                moms = all_momentums.get(lookback, {})
                if ticker not in moms:
                    continue

                # z-score 정규화
                values = list(moms.values())
                if len(values) < 2:
                    z = 0.0
                else:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    if std_val > 0:
                        z = (moms[ticker] - mean_val) / std_val
                    else:
                        z = 0.0

                score += z * weight
                total_weight += weight

            if total_weight > 0:
                composite_scores[ticker] = score / total_weight

        if not composite_scores:
            return pd.Series(dtype=float)

        return pd.Series(composite_scores, name="composite_momentum")

    def _calculate_raw_momentum(
        self,
        etf_prices: dict[str, pd.DataFrame],
        lookback: int = 63,
    ) -> dict[str, float]:
        """단순 모멘텀 (절대 모멘텀 체크용)."""
        momentums = {}
        for ticker in self.etf_universe:
            if ticker == self.safe_asset:
                continue
            if ticker not in etf_prices:
                continue
            price_df = etf_prices[ticker]
            if price_df.empty or "close" not in price_df.columns:
                continue
            closes = price_df["close"]
            if len(closes) < lookback:
                continue
            price_current = float(closes.iloc[-1])
            price_past = float(closes.iloc[-lookback])
            if price_past <= 0 or np.isnan(price_past) or np.isnan(price_current):
                continue
            momentums[ticker] = price_current / price_past - 1
        return momentums

    def _check_market_regime(
        self, etf_prices: dict[str, pd.DataFrame],
    ) -> str:
        """시장 레짐을 판단한다.

        KOSPI 대표 ETF(KODEX 200)의 종가가 200일 이동평균선 위인지 확인한다.

        Returns:
            'RISK_ON' or 'RISK_OFF'
        """
        if self.market_proxy not in etf_prices:
            return "RISK_ON"

        price_df = etf_prices[self.market_proxy]
        if price_df.empty or "close" not in price_df.columns:
            return "RISK_ON"

        closes = price_df["close"]
        if len(closes) < self.market_ma_period:
            return "RISK_ON"

        ma = closes.rolling(window=self.market_ma_period).mean()
        current_price = float(closes.iloc[-1])
        current_ma = float(ma.iloc[-1])

        if np.isnan(current_ma) or current_ma <= 0:
            return "RISK_ON"

        regime = "RISK_ON" if current_price > current_ma else "RISK_OFF"
        logger.info(
            f"시장 레짐: price={current_price:.0f}, "
            f"MA{self.market_ma_period}={current_ma:.0f}, regime={regime}"
        )
        return regime

    def _check_trend(
        self, etf_prices: dict[str, pd.DataFrame], ticker: str,
    ) -> bool:
        """개별 ETF의 추세를 확인한다.

        단기 이동평균(20일) > 장기 이동평균(60일)이면 상승 추세.

        Returns:
            True if uptrend, False otherwise
        """
        if ticker not in etf_prices:
            return True  # 데이터 없으면 통과시킴

        price_df = etf_prices[ticker]
        if price_df.empty or "close" not in price_df.columns:
            return True

        closes = price_df["close"]
        if len(closes) < self.trend_long_ma:
            return True

        short_ma = float(closes.rolling(window=self.trend_short_ma).mean().iloc[-1])
        long_ma = float(closes.rolling(window=self.trend_long_ma).mean().iloc[-1])

        if np.isnan(short_ma) or np.isnan(long_ma) or long_ma <= 0:
            return True

        return short_ma > long_ma

    def _check_drawdown(
        self, etf_prices: dict[str, pd.DataFrame], ticker: str,
    ) -> bool:
        """최근 고점 대비 하락 비율을 확인한다.

        max_drawdown_filter 이상 하락한 ETF는 제외한다.

        Returns:
            True if OK (drawdown within limit), False if excessive drawdown
        """
        if self.max_drawdown_filter <= 0:
            return True

        if ticker not in etf_prices:
            return True

        price_df = etf_prices[ticker]
        if price_df.empty or "close" not in price_df.columns:
            return True

        closes = price_df["close"].tail(252)  # 최근 1년 기준
        if len(closes) < 20:
            return True

        peak = float(closes.max())
        current = float(closes.iloc[-1])

        if peak <= 0:
            return True

        drawdown = (current - peak) / peak
        return drawdown > -self.max_drawdown_filter

    def _calculate_inverse_vol_weights(
        self, etf_prices: dict[str, pd.DataFrame], tickers: list[str],
    ) -> dict[str, float]:
        """역변동성 가중 비중을 계산한다."""
        volatilities = {}

        for ticker in tickers:
            if ticker not in etf_prices:
                continue
            price_df = etf_prices[ticker]
            if price_df.empty or "close" not in price_df.columns:
                continue

            closes = price_df["close"].tail(self.vol_lookback)
            returns = closes.pct_change().dropna()

            if len(returns) < 10:
                continue

            vol = float(returns.std())
            if vol > 0:
                volatilities[ticker] = vol

        if not volatilities:
            weight = 1.0 / len(tickers) if tickers else 0.0
            return {t: weight for t in tickers}

        inv_vols = {t: 1.0 / v for t, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol <= 0:
            weight = 1.0 / len(tickers)
            return {t: weight for t in tickers}

        return {t: iv / total_inv_vol for t, iv in inv_vols.items()}

    def _apply_sector_filter(
        self, scored_tickers: list[str], num_etfs: int,
    ) -> list[str]:
        """섹터 집중도 제한 필터."""
        if self.max_same_sector <= 0:
            return scored_tickers[:num_etfs]

        selected = []
        sector_counts: dict[str, int] = {}

        for ticker in scored_tickers:
            if len(selected) >= num_etfs:
                break

            sector = ETF_SECTOR_MAP.get(ticker, "기타")
            current_count = sector_counts.get(sector, 0)

            if current_count >= self.max_same_sector:
                continue

            selected.append(ticker)
            sector_counts[sector] = current_count + 1

        return selected

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        1. 복합 모멘텀 스코어 계산 (1M/3M/6M/12M 가중 합산)
        2. 추세 필터 적용 (단기MA > 장기MA)
        3. 최대 하락 필터 적용
        4. 섹터 집중도 제한
        5. 마켓 레짐 필터 (RISK_OFF 시 현금 비중 증가)
        6. 절대 모멘텀 필터
        7. 비중 할당 (역변동성 가중 또는 동일 가중)

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {'etf_prices': dict[ticker, DataFrame]} 또는 {'prices': ...}

        Returns:
            종목코드: 비중 딕셔너리
        """
        etf_prices = data.get("etf_prices", data.get("prices", {}))

        if not etf_prices:
            logger.warning(f"ETF 가격 데이터 없음 ({date})")
            self.last_diagnostics = {"status": "DATA_UNAVAILABLE"}
            return {}

        # 1. 복합 모멘텀 스코어 계산
        composite_momentum = self._calculate_composite_momentum(etf_prices)
        if composite_momentum.empty:
            logger.warning(f"복합 모멘텀 계산 실패 ({date})")
            self.last_diagnostics = {"status": "MOMENTUM_FAIL"}
            return {}

        # 2. 순위 정렬 (높은 스코어 우선)
        ranked = composite_momentum.sort_values(ascending=False)

        # 3. 추세 필터 적용
        if self.use_trend_filter:
            trend_pass = [t for t in ranked.index if self._check_trend(etf_prices, t)]
            ranked = ranked[ranked.index.isin(trend_pass)]

        # 4. 최대 하락 필터 적용
        dd_pass = [t for t in ranked.index if self._check_drawdown(etf_prices, t)]
        ranked = ranked[ranked.index.isin(dd_pass)]

        if ranked.empty:
            logger.info(f"필터 후 남은 ETF 없음 ({date}), 안전자산 100%")
            self.last_diagnostics = {"status": "ALL_FILTERED"}
            return {self.safe_asset: 1.0}

        # 5. 섹터 집중도 제한
        selected_tickers = self._apply_sector_filter(
            ranked.index.tolist(), self.num_etfs
        )

        # 6. 마켓 레짐 필터
        market_regime = "RISK_ON"
        if self.use_market_filter:
            market_regime = self._check_market_regime(etf_prices)

        # 7. 절대 모멘텀 필터
        raw_momentum = self._calculate_raw_momentum(etf_prices, lookback=63)
        final_risky = []
        safe_count = 0

        for ticker in selected_tickers:
            raw_mom = raw_momentum.get(ticker, 0.0)
            if self.abs_momentum and raw_mom <= 0:
                safe_count += 1
            else:
                final_risky.append(ticker)

        total_slots = len(final_risky) + safe_count
        if total_slots == 0:
            return {self.safe_asset: 1.0}

        # 8. 비중 할당
        if self.use_vol_weight and final_risky:
            risky_weights = self._calculate_inverse_vol_weights(
                etf_prices, final_risky
            )
            risky_portion = len(final_risky) / total_slots
            signals = {t: w * risky_portion for t, w in risky_weights.items()}
        else:
            slot_weight = 1.0 / total_slots
            signals = {t: slot_weight for t in final_risky}

        # 안전자산 비중 추가
        if safe_count > 0:
            safe_portion = safe_count / total_slots
            signals[self.safe_asset] = signals.get(self.safe_asset, 0.0) + safe_portion

        # 9. 마켓 레짐에 따른 비중 조절
        if market_regime == "RISK_OFF" and self.use_market_filter:
            cash_ratio = self.cash_ratio_risk_off
            adjusted = {}
            for ticker, weight in signals.items():
                if ticker == self.safe_asset:
                    adjusted[ticker] = weight
                else:
                    adjusted[ticker] = weight * (1 - cash_ratio)

            # RISK_OFF로 줄어든 비중을 안전자산에 추가
            total_risky_reduced = sum(
                w * cash_ratio for t, w in signals.items() if t != self.safe_asset
            )
            adjusted[self.safe_asset] = adjusted.get(self.safe_asset, 0.0) + total_risky_reduced
            signals = adjusted

        # 진단 정보
        self.last_diagnostics = {
            "status": "OK",
            "date": date,
            "market_regime": market_regime,
            "composite_scores": {t: float(v) for t, v in composite_momentum.items()},
            "selected": selected_tickers,
            "final_risky": final_risky,
            "safe_count": safe_count,
            "signals": {t: round(w, 4) for t, w in signals.items()},
        }

        logger.info(
            f"Enhanced ETF 시그널 ({date}): "
            f"{len(signals)}개 ETF, regime={market_regime}, "
            f"risky={len(final_risky)}, safe={safe_count}"
        )

        return signals
