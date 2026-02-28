"""Cross-Asset Momentum (교차 자산 모멘텀) 전략 모듈.

서로 상관관계가 낮은 다양한 자산군(주식, 채권, 금, 달러) ETF를
대상으로 모멘텀 기반 로테이션을 수행한다.

핵심 아이디어:
- 한국 주식 하락 시 달러/금/채권이 상승하는 역상관관계 활용
- 절대 모멘텀으로 하락 자산군 회피
- 역변동성 가중으로 리스크 평탄화
- 시장 충격 시 안전자산 자동 전환

ETF 유니버스 (자산군별):
- 국내주식: KODEX 200 (069500)
- 미국주식: TIGER 미국S&P500 (371460), TIGER 미국나스닥100 (133690)
- 반도체: KODEX 반도체 (091160)
- 금: KODEX 골드선물 (132030)
- 채권: KODEX 단기채권 (439870)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.strategy.etf_rotation import ETF_SECTOR_MAP
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 교차 자산 유니버스: 상관관계가 낮은 자산군으로 구성
CROSS_ASSET_UNIVERSE: dict[str, str] = {
    "069500": "KODEX 200",             # 국내주식
    "371460": "TIGER 미국S&P500",       # 미국주식
    "133690": "TIGER 미국나스닥100",     # 미국기술주
    "091160": "KODEX 반도체",           # 섹터(반도체)
    "132030": "KODEX 골드선물(H)",       # 원자재(금)
    "439870": "KODEX 단기채권",         # 채권(안전자산)
}

# 자산군 카테고리
ASSET_CLASS_MAP: dict[str, str] = {
    "069500": "equity_kr",
    "371460": "equity_us",
    "133690": "equity_us",
    "091160": "equity_kr",
    "132030": "commodity",
    "439870": "bond",
}

DEFAULT_SAFE_ASSET = "439870"


class CrossAssetMomentumStrategy(Strategy):
    """교차 자산 모멘텀 전략.

    서로 상관관계가 낮은 자산군 ETF에 모멘텀 기반으로 투자한다.
    하락 자산군은 안전자산으로 자동 전환하여 MDD를 방어한다.

    독특한 특징:
    - 자산군 분산: 같은 자산군에서 최대 1개만 선정
    - SMA 크로스오버: 각 ETF의 SMA(20) > SMA(60)일 때만 투자
    - 상관관계 기반 선정: 포트폴리오 내 자산 간 상관관계가 낮도록 조정
    - 변동성 패리티: 변동성 역비례로 비중 배분

    Args:
        num_assets: 선정할 자산 수 (기본 3)
        safe_asset: 안전자산 종목코드
        etf_universe: ETF 유니버스
        lookback_short: 단기 모멘텀 (기본 63일 = 3M)
        lookback_long: 장기 모멘텀 (기본 252일 = 12M)
        short_weight: 단기 모멘텀 가중치 (기본 0.6)
        vol_lookback: 변동성 계산 기간 (기본 60일)
        use_trend_filter: 추세 필터 사용 여부 (기본 True)
        use_correlation_filter: 상관관계 필터 (기본 True)
        max_per_asset_class: 자산군당 최대 선정 수 (기본 1)
        abs_momentum_threshold: 절대 모멘텀 기준 (기본 0.0)
        max_correlation: 포트폴리오 내 최대 상관관계 (기본 0.7)
    """

    def __init__(
        self,
        num_assets: int = 3,
        safe_asset: str = DEFAULT_SAFE_ASSET,
        etf_universe: Optional[dict[str, str]] = None,
        lookback_short: int = 63,
        lookback_long: int = 252,
        short_weight: float = 0.6,
        vol_lookback: int = 60,
        use_trend_filter: bool = True,
        use_correlation_filter: bool = True,
        max_per_asset_class: int = 1,
        abs_momentum_threshold: float = 0.0,
        max_correlation: float = 0.7,
    ):
        self.num_assets = num_assets
        self.safe_asset = safe_asset
        self.etf_universe = etf_universe or CROSS_ASSET_UNIVERSE.copy()
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.short_weight = short_weight
        self.long_weight = 1.0 - short_weight
        self.vol_lookback = vol_lookback
        self.use_trend_filter = use_trend_filter
        self.use_correlation_filter = use_correlation_filter
        self.max_per_asset_class = max_per_asset_class
        self.abs_momentum_threshold = abs_momentum_threshold
        self.max_correlation = max_correlation

        self.last_diagnostics: dict = {}

        logger.info(
            f"CrossAssetMomentumStrategy 초기화: "
            f"num_assets={num_assets}, "
            f"lookback=({lookback_short},{lookback_long}), "
            f"short_weight={short_weight}, "
            f"trend_filter={use_trend_filter}, "
            f"corr_filter={use_correlation_filter}, "
            f"universe={len(self.etf_universe)}개"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"CrossAssetMom(top{self.num_assets})"

    def _get_returns(
        self,
        etf_prices: dict[str, pd.DataFrame],
        ticker: str,
        lookback: int,
    ) -> Optional[float]:
        """특정 기간의 수익률을 계산한다."""
        if ticker not in etf_prices:
            return None
        df = etf_prices[ticker]
        if df.empty or "close" not in df.columns:
            return None
        closes = df["close"]
        if len(closes) < lookback:
            return None
        current = float(closes.iloc[-1])
        past = float(closes.iloc[-lookback])
        if past <= 0 or np.isnan(past) or np.isnan(current):
            return None
        return current / past - 1

    def _calculate_dual_momentum_score(
        self,
        etf_prices: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """듀얼 모멘텀 스코어를 계산한다.

        단기 모멘텀과 장기 모멘텀의 가중 합산 스코어를 반환한다.
        """
        scores = {}

        for ticker in self.etf_universe:
            if ticker == self.safe_asset:
                continue

            short_ret = self._get_returns(etf_prices, ticker, self.lookback_short)
            long_ret = self._get_returns(etf_prices, ticker, self.lookback_long)

            if short_ret is None and long_ret is None:
                continue

            # 사용 가능한 것만으로 스코어 계산
            score = 0.0
            total_w = 0.0

            if short_ret is not None:
                score += short_ret * self.short_weight
                total_w += self.short_weight
            if long_ret is not None:
                score += long_ret * self.long_weight
                total_w += self.long_weight

            if total_w > 0:
                scores[ticker] = score / total_w

        return scores

    def _check_sma_crossover(
        self, etf_prices: dict[str, pd.DataFrame], ticker: str,
    ) -> bool:
        """SMA 크로스오버 확인 (SMA20 > SMA60)."""
        if ticker not in etf_prices:
            return True
        df = etf_prices[ticker]
        if df.empty or "close" not in df.columns:
            return True
        closes = df["close"]
        if len(closes) < 60:
            return True

        sma20 = float(closes.rolling(20).mean().iloc[-1])
        sma60 = float(closes.rolling(60).mean().iloc[-1])

        if np.isnan(sma20) or np.isnan(sma60) or sma60 <= 0:
            return True

        return sma20 > sma60

    def _apply_asset_class_filter(
        self, ranked_tickers: list[str],
    ) -> list[str]:
        """자산군당 최대 선정 수를 적용한다."""
        if self.max_per_asset_class <= 0:
            return ranked_tickers[:self.num_assets]

        selected = []
        class_counts: dict[str, int] = {}

        for ticker in ranked_tickers:
            if len(selected) >= self.num_assets:
                break

            asset_class = ASSET_CLASS_MAP.get(ticker, "other")
            count = class_counts.get(asset_class, 0)

            if count >= self.max_per_asset_class:
                continue

            selected.append(ticker)
            class_counts[asset_class] = count + 1

        return selected

    def _apply_correlation_filter(
        self,
        etf_prices: dict[str, pd.DataFrame],
        candidates: list[str],
    ) -> list[str]:
        """포트폴리오 내 상관관계가 높은 자산을 제거한다.

        그리디 방식: 모멘텀 순위가 높은 것부터 추가하면서,
        기존 선정 자산과 상관관계가 max_correlation 이상이면 스킵.
        """
        if not self.use_correlation_filter:
            return candidates

        # 일간 수익률 매트릭스 구성
        returns_data = {}
        for ticker in candidates:
            if ticker not in etf_prices:
                continue
            df = etf_prices[ticker]
            if df.empty or "close" not in df.columns:
                continue
            closes = df["close"].tail(self.vol_lookback)
            rets = closes.pct_change().dropna()
            if len(rets) >= 20:
                returns_data[ticker] = rets

        if len(returns_data) < 2:
            return candidates[:self.num_assets]

        selected = []
        for ticker in candidates:
            if len(selected) >= self.num_assets:
                break

            if ticker not in returns_data:
                selected.append(ticker)
                continue

            # 기존 선정 자산과의 상관관계 체크
            too_correlated = False
            for existing in selected:
                if existing not in returns_data:
                    continue

                # 인덱스 정렬
                common_idx = returns_data[ticker].index.intersection(
                    returns_data[existing].index
                )
                if len(common_idx) < 20:
                    continue

                corr = float(
                    returns_data[ticker].loc[common_idx].corr(
                        returns_data[existing].loc[common_idx]
                    )
                )

                if not np.isnan(corr) and abs(corr) > self.max_correlation:
                    too_correlated = True
                    logger.debug(
                        f"상관관계 필터: {ticker} vs {existing} "
                        f"corr={corr:.2f} > {self.max_correlation}"
                    )
                    break

            if not too_correlated:
                selected.append(ticker)

        return selected

    def _calculate_vol_parity_weights(
        self,
        etf_prices: dict[str, pd.DataFrame],
        tickers: list[str],
    ) -> dict[str, float]:
        """변동성 패리티(역변동성) 비중을 계산한다."""
        volatilities = {}

        for ticker in tickers:
            if ticker not in etf_prices:
                continue
            df = etf_prices[ticker]
            if df.empty or "close" not in df.columns:
                continue

            closes = df["close"].tail(self.vol_lookback)
            returns = closes.pct_change().dropna()

            if len(returns) < 10:
                continue

            vol = float(returns.std() * np.sqrt(252))  # 연환산
            if vol > 0:
                volatilities[ticker] = vol

        if not volatilities:
            weight = 1.0 / len(tickers) if tickers else 0.0
            return {t: weight for t in tickers}

        # 역변동성 정규화
        inv_vols = {t: 1.0 / v for t, v in volatilities.items()}
        total = sum(inv_vols.values())

        if total <= 0:
            weight = 1.0 / len(tickers)
            return {t: weight for t in tickers}

        return {t: iv / total for t, iv in inv_vols.items()}

    def generate_signals(self, date: str, data: dict) -> dict:
        """포트폴리오 비중을 생성한다.

        1. 듀얼 모멘텀 스코어 계산 (단기 + 장기)
        2. SMA 크로스오버 필터
        3. 자산군 분산 필터
        4. 상관관계 필터
        5. 절대 모멘텀 필터 (음수 → 안전자산)
        6. 변동성 패리티 비중 배분

        Args:
            date: 리밸런싱 날짜
            data: {'etf_prices': dict} 또는 {'prices': dict}

        Returns:
            {종목코드: 비중} 딕셔너리
        """
        etf_prices = data.get("etf_prices", data.get("prices", {}))

        if not etf_prices:
            logger.warning(f"가격 데이터 없음 ({date})")
            self.last_diagnostics = {"status": "DATA_UNAVAILABLE"}
            return {}

        # 1. 듀얼 모멘텀 스코어
        scores = self._calculate_dual_momentum_score(etf_prices)
        if not scores:
            logger.warning(f"모멘텀 스코어 계산 실패 ({date})")
            self.last_diagnostics = {"status": "SCORE_FAIL"}
            return {}

        # 2. 스코어 순 정렬
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_tickers = [t for t, _ in ranked]

        # 3. 추세 필터 (SMA 크로스오버)
        if self.use_trend_filter:
            ranked_tickers = [
                t for t in ranked_tickers
                if self._check_sma_crossover(etf_prices, t)
            ]

        if not ranked_tickers:
            return {self.safe_asset: 1.0}

        # 4. 자산군 분산 필터
        ranked_tickers = self._apply_asset_class_filter(ranked_tickers)

        # 5. 상관관계 필터
        ranked_tickers = self._apply_correlation_filter(etf_prices, ranked_tickers)

        if not ranked_tickers:
            return {self.safe_asset: 1.0}

        # 6. 절대 모멘텀 필터
        final_risky = []
        safe_count = 0

        for ticker in ranked_tickers:
            # 단기 모멘텀 기준으로 절대 모멘텀 판단
            short_ret = self._get_returns(etf_prices, ticker, self.lookback_short)
            if short_ret is not None and short_ret <= self.abs_momentum_threshold:
                safe_count += 1
            else:
                final_risky.append(ticker)

        total_slots = len(final_risky) + safe_count
        if total_slots == 0:
            return {self.safe_asset: 1.0}

        # 7. 변동성 패리티 비중 배분
        if final_risky:
            vol_weights = self._calculate_vol_parity_weights(etf_prices, final_risky)
            risky_portion = len(final_risky) / total_slots
            signals = {t: w * risky_portion for t, w in vol_weights.items()}
        else:
            signals = {}

        # 안전자산 비중
        if safe_count > 0:
            safe_portion = safe_count / total_slots
            signals[self.safe_asset] = signals.get(self.safe_asset, 0.0) + safe_portion

        # 진단 정보
        self.last_diagnostics = {
            "status": "OK",
            "date": date,
            "scores": {t: round(s, 4) for t, s in scores.items()},
            "final_risky": final_risky,
            "safe_count": safe_count,
            "signals": {t: round(w, 4) for t, w in signals.items()},
        }

        logger.info(
            f"CrossAssetMomentum 시그널 ({date}): "
            f"{len(signals)}개, risky={len(final_risky)}, safe={safe_count}"
        )

        return signals
