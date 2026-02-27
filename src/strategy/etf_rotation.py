"""ETF 로테이션 전략 모듈.

ETF 기반 섹터/테마/글로벌 로테이션 전략이다.
모멘텀 기반으로 상위 ETF를 선별하고, 절대모멘텀이 음수이면 안전자산으로 전환한다.

데이터 소스:
- pykrx ETF 가격 데이터 (etf_collector.get_etf_price)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 기본 ETF 유니버스
DEFAULT_ETF_UNIVERSE: dict[str, str] = {
    "069500": "KODEX 200",
    "371460": "TIGER 미국S&P500",
    "133690": "TIGER 미국나스닥100",
    "091160": "KODEX 반도체",
    "091170": "KODEX 은행",
    "117700": "KODEX 건설",
    "132030": "KODEX 골드선물(H)",
    "439870": "KODEX 단기채권",
}

# 기본 안전자산 (단기채권 ETF)
DEFAULT_SAFE_ASSET = "439870"


class ETFRotationStrategy(Strategy):
    """ETF 기반 섹터/테마 로테이션 전략.

    모멘텀 기반으로 상위 ETF를 선별하고,
    절대모멘텀이 음수이면 안전자산(단기채권)으로 전환한다.

    Args:
        lookback: 모멘텀 계산 기간 (기본 60거래일)
        num_etfs: 선정할 ETF 수 (기본 3)
        safe_asset: 안전자산 ETF 종목코드 (기본 '439870' = KODEX 단기채권)
        etf_universe: ETF 유니버스 딕셔너리 {종목코드: 이름}
        weighting: 가중 방식 ('equal' 또는 'inverse_vol')
        abs_momentum: 절대모멘텀 적용 여부 (기본 True)
    """

    def __init__(
        self,
        lookback: int = 60,
        num_etfs: int = 3,
        safe_asset: str = DEFAULT_SAFE_ASSET,
        etf_universe: Optional[dict[str, str]] = None,
        weighting: str = "equal",
        abs_momentum: bool = True,
    ):
        self.lookback = lookback
        self.num_etfs = num_etfs
        self.safe_asset = safe_asset
        self.etf_universe = etf_universe or DEFAULT_ETF_UNIVERSE.copy()
        self.weighting = weighting
        self.abs_momentum = abs_momentum

        if weighting not in ("equal", "inverse_vol"):
            raise ValueError(
                f"지원하지 않는 가중 방식: {weighting}. "
                "'equal' 또는 'inverse_vol' 중 선택하세요."
            )

        logger.info(
            f"ETFRotationStrategy 초기화: lookback={lookback}, "
            f"num_etfs={num_etfs}, safe_asset={safe_asset}, "
            f"weighting={weighting}, universe={len(self.etf_universe)}개"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"ETFRotation(top{self.num_etfs}, {self.lookback}d)"

    def _calculate_momentum(
        self, etf_prices: dict[str, pd.DataFrame]
    ) -> pd.Series:
        """ETF별 모멘텀을 계산한다.

        모멘텀 = (현재가 / lookback일 전 가격) - 1

        Args:
            etf_prices: {ticker: OHLCV DataFrame} 딕셔너리

        Returns:
            pd.Series (index=ticker, values=momentum)
        """
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
            if len(closes) < self.lookback:
                continue

            price_current = closes.iloc[-1]
            price_past = closes.iloc[-self.lookback]

            if price_past <= 0 or np.isnan(price_past) or np.isnan(price_current):
                continue

            momentum = float(price_current / price_past - 1)
            momentums[ticker] = momentum

        if not momentums:
            return pd.Series(dtype=float)

        return pd.Series(momentums, name="momentum")

    def _calculate_inverse_vol_weights(
        self, etf_prices: dict[str, pd.DataFrame], tickers: list[str]
    ) -> dict[str, float]:
        """역변동성 가중 비중을 계산한다.

        Args:
            etf_prices: {ticker: OHLCV DataFrame} 딕셔너리
            tickers: 대상 종목코드 리스트

        Returns:
            {ticker: weight} 딕셔너리
        """
        volatilities = {}

        for ticker in tickers:
            if ticker not in etf_prices:
                continue

            price_df = etf_prices[ticker]
            if price_df.empty or "close" not in price_df.columns:
                continue

            closes = price_df["close"].tail(self.lookback)
            returns = closes.pct_change().dropna()

            if len(returns) < 10:
                continue

            vol = float(returns.std())
            if vol > 0:
                volatilities[ticker] = vol

        if not volatilities:
            # fallback to equal weight
            weight = 1.0 / len(tickers) if tickers else 0.0
            return {t: weight for t in tickers}

        # 역변동성 = 1 / vol, 정규화
        inv_vols = {t: 1.0 / v for t, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol <= 0:
            weight = 1.0 / len(tickers)
            return {t: weight for t in tickers}

        return {t: iv / total_inv_vol for t, iv in inv_vols.items()}

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        1. ETF별 모멘텀 계산
        2. 상위 num_etfs개 선정
        3. 절대모멘텀 음수 ETF는 안전자산으로 대체
        4. 가중 방식에 따라 비중 할당

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {
                'etf_prices': dict[ticker, DataFrame],
                또는 'prices': dict[ticker, DataFrame] (호환용),
            }

        Returns:
            종목코드: 비중 딕셔너리
        """
        etf_prices = data.get("etf_prices", data.get("prices", {}))

        if not etf_prices:
            logger.warning(f"ETF 가격 데이터 없음 ({date}), 빈 시그널 반환")
            return {}

        # 모멘텀 계산
        momentum = self._calculate_momentum(etf_prices)

        if momentum.empty:
            logger.warning(f"모멘텀 계산 실패 ({date})")
            return {}

        # 모멘텀 상위 ETF 선정
        top_momentum = momentum.sort_values(ascending=False).head(self.num_etfs)

        # 절대모멘텀 적용: 음수 모멘텀 ETF는 안전자산으로 대체
        selected_tickers = []
        safe_asset_count = 0

        for ticker in top_momentum.index:
            if self.abs_momentum and top_momentum[ticker] <= 0:
                safe_asset_count += 1
            else:
                selected_tickers.append(ticker)

        # 비중 할당
        total_slots = len(selected_tickers) + safe_asset_count

        if total_slots == 0:
            return {}

        if self.weighting == "equal":
            slot_weight = 1.0 / total_slots
            signals = {t: slot_weight for t in selected_tickers}
            if safe_asset_count > 0:
                signals[self.safe_asset] = slot_weight * safe_asset_count
        elif self.weighting == "inverse_vol":
            if selected_tickers:
                risk_weights = self._calculate_inverse_vol_weights(
                    etf_prices, selected_tickers
                )
                # 안전자산 비중 = safe_asset_count / total_slots
                risk_portion = len(selected_tickers) / total_slots
                signals = {t: w * risk_portion for t, w in risk_weights.items()}
                if safe_asset_count > 0:
                    safe_portion = safe_asset_count / total_slots
                    signals[self.safe_asset] = safe_portion
            else:
                signals = {self.safe_asset: 1.0}
        else:
            signals = {}

        logger.info(
            f"ETF 로테이션 시그널 생성 ({date}): {len(signals)}개 ETF, "
            f"안전자산 배분={safe_asset_count}/{total_slots}"
        )

        return signals
