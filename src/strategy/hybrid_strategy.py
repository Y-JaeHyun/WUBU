"""하이브리드 전략 모듈.

코어 팩터 전략(ThreeFactorStrategy)과 ETF 헤지(DualMomentum 로직)를
결합하여 포트폴리오를 구성한다.

코어 전략이 개별 종목 팩터 스코어링을 담당하고,
헤지 파트가 ETF 듀얼 모멘텀으로 하방 리스크를 관리한다.
"""

from typing import Optional

import pandas as pd

from src.backtest.engine import Strategy
from src.strategy.three_factor import ThreeFactorStrategy
from src.strategy.dual_momentum import DualMomentumStrategy
from src.data.etf_collector import ETF_UNIVERSE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 헤지용 기본 위험자산 (모멘텀 비교 대상)
DEFAULT_HEDGE_RISKY: dict[str, str] = {
    "domestic": ETF_UNIVERSE["domestic_equity"]["ticker"],  # KODEX 200
    "us": ETF_UNIVERSE["us_equity"]["ticker"],              # TIGER S&P500
    "gold": ETF_UNIVERSE["gold"]["ticker"],                 # KODEX 골드선물
}

# 헤지용 안전자산
DEFAULT_HEDGE_SAFE: str = ETF_UNIVERSE["short_bond"]["ticker"]  # 단기채권


class HybridStrategy(Strategy):
    """코어 팩터 + ETF 헤지 하이브리드 전략.

    코어 비중(core_weight)은 ThreeFactorStrategy가 생성한 개별 종목 시그널을,
    헤지 비중(1 - core_weight)은 DualMomentum 로직으로 ETF 시그널을 생성한다.

    Args:
        core_strategy: 코어 팩터 전략 (ThreeFactorStrategy)
        core_weight: 코어 전략 배분 비중 (기본 0.75)
        hedge_risky_assets: 헤지 위험자산 딕셔너리 {이름: ETF 종목코드}
        hedge_safe_asset: 헤지 안전자산 ETF 종목코드
        hedge_lookback_months: 헤지 모멘텀 룩백 기간 (월, 기본 12)
        hedge_n_select: 헤지 상대 모멘텀 선택 수 (기본 1)
    """

    def __init__(
        self,
        core_strategy: ThreeFactorStrategy,
        core_weight: float = 0.75,
        hedge_risky_assets: Optional[dict[str, str]] = None,
        hedge_safe_asset: Optional[str] = None,
        hedge_lookback_months: int = 12,
        hedge_n_select: int = 1,
    ):
        # M10: core_weight 범위 검증
        if not (0.0 <= core_weight <= 1.0):
            raise ValueError(
                f"core_weight는 0.0~1.0 범위여야 합니다: {core_weight}"
            )
        self.core_strategy = core_strategy
        self.core_weight = core_weight
        self.hedge_weight = 1.0 - core_weight

        self._hedge = DualMomentumStrategy(
            risky_assets=hedge_risky_assets or DEFAULT_HEDGE_RISKY.copy(),
            safe_asset=hedge_safe_asset or DEFAULT_HEDGE_SAFE,
            lookback_months=hedge_lookback_months,
            n_select=hedge_n_select,
        )

        logger.info(
            f"HybridStrategy 초기화: core_weight={core_weight:.0%}, "
            f"hedge_weight={self.hedge_weight:.0%}, "
            f"hedge_risky={list(self._hedge.risky_assets.keys())}"
        )

    def update_holdings(self, holdings: set[str]) -> None:
        """현재 보유 종목 집합을 코어 전략에 전달한다."""
        if hasattr(self.core_strategy, "update_holdings"):
            self.core_strategy.update_holdings(holdings)

    @property
    def name(self) -> str:
        """전략 이름."""
        core_pct = int(self.core_weight * 100)
        hedge_pct = int(self.hedge_weight * 100)
        return f"Hybrid({core_pct}/{hedge_pct}, {self.core_strategy.name})"

    def generate_signals(self, date: str, data: dict) -> dict:
        """코어 시그널과 헤지 시그널을 병합하여 반환한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {'fundamentals': DataFrame, 'prices': dict, 'index_prices': Series}

        Returns:
            종목코드: 비중 딕셔너리 (코어 종목 + ETF 합산)
        """
        signals: dict[str, float] = {}

        # 1. 코어 전략 시그널 (개별 종목)
        try:
            core_signals = self.core_strategy.generate_signals(date, data)
        except Exception as e:
            logger.warning(f"코어 전략 시그널 생성 실패 ({date}): {e}")
            core_signals = {}

        if core_signals:
            for ticker, weight in core_signals.items():
                signals[ticker] = weight * self.core_weight

        # 2. 헤지 ETF 시그널 (듀얼 모멘텀)
        hedge_allocation = self._generate_hedge_signals(date, data)

        if hedge_allocation:
            for ticker, weight in hedge_allocation.items():
                signals[ticker] = signals.get(ticker, 0.0) + weight * self.hedge_weight
        else:
            # 헤지 데이터 없으면 안전자산 100%로 fallback
            safe = self._hedge.safe_asset
            signals[safe] = signals.get(safe, 0.0) + self.hedge_weight

        logger.info(
            f"하이브리드 시그널 생성 ({date}): "
            f"코어={len(core_signals)}종목, 헤지={len(hedge_allocation)}종목, "
            f"합계={len(signals)}종목"
        )

        return signals

    def _generate_hedge_signals(self, date: str, data: dict) -> dict[str, float]:
        """헤지 파트의 듀얼 모멘텀 자산배분을 생성한다.

        price_cache에서 ETF 가격 데이터를 추출하여
        DualMomentumStrategy.generate_allocation()을 호출한다.
        """
        price_cache: dict[str, pd.DataFrame] = data.get("prices", {})

        # ETF 종목의 close 시계열 추출
        etf_prices: dict[str, pd.Series] = {}

        for asset_name, ticker in self._hedge.risky_assets.items():
            if ticker in price_cache and not price_cache[ticker].empty:
                df = price_cache[ticker]
                if "close" in df.columns:
                    # 현재 날짜까지의 데이터만 사용
                    target_date = pd.Timestamp(date)
                    mask = df.index <= target_date
                    available = df.loc[mask, "close"]
                    if not available.empty:
                        etf_prices[asset_name] = available

        if not etf_prices:
            logger.debug(f"헤지 ETF 가격 데이터 없음 ({date}): 안전자산 fallback")
            return {self._hedge.safe_asset: 1.0}

        try:
            allocation = self._hedge.generate_allocation(etf_prices)
            return allocation
        except Exception as e:
            logger.warning(f"헤지 자산배분 실패 ({date}): {e}")
            return {self._hedge.safe_asset: 1.0}
