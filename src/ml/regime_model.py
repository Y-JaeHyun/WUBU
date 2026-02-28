"""시장 레짐 감지 및 팩터 가중치 동적 조절 모듈.

ML을 종목 수익률 예측이 아닌 '팩터 가중치 조절기'로 사용한다.
시장 레짐(변동성/모멘텀 조합)을 감지하고, 레짐에 따라
최적의 팩터 가중치를 제안한다.

Level 1: 룰 기반 레짐 감지 (즉시 사용 가능)
Level 2: ML 기반 가중치 예측 (데이터 축적 후)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 레짐별 사전 정의 가중치 (Level 1)
REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "bull_low_vol":  {"value": 0.35, "momentum": 0.30, "quality": 0.20, "low_vol": 0.15},
    "bull_high_vol": {"value": 0.15, "momentum": 0.35, "quality": 0.25, "low_vol": 0.25},
    "bear_low_vol":  {"value": 0.25, "momentum": 0.10, "quality": 0.35, "low_vol": 0.30},
    "bear_high_vol": {"value": 0.10, "momentum": 0.10, "quality": 0.35, "low_vol": 0.45},
}

# 기본 팩터 (low_vol 없을 때 3팩터용)
DEFAULT_FACTOR_NAMES = ["value", "momentum", "quality"]


def detect_market_regime(
    market_vol: float,
    market_momentum: float,
    vol_threshold: float = 0.20,
) -> str:
    """시장 레짐을 감지한다 (Level 1: 룰 기반).

    Args:
        market_vol: 시장 연율화 변동성 (예: 0.15 = 15%)
        market_momentum: 시장 모멘텀 (예: 0.05 = 5% 수익률)
        vol_threshold: 고/저변동 경계 (기본 0.20 = 20%)

    Returns:
        레짐 문자열: "bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"
    """
    is_bull = market_momentum >= 0
    is_low_vol = market_vol < vol_threshold

    if is_bull and is_low_vol:
        regime = "bull_low_vol"
    elif is_bull and not is_low_vol:
        regime = "bull_high_vol"
    elif not is_bull and is_low_vol:
        regime = "bear_low_vol"
    else:
        regime = "bear_high_vol"

    logger.info(f"레짐 감지: {regime} (vol={market_vol:.4f}, mom={market_momentum:.4f})")
    return regime


def get_regime_weights(
    regime: str,
    factor_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """레짐에 해당하는 팩터 가중치를 반환한다.

    Args:
        regime: 레짐 문자열
        factor_names: 사용할 팩터 이름 리스트. None이면 전체 4팩터.
            3팩터만 사용할 경우 ["value", "momentum", "quality"]

    Returns:
        {factor_name: weight} 딕셔너리 (합 = 1.0)
    """
    if regime not in REGIME_WEIGHTS:
        logger.warning(f"알 수 없는 레짐 '{regime}', 균등 가중치 반환")
        if factor_names is None:
            factor_names = list(REGIME_WEIGHTS["bull_low_vol"].keys())
        equal_w = 1.0 / len(factor_names)
        return {f: equal_w for f in factor_names}

    weights = REGIME_WEIGHTS[regime].copy()

    # 팩터 필터링 (예: low_vol 미사용 시)
    if factor_names is not None:
        filtered = {f: weights.get(f, 0.0) for f in factor_names}
        # 합 = 1.0으로 재정규화
        total = sum(filtered.values())
        if total > 0:
            filtered = {f: w / total for f, w in filtered.items()}
        else:
            equal_w = 1.0 / len(factor_names)
            filtered = {f: equal_w for f in factor_names}
        weights = filtered

    return weights


class RuleBasedRegimeModel:
    """룰 기반 레짐 모델 (Level 1).

    시장 변동성과 모멘텀으로 레짐을 감지하고,
    사전 정의된 팩터 가중치를 반환한다.

    Args:
        vol_threshold: 고/저변동 경계 (기본 0.20)
        vol_lookback: 변동성 계산 기간 (거래일, 기본 60)
        momentum_lookback: 모멘텀 계산 기간 (거래일, 기본 60)
        min_weight: 팩터 최소 가중치 (기본 0.10)
        max_weight: 팩터 최대 가중치 (기본 0.50)
        factor_names: 사용할 팩터 목록
    """

    def __init__(
        self,
        vol_threshold: float = 0.20,
        vol_lookback: int = 60,
        momentum_lookback: int = 60,
        min_weight: float = 0.10,
        max_weight: float = 0.50,
        factor_names: Optional[List[str]] = None,
    ):
        self.vol_threshold = vol_threshold
        self.vol_lookback = vol_lookback
        self.momentum_lookback = momentum_lookback
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.factor_names = factor_names or list(REGIME_WEIGHTS["bull_low_vol"].keys())
        self._last_regime: Optional[str] = None
        self._last_weights: Optional[Dict[str, float]] = None

    def compute_market_features(
        self,
        market_prices: pd.Series,
    ) -> Dict[str, float]:
        """시장 지수 가격으로부터 레짐 판별용 피쳐를 계산한다.

        Args:
            market_prices: 시장 지수 종가 시리즈 (DatetimeIndex)

        Returns:
            {"market_vol": float, "market_momentum": float}
        """
        if len(market_prices) < max(self.vol_lookback, self.momentum_lookback) + 1:
            logger.warning("시장 데이터 부족, 기본 피쳐 반환")
            return {"market_vol": self.vol_threshold, "market_momentum": 0.0}

        returns = market_prices.pct_change().dropna()

        # 연율화 변동성
        recent_vol_returns = returns.iloc[-self.vol_lookback:]
        market_vol = float(recent_vol_returns.std() * np.sqrt(252))

        # 모멘텀 (lookback 기간 수익률)
        if len(market_prices) > self.momentum_lookback:
            market_momentum = float(
                market_prices.iloc[-1] / market_prices.iloc[-self.momentum_lookback] - 1
            )
        else:
            market_momentum = 0.0

        return {"market_vol": market_vol, "market_momentum": market_momentum}

    def predict(
        self,
        market_prices: Optional[pd.Series] = None,
        market_vol: Optional[float] = None,
        market_momentum: Optional[float] = None,
    ) -> Dict[str, float]:
        """현재 시장 상태에 맞는 팩터 가중치를 예측한다.

        market_prices가 주어지면 자동으로 피쳐 계산.
        또는 market_vol, market_momentum을 직접 전달.

        Args:
            market_prices: 시장 지수 가격 시리즈
            market_vol: 시장 변동성 (직접 지정)
            market_momentum: 시장 모멘텀 (직접 지정)

        Returns:
            {factor_name: weight} (합 = 1.0)
        """
        if market_prices is not None:
            features = self.compute_market_features(market_prices)
            market_vol = features["market_vol"]
            market_momentum = features["market_momentum"]

        if market_vol is None or market_momentum is None:
            logger.warning("시장 피쳐 없음, 균등 가중치 반환")
            equal_w = 1.0 / len(self.factor_names)
            return {f: equal_w for f in self.factor_names}

        regime = detect_market_regime(market_vol, market_momentum, self.vol_threshold)
        weights = get_regime_weights(regime, self.factor_names)

        # 가중치 범위 제한
        weights = self._clip_weights(weights)

        self._last_regime = regime
        self._last_weights = weights

        return weights

    def _clip_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """가중치를 min/max 범위로 클리핑하고 재정규화한다."""
        clipped = {f: max(self.min_weight, min(self.max_weight, w))
                   for f, w in weights.items()}
        total = sum(clipped.values())
        if total > 0:
            clipped = {f: w / total for f, w in clipped.items()}
        return clipped

    @property
    def last_regime(self) -> Optional[str]:
        """마지막 감지된 레짐."""
        return self._last_regime
