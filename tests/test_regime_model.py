"""ML 레짐 메타 모델 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.ml.regime_model import (
    detect_market_regime,
    get_regime_weights,
    RuleBasedRegimeModel,
    REGIME_WEIGHTS,
)


class TestDetectMarketRegime:
    """detect_market_regime 테스트."""

    def test_bull_low_vol(self):
        assert detect_market_regime(0.10, 0.05) == "bull_low_vol"

    def test_bull_high_vol(self):
        assert detect_market_regime(0.30, 0.05) == "bull_high_vol"

    def test_bear_low_vol(self):
        assert detect_market_regime(0.10, -0.05) == "bear_low_vol"

    def test_bear_high_vol(self):
        assert detect_market_regime(0.30, -0.05) == "bear_high_vol"

    def test_boundary_vol(self):
        """경계값: vol=0.20 -> 고변동."""
        assert detect_market_regime(0.20, 0.05) == "bull_high_vol"

    def test_boundary_momentum(self):
        """경계값: mom=0.0 -> 상승장."""
        assert detect_market_regime(0.10, 0.0) == "bull_low_vol"

    def test_custom_threshold(self):
        assert detect_market_regime(0.25, 0.05, vol_threshold=0.30) == "bull_low_vol"


class TestGetRegimeWeights:
    """get_regime_weights 테스트."""

    def test_known_regime_returns_weights(self):
        weights = get_regime_weights("bull_low_vol")
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_all_regimes_sum_to_one(self):
        for regime in REGIME_WEIGHTS:
            weights = get_regime_weights(regime)
            assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_unknown_regime_returns_equal(self):
        weights = get_regime_weights("unknown_regime", ["value", "momentum", "quality"])
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_filter_to_3_factors(self):
        weights = get_regime_weights("bull_low_vol", ["value", "momentum", "quality"])
        assert "low_vol" not in weights
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_bear_high_vol_favors_low_vol(self):
        """고변동 하락장 -> low_vol 비중 최고."""
        weights = get_regime_weights("bear_high_vol")
        assert weights["low_vol"] == max(weights.values())

    def test_bull_low_vol_favors_value(self):
        """저변동 상승장 -> value 비중 최고."""
        weights = get_regime_weights("bull_low_vol")
        assert weights["value"] == max(weights.values())


class TestRuleBasedRegimeModel:
    """RuleBasedRegimeModel 테스트."""

    def test_predict_with_direct_values(self):
        model = RuleBasedRegimeModel()
        weights = model.predict(market_vol=0.10, market_momentum=0.05)
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert model.last_regime == "bull_low_vol"

    def test_predict_with_market_prices(self):
        """시장 가격 시리즈로 예측."""
        model = RuleBasedRegimeModel(vol_lookback=20, momentum_lookback=20)
        dates = pd.bdate_range("2024-01-01", periods=100)
        # 안정적 상승 시장 시뮬레이션
        np.random.seed(42)
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(100) * 0.005 + 0.001)),
            index=dates,
        )
        weights = model.predict(market_prices=prices)
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert model.last_regime is not None

    def test_insufficient_data_returns_equal(self):
        """데이터 부족 -> 균등 가중치 (기본 피쳐 사용)."""
        model = RuleBasedRegimeModel(vol_lookback=60)
        dates = pd.bdate_range("2024-01-01", periods=10)
        prices = pd.Series(range(100, 110), index=dates, dtype=float)
        weights = model.predict(market_prices=prices)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_no_features_returns_equal(self):
        """피쳐 없음 -> 균등 가중치."""
        model = RuleBasedRegimeModel()
        weights = model.predict()
        equal_w = 1.0 / 4
        for w in weights.values():
            assert abs(w - equal_w) < 1e-9

    def test_weight_clipping(self):
        """모든 가중치가 [min_weight, max_weight] 범위 내."""
        model = RuleBasedRegimeModel(min_weight=0.10, max_weight=0.50)
        for regime in REGIME_WEIGHTS:
            weights = model.predict(
                market_vol=0.10 if "low" in regime else 0.30,
                market_momentum=0.05 if "bull" in regime else -0.05,
            )
            for w in weights.values():
                assert w >= model.min_weight - 1e-9
                assert w <= model.max_weight + 1e-9

    def test_3_factor_mode(self):
        """3팩터 모드 (low_vol 제외)."""
        model = RuleBasedRegimeModel(factor_names=["value", "momentum", "quality"])
        weights = model.predict(market_vol=0.10, market_momentum=0.05)
        assert len(weights) == 3
        assert "low_vol" not in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_compute_market_features(self):
        """시장 피쳐 계산."""
        model = RuleBasedRegimeModel(vol_lookback=20, momentum_lookback=20)
        dates = pd.bdate_range("2024-01-01", periods=100)
        np.random.seed(42)
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(100) * 0.01)),
            index=dates,
        )
        features = model.compute_market_features(prices)
        assert "market_vol" in features
        assert "market_momentum" in features
        assert features["market_vol"] > 0

    def test_high_vol_increases_low_vol_weight(self):
        """고변동 레짐 -> low_vol 비중 증가."""
        model = RuleBasedRegimeModel()
        w_low = model.predict(market_vol=0.10, market_momentum=0.05)
        w_high = model.predict(market_vol=0.30, market_momentum=0.05)
        assert w_high["low_vol"] > w_low["low_vol"]

    def test_bear_market_increases_quality_weight(self):
        """하락장 -> quality 비중 증가."""
        model = RuleBasedRegimeModel()
        w_bull = model.predict(market_vol=0.10, market_momentum=0.05)
        w_bear = model.predict(market_vol=0.10, market_momentum=-0.05)
        assert w_bear["quality"] > w_bull["quality"]
