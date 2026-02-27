"""ETF 로테이션 전략 모듈 테스트.

ETFRotationStrategy의 초기화, 모멘텀 계산, 역변동성 가중,
절대모멘텀 전환, generate_signals 반환값 형식 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.etf_rotation import ETFRotationStrategy, DEFAULT_ETF_UNIVERSE


# ===================================================================
# 초기화 검증
# ===================================================================

class TestETFRotationInit:
    """ETFRotationStrategy 초기화 검증."""

    def test_default_parameters(self):
        """기본 파라미터로 생성된다."""
        s = ETFRotationStrategy()
        assert s.lookback == 60
        assert s.num_etfs == 3
        assert s.safe_asset == "439870"
        assert s.weighting == "equal"
        assert s.abs_momentum is True
        assert len(s.etf_universe) == len(DEFAULT_ETF_UNIVERSE)

    def test_custom_parameters(self):
        """사용자 정의 파라미터가 올바르게 반영된다."""
        custom_universe = {"069500": "KODEX 200", "133690": "TIGER 나스닥"}
        s = ETFRotationStrategy(
            lookback=90,
            num_etfs=2,
            safe_asset="214980",
            etf_universe=custom_universe,
            weighting="inverse_vol",
        )
        assert s.lookback == 90
        assert s.num_etfs == 2
        assert s.safe_asset == "214980"
        assert s.weighting == "inverse_vol"
        assert len(s.etf_universe) == 2

    def test_invalid_weighting_raises(self):
        """지원하지 않는 가중 방식은 ValueError를 발생시킨다."""
        with pytest.raises(ValueError, match="지원하지 않는 가중 방식"):
            ETFRotationStrategy(weighting="market_cap")

    def test_name_property(self):
        """name 프로퍼티가 올바른 형식을 반환한다."""
        s = ETFRotationStrategy(num_etfs=4, lookback=90)
        assert s.name == "ETFRotation(top4, 90d)"


# ===================================================================
# 모멘텀 계산 테스트
# ===================================================================

class TestMomentumCalculation:
    """_calculate_momentum 메서드 검증."""

    def _make_etf_price(self, start_price, end_price, n_days=100):
        """선형 보간으로 ETF 가격 데이터를 생성한다."""
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        prices = np.linspace(start_price, end_price, n_days)
        return pd.DataFrame(
            {"close": prices, "volume": [1_000_000] * n_days},
            index=dates,
        )

    def test_positive_momentum(self):
        """양의 모멘텀이 정확히 계산된다."""
        s = ETFRotationStrategy(
            lookback=60,
            etf_universe={"069500": "KODEX 200", "439870": "단기채권"},
            safe_asset="439870",
        )

        etf_prices = {
            "069500": self._make_etf_price(10000, 12000),  # +20%
        }

        momentum = s._calculate_momentum(etf_prices)

        assert "069500" in momentum.index
        assert momentum["069500"] > 0

    def test_negative_momentum(self):
        """음의 모멘텀이 정확히 계산된다."""
        s = ETFRotationStrategy(
            lookback=60,
            etf_universe={"069500": "KODEX 200", "439870": "단기채권"},
            safe_asset="439870",
        )

        etf_prices = {
            "069500": self._make_etf_price(12000, 10000),  # 하락
        }

        momentum = s._calculate_momentum(etf_prices)

        assert momentum["069500"] < 0

    def test_safe_asset_excluded_from_momentum(self):
        """안전자산은 모멘텀 계산에서 제외된다."""
        s = ETFRotationStrategy(
            lookback=60,
            etf_universe={"069500": "KODEX 200", "439870": "단기채권"},
            safe_asset="439870",
        )

        etf_prices = {
            "069500": self._make_etf_price(10000, 12000),
            "439870": self._make_etf_price(10000, 10100),
        }

        momentum = s._calculate_momentum(etf_prices)

        assert "439870" not in momentum.index

    def test_insufficient_data_excluded(self):
        """데이터가 부족한 ETF는 제외된다."""
        s = ETFRotationStrategy(
            lookback=60,
            etf_universe={"069500": "KODEX 200", "133690": "나스닥", "439870": "채권"},
            safe_asset="439870",
        )

        etf_prices = {
            "069500": self._make_etf_price(10000, 12000, n_days=100),
            "133690": self._make_etf_price(10000, 11000, n_days=30),  # 부족
        }

        momentum = s._calculate_momentum(etf_prices)

        assert "069500" in momentum.index
        assert "133690" not in momentum.index

    def test_empty_prices(self):
        """빈 가격 데이터에서 빈 시리즈 반환."""
        s = ETFRotationStrategy()
        momentum = s._calculate_momentum({})
        assert momentum.empty


# ===================================================================
# 역변동성 가중 테스트
# ===================================================================

class TestInverseVolWeights:
    """_calculate_inverse_vol_weights 메서드 검증."""

    def _make_price_data(self, n_days=100, vol_level=0.02, seed=42):
        np.random.seed(seed)
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        returns = np.random.normal(0, vol_level, n_days)
        prices = 10000 * np.cumprod(1 + returns)
        return pd.DataFrame(
            {"close": prices, "volume": [1_000_000] * n_days},
            index=dates,
        )

    def test_weights_sum_to_one(self):
        """역변동성 가중의 합이 1.0이다."""
        s = ETFRotationStrategy(lookback=60)

        etf_prices = {
            "A": self._make_price_data(vol_level=0.01, seed=1),
            "B": self._make_price_data(vol_level=0.03, seed=2),
        }

        weights = s._calculate_inverse_vol_weights(etf_prices, ["A", "B"])

        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_low_vol_gets_higher_weight(self):
        """저변동성 ETF가 더 높은 비중을 받는다."""
        s = ETFRotationStrategy(lookback=60)

        etf_prices = {
            "LOW": self._make_price_data(vol_level=0.005, seed=1),
            "HIGH": self._make_price_data(vol_level=0.05, seed=2),
        }

        weights = s._calculate_inverse_vol_weights(etf_prices, ["LOW", "HIGH"])

        assert weights["LOW"] > weights["HIGH"]


# ===================================================================
# generate_signals 테스트
# ===================================================================

class TestETFRotationSignals:
    """generate_signals 메서드 검증."""

    def _make_etf_price(self, start_price, end_price, n_days=100):
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        prices = np.linspace(start_price, end_price, n_days)
        return pd.DataFrame(
            {"close": prices, "volume": [1_000_000] * n_days},
            index=dates,
        )

    def _make_data(self):
        """테스트용 ETF 가격 데이터를 생성한다."""
        etf_prices = {
            "069500": self._make_etf_price(10000, 13000),  # +30%
            "133690": self._make_etf_price(10000, 12000),  # +20%
            "091160": self._make_etf_price(10000, 11000),  # +10%
            "091170": self._make_etf_price(10000, 9000),   # -10%
            "132030": self._make_etf_price(10000, 10500),  # +5%
            "439870": self._make_etf_price(10000, 10100),  # +1% (안전자산)
        }
        return {"etf_prices": etf_prices}

    def test_returns_dict(self):
        """반환값이 dict 타입이다."""
        s = ETFRotationStrategy(
            lookback=60,
            num_etfs=3,
            etf_universe={
                "069500": "KODEX200", "133690": "나스닥",
                "091160": "반도체", "439870": "채권",
            },
        )
        data = self._make_data()
        signals = s.generate_signals("20240102", data)
        assert isinstance(signals, dict)

    def test_weights_sum_to_one(self):
        """비중의 합이 1.0이다."""
        s = ETFRotationStrategy(
            lookback=60,
            num_etfs=3,
            etf_universe={
                "069500": "KODEX200", "133690": "나스닥",
                "091160": "반도체", "439870": "채권",
            },
        )
        data = self._make_data()
        signals = s.generate_signals("20240102", data)
        if signals:
            total = sum(signals.values())
            assert abs(total - 1.0) < 1e-9

    def test_top_momentum_selected(self):
        """모멘텀 상위 ETF가 선택된다."""
        s = ETFRotationStrategy(
            lookback=60,
            num_etfs=2,
            abs_momentum=False,  # 절대모멘텀 비활성화
            etf_universe={
                "069500": "KODEX200", "133690": "나스닥",
                "091160": "반도체", "091170": "은행", "439870": "채권",
            },
        )
        data = self._make_data()
        signals = s.generate_signals("20240102", data)

        # 069500(+30%), 133690(+20%) 가 상위 2개
        assert "069500" in signals
        assert "133690" in signals

    def test_abs_momentum_converts_to_safe_asset(self):
        """절대모멘텀이 음수이면 안전자산으로 전환된다."""
        s = ETFRotationStrategy(
            lookback=60,
            num_etfs=2,
            abs_momentum=True,
            etf_universe={
                "069500": "KODEX200", "091170": "은행", "439870": "채권",
            },
        )

        etf_prices = {
            "069500": self._make_etf_price(10000, 13000),  # +30%
            "091170": self._make_etf_price(10000, 9000),   # -10%
            "439870": self._make_etf_price(10000, 10100),  # 안전자산
        }

        data = {"etf_prices": etf_prices}
        signals = s.generate_signals("20240102", data)

        # 069500은 양의 모멘텀, 091170은 음수 → 안전자산
        assert "069500" in signals
        assert "439870" in signals
        assert "091170" not in signals

    def test_all_negative_goes_full_safe_asset(self):
        """모든 ETF 모멘텀이 음수이면 전부 안전자산으로 전환된다."""
        s = ETFRotationStrategy(
            lookback=60,
            num_etfs=2,
            abs_momentum=True,
            etf_universe={
                "069500": "KODEX200", "133690": "나스닥", "439870": "채권",
            },
        )

        etf_prices = {
            "069500": self._make_etf_price(10000, 8000),   # -20%
            "133690": self._make_etf_price(10000, 9000),   # -10%
            "439870": self._make_etf_price(10000, 10100),
        }

        data = {"etf_prices": etf_prices}
        signals = s.generate_signals("20240102", data)

        assert "439870" in signals
        assert abs(signals["439870"] - 1.0) < 1e-9

    def test_empty_prices_returns_empty(self):
        """가격 데이터 없으면 빈 dict 반환."""
        s = ETFRotationStrategy()
        signals = s.generate_signals("20240102", {"etf_prices": {}})
        assert signals == {}

    def test_no_data_returns_empty(self):
        """data가 비어있으면 빈 dict 반환."""
        s = ETFRotationStrategy()
        signals = s.generate_signals("20240102", {})
        assert signals == {}

    def test_equal_weighting(self):
        """동일 비중 가중이 올바르게 적용된다."""
        s = ETFRotationStrategy(
            lookback=60,
            num_etfs=3,
            weighting="equal",
            abs_momentum=False,
            etf_universe={
                "069500": "KODEX200", "133690": "나스닥",
                "091160": "반도체", "439870": "채권",
            },
        )
        data = self._make_data()
        signals = s.generate_signals("20240102", data)

        if signals:
            weights = list(signals.values())
            expected = 1.0 / len(signals)
            for w in weights:
                assert abs(w - expected) < 1e-9

    def test_inverse_vol_weighting(self):
        """역변동성 가중이 동작한다."""
        s = ETFRotationStrategy(
            lookback=60,
            num_etfs=2,
            weighting="inverse_vol",
            abs_momentum=False,
            etf_universe={
                "069500": "KODEX200", "133690": "나스닥", "439870": "채권",
            },
        )
        data = self._make_data()
        signals = s.generate_signals("20240102", data)

        if signals:
            total = sum(signals.values())
            assert abs(total - 1.0) < 1e-9

    def test_prices_key_compatibility(self):
        """'prices' 키로도 데이터를 받을 수 있다 (호환용)."""
        s = ETFRotationStrategy(
            lookback=60,
            num_etfs=2,
            abs_momentum=False,
            etf_universe={
                "069500": "KODEX200", "133690": "나스닥", "439870": "채권",
            },
        )

        prices = {
            "069500": self._make_etf_price(10000, 13000),
            "133690": self._make_etf_price(10000, 12000),
        }

        signals = s.generate_signals("20240102", {"prices": prices})

        assert isinstance(signals, dict)
        assert len(signals) > 0
