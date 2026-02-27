"""ETF 로테이션 전략 모듈 테스트.

ETFRotationStrategy의 초기화, 모멘텀 계산, 역변동성 가중,
절대모멘텀 전환, generate_signals 반환값 형식 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.etf_rotation import (
    ETFRotationStrategy,
    DEFAULT_ETF_UNIVERSE,
    ETF_SECTOR_MAP,
)


# ===================================================================
# 초기화 검증
# ===================================================================

class TestETFRotationInit:
    """ETFRotationStrategy 초기화 검증."""

    def test_default_parameters(self):
        """기본 파라미터로 생성된다."""
        s = ETFRotationStrategy()
        assert s.lookback == 252
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

    def test_ai_etfs_in_default_universe(self):
        """AI/로보틱스 ETF가 기본 유니버스에 포함된다."""
        assert "464310" in DEFAULT_ETF_UNIVERSE  # TIGER 글로벌AI&로보틱스INDXX
        assert "469150" in DEFAULT_ETF_UNIVERSE  # ACE AI반도체포커스

    def test_default_universe_size(self):
        """기본 유니버스는 10개 ETF를 포함한다."""
        assert len(DEFAULT_ETF_UNIVERSE) == 10

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


# ===================================================================
# Feature Flag config 주입 검증
# ===================================================================


class TestETFRotationConfigInjection:
    """Feature flag config에서 ETF 전략 파라미터가 올바르게 주입되는지 검증."""

    def test_config_lookback_months_to_days(self):
        """lookback_months=12는 252 거래일로 변환된다."""
        lookback_months = 12
        lookback_days = lookback_months * 21
        s = ETFRotationStrategy(lookback=lookback_days, num_etfs=2)
        assert s.lookback == 252
        assert s.num_etfs == 2

    def test_config_n_select(self):
        """n_select=2로 생성하면 num_etfs=2."""
        s = ETFRotationStrategy(num_etfs=2)
        assert s.num_etfs == 2

    def test_default_flag_has_etf_rotation_pct(self):
        """etf_rotation flag config에 etf_rotation_pct가 포함된다."""
        from src.utils.feature_flags import FeatureFlags

        defaults = FeatureFlags.DEFAULT_FLAGS["etf_rotation"]
        assert "etf_rotation_pct" in defaults["config"]
        assert defaults["config"]["etf_rotation_pct"] == 0.30

    def test_daily_simulation_includes_etf_rotation(self):
        """daily_simulation 기본 전략에 etf_rotation이 포함된다."""
        from src.utils.feature_flags import FeatureFlags

        strategies = FeatureFlags.DEFAULT_FLAGS["daily_simulation"]["config"]["strategies"]
        assert "etf_rotation" in strategies

    def test_daily_simulation_has_primary_strategy(self):
        """daily_simulation config에 primary_strategy가 포함된다."""
        from src.utils.feature_flags import FeatureFlags

        config = FeatureFlags.DEFAULT_FLAGS["daily_simulation"]["config"]
        assert "primary_strategy" in config
        assert config["primary_strategy"] == "multi_factor"


# ===================================================================
# Fallback Lookback + Diagnostics 테스트
# ===================================================================


class TestFallbackLookback:
    """단계적 fallback lookback 및 진단 정보 검증."""

    def _make_etf_price(self, start_price, end_price, n_days=300):
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        prices = np.linspace(start_price, end_price, n_days)
        return pd.DataFrame(
            {"close": prices, "volume": [1_000_000] * n_days},
            index=dates,
        )

    def test_build_fallback_lookbacks_default(self):
        """기본 lookback=252에서 fallback 목록이 [252, 126, 63]이다."""
        s = ETFRotationStrategy(lookback=252)
        assert s._fallback_lookbacks == [252, 126, 63]

    def test_build_fallback_lookbacks_short(self):
        """lookback=90이면 fallback 목록이 [90, 63]이다."""
        s = ETFRotationStrategy(lookback=90)
        assert s._fallback_lookbacks == [90, 63]

    def test_build_fallback_lookbacks_minimum(self):
        """lookback=63이면 fallback 목록이 [63]이다."""
        s = ETFRotationStrategy(lookback=63)
        assert s._fallback_lookbacks == [63]

    def test_fallback_uses_shorter_lookback(self):
        """252일 데이터 부족 시 126일로 fallback한다."""
        s = ETFRotationStrategy(
            lookback=252,
            num_etfs=2,
            abs_momentum=False,
            etf_universe={
                "A": "ETF_A", "B": "ETF_B", "C": "ETF_C", "439870": "채권",
            },
        )
        # 150일 데이터 → 252일 부족, 126일 충분
        etf_prices = {
            "A": self._make_etf_price(10000, 13000, n_days=150),
            "B": self._make_etf_price(10000, 12000, n_days=150),
            "C": self._make_etf_price(10000, 11000, n_days=150),
        }
        signals = s.generate_signals("20240101", {"etf_prices": etf_prices})
        assert len(signals) > 0
        assert s.last_diagnostics["status"] == "DEGRADED"
        assert s.last_diagnostics["lookback_used"] == 126

    def test_fallback_to_63_days(self):
        """126일도 부족하면 63일로 fallback한다."""
        s = ETFRotationStrategy(
            lookback=252,
            num_etfs=2,
            abs_momentum=False,
            etf_universe={
                "A": "ETF_A", "B": "ETF_B", "C": "ETF_C", "439870": "채권",
            },
        )
        # 80일 데이터 → 252/126 부족, 63 충분
        etf_prices = {
            "A": self._make_etf_price(10000, 13000, n_days=80),
            "B": self._make_etf_price(10000, 12000, n_days=80),
            "C": self._make_etf_price(10000, 11000, n_days=80),
        }
        signals = s.generate_signals("20240101", {"etf_prices": etf_prices})
        assert len(signals) > 0
        assert s.last_diagnostics["status"] == "DEGRADED"
        assert s.last_diagnostics["lookback_used"] == 63

    def test_all_fallbacks_exhausted(self):
        """모든 fallback을 소진하면 빈 시그널 + DATA_INSUFFICIENT."""
        s = ETFRotationStrategy(
            lookback=252,
            num_etfs=2,
            etf_universe={
                "A": "ETF_A", "B": "ETF_B", "439870": "채권",
            },
        )
        # 30일 데이터 → 63일 미만
        etf_prices = {
            "A": self._make_etf_price(10000, 13000, n_days=30),
            "B": self._make_etf_price(10000, 12000, n_days=30),
        }
        signals = s.generate_signals("20240101", {"etf_prices": etf_prices})
        assert signals == {}
        assert s.last_diagnostics["status"] == "DATA_INSUFFICIENT"

    def test_sufficient_data_no_fallback(self):
        """데이터 충분하면 원래 lookback 사용 + status=OK."""
        s = ETFRotationStrategy(
            lookback=60,
            num_etfs=2,
            abs_momentum=False,
            etf_universe={
                "A": "ETF_A", "B": "ETF_B", "C": "ETF_C", "439870": "채권",
            },
        )
        etf_prices = {
            "A": self._make_etf_price(10000, 13000, n_days=100),
            "B": self._make_etf_price(10000, 12000, n_days=100),
            "C": self._make_etf_price(10000, 11000, n_days=100),
        }
        signals = s.generate_signals("20240101", {"etf_prices": etf_prices})
        assert len(signals) > 0
        assert s.last_diagnostics["status"] == "OK"
        assert s.last_diagnostics["lookback_used"] == 60

    def test_diagnostics_per_ticker_data_short(self):
        """데이터 부족 ETF에 DATA_SHORT 진단이 기록된다."""
        s = ETFRotationStrategy(
            lookback=252,
            num_etfs=2,
            abs_momentum=False,
            etf_universe={
                "A": "ETF_A", "B": "ETF_B", "439870": "채권",
            },
        )
        etf_prices = {
            "A": self._make_etf_price(10000, 13000, n_days=300),
            "B": self._make_etf_price(10000, 12000, n_days=30),  # 부족
        }
        signals = s.generate_signals("20240101", {"etf_prices": etf_prices})
        per = s.last_diagnostics["per_ticker"]
        assert per["A"]["status"] == "OK"
        assert per["B"]["status"] == "DATA_SHORT"
        assert per["B"]["available_days"] == 30

    def test_diagnostics_per_ticker_data_missing(self):
        """가격 데이터 없는 ETF에 DATA_MISSING 진단이 기록된다."""
        s = ETFRotationStrategy(
            lookback=60,
            num_etfs=2,
            abs_momentum=False,
            etf_universe={
                "A": "ETF_A", "B": "ETF_B", "C": "ETF_C", "439870": "채권",
            },
        )
        etf_prices = {
            "A": self._make_etf_price(10000, 13000, n_days=100),
            "C": self._make_etf_price(10000, 11000, n_days=100),
            # B 누락
        }
        s.generate_signals("20240101", {"etf_prices": etf_prices})
        per = s.last_diagnostics["per_ticker"]
        assert per["B"]["status"] == "DATA_MISSING"

    def test_diagnostics_empty_prices(self):
        """빈 가격 데이터에서 DATA_UNAVAILABLE 진단이 기록된다."""
        s = ETFRotationStrategy()
        s.generate_signals("20240101", {"etf_prices": {}})
        assert s.last_diagnostics["status"] == "DATA_UNAVAILABLE"


# ===================================================================
# /balance 커맨드 테스트
# ===================================================================


class TestBalanceCommand:
    """/balance 커맨드 검증."""

    def _make_etf_price(self, start_price, end_price, n_days=300):
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        prices = np.linspace(start_price, end_price, n_days)
        return pd.DataFrame(
            {"close": prices, "volume": [1_000_000] * n_days},
            index=dates,
        )

    def _make_mock_bot(
        self,
        kis_configured=True,
        etf_flag=True,
        allocator=True,
        daily_sim_flag=False,
    ):
        """테스트용 mock TradingBot 객체를 생성한다."""
        import types
        from unittest.mock import MagicMock
        from src.scheduler.main import TradingBot

        bot = MagicMock()

        # KIS client
        bot.kis_client.is_configured.return_value = kis_configured
        bot.kis_client.get_balance.return_value = {
            "total_eval": 1_500_000,
            "cash": 200_000,
            "total_profit_pct": 3.45,
            "holdings": [
                {
                    "ticker": "371460",
                    "name": "TIGER미국S&P500",
                    "qty": 10,
                    "eval_amount": 700_000,
                    "pnl_pct": 5.0,
                },
                {
                    "ticker": "469150",
                    "name": "ACE AI반도체",
                    "qty": 20,
                    "eval_amount": 500_000,
                    "pnl_pct": 2.0,
                },
                {
                    "ticker": "005930",
                    "name": "삼성전자",
                    "qty": 5,
                    "eval_amount": 300_000,
                    "pnl_pct": -1.5,
                },
            ],
        }

        # Feature flags
        def _is_enabled(name):
            if name == "etf_rotation":
                return etf_flag
            if name == "daily_simulation":
                return daily_sim_flag
            return False

        bot.feature_flags.is_enabled.side_effect = _is_enabled

        def _get_config(name):
            if name == "etf_rotation":
                return {
                    "lookback_months": 12,
                    "n_select": 2,
                    "etf_rotation_pct": 0.30,
                }
            if name == "daily_simulation":
                return {
                    "strategies": ["multi_factor", "three_factor", "etf_rotation"],
                    "report_time": "16:00",
                    "primary_strategy": "multi_factor",
                }
            return {}

        bot.feature_flags.get_config.side_effect = _get_config

        # Allocator
        if allocator:
            bot.allocator = MagicMock()
            bot.allocator.rebalance_allocation.return_value = {
                "long_term_target": 0.70,
                "long_term_actual": 0.68,
                "long_term_eval": 1_020_000,
                "etf_rotation_target": 0.30,
                "etf_rotation_actual": 0.32,
                "etf_rotation_eval": 480_000,
                "short_term_target": 0.0,
                "short_term_actual": 0.0,
                "short_term_eval": 0,
            }
            # 풀별 포지션
            bot.allocator.get_positions_by_pool.side_effect = (
                lambda pool: {
                    "long_term": [
                        {
                            "ticker": "005930",
                            "qty": 5,
                            "eval_amount": 300_000,
                            "pnl_pct": -1.5,
                        },
                    ],
                    "etf_rotation": [
                        {
                            "ticker": "371460",
                            "qty": 10,
                            "eval_amount": 700_000,
                            "pnl_pct": 5.0,
                        },
                        {
                            "ticker": "469150",
                            "qty": 20,
                            "eval_amount": 500_000,
                            "pnl_pct": 2.0,
                        },
                    ],
                    "short_term": [],
                }.get(pool, [])
            )
        else:
            bot.allocator = None

        # Executor
        bot.executor = MagicMock()
        bot.executor.dry_run.return_value = {
            "sell_orders": [{"ticker": "069500", "qty": 5}],
            "buy_orders": [
                {"ticker": "133690", "qty": 3},
                {"ticker": "464310", "qty": 2},
            ],
        }

        # Bind real helper methods to mock bot
        bot._format_merged_holdings = types.MethodType(
            TradingBot._format_merged_holdings, bot
        )
        bot._generate_long_term_preview = types.MethodType(
            TradingBot._generate_long_term_preview, bot
        )
        bot._find_latest_simulation = TradingBot._find_latest_simulation

        return bot

    def test_balance_kis_not_configured(self):
        """KIS 미설정 시 안내 메시지를 반환한다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(kis_configured=False)
        result = TradingBot._cmd_balance(bot, "")
        assert "미설정" in result

    def test_balance_basic_info(self):
        """잔고 기본 정보가 포함된다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=False, allocator=False)
        result = TradingBot._cmd_balance(bot, "")
        assert "총 평가" in result
        assert "1,500,000" in result
        assert "현금" in result
        assert "+3.45%" in result

    def test_balance_merged_holdings(self):
        """통합 보유 종목 리스트에 풀 태그가 포함된다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=False)
        result = TradingBot._cmd_balance(bot, "")
        # 통합 헤더
        assert "현재 보유 종목" in result
        # 풀 태그 확인
        assert "장기|" in result
        assert "ETF|" in result
        # 종목 상세 확인 (수량, 금액)
        assert "주" in result
        assert "원" in result

    def test_balance_pool_tag_details(self):
        """풀 태그와 함께 종목 이름, 수량, 평가금, 수익률이 출력된다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=False)
        result = TradingBot._cmd_balance(bot, "")
        # 장기 풀 종목 (삼성전자)
        assert "삼성전자" in result
        assert "300,000" in result
        # ETF 풀 종목
        assert "TIGER미국S&" in result or "TIGER미국" in result
        assert "700,000" in result

    def test_balance_etf_preview_with_flag_on(self):
        """ETF 리밸런싱 프리뷰가 포함된다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=True)

        # Mock _generate_etf_preview
        preview = [
            "  모멘텀 순위:",
            "    1. TIGER 미국나스닥100: +18.5%",
            "  선정 ETF:",
            "    TIGER 미국나스닥100: 50.0%",
            "  예상 변경: 매도 1건, 매수 2건",
        ]
        bot._generate_etf_preview = lambda: preview

        result = TradingBot._cmd_balance(bot, "")
        assert "ETF 리밸런싱 프리뷰" in result
        assert "모멘텀 순위" in result

    def test_balance_etf_preview_disabled(self):
        """ETF 플래그 OFF 시 프리뷰가 없다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=False)
        result = TradingBot._cmd_balance(bot, "")
        assert "ETF 리밸런싱 프리뷰" not in result

    def test_balance_without_allocator(self):
        """allocator 없을 때 전체 보유목록이 풀 태그 없이 표시된다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=False, allocator=False)
        result = TradingBot._cmd_balance(bot, "")
        # 풀 태그 없이 종목 표시
        assert "장기|" not in result
        assert "ETF|" not in result
        # 보유 종목은 표시됨
        assert "현재 보유 종목" in result
        assert "TIGER미국S&" in result or "TIGER미국" in result
        assert "삼성전자" in result

    def test_balance_pool_allocation_summary(self):
        """풀 배분 요약라인이 계좌 총괄에 포함된다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=False)
        result = TradingBot._cmd_balance(bot, "")
        assert "풀 배분" in result
        assert "장기" in result
        assert "ETF" in result

    def test_balance_long_term_preview_with_sim_data(self):
        """시뮬레이션 캐시가 있으면 장기 전략 프리뷰가 표시된다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=False, daily_sim_flag=True)

        sim_data = {
            "date": "2026-02-25",
            "strategy": "multi_factor",
            "selected": [
                {"ticker": "005930", "name": "삼성전자", "weight": 0.10, "rank": 1},
                {"ticker": "000660", "name": "SK하이닉스", "weight": 0.10, "rank": 2},
                {"ticker": "005380", "name": "현대차", "weight": 0.10, "rank": 3},
            ],
        }

        bot._find_latest_simulation = lambda name: sim_data
        result = TradingBot._cmd_balance(bot, "")

        assert "장기 전략 프리뷰" in result
        assert "2026-02-25" in result
        assert "삼성전자" in result
        assert "(보유중)" in result
        assert "SK하이닉스" in result
        assert "(신규)" in result
        # dry_run 호출 확인
        assert "예상 변경" in result

    def test_balance_long_term_preview_no_sim_data(self):
        """시뮬레이션 캐시가 없으면 안내 메시지가 표시된다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=False, daily_sim_flag=True)

        bot._find_latest_simulation = lambda name: None
        result = TradingBot._cmd_balance(bot, "")

        assert "장기 전략 프리뷰" in result
        assert "시뮬레이션 데이터 없음" in result
        assert "16:05" in result

    def test_balance_long_term_preview_disabled(self):
        """daily_simulation OFF 시 장기 전략 프리뷰가 없다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=False, daily_sim_flag=False)
        result = TradingBot._cmd_balance(bot, "")
        assert "장기 전략 프리뷰" not in result

    def test_balance_untagged_holdings(self):
        """allocator에 태깅되지 않은 종목에 '미분류' 태그가 붙는다."""
        from unittest.mock import MagicMock
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=False)
        # 005930만 long_term, 나머지는 미태깅
        bot.allocator.get_positions_by_pool.side_effect = (
            lambda pool: {
                "long_term": [
                    {"ticker": "005930", "qty": 5, "eval_amount": 300_000, "pnl_pct": -1.5},
                ],
                "etf_rotation": [],
                "short_term": [],
            }.get(pool, [])
        )

        result = TradingBot._cmd_balance(bot, "")
        assert "미분류|" in result
        assert "장기|" in result


# ===================================================================
# ETF 리밸런싱 E2E 통합 테스트
# ===================================================================


class TestETFRotationRebalanceE2E:
    """ETF 로테이션 리밸런싱 전체 플로우 통합 테스트.

    execute_etf_rotation_rebalance()의 전체 흐름을 검증한다.
    KIS API 호출 직전까지의 시그널 생성 → 스케일링 → 주문 계산을 검증.
    """

    def _make_etf_price(self, start_price, end_price, n_days=300):
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        prices = np.linspace(start_price, end_price, n_days)
        return pd.DataFrame(
            {"close": prices, "volume": [1_000_000] * n_days},
            index=dates,
        )

    def _make_mock_bot(self, is_rebalance_day=True, etf_flag=True):
        """통합 테스트용 mock TradingBot을 생성한다."""
        from unittest.mock import MagicMock, patch

        bot = MagicMock()

        # Feature flags
        bot.feature_flags.is_enabled.side_effect = lambda name: (
            etf_flag if name == "etf_rotation" else False
        )
        bot.feature_flags.get_config.return_value = {
            "lookback_months": 12,
            "n_select": 2,
            "rebalance_freq": "monthly",
            "etf_rotation_pct": 0.30,
        }

        # Holidays
        bot.holidays.is_rebalance_day.return_value = is_rebalance_day
        bot._is_trading_day.return_value = True

        # KIS client
        bot.kis_client.is_configured.return_value = True

        # Allocator (real-like)
        bot.allocator = MagicMock()
        bot.allocator._etf_rotation_pct = 0.30

        # Executor
        bot.executor.execute_rebalance.return_value = {
            "success": True,
            "sells": [],
            "buys": [
                {"ticker": "133690", "qty": 3, "amount": 150_000},
                {"ticker": "069500", "qty": 5, "amount": 200_000},
            ],
            "total_sell_amount": 0,
            "total_buy_amount": 350_000,
            "errors": [],
        }

        # Notification
        bot._send_notification = MagicMock()

        # Mock _fetch_etf_prices
        bot._fetch_etf_prices = MagicMock(return_value={
            "069500": self._make_etf_price(10000, 13000),  # +30%
            "133690": self._make_etf_price(10000, 12000),  # +20%
            "371460": self._make_etf_price(10000, 11000),  # +10%
            "091160": self._make_etf_price(10000, 9000),   # -10%
            "091170": self._make_etf_price(10000, 8500),   # -15%
            "117700": self._make_etf_price(10000, 9500),   # -5%
            "132030": self._make_etf_price(10000, 10500),  # +5%
            "464310": self._make_etf_price(10000, 11500),  # +15%
            "469150": self._make_etf_price(10000, 10200),  # +2%
        })

        return bot

    def test_full_flow_monthly_first_day(self):
        """월초 거래일에 정상 리밸런싱이 실행된다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(is_rebalance_day=True)

        # 실행
        TradingBot.execute_etf_rotation_rebalance(bot)

        # 검증: executor.execute_rebalance가 호출되었다
        bot.executor.execute_rebalance.assert_called_once()
        call_args = bot.executor.execute_rebalance.call_args
        signals = call_args[0][0]
        pool = call_args[1].get("pool") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("pool")

        # 시그널이 dict이고 비어있지 않다
        assert isinstance(signals, dict)
        assert len(signals) > 0

        # pool은 etf_rotation이다
        assert pool == "etf_rotation"

        # 비중 합이 1.0이다
        total_weight = sum(signals.values())
        assert abs(total_weight - 1.0) < 1e-6

        # tag_position이 시그널 수만큼 호출되었다
        assert bot.allocator.tag_position.call_count == len(signals)

        # 알림이 발송되었다 (시작 + 완료 = 최소 2회)
        assert bot._send_notification.call_count >= 2

    def test_top_momentum_selected(self):
        """모멘텀 상위 2개 ETF가 선정된다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(is_rebalance_day=True)

        TradingBot.execute_etf_rotation_rebalance(bot)

        call_args = bot.executor.execute_rebalance.call_args
        signals = call_args[0][0]

        # 069500(+30%)과 133690(+20%)이 상위 2개
        assert "069500" in signals
        assert "133690" in signals

    def test_skipped_non_rebalance_day(self):
        """리밸런싱일이 아니면 executor가 호출되지 않는다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(is_rebalance_day=False)

        TradingBot.execute_etf_rotation_rebalance(bot)

        bot.executor.execute_rebalance.assert_not_called()

    def test_flag_disabled_skips(self):
        """etf_rotation 플래그 OFF 시 즉시 리턴한다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(etf_flag=False)

        TradingBot.execute_etf_rotation_rebalance(bot)

        bot.executor.execute_rebalance.assert_not_called()
        bot._fetch_etf_prices.assert_not_called()

    def test_empty_prices_skips(self):
        """ETF 가격 데이터가 없으면 스킵한다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(is_rebalance_day=True)
        bot._fetch_etf_prices.return_value = {}

        TradingBot.execute_etf_rotation_rebalance(bot)

        bot.executor.execute_rebalance.assert_not_called()

    def test_all_negative_momentum_uses_safe_asset(self):
        """모든 ETF가 하락 시 안전자산(단기채권)으로 전환한다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(is_rebalance_day=True)
        # 모든 ETF 하락
        bot._fetch_etf_prices.return_value = {
            "069500": self._make_etf_price(10000, 8000),   # -20%
            "133690": self._make_etf_price(10000, 7000),   # -30%
            "371460": self._make_etf_price(10000, 9000),   # -10%
            "091160": self._make_etf_price(10000, 8500),   # -15%
            "091170": self._make_etf_price(10000, 7500),   # -25%
            "117700": self._make_etf_price(10000, 8200),   # -18%
            "132030": self._make_etf_price(10000, 9500),   # -5%
            "464310": self._make_etf_price(10000, 8800),   # -12%
            "469150": self._make_etf_price(10000, 9200),   # -8%
        }

        TradingBot.execute_etf_rotation_rebalance(bot)

        call_args = bot.executor.execute_rebalance.call_args
        signals = call_args[0][0]

        # 안전자산(439870)이 포함되어야 한다
        assert "439870" in signals
        # 안전자산 비중이 1.0이어야 한다
        assert abs(signals["439870"] - 1.0) < 1e-6

    def test_notification_contains_result_info(self):
        """알림에 결과 정보가 포함된다."""
        from src.scheduler.main import TradingBot

        bot = self._make_mock_bot(is_rebalance_day=True)

        TradingBot.execute_etf_rotation_rebalance(bot)

        # 완료 알림 검증
        calls = bot._send_notification.call_args_list
        messages = [str(c) for c in calls]
        combined = " ".join(messages)
        assert "ETF 로테이션" in combined


# ===================================================================
# 섹터 집중도 필터 테스트
# ===================================================================

class TestSectorFilter:
    """ETF 섹터 집중도 제한 검증."""

    @staticmethod
    def _make_etf_price(start, end, n_days=300):
        prices = np.linspace(start, end, n_days)
        dates = pd.date_range(end="2025-02-25", periods=n_days, freq="B")
        return pd.DataFrame({"close": prices}, index=dates)

    def test_same_sector_limited(self):
        """같은 섹터(반도체) ETF가 max_same_sector=1로 제한된다."""
        universe = {
            "091160": "KODEX 반도체",
            "469150": "ACE AI반도체포커스",
            "069500": "KODEX 200",
            "117700": "KODEX 건설",
            "439870": "KODEX 단기채권",
        }
        s = ETFRotationStrategy(
            lookback=60, num_etfs=3, etf_universe=universe,
            max_same_sector=1, abs_momentum=False,
        )

        etf_prices = {
            "091160": self._make_etf_price(10000, 20000),   # 반도체 +100%
            "469150": self._make_etf_price(10000, 23000),   # 반도체 +130% (상위)
            "069500": self._make_etf_price(10000, 17000),   # 국내지수 +70%
            "117700": self._make_etf_price(10000, 15000),   # 건설 +50%
        }

        signals = s.generate_signals("20250225", {"etf_prices": etf_prices})

        # 반도체 1개만 선정 (469150이 모멘텀 상위)
        assert "469150" in signals
        assert "091160" not in signals
        # 나머지 2 슬롯은 다른 섹터
        assert "069500" in signals
        assert "117700" in signals

    def test_sector_filter_fills_from_next(self):
        """섹터 스킵 후 다음 순위 ETF로 대체된다."""
        universe = {
            "091160": "KODEX 반도체",
            "469150": "ACE AI반도체포커스",
            "069500": "KODEX 200",
            "117700": "KODEX 건설",
            "091170": "KODEX 은행",
            "439870": "KODEX 단기채권",
        }
        s = ETFRotationStrategy(
            lookback=60, num_etfs=3, etf_universe=universe,
            max_same_sector=1, abs_momentum=False,
        )

        etf_prices = {
            "469150": self._make_etf_price(10000, 25000),   # 반도체 1위
            "091160": self._make_etf_price(10000, 22000),   # 반도체 2위 → 스킵
            "069500": self._make_etf_price(10000, 18000),   # 국내지수 3위
            "117700": self._make_etf_price(10000, 15000),   # 건설 4위 → 대체 선정
            "091170": self._make_etf_price(10000, 12000),   # 금융 5위
        }

        signals = s.generate_signals("20250225", {"etf_prices": etf_prices})

        assert len(signals) == 3
        assert "469150" in signals  # 반도체 1개
        assert "069500" in signals  # 국내지수
        assert "117700" in signals  # 건설 (대체)
        assert "091160" not in signals  # 반도체 스킵

    def test_sector_filter_disabled(self):
        """max_same_sector=0이면 섹터 제한 없이 기존 동작."""
        universe = {
            "091160": "KODEX 반도체",
            "469150": "ACE AI반도체포커스",
            "069500": "KODEX 200",
            "439870": "KODEX 단기채권",
        }
        s = ETFRotationStrategy(
            lookback=60, num_etfs=2, etf_universe=universe,
            max_same_sector=0, abs_momentum=False,
        )

        etf_prices = {
            "091160": self._make_etf_price(10000, 20000),
            "469150": self._make_etf_price(10000, 23000),
            "069500": self._make_etf_price(10000, 17000),
        }

        signals = s.generate_signals("20250225", {"etf_prices": etf_prices})

        # 섹터 제한 없이 모멘텀 상위 2개 (반도체 2개 모두 선정)
        assert "469150" in signals
        assert "091160" in signals
        assert "069500" not in signals

    def test_diagnostics_includes_sector_info(self):
        """진단 정보에 섹터 스킵 기록이 포함된다."""
        universe = {
            "091160": "KODEX 반도체",
            "469150": "ACE AI반도체포커스",
            "069500": "KODEX 200",
            "117700": "KODEX 건설",
            "439870": "KODEX 단기채권",
        }
        s = ETFRotationStrategy(
            lookback=60, num_etfs=2, etf_universe=universe,
            max_same_sector=1, abs_momentum=False,
        )

        etf_prices = {
            "469150": self._make_etf_price(10000, 25000),
            "091160": self._make_etf_price(10000, 22000),
            "069500": self._make_etf_price(10000, 18000),
            "117700": self._make_etf_price(10000, 15000),
        }

        s.generate_signals("20250225", {"etf_prices": etf_prices})

        assert "sector_skipped" in s.last_diagnostics
        skipped = s.last_diagnostics["sector_skipped"]
        assert len(skipped) == 1
        assert skipped[0]["ticker"] == "091160"
        assert skipped[0]["sector"] == "반도체"


# ===================================================================
# 모멘텀 캡 (Winsorization) 테스트
# ===================================================================

class TestMomentumCap:
    """ETF 모멘텀 캡 검증."""

    @staticmethod
    def _make_etf_price(start, end, n_days=300):
        prices = np.linspace(start, end, n_days)
        dates = pd.date_range(end="2025-02-25", periods=n_days, freq="B")
        return pd.DataFrame({"close": prices}, index=dates)

    def test_extreme_momentum_is_capped(self):
        """극단적 모멘텀(+400%)이 momentum_cap(300%)으로 캡된다."""
        universe = {
            "A": "ETF_A", "B": "ETF_B", "C": "ETF_C", "439870": "채권",
        }
        s = ETFRotationStrategy(
            lookback=60, num_etfs=2, etf_universe=universe,
            abs_momentum=False, momentum_cap=3.0,
        )

        # n_days=lookback과 동일하게 설정하여 전체 기간 수익률 계산
        etf_prices = {
            "A": self._make_etf_price(10000, 50000, n_days=60),   # +400% → 캡 300%
            "B": self._make_etf_price(10000, 25000, n_days=60),   # +150%
            "C": self._make_etf_price(10000, 15000, n_days=60),   # +50%
        }

        s._calculate_momentum(etf_prices, lookback_override=60)

        per = s._last_per_ticker
        assert per["A"]["capped"] is True
        assert per["A"]["momentum"] == 3.0
        assert per["B"]["capped"] is False

    def test_momentum_cap_disabled(self):
        """momentum_cap=0이면 캡이 비활성화된다."""
        universe = {
            "A": "ETF_A", "439870": "채권",
        }
        s = ETFRotationStrategy(
            lookback=60, num_etfs=1, etf_universe=universe,
            abs_momentum=False, momentum_cap=0,
        )

        etf_prices = {
            "A": self._make_etf_price(10000, 50000, n_days=60),   # +400%
        }

        momentum = s._calculate_momentum(etf_prices, lookback_override=60)

        assert momentum["A"] > 3.0  # 캡 없이 원래 값
        assert s._last_per_ticker["A"]["capped"] is False

    def test_momentum_cap_preserves_ranking(self):
        """모멘텀 캡 후에도 랭킹이 올바르게 유지된다."""
        universe = {
            "A": "ETF_A", "B": "ETF_B", "C": "ETF_C",
            "D": "ETF_D", "439870": "채권",
        }
        s = ETFRotationStrategy(
            lookback=60, num_etfs=3, etf_universe=universe,
            abs_momentum=False, momentum_cap=2.0,
            max_same_sector=0,
        )

        etf_prices = {
            "A": self._make_etf_price(10000, 50000),   # +400% → 캡 200%
            "B": self._make_etf_price(10000, 40000),   # +300% → 캡 200%
            "C": self._make_etf_price(10000, 25000),   # +150% (캡 미적용)
            "D": self._make_etf_price(10000, 12000),   # +20%
        }

        signals = s.generate_signals("20250225", {"etf_prices": etf_prices})

        # A, B는 캡 후 동률 → C가 3순위
        assert "C" in signals
        assert len(signals) == 3

    def test_default_momentum_cap(self):
        """기본 momentum_cap이 3.0이다."""
        s = ETFRotationStrategy()
        assert s.momentum_cap == 3.0

    def test_feature_flag_has_momentum_cap(self):
        """etf_rotation flag config에 momentum_cap이 포함된다."""
        from src.utils.feature_flags import FeatureFlags

        defaults = FeatureFlags.DEFAULT_FLAGS["etf_rotation"]
        assert "momentum_cap" in defaults["config"]
        assert defaults["config"]["momentum_cap"] == 3.0
