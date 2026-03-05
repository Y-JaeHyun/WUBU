"""strategy_config 모듈 테스트."""

import pytest

from src.strategy.strategy_config import (
    MULTI_FACTOR_BASE,
    MULTI_FACTOR_PROFILES,
    get_multi_factor_config,
    create_multi_factor,
)


class TestMultiFactorBase:
    """MULTI_FACTOR_BASE 기본 설정 테스트."""

    def test_base_config_has_all_keys(self):
        required_keys = {
            "factors",
            "weights",
            "combine_method",
            "turnover_penalty",
            "max_group_weight",
            "max_stocks_per_conglomerate",
            "spike_filter",
            "spike_threshold_1d",
            "spike_threshold_5d",
            "value_trap_filter",
            "min_roe",
            "min_f_score",
        }
        assert required_keys.issubset(set(MULTI_FACTOR_BASE.keys()))

    def test_base_spike_filter_enabled(self):
        assert MULTI_FACTOR_BASE["spike_filter"] is True

    def test_base_value_trap_disabled(self):
        assert MULTI_FACTOR_BASE["value_trap_filter"] is False


class TestProfiles:
    """프로필 설정 테스트."""

    def test_live_profile_defaults(self):
        config = get_multi_factor_config("live")
        assert config["num_stocks"] == 7
        assert config["apply_market_timing"] is True

    def test_backtest_profile_defaults(self):
        config = get_multi_factor_config("backtest")
        assert config["num_stocks"] == 10
        assert config["apply_market_timing"] is False

    def test_profiles_share_common_base(self):
        live = get_multi_factor_config("live")
        backtest = get_multi_factor_config("backtest")
        for key in MULTI_FACTOR_BASE:
            assert live[key] == backtest[key], f"Mismatch on base key: {key}"

    def test_unknown_profile_uses_base_only(self):
        config = get_multi_factor_config("nonexistent")
        assert config == MULTI_FACTOR_BASE


class TestOverrides:
    """오버라이드 우선순위 테스트."""

    def test_override_takes_precedence_over_profile(self):
        config = get_multi_factor_config("live", num_stocks=20)
        assert config["num_stocks"] == 20

    def test_override_takes_precedence_over_base(self):
        config = get_multi_factor_config("live", spike_filter=False)
        assert config["spike_filter"] is False

    def test_multiple_overrides(self):
        config = get_multi_factor_config(
            "backtest",
            num_stocks=5,
            turnover_penalty=0.0,
            spike_filter=False,
        )
        assert config["num_stocks"] == 5
        assert config["turnover_penalty"] == 0.0
        assert config["spike_filter"] is False


class TestFactory:
    """create_multi_factor 팩토리 테스트."""

    def test_create_returns_strategy(self):
        from src.strategy.multi_factor import MultiFactorStrategy

        s = create_multi_factor("live")
        assert isinstance(s, MultiFactorStrategy)

    def test_create_live_profile(self):
        s = create_multi_factor("live")
        assert s.num_stocks == 7

    def test_create_backtest_profile(self):
        s = create_multi_factor("backtest")
        assert s.num_stocks == 10

    def test_create_with_overrides(self):
        s = create_multi_factor("backtest", num_stocks=15)
        assert s.num_stocks == 15

    def test_create_applies_spike_filter(self):
        s = create_multi_factor("live")
        assert s.spike_filter is True
