"""단기 트레이딩 모드 필터링 테스트."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.execution.short_term_trader import ShortTermTrader
from src.strategy.short_term_base import ShortTermSignal, ShortTermStrategy


# ──────────────────────────────────────────────────────────
# Fake strategies
# ──────────────────────────────────────────────────────────


class FakeSwingStrategy(ShortTermStrategy):
    name = "fake_swing"
    mode = "swing"

    def scan_signals(self, market_data):
        return [
            ShortTermSignal(
                id="",
                ticker="005930",
                strategy=self.name,
                side="buy",
                mode=self.mode,
                confidence=0.8,
                reason="test swing signal",
            )
        ]

    def check_exit(self, position, market_data):
        return None


class FakeDaytradingStrategy(ShortTermStrategy):
    name = "fake_daytrading"
    mode = "daytrading"

    def scan_signals(self, market_data):
        return [
            ShortTermSignal(
                id="",
                ticker="035720",
                strategy=self.name,
                side="buy",
                mode=self.mode,
                confidence=0.7,
                reason="test daytrading signal",
            )
        ]

    def check_exit(self, position, market_data):
        return None


# ──────────────────────────────────────────────────────────
# 1. 모드 프로퍼티 테스트
# ──────────────────────────────────────────────────────────


class TestModeProperty:
    """mode 프로퍼티 테스트."""

    def test_default_mode(self, tmp_path):
        """기본 모드는 swing."""
        trader = ShortTermTrader(signals_path=str(tmp_path / "signals.json"))
        assert trader.mode == "swing"

    def test_set_mode_swing(self, tmp_path):
        """mode를 swing으로 설정."""
        trader = ShortTermTrader(signals_path=str(tmp_path / "signals.json"))
        trader.mode = "swing"
        assert trader.mode == "swing"

    def test_set_mode_daytrading(self, tmp_path):
        """mode를 daytrading으로 설정."""
        trader = ShortTermTrader(signals_path=str(tmp_path / "signals.json"))
        trader.mode = "daytrading"
        assert trader.mode == "daytrading"

    def test_set_mode_multi(self, tmp_path):
        """mode를 multi로 설정."""
        trader = ShortTermTrader(signals_path=str(tmp_path / "signals.json"))
        trader.mode = "multi"
        assert trader.mode == "multi"

    def test_set_invalid_mode(self, tmp_path):
        """유효하지 않은 모드 설정 시 ValueError."""
        trader = ShortTermTrader(signals_path=str(tmp_path / "signals.json"))
        with pytest.raises(ValueError, match="Invalid mode"):
            trader.mode = "invalid_mode"


# ──────────────────────────────────────────────────────────
# 2. 모드 필터링 테스트
# ──────────────────────────────────────────────────────────


class TestModeFiltering:
    """모드별 시그널 필터링 테스트."""

    def test_swing_mode_only_swing_signals(self, tmp_path):
        """swing 모드에서는 swing 전략만 스캔."""
        trader = ShortTermTrader(
            strategies=[FakeSwingStrategy(), FakeDaytradingStrategy()],
            signals_path=str(tmp_path / "signals.json"),
            mode="swing",
        )
        signals = trader.scan_for_signals()
        assert len(signals) == 1
        assert signals[0].ticker == "005930"
        assert signals[0].strategy == "fake_swing"

    def test_daytrading_mode_only_daytrading_signals(self, tmp_path):
        """daytrading 모드에서는 daytrading 전략만 스캔."""
        trader = ShortTermTrader(
            strategies=[FakeSwingStrategy(), FakeDaytradingStrategy()],
            signals_path=str(tmp_path / "signals.json"),
            mode="daytrading",
        )
        signals = trader.scan_for_signals()
        assert len(signals) == 1
        assert signals[0].ticker == "035720"
        assert signals[0].strategy == "fake_daytrading"

    def test_multi_mode_all_signals(self, tmp_path):
        """multi 모드에서는 모든 전략 스캔."""
        trader = ShortTermTrader(
            strategies=[FakeSwingStrategy(), FakeDaytradingStrategy()],
            signals_path=str(tmp_path / "signals.json"),
            mode="multi",
        )
        signals = trader.scan_for_signals()
        assert len(signals) == 2
        tickers = {s.ticker for s in signals}
        assert tickers == {"005930", "035720"}

    def test_mode_change_affects_scanning(self, tmp_path):
        """모드 변경이 스캔 결과에 즉시 반영."""
        trader = ShortTermTrader(
            strategies=[FakeSwingStrategy(), FakeDaytradingStrategy()],
            signals_path=str(tmp_path / "signals.json"),
            mode="swing",
        )
        # swing 모드: swing 시그널만
        signals_swing = trader.scan_for_signals()
        assert len(signals_swing) == 1
        assert signals_swing[0].strategy == "fake_swing"

        # daytrading 모드로 변경
        trader.mode = "daytrading"
        signals_day = trader.scan_for_signals()
        assert len(signals_day) == 1
        assert signals_day[0].strategy == "fake_daytrading"


# ──────────────────────────────────────────────────────────
# 3. 활성 전략 목록 테스트
# ──────────────────────────────────────────────────────────


class TestGetActiveStrategies:
    """get_active_strategies 테스트."""

    def test_swing_mode_active_strategies(self, tmp_path):
        """swing 모드에서 활성 전략은 swing 전략만."""
        swing = FakeSwingStrategy()
        day = FakeDaytradingStrategy()
        trader = ShortTermTrader(
            strategies=[swing, day],
            signals_path=str(tmp_path / "signals.json"),
            mode="swing",
        )
        active = trader.get_active_strategies()
        assert len(active) == 1
        assert active[0].name == "fake_swing"

    def test_daytrading_mode_active_strategies(self, tmp_path):
        """daytrading 모드에서 활성 전략은 daytrading 전략만."""
        swing = FakeSwingStrategy()
        day = FakeDaytradingStrategy()
        trader = ShortTermTrader(
            strategies=[swing, day],
            signals_path=str(tmp_path / "signals.json"),
            mode="daytrading",
        )
        active = trader.get_active_strategies()
        assert len(active) == 1
        assert active[0].name == "fake_daytrading"

    def test_multi_mode_active_strategies(self, tmp_path):
        """multi 모드에서는 모든 전략이 활성."""
        swing = FakeSwingStrategy()
        day = FakeDaytradingStrategy()
        trader = ShortTermTrader(
            strategies=[swing, day],
            signals_path=str(tmp_path / "signals.json"),
            mode="multi",
        )
        active = trader.get_active_strategies()
        assert len(active) == 2
        names = {s.name for s in active}
        assert names == {"fake_swing", "fake_daytrading"}


# ──────────────────────────────────────────────────────────
# 4. Feature Flag 모드 설정 테스트
# ──────────────────────────────────────────────────────────


class TestFeatureFlagMode:
    """FeatureFlags의 short_term_trading mode 설정 테스트."""

    def test_mode_config_value(self, tmp_path):
        """short_term_trading의 기본 mode 설정이 swing인지 확인."""
        from src.utils.feature_flags import FeatureFlags

        ff = FeatureFlags(flags_path=str(tmp_path / "flags.json"))
        config = ff.get_config("short_term_trading")
        assert config["mode"] == "swing"

    def test_mode_config_change(self, tmp_path):
        """mode 설정 변경이 반영되는지 확인."""
        from src.utils.feature_flags import FeatureFlags

        ff = FeatureFlags(flags_path=str(tmp_path / "flags.json"))
        ff.set_config("short_term_trading", "mode", "multi")
        config = ff.get_config("short_term_trading")
        assert config["mode"] == "multi"

    def test_mode_config_persists(self, tmp_path):
        """mode 변경이 파일에 영속화되는지 확인."""
        from src.utils.feature_flags import FeatureFlags

        flags_file = str(tmp_path / "flags.json")
        ff1 = FeatureFlags(flags_path=flags_file)
        ff1.set_config("short_term_trading", "mode", "daytrading")

        # 새 인스턴스로 로드
        ff2 = FeatureFlags(flags_path=flags_file)
        config = ff2.get_config("short_term_trading")
        assert config["mode"] == "daytrading"

    def test_mode_config_toggle_does_not_affect_mode(self, tmp_path):
        """피처 토글이 mode config에 영향을 주지 않는지 확인."""
        from src.utils.feature_flags import FeatureFlags

        ff = FeatureFlags(flags_path=str(tmp_path / "flags.json"))
        ff.set_config("short_term_trading", "mode", "multi")
        ff.toggle("short_term_trading", enabled=True)
        config = ff.get_config("short_term_trading")
        assert config["mode"] == "multi"
