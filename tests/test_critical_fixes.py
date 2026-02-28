"""CRITICAL/MAJOR 이슈 수정 검증 테스트.

C1: emergency_monitor 스케줄 등록
C2: 오버레이 라이브 리밸런싱 적용
C3: PortfolioTracker EOD 스냅샷
M1: auto_backtest 전략 확장
M2: MIN_ORDER_AMOUNT 정합성
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _import_trading_bot():
    """TradingBot 클래스를 임포트한다."""
    from src.scheduler.main import TradingBot

    return TradingBot


def _make_bot():
    """모든 외부 의존성을 mock하여 TradingBot을 생성한다."""
    TradingBot = _import_trading_bot()

    with patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
    }):
        with (
            patch("src.scheduler.main.KISClient") as mock_kis_cls,
            patch("src.scheduler.main.TelegramNotifier") as mock_notifier_cls,
            patch("src.scheduler.main.PortfolioAllocator"),
            patch("src.scheduler.main.ShortTermTrader"),
            patch("src.scheduler.main.SwingReversionStrategy") as mock_swing_cls,
            patch("src.scheduler.main.FeatureFlags") as mock_flags_cls,
            patch("src.scheduler.main.DataCache"),
            patch("src.scheduler.main.TelegramCommander"),
            patch("src.scheduler.main.PortfolioTracker"),
            patch("src.scheduler.main.StockReviewer"),
            patch("src.scheduler.main.NightResearcher"),
        ):
            mock_kis = MagicMock()
            mock_kis.is_paper = True
            mock_kis.is_configured.return_value = True
            mock_kis.mode_tag = "[모의]"
            mock_kis.trading_mode = "모의"
            mock_kis_cls.return_value = mock_kis

            mock_notifier = MagicMock()
            mock_notifier.is_configured.return_value = False
            mock_notifier_cls.return_value = mock_notifier

            mock_flags = MagicMock()
            mock_flags.is_enabled.return_value = False
            mock_flags.get_config.return_value = {}
            mock_flags.get_all_status.return_value = {}
            mock_flags.get_summary.return_value = "[Feature Flags]"
            mock_flags_cls.return_value = mock_flags

            mock_swing = MagicMock()
            mock_swing.name = "swing_reversion"
            mock_swing.mode = "swing"
            mock_swing_cls.return_value = mock_swing

            bot = TradingBot()

    return bot


# ── C1: emergency_monitor 스케줄 등록 ─────────────────────


class TestEmergencyMonitorSchedule:
    """C1: emergency_monitor가 setup_schedule()에 등록되는지 검증."""

    def test_emergency_monitor_registered(self):
        """setup_schedule() 후 emergency_monitor 작업이 존재한다."""
        bot = _make_bot()
        bot.setup_schedule()

        job_ids = [j.id for j in bot.scheduler.get_jobs()]
        assert "emergency_monitor" in job_ids

    def test_total_job_count_increased(self):
        """스케줄 등록 후 20개 작업이 존재한다 (기존 19 + emergency_monitor)."""
        bot = _make_bot()
        bot.setup_schedule()

        assert len(bot.scheduler.get_jobs()) == 20


# ── C2: 오버레이 라이브 적용 ─────────────────────────────


class TestApplyLiveOverlays:
    """C2: 라이브 리밸런싱에서 오버레이가 적용되는지 검증."""

    def test_no_overlays_passthrough(self):
        """오버레이가 없으면 시그널이 그대로 통과한다."""
        bot = _make_bot()
        signals = {"005930": 0.3, "000660": 0.2}
        result = bot._apply_live_overlays(signals)
        assert result == signals

    def test_drawdown_overlay_scales_signals(self):
        """drawdown_overlay가 활성이면 시그널 비중이 스케일링된다."""
        bot = _make_bot()

        mock_overlay = MagicMock()
        mock_overlay.apply_overlay.return_value = {
            "005930": 0.225, "000660": 0.15,
        }
        bot._drawdown_overlay = mock_overlay
        bot.kis_client.get_balance.return_value = {"total_eval": 1_500_000}

        signals = {"005930": 0.3, "000660": 0.2}
        result = bot._apply_live_overlays(signals)

        mock_overlay.apply_overlay.assert_called_once_with(signals, 1_500_000)
        assert result == {"005930": 0.225, "000660": 0.15}

    def test_vol_targeting_overlay_scales_signals(self):
        """vol_targeting이 활성이면 시그널 비중이 스케일링된다."""
        bot = _make_bot()

        mock_overlay = MagicMock()
        mock_overlay.apply.return_value = {"005930": 0.24, "000660": 0.16}
        bot._vol_targeting_overlay = mock_overlay

        mock_series = pd.Series(
            [1_500_000, 1_480_000, 1_510_000],
            index=pd.date_range("2026-02-27", periods=3, freq="h"),
        )
        bot.portfolio_tracker.get_values_series.return_value = mock_series

        signals = {"005930": 0.3, "000660": 0.2}
        result = bot._apply_live_overlays(signals)

        mock_overlay.apply.assert_called_once_with(signals, mock_series)
        assert result == {"005930": 0.24, "000660": 0.16}

    def test_minimum_exposure_floor(self):
        """오버레이 후 비중 합이 10% 미만이면 하한이 적용된다."""
        bot = _make_bot()

        # 드로다운 오버레이가 극단적으로 축소한 케이스
        mock_overlay = MagicMock()
        mock_overlay.apply_overlay.return_value = {
            "005930": 0.02, "000660": 0.01,
        }
        bot._drawdown_overlay = mock_overlay
        bot.kis_client.get_balance.return_value = {"total_eval": 1_500_000}

        signals = {"005930": 0.3, "000660": 0.2}
        result = bot._apply_live_overlays(signals)

        # 0.03 < 0.10 → 스케일업
        total = sum(result.values())
        assert abs(total - 0.10) < 0.001

    def test_empty_signals_passthrough(self):
        """빈 시그널은 그대로 반환된다."""
        bot = _make_bot()
        assert bot._apply_live_overlays({}) == {}

    def test_overlay_error_graceful(self):
        """오버레이 에러 시 원본 시그널이 유지된다."""
        bot = _make_bot()

        mock_overlay = MagicMock()
        mock_overlay.apply_overlay.side_effect = RuntimeError("API error")
        bot._drawdown_overlay = mock_overlay
        bot.kis_client.get_balance.return_value = {"total_eval": 1_500_000}

        signals = {"005930": 0.3, "000660": 0.2}
        result = bot._apply_live_overlays(signals)

        # 에러 시 원본 유지 (drawdown 실패 후 그대로)
        assert result == signals


# ── C3: PortfolioTracker EOD 스냅샷 ─────────────────────


class TestRecordDailyPerformanceTracker:
    """C3: record_daily_performance()가 portfolio_tracker.update()를 호출하는지."""

    def test_tracker_update_called(self):
        """balance 조회 후 portfolio_tracker.update()가 호출된다."""
        bot = _make_bot()
        bot.kis_client.is_configured.return_value = True
        bot.kis_client.get_balance.return_value = {
            "total_eval": 1_500_000,
            "cash": 500_000,
            "holdings": [],
        }

        with patch("src.scheduler.main.PerformanceDB", create=True):
            with patch("src.data.performance_db.PerformanceDB"):
                bot.record_daily_performance()

        bot.portfolio_tracker.update.assert_called_with(1_500_000)


# ── M1: auto_backtest 전략 확장 ──────────────────────────


class TestAutoBacktesterStrategies:
    """M1: AutoBacktester._create_strategy() 전략 확장 검증."""

    @pytest.mark.parametrize(
        "name",
        [
            "value",
            "momentum",
            "multi_factor",
            "quality",
            "three_factor",
            "pead",
            "shareholder_yield",
            "low_vol_quality",
            "accrual",
            "dual_momentum",
            "etf_rotation",
            "enhanced_etf_rotation",
            "cross_asset_momentum",
        ],
    )
    def test_create_strategy_supported(self, name):
        """지원하는 전략이 정상 생성된다."""
        from src.backtest.auto_runner import AutoBacktester

        strategy = AutoBacktester._create_strategy(name)
        assert strategy is not None, f"{name} 전략이 생성되어야 합니다."

    def test_unknown_strategy_returns_none(self):
        """알 수 없는 전략은 None을 반환한다."""
        from src.backtest.auto_runner import AutoBacktester

        assert AutoBacktester._create_strategy("unknown_xyz") is None

    def test_default_strategies_expanded(self):
        """기본 전략 리스트가 5개로 확장되었다."""
        from src.backtest.auto_runner import AutoBacktester

        bt = AutoBacktester()
        assert len(bt.strategies) == 5
        assert "three_factor" in bt.strategies
        assert "quality" in bt.strategies


# ── M2: MIN_ORDER_AMOUNT 정합성 ──────────────────────────


class TestMinOrderAmountConsistency:
    """M2: MIN_ORDER_AMOUNT 정합성 검증."""

    def test_position_manager_min_order_matches_config(self):
        """PositionManager.MIN_ORDER_AMOUNT == SmallCapitalConfig.min_order_amount."""
        from src.execution.order_manager import SmallCapitalConfig
        from src.execution.position_manager import PositionManager

        assert PositionManager.MIN_ORDER_AMOUNT == SmallCapitalConfig().min_order_amount


# ── PortfolioTracker.get_values_series ───────────────────


class TestPortfolioTrackerValuesSeries:
    """PortfolioTracker.get_values_series() 검증."""

    def test_empty_history_returns_empty_series(self):
        """이력이 없으면 빈 Series를 반환한다."""
        from src.report.portfolio_tracker import PortfolioTracker

        with patch.object(PortfolioTracker, "__init__", lambda self: None):
            tracker = PortfolioTracker()
            tracker._history = []

            series = tracker.get_values_series()
            assert len(series) == 0

    def test_history_returns_correct_series(self):
        """이력이 있으면 올바른 pd.Series를 반환한다."""
        from src.report.portfolio_tracker import PortfolioTracker

        with patch.object(PortfolioTracker, "__init__", lambda self: None):
            tracker = PortfolioTracker()
            tracker._history = [
                {
                    "ts": "2026-03-01T10:00:00",
                    "eval": 1_500_000,
                    "peak": 1_500_000,
                    "mdd": 0.0,
                },
                {
                    "ts": "2026-03-01T11:00:00",
                    "eval": 1_480_000,
                    "peak": 1_500_000,
                    "mdd": -0.0133,
                },
            ]

            series = tracker.get_values_series()
            assert len(series) == 2
            assert series.iloc[0] == 1_500_000
            assert series.iloc[1] == 1_480_000

    def test_malformed_records_skipped(self):
        """잘못된 레코드는 건너뛴다."""
        from src.report.portfolio_tracker import PortfolioTracker

        with patch.object(PortfolioTracker, "__init__", lambda self: None):
            tracker = PortfolioTracker()
            tracker._history = [
                {"ts": "2026-03-01T10:00:00", "eval": 1_500_000},
                {"bad_key": "no_ts"},  # 잘못된 레코드
                {"ts": "2026-03-01T11:00:00", "eval": 1_480_000},
            ]

            series = tracker.get_values_series()
            assert len(series) == 2
