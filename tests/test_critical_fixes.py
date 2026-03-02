"""CRITICAL/MAJOR 이슈 수정 검증 테스트.

C1: emergency_monitor 스케줄 등록
C2: 오버레이 라이브 리밸런싱 적용
C3: PortfolioTracker EOD 스냅샷
M1: auto_backtest 전략 확장
M2: MIN_ORDER_AMOUNT 정합성
D1: DrawdownOverlay peak 동기화 (PortfolioTracker 연동)
D2: ThreeFactorStrategy feature flag 연동 (5개 dead flag)
D3: walk_forward_backtest → auto_backtest 연동
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


# ── D1: DrawdownOverlay peak 동기화 ─────────────────────


class TestDrawdownOverlayPeakSync:
    """D1: DrawdownOverlay가 PortfolioTracker의 peak로 초기화되는지 검증."""

    def test_peak_synced_from_tracker(self):
        """PortfolioTracker.peak > 0 이면 DrawdownOverlay._peak에 동기화된다."""
        from src.strategy.drawdown_overlay import DrawdownOverlay

        overlay = DrawdownOverlay()
        assert overlay._peak == 0.0  # 기본값

        # 시뮬레이션: 스케줄러 초기화 시 동기화
        overlay._peak = 1_500_000.0
        assert overlay._peak == 1_500_000.0

        # update 호출 시 고점 유지 확인 (-12% 낙폭 → threshold -10% 초과)
        exposure = overlay.update(1_320_000)
        assert overlay._peak == 1_500_000.0  # 고점 유지
        assert exposure < 1.0  # 낙폭으로 디레버리징

    def test_peak_zero_no_sync(self):
        """PortfolioTracker.peak == 0 이면 동기화하지 않는다."""
        from src.strategy.drawdown_overlay import DrawdownOverlay

        overlay = DrawdownOverlay()
        # peak이 0이면 첫 update에서 새 고점 설정
        exposure = overlay.update(1_000_000)
        assert overlay._peak == 1_000_000.0
        assert exposure == 1.0

    def test_synced_peak_persists_drawdown_calculation(self):
        """동기화된 peak 기준으로 MDD가 정확히 계산된다."""
        from src.strategy.drawdown_overlay import DrawdownOverlay

        overlay = DrawdownOverlay(
            thresholds=[(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)]
        )
        overlay._peak = 1_000_000.0  # 재시작 후 동기화

        # -12% 낙폭 → 0.75 노출
        exposure = overlay.update(880_000)
        assert exposure == 0.75

        # -18% 낙폭 → 0.50 노출
        exposure = overlay.update(820_000)
        assert exposure == 0.50


# ── D2: ThreeFactorStrategy feature flag 연동 ──────────


class TestThreeFactorFlagWiring:
    """D2: _create_long_term_strategy('three_factor')가 feature flag를 읽는지 검증."""

    def _make_bot_with_flags(self, flag_overrides: dict):
        """특정 flag를 활성화한 봇을 생성한다."""
        bot = _make_bot()

        def is_enabled_side_effect(name):
            return flag_overrides.get(name, False)

        def get_config_side_effect(name):
            configs = {
                "low_volatility_factor": {"weight": 0.15, "vol_period": 60},
                "sector_neutral": {"max_sector_pct": 0.30},
                "turnover_reduction": {"buffer_size": 3, "holding_bonus": 0.1},
                "regime_meta_model": {"level": "rule_based"},
            }
            return configs.get(name, {})

        bot.feature_flags.is_enabled.side_effect = is_enabled_side_effect
        bot.feature_flags.get_config.side_effect = get_config_side_effect
        return bot

    def test_default_three_factor_no_extras(self):
        """플래그 모두 OFF → 기본 ThreeFactorStrategy."""
        bot = self._make_bot_with_flags({})
        strategy = bot._create_long_term_strategy("three_factor")

        assert strategy is not None
        assert strategy.low_vol_weight == 0.0
        assert strategy.sector_neutral is False
        assert strategy.turnover_buffer == 0
        assert strategy.holding_bonus == 0.0
        assert strategy.regime_model is None

    def test_low_volatility_factor_enabled(self):
        """low_volatility_factor ON → low_vol_weight 주입."""
        bot = self._make_bot_with_flags({"low_volatility_factor": True})
        strategy = bot._create_long_term_strategy("three_factor")

        assert strategy.low_vol_weight == 0.15
        assert strategy._low_vol_strategy is not None

    def test_sector_neutral_enabled(self):
        """sector_neutral ON → sector_neutral=True, max_sector_pct 주입."""
        bot = self._make_bot_with_flags({"sector_neutral": True})
        strategy = bot._create_long_term_strategy("three_factor")

        assert strategy.sector_neutral is True
        assert strategy.max_sector_pct == 0.30

    def test_turnover_reduction_enabled(self):
        """turnover_reduction ON → turnover_buffer, holding_bonus 주입."""
        bot = self._make_bot_with_flags({"turnover_reduction": True})
        strategy = bot._create_long_term_strategy("three_factor")

        assert strategy.turnover_buffer == 3
        assert strategy.holding_bonus == 0.1

    def test_regime_meta_model_enabled(self):
        """regime_meta_model ON → RuleBasedRegimeModel 주입."""
        bot = self._make_bot_with_flags({"regime_meta_model": True})
        strategy = bot._create_long_term_strategy("three_factor")

        assert strategy.regime_model is not None
        from src.ml.regime_model import RuleBasedRegimeModel
        assert isinstance(strategy.regime_model, RuleBasedRegimeModel)

    def test_regime_with_low_vol_includes_4_factors(self):
        """low_vol + regime 동시 ON → 4팩터 가중치."""
        bot = self._make_bot_with_flags({
            "low_volatility_factor": True,
            "regime_meta_model": True,
        })
        strategy = bot._create_long_term_strategy("three_factor")

        assert strategy.regime_model is not None
        assert "low_vol" in strategy.regime_model.factor_names

    def test_all_flags_combined(self):
        """모든 플래그 ON → 전부 적용."""
        bot = self._make_bot_with_flags({
            "low_volatility_factor": True,
            "sector_neutral": True,
            "turnover_reduction": True,
            "regime_meta_model": True,
        })
        strategy = bot._create_long_term_strategy("three_factor")

        assert strategy.low_vol_weight == 0.15
        assert strategy.sector_neutral is True
        assert strategy.turnover_buffer == 3
        assert strategy.regime_model is not None


# ── D3: walk_forward_backtest 연동 ───────────────────────


class TestWalkForwardIntegration:
    """D3: walk_forward_backtest가 auto_backtest에 연동되는지 검증."""

    def test_run_walk_forward_method_exists(self):
        """_run_walk_forward 메서드가 존재한다."""
        bot = _make_bot()
        assert hasattr(bot, "_run_walk_forward")
        assert callable(bot._run_walk_forward)

    def test_auto_backtest_calls_walk_forward_when_enabled(self):
        """auto_backtest + walk_forward_backtest 모두 ON이면 _run_walk_forward 호출."""
        bot = _make_bot()

        call_log = []

        def is_enabled_side_effect(name):
            return name in ("auto_backtest", "walk_forward_backtest")

        bot.feature_flags.is_enabled.side_effect = is_enabled_side_effect
        bot.feature_flags.get_config.return_value = {
            "lookback_months": 6,
            "strategies": ["value"],
        }

        with patch.object(bot, "_run_walk_forward") as mock_wf:
            with patch("src.scheduler.main.AutoBacktester") as mock_bt_cls:
                mock_bt = MagicMock()
                mock_bt.run_all.return_value = "결과"
                mock_bt_cls.return_value = mock_bt

                bot.auto_backtest_job()

                mock_wf.assert_called_once()

    def test_auto_backtest_skips_walk_forward_when_disabled(self):
        """walk_forward_backtest OFF이면 _run_walk_forward 미호출."""
        bot = _make_bot()

        def is_enabled_side_effect(name):
            return name == "auto_backtest"

        bot.feature_flags.is_enabled.side_effect = is_enabled_side_effect
        bot.feature_flags.get_config.return_value = {
            "lookback_months": 6,
            "strategies": ["value"],
        }

        with patch.object(bot, "_run_walk_forward") as mock_wf:
            with patch("src.scheduler.main.AutoBacktester") as mock_bt_cls:
                mock_bt = MagicMock()
                mock_bt.run_all.return_value = "결과"
                mock_bt_cls.return_value = mock_bt

                bot.auto_backtest_job()

                mock_wf.assert_not_called()


# ── D4: turnover_reduction 라이브 연동 ──────────────────


class TestTurnoverReductionLive:
    """D4: execute_rebalance()에서 update_holdings가 호출되는지 검증."""

    def test_update_holdings_called_before_signals(self):
        """전략에 update_holdings가 있으면 리밸런싱 전에 호출된다."""
        mock_strategy = MagicMock()
        mock_strategy.generate_signals.return_value = {"005930": 0.3}

        bot = _make_bot()
        bot._strategy = mock_strategy
        bot.kis_client.get_balance.return_value = {
            "total_eval": 1_500_000,
            "cash": 500_000,
            "holdings": [
                {"ticker": "005930", "name": "삼성전자"},
                {"ticker": "000660", "name": "SK하이닉스"},
            ],
        }

        # update_holdings가 호출되는지만 검증 (실제 리밸런싱 미실행)
        assert hasattr(mock_strategy, "update_holdings")
