"""ShortTermRiskManager 테스트."""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from src.execution.short_term_risk import (
    DataHealthStatus,
    ShortTermRiskConfig,
    ShortTermRiskManager,
)


# ── 1. 기본 설정 테스트 ──────────────────────────────────────────────


class TestShortTermRiskConfig:
    """ShortTermRiskConfig 기본값/커스텀 값 테스트."""

    def test_default_values(self):
        cfg = ShortTermRiskConfig()
        assert cfg.stop_loss_pct == -0.05
        assert cfg.take_profit_pct == 0.10
        assert cfg.max_concurrent_positions == 3
        assert cfg.max_daily_loss_pct == -0.03
        assert cfg.max_single_position_pct == 0.50
        assert cfg.daytrading_close_time == "15:20"
        assert cfg.data_warning_seconds == 180
        assert cfg.data_emergency_seconds == 600

    def test_custom_values(self):
        cfg = ShortTermRiskConfig(
            stop_loss_pct=-0.03,
            take_profit_pct=0.08,
            max_concurrent_positions=5,
            max_daily_loss_pct=-0.05,
            max_single_position_pct=0.30,
            daytrading_close_time="15:10",
            data_warning_seconds=120,
            data_emergency_seconds=300,
        )
        assert cfg.stop_loss_pct == -0.03
        assert cfg.take_profit_pct == 0.08
        assert cfg.max_concurrent_positions == 5
        assert cfg.max_daily_loss_pct == -0.05
        assert cfg.max_single_position_pct == 0.30
        assert cfg.daytrading_close_time == "15:10"
        assert cfg.data_warning_seconds == 120
        assert cfg.data_emergency_seconds == 300


class TestShortTermRiskManagerInit:
    """초기화 테스트."""

    def test_default_config(self):
        mgr = ShortTermRiskManager()
        assert mgr.config.stop_loss_pct == -0.05

    def test_custom_config(self):
        cfg = ShortTermRiskConfig(stop_loss_pct=-0.10)
        mgr = ShortTermRiskManager(config=cfg)
        assert mgr.config.stop_loss_pct == -0.10


# ── 2. check_can_open 테스트 ─────────────────────────────────────────


class TestCheckCanOpen:
    """신규 진입 가능 여부 체크."""

    def setup_method(self):
        self.mgr = ShortTermRiskManager()
        self.mgr.reset_daily()

    def test_normal_open_allowed(self):
        """정상 진입 허용."""
        ok, reason = self.mgr.check_can_open(
            current_position_count=1,
            order_amount=300_000,
            short_term_budget=1_000_000,
        )
        assert ok is True
        assert reason == ""

    def test_max_positions_exceeded(self):
        """포지션 수 초과 시 거절."""
        ok, reason = self.mgr.check_can_open(
            current_position_count=3,
            order_amount=300_000,
            short_term_budget=1_000_000,
        )
        assert ok is False
        assert "최대 동시 포지션 초과" in reason

    def test_daily_loss_limit_reached(self):
        """일일 손실 한도 도달 시 거절."""
        self.mgr.record_trade_pnl(-0.02)
        self.mgr.record_trade_pnl(-0.01)  # 합계 -3% = 한도와 동일
        ok, reason = self.mgr.check_can_open(
            current_position_count=0,
            order_amount=300_000,
            short_term_budget=1_000_000,
        )
        assert ok is False
        assert "일일 손실 한도 도달" in reason

    def test_single_position_weight_exceeded(self):
        """단일 포지션 비중 초과 시 거절."""
        ok, reason = self.mgr.check_can_open(
            current_position_count=0,
            order_amount=600_000,
            short_term_budget=1_000_000,
        )
        assert ok is False
        assert "단일 포지션 비중 초과" in reason

    def test_zero_budget_skips_weight_check(self):
        """예산 0이면 비중 체크 스킵."""
        ok, reason = self.mgr.check_can_open(
            current_position_count=0,
            order_amount=600_000,
            short_term_budget=0,
        )
        assert ok is True


# ── 3. check_stop_loss 테스트 ────────────────────────────────────────


class TestCheckStopLoss:
    """손절 체크."""

    def setup_method(self):
        self.mgr = ShortTermRiskManager()

    def test_no_stop_loss(self):
        """손절 미도달: 진입 10000원, 현재 9600원 (-4%) → False."""
        triggered, reason = self.mgr.check_stop_loss(10000, 9600)
        assert triggered is False
        assert reason == ""

    def test_stop_loss_triggered(self):
        """손절 도달: 진입 10000원, 현재 9400원 (-6%) → True."""
        triggered, reason = self.mgr.check_stop_loss(10000, 9400)
        assert triggered is True
        assert "손절 트리거" in reason

    def test_exact_boundary(self):
        """정확히 -5%: 진입 10000원, 현재 9500원 → True (<=)."""
        triggered, reason = self.mgr.check_stop_loss(10000, 9500)
        assert triggered is True

    def test_entry_price_zero(self):
        """진입가 0 → False."""
        triggered, reason = self.mgr.check_stop_loss(0, 9400)
        assert triggered is False
        assert reason == ""

    def test_negative_entry_price(self):
        """진입가 음수 → False."""
        triggered, reason = self.mgr.check_stop_loss(-100, 9400)
        assert triggered is False


# ── 4. check_take_profit 테스트 ──────────────────────────────────────


class TestCheckTakeProfit:
    """익절 체크."""

    def setup_method(self):
        self.mgr = ShortTermRiskManager()

    def test_no_take_profit(self):
        """익절 미도달: 진입 10000원, 현재 10800원 (+8%) → False."""
        triggered, reason = self.mgr.check_take_profit(10000, 10800)
        assert triggered is False
        assert reason == ""

    def test_take_profit_triggered(self):
        """익절 도달: 진입 10000원, 현재 11100원 (+11%) → True."""
        triggered, reason = self.mgr.check_take_profit(10000, 11100)
        assert triggered is True
        assert "익절 트리거" in reason

    def test_exact_boundary(self):
        """정확히 +10%: 진입 10000원, 현재 11000원 → True (>=)."""
        triggered, reason = self.mgr.check_take_profit(10000, 11000)
        assert triggered is True

    def test_entry_price_zero(self):
        """진입가 0 → False."""
        triggered, reason = self.mgr.check_take_profit(0, 11000)
        assert triggered is False


# ── 5. check_time_stop 테스트 ────────────────────────────────────────


class TestCheckTimeStop:
    """시간 손절 체크."""

    def setup_method(self):
        self.mgr = ShortTermRiskManager()

    def test_within_holding_period(self):
        """5일 미경과 → False."""
        yesterday = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        triggered, reason = self.mgr.check_time_stop(yesterday, max_holding_days=5)
        assert triggered is False

    def test_exceeded_holding_period(self):
        """5일 이상 경과 → True."""
        old_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        triggered, reason = self.mgr.check_time_stop(old_date, max_holding_days=5)
        assert triggered is True
        assert "시간 손절" in reason

    def test_exact_boundary(self):
        """정확히 5일: → True (>=)."""
        exact_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        triggered, reason = self.mgr.check_time_stop(exact_date, max_holding_days=5)
        assert triggered is True

    def test_invalid_date_format(self):
        """잘못된 날짜 포맷 → False."""
        triggered, reason = self.mgr.check_time_stop("not-a-date")
        assert triggered is False
        assert reason == ""

    def test_empty_date(self):
        """빈 문자열 → False."""
        triggered, reason = self.mgr.check_time_stop("")
        assert triggered is False


# ── 6. check_data_health 테스트 ──────────────────────────────────────


class TestCheckDataHealth:
    """데이터 수신 상태 체크."""

    def setup_method(self):
        self.mgr = ShortTermRiskManager()

    def test_none_returns_emergency(self):
        """None → EMERGENCY."""
        status = self.mgr.check_data_health(None)
        assert status == DataHealthStatus.EMERGENCY

    def test_recent_tick_ok(self):
        """1분 전 → OK."""
        tick_time = datetime.now() - timedelta(seconds=60)
        status = self.mgr.check_data_health(tick_time)
        assert status == DataHealthStatus.OK

    def test_warning_threshold(self):
        """5분 전 (300초) → WARNING."""
        tick_time = datetime.now() - timedelta(seconds=300)
        status = self.mgr.check_data_health(tick_time)
        assert status == DataHealthStatus.WARNING

    def test_emergency_threshold(self):
        """15분 전 (900초) → EMERGENCY."""
        tick_time = datetime.now() - timedelta(seconds=900)
        status = self.mgr.check_data_health(tick_time)
        assert status == DataHealthStatus.EMERGENCY

    def test_exact_warning_boundary(self):
        """정확히 180초 → WARNING (>=)."""
        tick_time = datetime.now() - timedelta(seconds=180)
        status = self.mgr.check_data_health(tick_time)
        assert status == DataHealthStatus.WARNING

    def test_exact_emergency_boundary(self):
        """정확히 600초 → EMERGENCY (>=)."""
        tick_time = datetime.now() - timedelta(seconds=600)
        status = self.mgr.check_data_health(tick_time)
        assert status == DataHealthStatus.EMERGENCY


# ── 7. should_force_close_daytrading 테스트 ──────────────────────────


class TestShouldForceCloseDaytrading:
    """데이트레이딩 강제 청산 시각."""

    def test_before_close_time(self):
        """15:19 → False."""
        fake_now = datetime(2026, 2, 22, 15, 19, 0)
        with patch("src.execution.short_term_risk.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mgr = ShortTermRiskManager()
            assert mgr.should_force_close_daytrading() is False

    def test_at_close_time(self):
        """15:20 → True."""
        fake_now = datetime(2026, 2, 22, 15, 20, 0)
        with patch("src.execution.short_term_risk.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mgr = ShortTermRiskManager()
            assert mgr.should_force_close_daytrading() is True

    def test_after_close_time(self):
        """15:30 → True."""
        fake_now = datetime(2026, 2, 22, 15, 30, 0)
        with patch("src.execution.short_term_risk.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mgr = ShortTermRiskManager()
            assert mgr.should_force_close_daytrading() is True

    def test_invalid_close_time_config(self):
        """잘못된 설정 → False."""
        cfg = ShortTermRiskConfig(daytrading_close_time="invalid")
        mgr = ShortTermRiskManager(config=cfg)
        assert mgr.should_force_close_daytrading() is False


# ── 8. check_position 종합 테스트 ────────────────────────────────────


class TestCheckPosition:
    """포지션 종합 리스크 체크."""

    def test_normal_hold(self):
        """정상 보유 → should_close False."""
        mgr = ShortTermRiskManager()
        today = datetime.now().strftime("%Y-%m-%d")
        result = mgr.check_position(
            entry_price=10000,
            current_price=10200,
            entry_date=today,
            mode="swing",
            last_tick_time=datetime.now(),
        )
        assert result["should_close"] is False
        assert result["reasons"] == []
        assert abs(result["pnl_pct"] - 0.02) < 1e-9
        assert result["data_health"] == "ok"

    def test_stop_loss_triggers_close(self):
        """손절 트리거 → should_close True."""
        mgr = ShortTermRiskManager()
        today = datetime.now().strftime("%Y-%m-%d")
        result = mgr.check_position(
            entry_price=10000,
            current_price=9400,
            entry_date=today,
            mode="swing",
            last_tick_time=datetime.now(),
        )
        assert result["should_close"] is True
        assert result["checks"]["stop_loss"] is True
        assert any("손절" in r for r in result["reasons"])

    def test_take_profit_triggers_close(self):
        """익절 트리거 → should_close True."""
        mgr = ShortTermRiskManager()
        today = datetime.now().strftime("%Y-%m-%d")
        result = mgr.check_position(
            entry_price=10000,
            current_price=11100,
            entry_date=today,
            mode="swing",
            last_tick_time=datetime.now(),
        )
        assert result["should_close"] is True
        assert result["checks"]["take_profit"] is True

    def test_daytrading_time_close(self):
        """데이트레이딩 시간 청산 → should_close True."""
        fake_now = datetime(2026, 2, 22, 15, 25, 0)
        with patch("src.execution.short_term_risk.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            mgr = ShortTermRiskManager()
            result = mgr.check_position(
                entry_price=10000,
                current_price=10100,
                entry_date="2026-02-22",
                mode="daytrading",
                last_tick_time=fake_now,
            )
            assert result["should_close"] is True
            assert result["checks"]["daytrading_close"] is True

    def test_daytrading_no_time_stop(self):
        """데이트레이딩 모드에서 시간 손절(swing) 미적용."""
        mgr = ShortTermRiskManager()
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        result = mgr.check_position(
            entry_price=10000,
            current_price=10100,
            entry_date=old_date,
            mode="daytrading",
            last_tick_time=datetime.now(),
        )
        # 데이트레이딩이므로 time_stop은 False
        assert result["checks"]["time_stop"] is False

    def test_data_emergency_triggers_close(self):
        """데이터 긴급 → should_close True."""
        mgr = ShortTermRiskManager()
        today = datetime.now().strftime("%Y-%m-%d")
        result = mgr.check_position(
            entry_price=10000,
            current_price=10100,
            entry_date=today,
            mode="swing",
            last_tick_time=None,  # None = EMERGENCY
        )
        assert result["should_close"] is True
        assert result["checks"]["data_emergency"] is True
        assert result["data_health"] == "emergency"

    def test_swing_time_stop(self):
        """스윙 모드에서 시간 손절 동작."""
        mgr = ShortTermRiskManager()
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        result = mgr.check_position(
            entry_price=10000,
            current_price=10100,
            entry_date=old_date,
            mode="swing",
            last_tick_time=datetime.now(),
        )
        assert result["should_close"] is True
        assert result["checks"]["time_stop"] is True

    def test_multiple_triggers(self):
        """여러 트리거 동시 발생."""
        mgr = ShortTermRiskManager()
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        result = mgr.check_position(
            entry_price=10000,
            current_price=9400,  # 손절
            entry_date=old_date,  # 시간 손절
            mode="swing",
            last_tick_time=None,  # 데이터 긴급
        )
        assert result["should_close"] is True
        assert result["checks"]["stop_loss"] is True
        assert result["checks"]["time_stop"] is True
        assert result["checks"]["data_emergency"] is True
        assert len(result["reasons"]) >= 3

    def test_entry_price_zero(self):
        """진입가 0 → pnl_pct 0."""
        mgr = ShortTermRiskManager()
        today = datetime.now().strftime("%Y-%m-%d")
        result = mgr.check_position(
            entry_price=0,
            current_price=10000,
            entry_date=today,
            mode="swing",
            last_tick_time=datetime.now(),
        )
        assert result["pnl_pct"] == 0.0
        assert result["checks"]["stop_loss"] is False
        assert result["checks"]["take_profit"] is False


# ── 9. reset_daily / record_trade_pnl 테스트 ────────────────────────


class TestDailyTracking:
    """일일 거래 추적 테스트."""

    def test_record_and_summary(self):
        """기록 후 일일 요약 확인."""
        mgr = ShortTermRiskManager()
        mgr.reset_daily()
        mgr.record_trade_pnl(0.02)
        mgr.record_trade_pnl(-0.01)

        summary = mgr.get_daily_summary()
        assert summary["daily_pnl_pct"] == pytest.approx(0.01, abs=1e-4)
        assert summary["trade_count"] == 2
        assert summary["can_trade"] is True

    def test_auto_reset_on_date_change(self):
        """날짜 변경 시 자동 리셋."""
        mgr = ShortTermRiskManager()
        mgr._trade_date = "2020-01-01"  # 과거 날짜 설정
        mgr._daily_pnl = -0.05
        mgr._daily_trade_count = 10

        # _ensure_today 호출 시 리셋됨
        mgr.record_trade_pnl(0.01)

        assert mgr._daily_trade_count == 1  # 리셋 후 1건
        assert mgr._daily_pnl == pytest.approx(0.01, abs=1e-9)
        assert mgr._trade_date == datetime.now().strftime("%Y-%m-%d")

    def test_reset_daily_clears_state(self):
        """reset_daily 호출 시 초기화."""
        mgr = ShortTermRiskManager()
        mgr.record_trade_pnl(-0.02)
        mgr.record_trade_pnl(-0.01)

        mgr.reset_daily()
        assert mgr._daily_pnl == 0.0
        assert mgr._daily_trade_count == 0

    def test_daily_loss_blocks_trading(self):
        """일일 손실 한도 도달 → can_trade False."""
        mgr = ShortTermRiskManager()
        mgr.reset_daily()
        mgr.record_trade_pnl(-0.03)  # 정확히 -3%

        summary = mgr.get_daily_summary()
        assert summary["can_trade"] is False


# ── 10. get_daily_summary 테스트 ─────────────────────────────────────


class TestGetDailySummary:
    """일일 요약 테스트."""

    def test_initial_summary(self):
        """초기 상태의 요약."""
        mgr = ShortTermRiskManager()
        mgr.reset_daily()
        summary = mgr.get_daily_summary()

        assert summary["date"] == datetime.now().strftime("%Y-%m-%d")
        assert summary["daily_pnl_pct"] == 0.0
        assert summary["trade_count"] == 0
        assert summary["daily_loss_limit"] == -0.03
        assert summary["daily_loss_remaining"] == -0.03
        assert summary["can_trade"] is True

    def test_summary_with_profit(self):
        """수익 발생 시 요약."""
        mgr = ShortTermRiskManager()
        mgr.reset_daily()
        mgr.record_trade_pnl(0.05)

        summary = mgr.get_daily_summary()
        assert summary["daily_pnl_pct"] == pytest.approx(0.05, abs=1e-4)
        assert summary["daily_loss_remaining"] == -0.03  # 수익이면 한도 그대로
        assert summary["can_trade"] is True

    def test_summary_with_loss(self):
        """손실 발생 시 요약."""
        mgr = ShortTermRiskManager()
        mgr.reset_daily()
        mgr.record_trade_pnl(-0.02)

        summary = mgr.get_daily_summary()
        assert summary["daily_pnl_pct"] == pytest.approx(-0.02, abs=1e-4)
        assert summary["daily_loss_remaining"] == pytest.approx(-0.01, abs=1e-4)
        assert summary["can_trade"] is True

    def test_summary_ensures_today(self):
        """get_daily_summary가 _ensure_today를 호출."""
        mgr = ShortTermRiskManager()
        mgr._trade_date = "1999-12-31"
        mgr._daily_pnl = -0.99
        mgr._daily_trade_count = 999

        summary = mgr.get_daily_summary()
        # 날짜 변경으로 리셋됨
        assert summary["date"] == datetime.now().strftime("%Y-%m-%d")
        assert summary["daily_pnl_pct"] == 0.0
        assert summary["trade_count"] == 0

    def test_custom_config_reflected(self):
        """커스텀 설정이 요약에 반영."""
        cfg = ShortTermRiskConfig(max_daily_loss_pct=-0.05)
        mgr = ShortTermRiskManager(config=cfg)
        mgr.reset_daily()

        summary = mgr.get_daily_summary()
        assert summary["daily_loss_limit"] == -0.05


# ── DataHealthStatus enum 테스트 ─────────────────────────────────────


class TestDataHealthStatus:
    """DataHealthStatus 열거형 테스트."""

    def test_values(self):
        assert DataHealthStatus.OK.value == "ok"
        assert DataHealthStatus.WARNING.value == "warning"
        assert DataHealthStatus.EMERGENCY.value == "emergency"
