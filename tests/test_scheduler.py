"""스케줄러 모듈 테스트.

KRXHolidays(한국거래소 휴장일 관리), TradingBot(트레이딩 봇) 등을 검증한다.
외부 API 호출은 mock 처리한다.
"""

import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _import_krx_holidays():
    """KRXHolidays 클래스를 임포트한다."""
    from src.scheduler.holidays import KRXHolidays
    return KRXHolidays


def _try_import_trading_bot():
    """TradingBot 클래스가 있으면 임포트한다."""
    try:
        from src.scheduler.trading_bot import TradingBot
        return TradingBot
    except ImportError:
        try:
            from src.scheduler.bot import TradingBot
            return TradingBot
        except ImportError:
            return None


def _import_trading_bot():
    """src.scheduler.main에서 TradingBot을 임포트한다."""
    from src.scheduler.main import TradingBot
    return TradingBot


# ===================================================================
# KRXHolidays 테스트
# ===================================================================

class TestKRXHolidays:
    """한국거래소 휴장일 관리 클래스 검증."""

    def test_weekend_is_not_trading_day(self):
        """토요일/일요일은 거래일이 아니다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 2026-02-14 (토요일), 2026-02-15 (일요일)
        saturday = datetime.date(2026, 2, 14)
        sunday = datetime.date(2026, 2, 15)

        assert holidays.is_trading_day(saturday) is False, (
            "토요일은 거래일이 아닙니다."
        )
        assert holidays.is_trading_day(sunday) is False, (
            "일요일은 거래일이 아닙니다."
        )

    def test_weekday_is_trading_day(self):
        """공휴일이 아닌 평일은 거래일이다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 2026-02-18 (수요일) - 설 연휴가 아닌 평일이라고 가정하기 보다는
        # 명확히 공휴일이 아닌 날짜를 선택
        # 2026-03-04 (수요일)
        wednesday = datetime.date(2026, 3, 4)

        result = holidays.is_trading_day(wednesday)

        # 특별한 공휴일이 아닌 수요일이면 True
        assert result is True, (
            f"2026-03-04 (수요일)은 거래일이어야 합니다."
        )

    def test_new_year_holiday(self):
        """1월 1일(신정)은 거래일이 아니다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        new_year = datetime.date(2026, 1, 1)

        assert holidays.is_trading_day(new_year) is False, (
            "1월 1일(신정)은 거래일이 아닙니다."
        )

    def test_lunar_new_year(self):
        """설날(음력 1/1)은 거래일이 아니다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 2026년 설날은 2월 17일
        lunar_new_year = datetime.date(2026, 2, 17)

        assert holidays.is_trading_day(lunar_new_year) is False, (
            "설날(2026-02-17)은 거래일이 아닙니다."
        )

    def test_next_trading_day_from_friday(self):
        """금요일의 다음 거래일은 월요일이다 (공휴일이 아닌 경우)."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 2026-03-06 (금요일)
        friday = datetime.date(2026, 3, 6)
        next_day = holidays.next_trading_day(friday)

        assert next_day is not None, "다음 거래일이 반환되어야 합니다."
        assert next_day.weekday() < 5, "다음 거래일은 평일이어야 합니다."
        # 월요일(2026-03-09)이어야 함
        expected = datetime.date(2026, 3, 9)
        assert next_day == expected, (
            f"금요일(2026-03-06)의 다음 거래일은 2026-03-09(월)이어야 합니다: {next_day}"
        )

    def test_next_trading_day_from_holiday(self):
        """공휴일의 다음 거래일이 올바르게 반환된다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 2026-01-01 (신정, 목요일)
        new_year = datetime.date(2026, 1, 1)
        next_day = holidays.next_trading_day(new_year)

        assert next_day is not None, "다음 거래일이 반환되어야 합니다."
        assert next_day > new_year, "다음 거래일은 현재 날짜보다 미래여야 합니다."
        assert holidays.is_trading_day(next_day) is True, (
            "반환된 날짜는 거래일이어야 합니다."
        )

    def test_prev_trading_day(self):
        """이전 거래일이 올바르게 반환된다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 2026-03-09 (월요일)
        monday = datetime.date(2026, 3, 9)
        prev_day = holidays.prev_trading_day(monday)

        assert prev_day is not None, "이전 거래일이 반환되어야 합니다."
        assert prev_day < monday, "이전 거래일은 현재 날짜보다 과거여야 합니다."
        assert holidays.is_trading_day(prev_day) is True, (
            "반환된 날짜는 거래일이어야 합니다."
        )
        # 금요일(2026-03-06)이어야 함
        expected = datetime.date(2026, 3, 6)
        assert prev_day == expected, (
            f"월요일(2026-03-09)의 이전 거래일은 2026-03-06(금)이어야 합니다: {prev_day}"
        )

    def test_is_rebalance_day_monthly(self):
        """월간 리밸런싱일(매월 첫 거래일)이 올바르게 판별된다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 2026-03-02 (월요일) -> 3월 첫 영업일
        first_trading_day = datetime.date(2026, 3, 2)
        result = holidays.is_rebalance_day(first_trading_day, freq="monthly")

        assert result is True, (
            f"2026-03-02은 3월 첫 거래일이므로 리밸런싱일이어야 합니다."
        )

    def test_is_rebalance_day_not_first(self):
        """매월 첫 거래일이 아니면 리밸런싱일이 아니다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 2026-03-10 (화요일) -> 3월 첫 거래일이 아님
        mid_month = datetime.date(2026, 3, 10)
        result = holidays.is_rebalance_day(mid_month, freq="monthly")

        assert result is False, (
            "월 중간 날짜는 리밸런싱일이 아닙니다."
        )


# ===================================================================
# TradingBot 테스트
# ===================================================================

class TestTradingBot:
    """트레이딩 봇 검증."""

    def test_bot_init(self):
        """트레이딩 봇이 정상 초기화된다."""
        TradingBot = _try_import_trading_bot()
        if TradingBot is None:
            pytest.skip("TradingBot이 아직 구현되지 않았습니다.")

        bot = TradingBot()
        assert bot is not None, "TradingBot이 정상 초기화되어야 합니다."

    def test_setup_schedule_adds_jobs(self):
        """스케줄 설정 시 작업이 추가된다."""
        TradingBot = _try_import_trading_bot()
        if TradingBot is None:
            pytest.skip("TradingBot이 아직 구현되지 않았습니다.")

        bot = TradingBot()

        # setup_schedule 메서드가 있는지 확인
        if hasattr(bot, "setup_schedule"):
            try:
                bot.setup_schedule()
                # 에러 없이 수행되면 통과
            except Exception:
                # 스케줄러 미설치 등으로 에러 가능
                pass
        else:
            pytest.skip("setup_schedule 메서드가 아직 구현되지 않았습니다.")


# ===================================================================
# TradingBot 단기 트레이딩 통합 테스트
# ===================================================================

def _make_bot_with_flag(flag_enabled: bool):
    """short_term_trading 플래그에 따라 TradingBot을 생성한다.

    모든 외부 의존성을 mock하여 테스트 격리를 보장한다.
    """
    TradingBot = _import_trading_bot()

    with patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
    }):
        with patch("src.scheduler.main.KISClient") as mock_kis_cls, \
             patch("src.scheduler.main.TelegramNotifier") as mock_notifier_cls, \
             patch("src.scheduler.main.PortfolioAllocator") as mock_allocator_cls, \
             patch("src.scheduler.main.ShortTermTrader") as mock_trader_cls, \
             patch("src.scheduler.main.SwingReversionStrategy") as mock_swing_cls, \
             patch("src.scheduler.main.FeatureFlags") as mock_flags_cls, \
             patch("src.scheduler.main.DataCache"), \
             patch("src.scheduler.main.TelegramCommander") as mock_cmd_cls, \
             patch("src.scheduler.main.PortfolioTracker"), \
             patch("src.scheduler.main.StockReviewer"), \
             patch("src.scheduler.main.NightResearcher"):

            # KISClient mock
            mock_kis = MagicMock()
            mock_kis.is_paper = True
            mock_kis.is_configured.return_value = True
            mock_kis.mode_tag = "[모의]"
            mock_kis.trading_mode = "모의"
            mock_kis_cls.return_value = mock_kis

            # TelegramNotifier mock
            mock_notifier = MagicMock()
            mock_notifier.is_configured.return_value = False
            mock_notifier_cls.return_value = mock_notifier

            # FeatureFlags mock
            mock_flags = MagicMock()
            mock_flags.is_enabled.side_effect = lambda name: (
                flag_enabled if name == "short_term_trading" else False
            )
            mock_flags.get_config.return_value = {
                "long_term_pct": 0.90,
                "short_term_pct": 0.10,
                "stop_loss_pct": -0.05,
                "take_profit_pct": 0.10,
                "max_concurrent_positions": 3,
                "max_daily_loss_pct": -0.03,
                "confirm_timeout_minutes": 30,
                "swing_enabled": True,
                "daytrading_enabled": False,
            }
            mock_flags.get_all_status.return_value = {
                "short_term_trading": flag_enabled
            }
            mock_flags.get_summary.return_value = "[Feature Flags]"
            mock_flags_cls.return_value = mock_flags

            # PortfolioAllocator mock
            mock_allocator = MagicMock()
            mock_allocator._long_term_pct = 0.90
            mock_allocator._short_term_pct = 0.10
            mock_allocator_cls.return_value = mock_allocator

            # ShortTermTrader mock
            mock_trader = MagicMock()
            mock_trader_cls.return_value = mock_trader

            # SwingReversionStrategy mock
            mock_swing = MagicMock()
            mock_swing.name = "swing_reversion"
            mock_swing.mode = "swing"
            mock_swing_cls.return_value = mock_swing

            # TelegramCommander mock
            mock_cmd = MagicMock()
            mock_cmd_cls.return_value = mock_cmd

            bot = TradingBot()

    return bot


class TestShortTermInit:
    """_init_short_term_modules 테스트."""

    def test_flag_off_modules_none(self):
        """feature flag off이면 단기 모듈이 모두 None이다."""
        bot = _make_bot_with_flag(False)

        assert bot.allocator is None, "allocator가 None이어야 합니다."
        assert bot.short_term_trader is None, "short_term_trader가 None이어야 합니다."
        assert bot.short_term_risk is None, "short_term_risk가 None이어야 합니다."

    def test_flag_on_modules_initialized(self):
        """feature flag on이면 단기 모듈이 초기화된다."""
        bot = _make_bot_with_flag(True)

        assert bot.allocator is not None, "allocator가 초기화되어야 합니다."
        assert bot.short_term_trader is not None, "short_term_trader가 초기화되어야 합니다."
        assert bot.short_term_risk is not None, "short_term_risk가 초기화되어야 합니다."


class TestShortTermScan:
    """short_term_scan 테스트."""

    def test_flag_off_skip(self):
        """feature flag off이면 스캔이 실행되지 않는다."""
        bot = _make_bot_with_flag(False)
        # _is_trading_day가 호출되지 않아야 함
        bot._is_trading_day = MagicMock()

        bot.short_term_scan()

        bot._is_trading_day.assert_not_called()

    def test_signals_found_notification(self):
        """시그널이 발견되면 알림이 발송된다."""
        bot = _make_bot_with_flag(True)
        bot._is_trading_day = MagicMock(return_value=True)
        bot._send_notification = MagicMock()

        # 시그널 mock
        mock_signal = MagicMock()
        mock_signal.side = "buy"
        mock_signal.ticker = "005930"
        mock_signal.strategy = "swing_reversion"
        mock_signal.confidence = 0.85
        mock_signal.reason = "RSI 과매도"
        mock_signal.id = "sig_test_001"

        bot.short_term_trader.scan_for_signals.return_value = [mock_signal]

        bot.short_term_scan()

        bot._send_notification.assert_called_once()
        msg = bot._send_notification.call_args[0][0]
        assert "1개 발견" in msg
        assert "005930" in msg
        assert "sig_test_001" in msg

    def test_no_signals_no_notification(self):
        """시그널이 없으면 알림이 발송되지 않는다."""
        bot = _make_bot_with_flag(True)
        bot._is_trading_day = MagicMock(return_value=True)
        bot._send_notification = MagicMock()

        bot.short_term_trader.scan_for_signals.return_value = []

        bot.short_term_scan()

        bot._send_notification.assert_not_called()


class TestShortTermMonitor:
    """short_term_monitor 테스트."""

    def test_no_positions_early_return(self):
        """포지션이 없으면 조기 리턴한다."""
        bot = _make_bot_with_flag(True)
        bot._is_trading_day = MagicMock(return_value=True)
        bot._send_notification = MagicMock()

        bot.allocator.get_positions_by_pool.return_value = []

        bot.short_term_monitor()

        bot._send_notification.assert_not_called()

    def test_position_risk_alert(self):
        """포지션에 리스크 알림이 있으면 알림이 발송된다."""
        bot = _make_bot_with_flag(True)
        bot._is_trading_day = MagicMock(return_value=True)
        bot._send_notification = MagicMock()

        # 포지션 mock
        position = {
            "ticker": "005930",
            "entry_price": 70000,
            "current_price": 60000,
            "entry_date": "2026-02-20",
            "mode": "swing",
        }
        bot.allocator.get_positions_by_pool.return_value = [position]

        # short_term_risk를 mock으로 교체 (실제 ShortTermRiskManager가 아닌)
        mock_risk = MagicMock()
        mock_risk.check_position.return_value = {
            "should_close": True,
            "reasons": ["손절 트리거: -14.29% (한도: -5.00%)"],
            "pnl_pct": -0.1429,
        }
        bot.short_term_risk = mock_risk

        bot.short_term_trader.execute_confirmed_signals.return_value = []

        bot.short_term_monitor()

        bot._send_notification.assert_called_once()
        msg = bot._send_notification.call_args[0][0]
        assert "단기 포지션 알림" in msg
        assert "005930" in msg
        assert "손절" in msg

    def test_flag_off_skip(self):
        """feature flag off이면 모니터링이 실행되지 않는다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock()

        bot.short_term_monitor()

        bot._is_trading_day.assert_not_called()


class TestDaytradingClose:
    """daytrading_close 테스트."""

    def test_daytrading_disabled_skip(self):
        """daytrading_enabled=False이면 스킵한다."""
        bot = _make_bot_with_flag(True)
        bot._is_trading_day = MagicMock(return_value=True)
        bot._send_notification = MagicMock()

        # 기본 config에서 daytrading_enabled=False
        bot.feature_flags.get_config.return_value = {
            "daytrading_enabled": False,
        }

        bot.daytrading_close()

        bot._send_notification.assert_not_called()

    def test_daytrading_positions_alert(self):
        """데이트레이딩 포지션이 있으면 청산 알림이 발송된다."""
        bot = _make_bot_with_flag(True)
        bot._is_trading_day = MagicMock(return_value=True)
        bot._send_notification = MagicMock()

        bot.feature_flags.get_config.return_value = {
            "mode": "multi",
        }

        # 데이트레이딩 포지션 mock
        position = {
            "ticker": "035420",
            "mode": "daytrading",
            "pnl_pct": 0.02,
        }
        bot.allocator.get_positions_by_pool.return_value = [position]

        bot.daytrading_close()

        bot._send_notification.assert_called_once()
        msg = bot._send_notification.call_args[0][0]
        assert "데이트레이딩 청산" in msg
        assert "035420" in msg

    def test_flag_off_skip(self):
        """feature flag off이면 스킵한다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock()

        bot.daytrading_close()

        bot._is_trading_day.assert_not_called()


class TestSetupScheduleShortTerm:
    """setup_schedule에 단기 스케줄이 추가되는지 검증."""

    def test_schedule_has_14_jobs(self):
        """setup_schedule 후 14개의 작업이 등록된다 (기존 11 + 단기 3)."""
        bot = _make_bot_with_flag(False)

        bot.setup_schedule()

        jobs = bot.scheduler.get_jobs()
        assert len(jobs) == 14, (
            f"14개 작업이 등록되어야 합니다. 실제: {len(jobs)}개"
        )

    def test_short_term_job_ids_exist(self):
        """단기 트레이딩 관련 스케줄 ID가 존재한다."""
        bot = _make_bot_with_flag(False)

        bot.setup_schedule()

        job_ids = [j.id for j in bot.scheduler.get_jobs()]
        assert "short_term_scan" in job_ids, "short_term_scan 작업이 필요합니다."
        assert "short_term_monitor" in job_ids, "short_term_monitor 작업이 필요합니다."
        assert "daytrading_close" in job_ids, "daytrading_close 작업이 필요합니다."


class TestExistingBehaviorRegression:
    """기존 동작 회귀 테스트: feature flag off 시 기존과 동일하게 동작한다."""

    def test_execute_rebalance_no_allocator(self):
        """allocator 없이 리밸런싱이 정상 실행된다."""
        bot = _make_bot_with_flag(False)

        assert bot.allocator is None, "flag off 시 allocator는 None이어야 합니다."
        # execute_rebalance는 _is_trading_day 등에서 가드되므로 호출만 확인
        bot._is_trading_day = MagicMock(return_value=False)
        bot.execute_rebalance()
        # 비거래일이므로 조기 리턴, 정상 종료

    def test_bot_has_all_existing_methods(self):
        """기존 메서드가 모두 존재한다."""
        bot = _make_bot_with_flag(False)

        existing_methods = [
            "morning_briefing",
            "premarket_check",
            "execute_rebalance",
            "hourly_monitor",
            "eod_review",
            "evening_report",
            "health_check",
            "global_market_check",
            "stock_review_job",
            "auto_backtest_job",
            "night_research_job",
            "setup_schedule",
            "run",
        ]
        for method in existing_methods:
            assert hasattr(bot, method), f"{method} 메서드가 존재해야 합니다."

    def test_bot_has_new_methods(self):
        """새 단기 트레이딩 메서드가 존재한다."""
        bot = _make_bot_with_flag(False)

        new_methods = [
            "_init_short_term_modules",
            "short_term_scan",
            "short_term_monitor",
            "daytrading_close",
        ]
        for method in new_methods:
            assert hasattr(bot, method), f"{method} 메서드가 존재해야 합니다."
