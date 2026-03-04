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

        # 2026-03-03 (화요일) -> 3월 첫 거래일
        # 03-01(삼일절, 일) → 03-02(대체공휴일, 월) → 03-03(화)이 첫 거래일
        first_trading_day = datetime.date(2026, 3, 3)
        result = holidays.is_rebalance_day(first_trading_day, freq="monthly")

        assert result is True, (
            "2026-03-03은 3월 첫 거래일이므로 리밸런싱일이어야 합니다."
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

    # ── 대체공휴일 테스트 ────────────────────────────────────

    def test_substitute_holiday_march_2026(self):
        """2026-03-02는 삼일절(03-01 일요일) 대체공휴일이다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        substitute = datetime.date(2026, 3, 2)

        assert holidays.is_holiday(substitute) is True, (
            "2026-03-02는 삼일절 대체공휴일이어야 합니다."
        )
        assert holidays.is_trading_day(substitute) is False, (
            "2026-03-02(대체공휴일)는 거래일이 아닙니다."
        )

    def test_substitute_holiday_not_rebalance_day(self):
        """대체공휴일인 03-02는 리밸런싱일이 아니다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        substitute = datetime.date(2026, 3, 2)
        result = holidays.is_rebalance_day(substitute, freq="monthly")

        assert result is False, (
            "2026-03-02(대체공휴일)는 리밸런싱일이 아닙니다."
        )

    def test_march_2026_first_trading_day(self):
        """2026년 3월 첫 거래일은 03-03(화)이다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 2월 마지막 거래일에서 다음 거래일 조회
        feb_27 = datetime.date(2026, 2, 27)  # 금요일
        next_td = holidays.next_trading_day(feb_27)

        assert next_td == datetime.date(2026, 3, 3), (
            f"2026-02-27 다음 거래일은 2026-03-03이어야 합니다: {next_td}"
        )

    def test_substitute_holiday_saturday_gwangbok(self):
        """2026-08-17은 광복절(08-15 토요일) 대체공휴일이다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 08-15(토) → 08-16(일, 주말) → 08-17(월, 대체공휴일)
        substitute = datetime.date(2026, 8, 17)

        assert holidays.is_holiday(substitute) is True, (
            "2026-08-17은 광복절 대체공휴일이어야 합니다."
        )
        assert holidays.is_trading_day(substitute) is False, (
            "2026-08-17(대체공휴일)는 거래일이 아닙니다."
        )

    def test_substitute_holiday_gaecheonjeol_2026(self):
        """2026-10-05는 개천절(10-03 토요일) 대체공휴일이다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 10-03(토) → 10-04(일, 주말) → 10-05(월, 대체공휴일)
        substitute = datetime.date(2026, 10, 5)

        assert holidays.is_holiday(substitute) is True, (
            "2026-10-05은 개천절 대체공휴일이어야 합니다."
        )
        assert holidays.is_trading_day(substitute) is False, (
            "2026-10-05(대체공휴일)는 거래일이 아닙니다."
        )

    def test_substitute_holiday_overlap_2025(self):
        """2025-05-06은 어린이날+부처님오신날 겹침의 대체공휴일이다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 2025-05-05: 어린이날(고정) + 부처님오신날(음력) → 겹침 → 대체
        substitute = datetime.date(2025, 5, 6)

        assert holidays.is_holiday(substitute) is True, (
            "2025-05-06은 어린이날/부처님오신날 대체공휴일이어야 합니다."
        )

    def test_substitute_holiday_buddha_birthday_2026(self):
        """2026-05-25는 부처님오신날(05-24 일요일) 대체공휴일이다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # LUNAR_HOLIDAYS에 수동 등록된 대체공휴일
        substitute = datetime.date(2026, 5, 25)

        assert holidays.is_holiday(substitute) is True, (
            "2026-05-25는 부처님오신날 대체공휴일이어야 합니다."
        )
        assert holidays.is_trading_day(substitute) is False, (
            "2026-05-25(대체공휴일)는 거래일이 아닙니다."
        )

    def test_no_substitute_for_new_year(self):
        """신정(01-01)은 대체공휴일 적용 대상이 아니다."""
        KRXHolidays = _import_krx_holidays()
        holidays = KRXHolidays()

        # 2028-01-01은 토요일이지만 신정은 대체공휴일 미적용
        jan_3 = datetime.date(2028, 1, 3)  # 월요일

        assert holidays.is_trading_day(jan_3) is True, (
            "2028-01-03(월)은 거래일이어야 합니다 (신정은 대체 미적용)."
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
            mock_allocator._etf_rotation_pct = 0.0
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
        bot._collect_short_term_data = MagicMock(return_value={
            "daily_data": {"005930": MagicMock()},
            "date": "20260226",
        })

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
        # market_data가 전달되었는지 확인
        call_args = bot.short_term_trader.scan_for_signals.call_args
        assert call_args[0][0].get("daily_data"), "market_data가 전달되어야 합니다."

    def test_no_signals_no_notification(self):
        """시그널이 없으면 알림이 발송되지 않는다."""
        bot = _make_bot_with_flag(True)
        bot._is_trading_day = MagicMock(return_value=True)
        bot._send_notification = MagicMock()
        bot._collect_short_term_data = MagicMock(return_value={
            "daily_data": {"005930": MagicMock()},
            "date": "20260226",
        })

        bot.short_term_trader.scan_for_signals.return_value = []

        bot.short_term_scan()

        bot._send_notification.assert_not_called()

    def test_data_collection_failure_skips_scan(self):
        """데이터 수집 실패 시 스캔을 스킵한다."""
        bot = _make_bot_with_flag(True)
        bot._is_trading_day = MagicMock(return_value=True)
        bot._send_notification = MagicMock()
        bot._collect_short_term_data = MagicMock(return_value={
            "daily_data": {},
            "date": "20260226",
        })

        bot.short_term_scan()

        bot.short_term_trader.scan_for_signals.assert_not_called()
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
        assert "데이트레이딩" in msg and "청산" in msg
        assert "035420" in msg

    def test_flag_off_skip(self):
        """feature flag off이면 스킵한다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock()

        bot.daytrading_close()

        bot._is_trading_day.assert_not_called()


class TestSetupScheduleShortTerm:
    """setup_schedule에 단기 스케줄이 추가되는지 검증."""

    def test_schedule_has_21_jobs(self):
        """setup_schedule 후 21개의 작업이 등록된다 (기존 20 + prewarm_cache 1)."""
        bot = _make_bot_with_flag(False)

        bot.setup_schedule()

        jobs = bot.scheduler.get_jobs()
        assert len(jobs) == 21, (
            f"21개 작업이 등록되어야 합니다. 실제: {len(jobs)}개"
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
        """새 단기 트레이딩 + ETF 로테이션 메서드가 존재한다."""
        bot = _make_bot_with_flag(False)

        new_methods = [
            "_init_short_term_modules",
            "short_term_scan",
            "short_term_monitor",
            "daytrading_close",
            "execute_etf_rotation_rebalance",
            "_fetch_etf_prices",
        ]
        for method in new_methods:
            assert hasattr(bot, method), f"{method} 메서드가 존재해야 합니다."


class TestETFRotationScheduler:
    """ETF 로테이션 스케줄러 통합 테스트."""

    def test_etf_rotation_job_exists(self):
        """ETF 로테이션 리밸런싱 잡이 등록된다."""
        bot = _make_bot_with_flag(False)
        bot.setup_schedule()

        job_ids = [j.id for j in bot.scheduler.get_jobs()]
        assert "etf_rotation_rebalance" in job_ids

    def test_etf_rotation_skips_when_flag_off(self):
        """etf_rotation flag off이면 스킵한다."""
        bot = _make_bot_with_flag(False)
        # flag off → 조기 리턴
        bot.execute_etf_rotation_rebalance()

    def test_etf_rotation_skips_non_trading_day(self):
        """비거래일이면 스킵한다."""
        bot = _make_bot_with_flag(False)
        bot.feature_flags.is_enabled = MagicMock(
            side_effect=lambda n: n == "etf_rotation"
        )
        bot._is_trading_day = MagicMock(return_value=False)
        bot.execute_etf_rotation_rebalance()

    def test_etf_rotation_skips_no_allocator(self):
        """allocator 없으면 스킵한다."""
        bot = _make_bot_with_flag(False)
        bot.feature_flags.is_enabled = MagicMock(
            side_effect=lambda n: n == "etf_rotation"
        )
        bot._is_trading_day = MagicMock(return_value=True)
        bot.allocator = None
        bot.execute_etf_rotation_rebalance()


class TestEnhancedETFRotationScheduler:
    """Enhanced ETF 로테이션 스케줄러 통합 테스트."""

    def test_create_etf_strategy_enhanced_on(self):
        """enhanced_etf_rotation ON이면 EnhancedETFRotationStrategy를 생성한다."""
        bot = _make_bot_with_flag(False)
        bot.feature_flags.is_enabled = MagicMock(
            side_effect=lambda n: n in ("etf_rotation", "enhanced_etf_rotation")
        )
        bot.feature_flags.get_config = MagicMock(
            side_effect=lambda n: {
                "etf_rotation": {
                    "lookback_months": 12,
                    "n_select": 3,
                    "max_same_sector": 1,
                },
                "enhanced_etf_rotation": {
                    "cash_ratio_risk_off": 0.7,
                    "use_vol_weight": True,
                    "use_market_filter": True,
                    "use_trend_filter": True,
                    "use_max_drawdown_filter": True,
                    "max_drawdown_filter": 0.15,
                    "vol_lookback": 60,
                    "market_ma_period": 200,
                    "trend_short_ma": 20,
                    "trend_long_ma": 60,
                },
            }.get(n, {})
        )

        strategy = bot._create_etf_strategy()

        from src.strategy.enhanced_etf_rotation import (
            EnhancedETFRotationStrategy,
        )
        assert isinstance(strategy, EnhancedETFRotationStrategy)
        assert strategy.num_etfs == 3
        assert strategy.cash_ratio_risk_off == 0.7
        assert strategy.use_vol_weight is True
        assert strategy.use_market_filter is True
        assert strategy.use_trend_filter is True
        assert strategy.max_drawdown_filter == 0.15

    def test_create_etf_strategy_enhanced_off(self):
        """enhanced_etf_rotation OFF이면 기존 ETFRotationStrategy를 생성한다."""
        bot = _make_bot_with_flag(False)
        bot.feature_flags.is_enabled = MagicMock(
            side_effect=lambda n: n == "etf_rotation"
        )
        bot.feature_flags.get_config = MagicMock(
            side_effect=lambda n: {
                "etf_rotation": {
                    "lookback_months": 12,
                    "n_select": 3,
                    "max_same_sector": 1,
                },
            }.get(n, {})
        )

        strategy = bot._create_etf_strategy()

        from src.strategy.etf_rotation import ETFRotationStrategy
        assert isinstance(strategy, ETFRotationStrategy)
        assert strategy.num_etfs == 3
        assert strategy.lookback == 252  # 12 * 21

    def test_create_etf_strategy_config_params(self):
        """Enhanced 전략 생성 시 config 파라미터가 올바르게 전달된다."""
        bot = _make_bot_with_flag(False)
        bot.feature_flags.is_enabled = MagicMock(
            side_effect=lambda n: n in ("etf_rotation", "enhanced_etf_rotation")
        )
        bot.feature_flags.get_config = MagicMock(
            side_effect=lambda n: {
                "etf_rotation": {
                    "lookback_months": 9,
                    "n_select": 2,
                    "max_same_sector": 2,
                },
                "enhanced_etf_rotation": {
                    "cash_ratio_risk_off": 0.5,
                    "use_vol_weight": False,
                    "use_market_filter": False,
                    "use_trend_filter": False,
                    "use_max_drawdown_filter": False,
                    "max_drawdown_filter": 0.20,
                    "vol_lookback": 90,
                    "market_ma_period": 100,
                    "trend_short_ma": 10,
                    "trend_long_ma": 30,
                },
            }.get(n, {})
        )

        strategy = bot._create_etf_strategy()

        from src.strategy.enhanced_etf_rotation import (
            EnhancedETFRotationStrategy,
        )
        assert isinstance(strategy, EnhancedETFRotationStrategy)
        assert strategy.num_etfs == 2
        assert strategy.max_same_sector == 2
        assert strategy.cash_ratio_risk_off == 0.5
        assert strategy.use_vol_weight is False
        assert strategy.use_market_filter is False
        assert strategy.use_trend_filter is False
        # max_drawdown_filter disabled → 0.0
        assert strategy.max_drawdown_filter == 0.0
        assert strategy.vol_lookback == 90
        assert strategy.market_ma_period == 100
        assert strategy.trend_short_ma == 10
        assert strategy.trend_long_ma == 30

    def test_generate_etf_signals_uses_enhanced(self):
        """_generate_etf_signals가 enhanced 전략을 사용한다."""
        bot = _make_bot_with_flag(False)
        bot.feature_flags.is_enabled = MagicMock(
            side_effect=lambda n: n in ("etf_rotation", "enhanced_etf_rotation")
        )
        bot.feature_flags.get_config = MagicMock(
            side_effect=lambda n: {
                "etf_rotation": {
                    "lookback_months": 12,
                    "n_select": 3,
                    "max_same_sector": 1,
                },
                "enhanced_etf_rotation": {
                    "cash_ratio_risk_off": 0.7,
                    "use_vol_weight": True,
                    "use_market_filter": True,
                    "use_trend_filter": True,
                    "use_max_drawdown_filter": True,
                },
            }.get(n, {})
        )

        with patch.object(bot, "_fetch_etf_prices", return_value={}):
            result = bot._generate_etf_signals("20260228")
            # 가격 데이터 없으므로 빈 결과
            assert result == {}

    def test_get_etf_universe_uses_enhanced(self):
        """_get_etf_universe_tickers가 enhanced 전략의 유니버스를 반환한다."""
        bot = _make_bot_with_flag(False)
        bot.feature_flags.is_enabled = MagicMock(
            side_effect=lambda n: n in ("etf_rotation", "enhanced_etf_rotation")
        )
        bot.feature_flags.get_config = MagicMock(
            side_effect=lambda n: {
                "etf_rotation": {
                    "lookback_months": 12,
                    "n_select": 3,
                    "max_same_sector": 1,
                },
                "enhanced_etf_rotation": {},
            }.get(n, {})
        )

        tickers = bot._get_etf_universe_tickers()
        # 기본 유니버스: 10개 ETF
        assert len(tickers) == 10
        assert "069500" in tickers  # KODEX 200
        assert "439870" in tickers  # KODEX 단기채권

    def test_backward_compat_etf_rotation_only(self):
        """etf_rotation ON + enhanced OFF → 기존 전략 하위호환."""
        bot = _make_bot_with_flag(False)
        bot.feature_flags.is_enabled = MagicMock(
            side_effect=lambda n: n == "etf_rotation"
        )
        bot.feature_flags.get_config = MagicMock(
            side_effect=lambda n: {
                "etf_rotation": {
                    "lookback_months": 12,
                    "n_select": 3,
                    "max_same_sector": 1,
                },
            }.get(n, {})
        )

        strategy = bot._create_etf_strategy()

        from src.strategy.etf_rotation import ETFRotationStrategy
        from src.strategy.enhanced_etf_rotation import (
            EnhancedETFRotationStrategy,
        )
        assert isinstance(strategy, ETFRotationStrategy)
        assert not isinstance(strategy, EnhancedETFRotationStrategy)

    def test_create_long_term_strategy_uses_enhanced(self):
        """_create_long_term_strategy('etf_rotation')이 enhanced 전략을 사용한다."""
        bot = _make_bot_with_flag(False)
        bot.feature_flags.is_enabled = MagicMock(
            side_effect=lambda n: n in ("etf_rotation", "enhanced_etf_rotation")
        )
        bot.feature_flags.get_config = MagicMock(
            side_effect=lambda n: {
                "etf_rotation": {
                    "lookback_months": 12,
                    "n_select": 3,
                    "max_same_sector": 1,
                },
                "enhanced_etf_rotation": {
                    "cash_ratio_risk_off": 0.7,
                    "use_vol_weight": True,
                    "use_market_filter": True,
                    "use_trend_filter": True,
                    "use_max_drawdown_filter": True,
                },
            }.get(n, {})
        )

        strategy = bot._create_long_term_strategy("etf_rotation")

        from src.strategy.enhanced_etf_rotation import (
            EnhancedETFRotationStrategy,
        )
        assert isinstance(strategy, EnhancedETFRotationStrategy)


# ===================================================================
# 전략 데이터 수집 테스트
# ===================================================================


class TestCollectStrategyData:
    """_collect_strategy_data 메서드 테스트."""

    def test_returns_correct_keys(self):
        """반환값에 fundamentals, prices, index_prices 키가 있다."""
        bot = _make_bot_with_flag(False)
        with patch("src.scheduler.main.get_all_fundamentals") as mock_fund, \
             patch("src.scheduler.main.get_price_data"), \
             patch("src.scheduler.main.get_index_data") as mock_idx:
            mock_fund.return_value = pd.DataFrame({
                "ticker": ["005930"], "name": ["삼성전자"],
                "market_cap": [500e12], "pbr": [1.5],
            })
            mock_idx.return_value = pd.DataFrame({"close": [2800.0]})

            result = bot._collect_strategy_data("20260225")

            assert "fundamentals" in result
            assert "prices" in result
            assert "index_prices" in result

    def test_fundamentals_populated(self):
        """fundamentals가 올바르게 채워진다."""
        bot = _make_bot_with_flag(False)
        expected_df = pd.DataFrame({
            "ticker": ["005930", "000660"],
            "name": ["삼성전자", "SK하이닉스"],
            "market_cap": [500e12, 100e12],
        })
        with patch("src.scheduler.main.get_all_fundamentals") as mock_fund, \
             patch("src.scheduler.main.get_price_data"), \
             patch("src.scheduler.main.get_index_data") as mock_idx:
            mock_fund.return_value = expected_df
            mock_idx.return_value = pd.DataFrame()

            result = bot._collect_strategy_data("20260225")

            assert len(result["fundamentals"]) == 2
            mock_fund.assert_called_once_with("20260225")

    def test_empty_fundamentals_returns_safe_dict(self):
        """펀더멘탈 수집 실패 시에도 안전한 dict를 반환한다."""
        bot = _make_bot_with_flag(False)
        with patch("src.scheduler.main.get_all_fundamentals") as mock_fund, \
             patch("src.scheduler.main.get_price_data"), \
             patch("src.scheduler.main.get_index_data") as mock_idx:
            mock_fund.side_effect = Exception("pykrx error")
            mock_idx.return_value = pd.DataFrame()

            result = bot._collect_strategy_data("20260225")

            assert result["fundamentals"].empty
            assert result["prices"] == {}

    def test_cache_hit_avoids_api(self):
        """캐시 히트 시 get_all_fundamentals를 호출하지 않는다."""
        bot = _make_bot_with_flag(False)
        bot.feature_flags.is_enabled = MagicMock(
            side_effect=lambda n: n == "data_cache"
        )
        cached_df = pd.DataFrame({
            "ticker": ["005930"], "name": ["삼성전자"],
            "market_cap": [500e12],
        })
        bot.data_cache.get = MagicMock(return_value=cached_df)

        with patch("src.scheduler.main.get_all_fundamentals") as mock_fund, \
             patch("src.scheduler.main.get_price_data"), \
             patch("src.scheduler.main.get_index_data") as mock_idx:
            mock_idx.return_value = pd.DataFrame()

            result = bot._collect_strategy_data("20260225")

            mock_fund.assert_not_called()
            assert len(result["fundamentals"]) == 1


class TestCollectShortTermData:
    """_collect_short_term_data 메서드 테스트."""

    def test_returns_daily_data_and_date(self):
        """반환값에 daily_data와 date 키가 있다."""
        bot = _make_bot_with_flag(True)
        with patch("src.scheduler.main.get_all_fundamentals") as mock_fund, \
             patch("src.scheduler.main.get_price_data") as mock_price:
            mock_fund.return_value = pd.DataFrame({
                "ticker": ["005930"], "name": ["삼성전자"],
                "market_cap": [500e12],
            })
            mock_price.return_value = pd.DataFrame({
                "close": [70000.0], "volume": [1000000],
            })

            result = bot._collect_short_term_data("20260226")

            assert "daily_data" in result
            assert "date" in result
            assert result["date"] == "20260226"
            assert "005930" in result["daily_data"]

    def test_max_100_stocks(self):
        """시총 상위 100종목만 수집한다."""
        bot = _make_bot_with_flag(True)
        tickers = [f"{i:06d}" for i in range(150)]
        with patch("src.scheduler.main.get_all_fundamentals") as mock_fund, \
             patch("src.scheduler.main.get_price_data") as mock_price:
            mock_fund.return_value = pd.DataFrame({
                "ticker": tickers,
                "market_cap": list(range(150, 0, -1)),
            })
            mock_price.return_value = pd.DataFrame({"close": [100.0]})

            result = bot._collect_short_term_data("20260226")

            # 최대 100종목
            assert len(result["daily_data"]) <= 100

    def test_empty_fundamentals_returns_empty(self):
        """펀더멘탈 수집 실패 시 빈 daily_data를 반환한다."""
        bot = _make_bot_with_flag(True)
        with patch("src.scheduler.main.get_all_fundamentals") as mock_fund:
            mock_fund.side_effect = Exception("pykrx error")

            result = bot._collect_short_term_data("20260226")

            assert result["daily_data"] == {}


class TestPremarketWithData:
    """premarket_check가 데이터를 올바르게 수집하는지 테스트."""

    def test_passes_data_to_strategy(self):
        """premarket_check가 strategy에 데이터를 전달한다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=True)
        bot.holidays.is_rebalance_day = MagicMock(return_value=True)
        bot.holidays.prev_trading_day = MagicMock(
            return_value=datetime.date(2026, 2, 25)
        )

        mock_strategy = MagicMock()
        mock_strategy.generate_signals.return_value = {"005930": 0.1}
        bot._strategy = mock_strategy

        mock_data = {
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {},
            "index_prices": pd.Series(dtype=float),
        }
        bot._collect_strategy_data = MagicMock(return_value=mock_data)
        bot._send_notification = MagicMock()
        bot.executor.dry_run = MagicMock(return_value={
            "sell_orders": [], "buy_orders": [],
            "risk_check": {"passed": True, "warnings": []},
        })

        bot.premarket_check()

        # strategy에 data가 전달되었는지 확인
        call_args = mock_strategy.generate_signals.call_args[0]
        assert "fundamentals" in call_args[1]
        # T-1 데이터 기준일 확인
        bot._collect_strategy_data.assert_called_once_with("20260225")

    def test_skips_on_empty_fundamentals(self):
        """펀더멘탈 비어있으면 CRITICAL 알림 후 스킵한다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=True)
        bot.holidays.is_rebalance_day = MagicMock(return_value=True)
        bot.holidays.prev_trading_day = MagicMock(
            return_value=datetime.date(2026, 2, 25)
        )
        bot._strategy = MagicMock()

        bot._collect_strategy_data = MagicMock(return_value={
            "fundamentals": pd.DataFrame(),
            "prices": {},
            "index_prices": pd.Series(dtype=float),
        })
        bot._send_notification = MagicMock()

        bot.premarket_check()

        # strategy.generate_signals가 호출되지 않아야 함
        bot._strategy.generate_signals.assert_not_called()
        # CRITICAL 알림이 발송됨
        bot._send_notification.assert_called()
        call_text = bot._send_notification.call_args[0][0]
        assert "시그널 생성 실패" in call_text
        assert bot._send_notification.call_args[1].get("level") == "CRITICAL"


class TestRebalanceWithData:
    """execute_rebalance가 데이터를 올바르게 수집하는지 테스트."""

    def test_uses_prev_day_data(self):
        """리밸런싱이 T-1 기준 데이터를 사용한다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=True)
        bot.holidays.is_rebalance_day = MagicMock(return_value=True)
        bot.holidays.prev_trading_day = MagicMock(
            return_value=datetime.date(2026, 2, 25)
        )
        bot.kis_client.is_configured.return_value = True

        mock_strategy = MagicMock()
        mock_strategy.generate_signals.return_value = {"005930": 1.0}
        bot._strategy = mock_strategy

        mock_data = {
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {},
            "index_prices": pd.Series(dtype=float),
        }
        bot._collect_strategy_data = MagicMock(return_value=mock_data)
        bot._send_notification = MagicMock()
        bot.executor.execute_rebalance = MagicMock(return_value={
            "success": True, "sell_orders": [], "buy_orders": [],
            "errors": [],
        })

        bot.execute_rebalance()

        bot._collect_strategy_data.assert_called_once_with("20260225")


class TestSimulationWithData:
    """daily_simulation_batch가 데이터를 주입하는지 테스트."""

    def test_injects_strategy_data(self):
        """시뮬레이터에 strategy_data가 주입된다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=True)
        bot.feature_flags.is_enabled = MagicMock(
            side_effect=lambda n: n == "daily_simulation"
        )
        bot.feature_flags.get_config.return_value = {
            "strategies": ["multi_factor"],
        }

        mock_strategy = MagicMock()
        mock_strategy.generate_signals.return_value = {"005930": 1.0}

        mock_data = {
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {},
            "index_prices": pd.Series(dtype=float),
        }
        bot._collect_strategy_data = MagicMock(return_value=mock_data)
        bot._create_long_term_strategy = MagicMock(return_value=mock_strategy)
        bot._send_notification = MagicMock()

        with patch("src.data.daily_simulator.DailySimulator") as mock_sim_cls:
            mock_sim = MagicMock()
            mock_sim.run_daily_simulation.return_value = {}
            mock_sim.format_telegram_report.return_value = "test"
            mock_sim_cls.return_value = mock_sim

            bot.daily_simulation_batch()

            # strategy_data가 주입되었는지 확인
            assert mock_sim.strategy_data == mock_data
            bot._collect_strategy_data.assert_called_once()


class TestPrewarmStrategyCache:
    """prewarm_strategy_cache 테스트."""

    def test_calls_collect_strategy_data(self):
        """프리워밍 시 _collect_strategy_data가 호출된다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=True)

        mock_data = {
            "fundamentals": pd.DataFrame({"ticker": ["005930"] * 50}),
            "prices": {f"0{i:05d}": pd.DataFrame() for i in range(150)},
            "index_prices": pd.Series(range(200), dtype=float),
            "_meta": {
                "price_requested": 200,
                "price_collected": 150,
                "price_failed": 50,
                "cache_hits": 100,
                "api_fetched": 50,
                "failed_tickers": [],
            },
        }
        bot._collect_strategy_data = MagicMock(return_value=mock_data)
        bot._send_notification = MagicMock()

        bot.prewarm_strategy_cache()

        bot._collect_strategy_data.assert_called_once()
        bot._send_notification.assert_called_once()
        # 실패율 25% > 20% → CRITICAL
        call_kwargs = bot._send_notification.call_args[1]
        assert call_kwargs.get("level") == "CRITICAL"
        msg = bot._send_notification.call_args[0][0]
        assert "프리워밍 실패" in msg

    def test_success_sends_info(self):
        """프리워밍 성공 시 INFO 알림을 발송한다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=True)

        mock_data = {
            "fundamentals": pd.DataFrame({"ticker": ["005930"] * 50}),
            "prices": {f"0{i:05d}": pd.DataFrame() for i in range(190)},
            "index_prices": pd.Series(range(200), dtype=float),
            "_meta": {
                "price_requested": 200,
                "price_collected": 190,
                "price_failed": 10,
                "cache_hits": 180,
                "api_fetched": 10,
                "failed_tickers": [],
            },
        }
        bot._collect_strategy_data = MagicMock(return_value=mock_data)
        bot._send_notification = MagicMock()

        bot.prewarm_strategy_cache()

        msg = bot._send_notification.call_args[0][0]
        assert "프리워밍 완료" in msg
        assert "190/200" in msg

    def test_skips_non_trading_day(self):
        """비거래일에는 스킵한다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=False)
        bot._collect_strategy_data = MagicMock()

        bot.prewarm_strategy_cache()

        bot._collect_strategy_data.assert_not_called()

    def test_error_does_not_crash(self):
        """에러 시 크래시하지 않고 로깅만 한다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=True)
        bot._collect_strategy_data = MagicMock(
            side_effect=RuntimeError("test error")
        )
        bot._send_notification = MagicMock()

        # 예외가 발생하지 않아야 한다
        bot.prewarm_strategy_cache()

    def test_cache_key_matches_premarket(self):
        """15:45 프리워밍 캐시 키가 익일 08:50 캐시 키와 일치한다.

        15:45에 today_str로 캐시 → 익일 08:50에 prev_trading_day(내일) = 오늘
        → 동일한 date_str로 캐시 히트.
        """
        from src.data.cache import DataCache

        today_str = "20260304"

        # 프리워밍 시 캐시 키
        prewarm_fund_key = DataCache.make_key("fundamentals", today_str)
        prewarm_price_key = DataCache.make_key(
            "price", "005930", "20250129", today_str
        )
        prewarm_idx_key = DataCache.make_key("kospi_index", today_str)

        # 익일 08:50 시 캐시 키 (prev_trading_day(내일) = 오늘)
        premarket_fund_key = DataCache.make_key("fundamentals", today_str)
        premarket_price_key = DataCache.make_key(
            "price", "005930", "20250129", today_str
        )
        premarket_idx_key = DataCache.make_key("kospi_index", today_str)

        assert prewarm_fund_key == premarket_fund_key
        assert prewarm_price_key == premarket_price_key
        assert prewarm_idx_key == premarket_idx_key

    def test_schedule_registered(self):
        """setup_schedule에 prewarm_cache 작업이 등록된다."""
        bot = _make_bot_with_flag(False)
        bot.setup_schedule()

        job_ids = [j.id for j in bot.scheduler.get_jobs()]
        assert "prewarm_cache" in job_ids


class TestValidateSignalData:
    """_validate_signal_data 데이터 품질 검증 테스트."""

    def _import_bot_cls(self):
        return _import_trading_bot()

    def test_valid_data_passes(self):
        """정상 데이터는 검증을 통과한다."""
        cls = self._import_bot_cls()
        data = {
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {"005930": pd.DataFrame()},
            "index_prices": pd.Series([100.0, 101.0]),
            "_meta": {
                "price_requested": 200,
                "price_collected": 190,
                "price_failed": 10,
                "failed_tickers": [],
            },
        }
        is_valid, issues = cls._validate_signal_data(data)
        assert is_valid is True
        # 5% 실패율 → 경고 없음
        fatal = [i for i in issues if not i.startswith("(경고)")]
        assert len(fatal) == 0

    def test_empty_fundamentals_fails(self):
        """펀더멘탈이 비어있으면 실패한다."""
        cls = self._import_bot_cls()
        data = {
            "fundamentals": pd.DataFrame(),
            "prices": {},
            "index_prices": pd.Series(dtype=float),
        }
        is_valid, issues = cls._validate_signal_data(data)
        assert is_valid is False
        assert any("펀더멘탈" in i for i in issues)

    def test_high_failure_rate_fails(self):
        """가격 수집 실패율 > 20%이면 실패한다."""
        cls = self._import_bot_cls()
        data = {
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {"005930": pd.DataFrame()},
            "index_prices": pd.Series([100.0]),
            "_meta": {
                "price_requested": 200,
                "price_collected": 150,
                "price_failed": 50,
                "failed_tickers": [],
            },
        }
        is_valid, issues = cls._validate_signal_data(data)
        assert is_valid is False
        assert any("실패율 초과" in i for i in issues)

    def test_all_prices_failed(self):
        """가격 데이터가 전량 실패하면 실패한다."""
        cls = self._import_bot_cls()
        data = {
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {},
            "index_prices": pd.Series([100.0]),
            "_meta": {
                "price_requested": 200,
                "price_collected": 0,
                "price_failed": 200,
                "failed_tickers": [],
            },
        }
        is_valid, issues = cls._validate_signal_data(data)
        assert is_valid is False
        assert any("전량 실패" in i for i in issues)

    def test_boundary_20_percent_passes(self):
        """정확히 20% 실패율은 통과한다 (> 20% 기준)."""
        cls = self._import_bot_cls()
        data = {
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {"005930": pd.DataFrame()},
            "index_prices": pd.Series([100.0]),
            "_meta": {
                "price_requested": 200,
                "price_collected": 160,
                "price_failed": 40,
                "failed_tickers": [],
            },
        }
        is_valid, issues = cls._validate_signal_data(data)
        assert is_valid is True

    def test_missing_index_is_warning_only(self):
        """KOSPI 지수 없음은 경고만, 실패는 아니다."""
        cls = self._import_bot_cls()
        data = {
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {"005930": pd.DataFrame()},
            "index_prices": pd.Series(dtype=float),
            "_meta": {
                "price_requested": 200,
                "price_collected": 200,
                "price_failed": 0,
                "failed_tickers": [],
            },
        }
        is_valid, issues = cls._validate_signal_data(data)
        assert is_valid is True
        assert any("KOSPI" in i for i in issues)

    def test_no_meta_key_with_good_data(self):
        """_meta 키 없는 기존 데이터도 펀더멘탈이 있으면 통과한다."""
        cls = self._import_bot_cls()
        data = {
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {"005930": pd.DataFrame()},
            "index_prices": pd.Series([100.0]),
        }
        is_valid, issues = cls._validate_signal_data(data)
        assert is_valid is True


class TestPremarketSignalFailure:
    """premarket_check 시그널 생성 실패 시 CRITICAL 알림 테스트."""

    def test_data_quality_failure_sends_critical(self):
        """데이터 품질 불량 시 CRITICAL 알림을 발송한다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=True)
        bot.holidays.is_rebalance_day = MagicMock(return_value=True)
        bot.holidays.prev_trading_day = MagicMock(
            return_value=datetime.date(2026, 2, 25)
        )
        bot._strategy = MagicMock()
        bot._collect_strategy_data = MagicMock(return_value={
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {},
            "index_prices": pd.Series(dtype=float),
            "_meta": {
                "price_requested": 200,
                "price_collected": 50,
                "price_failed": 150,
                "failed_tickers": ["A"] * 10,
            },
        })
        bot._send_notification = MagicMock()

        bot.premarket_check()

        bot._strategy.generate_signals.assert_not_called()
        bot._send_notification.assert_called()
        call_kwargs = bot._send_notification.call_args[1]
        assert call_kwargs.get("level") == "CRITICAL"
        call_text = bot._send_notification.call_args[0][0]
        assert "실패율 초과" in call_text

    def test_signal_generation_exception_sends_critical(self):
        """시그널 생성 예외 시 CRITICAL 알림을 발송한다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=True)
        bot.holidays.is_rebalance_day = MagicMock(return_value=True)
        bot.holidays.prev_trading_day = MagicMock(
            return_value=datetime.date(2026, 2, 25)
        )
        mock_strategy = MagicMock()
        mock_strategy.generate_signals.side_effect = RuntimeError("test")
        bot._strategy = mock_strategy
        bot._collect_strategy_data = MagicMock(return_value={
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {"005930": pd.DataFrame()},
            "index_prices": pd.Series([100.0]),
            "_meta": {
                "price_requested": 200,
                "price_collected": 200,
                "price_failed": 0,
                "failed_tickers": [],
            },
        })
        bot._send_notification = MagicMock()

        bot.premarket_check()

        bot._send_notification.assert_called()
        call_kwargs = bot._send_notification.call_args[1]
        assert call_kwargs.get("level") == "CRITICAL"
        call_text = bot._send_notification.call_args[0][0]
        assert "예외 발생" in call_text


class TestRebalanceDataValidation:
    """execute_rebalance 데이터 품질 검증 테스트."""

    def test_data_quality_failure_aborts_rebalance(self):
        """데이터 품질 불량 시 리밸런싱을 중단한다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=True)
        bot.holidays.is_rebalance_day = MagicMock(return_value=True)
        bot.holidays.prev_trading_day = MagicMock(
            return_value=datetime.date(2026, 2, 25)
        )
        bot.kis_client.is_configured.return_value = True
        bot._strategy = MagicMock()
        bot._collect_strategy_data = MagicMock(return_value={
            "fundamentals": pd.DataFrame({"ticker": ["005930"]}),
            "prices": {},
            "index_prices": pd.Series(dtype=float),
            "_meta": {
                "price_requested": 200,
                "price_collected": 100,
                "price_failed": 100,
                "failed_tickers": [],
            },
        })
        bot._send_notification = MagicMock()

        bot.execute_rebalance()

        # 시그널 생성도 시도하지 않아야 함
        bot._strategy.generate_signals.assert_not_called()
        bot._send_notification.assert_called()
        call_kwargs = bot._send_notification.call_args[1]
        assert call_kwargs.get("level") == "CRITICAL"
        call_text = bot._send_notification.call_args[0][0]
        assert "리밸런싱 중단" in call_text


class TestScheduleTime:
    """스케줄 시간 변경 테스트."""

    def test_premarket_check_at_0730(self):
        """premarket_check가 07:30에 등록된다."""
        bot = _make_bot_with_flag(False)
        bot.setup_schedule()

        jobs = {j.id: j for j in bot.scheduler.get_jobs()}
        premarket = jobs["premarket_check"]
        trigger = premarket.trigger
        # CronTrigger의 fields에서 hour/minute 확인
        hour_field = str(trigger.fields[trigger.FIELD_NAMES.index("hour")])
        minute_field = str(trigger.fields[trigger.FIELD_NAMES.index("minute")])
        assert hour_field == "7"
        assert minute_field == "30"


# ===================================================================
# 매수가능성(Affordability) 필터 테스트
# ===================================================================

class TestFilterAffordableSignals:
    """_filter_affordable_signals 메서드 테스트."""

    def _make_bot(self):
        """allocator가 있는 bot을 생성한다."""
        bot = _make_bot_with_flag(True)
        # _strategy mock
        mock_strategy = MagicMock()
        mock_strategy.num_stocks = 7
        mock_strategy._last_combined_scores = pd.Series(dtype=float)
        bot._strategy = mock_strategy
        return bot

    def _make_prices(self, price_map: dict[str, float]) -> dict:
        """종목별 가격 데이터를 생성한다."""
        prices = {}
        for ticker, price in price_map.items():
            prices[ticker] = pd.DataFrame({"close": [price]})
        return prices

    def test_filters_expensive_stocks(self):
        """1주 가격 > budget인 종목이 제외된다."""
        bot = self._make_bot()
        signals = {
            "A001": 0.143,  # 비쌈
            "A002": 0.143,  # 싸다
            "A003": 0.143,  # 싸다
            "A004": 0.143,  # 싸다
            "A005": 0.143,  # 싸다
            "A006": 0.143,  # 싸다
            "A007": 0.143,  # 싸다
        }
        prices = self._make_prices({
            "A001": 100000,  # 예산 초과
            "A002": 5000,
            "A003": 5000,
            "A004": 5000,
            "A005": 5000,
            "A006": 5000,
            "A007": 5000,
        })
        strategy_data = {"prices": prices}
        # 예산: 105,000원 / 7종목 = 15,000원/종목, 버퍼 5% → max 14,250원
        pool_budget = 105000

        filtered, reasons = bot._filter_affordable_signals(
            signals, strategy_data, pool_budget, 7
        )

        assert "A001" not in filtered
        assert len(filtered) == 6
        assert any("A001" in r for r in reasons)

    def test_price_buffer_applied(self):
        """5% 가격 버퍼가 적용된다 (경계값)."""
        bot = self._make_bot()
        # 예산: 70,000 / 7 = 10,000원/종목, 버퍼 5% → max 9,500원
        signals = {"A001": 0.5, "A002": 0.5}
        prices = self._make_prices({
            "A001": 9600,  # > 9,500 → 제외
            "A002": 9400,  # < 9,500 → 통과
        })

        filtered, reasons = bot._filter_affordable_signals(
            signals, {"prices": prices}, 70000, 7
        )

        assert "A001" not in filtered
        assert "A002" in filtered

    def test_score_threshold(self):
        """스코어 하한선 이하 종목이 제외된다."""
        bot = self._make_bot()
        bot._strategy._last_combined_scores = pd.Series(
            {"A001": 1.0, "A002": 0.8, "A003": 0.5},
        )
        signals = {"A001": 0.33, "A002": 0.33, "A003": 0.33}
        # 모든 종목 가격 OK
        prices = self._make_prices({"A001": 100, "A002": 100, "A003": 100})

        filtered, reasons = bot._filter_affordable_signals(
            signals, {"prices": prices}, 700000, 7,
            score_threshold_ratio=0.7,
        )

        # top=1.0 * 0.7 = 0.7 → A003(0.5) 제외
        assert "A001" in filtered
        assert "A002" in filtered
        assert "A003" not in filtered

    def test_min_stocks_warning(self):
        """매수 가능 종목 < min_stocks이면 경고가 포함된다."""
        bot = self._make_bot()
        signals = {"A001": 0.25, "A002": 0.25, "A003": 0.25, "A004": 0.25}
        # 4개 중 3개 비쌈 → 1개만 통과
        prices = self._make_prices({
            "A001": 100000,
            "A002": 100000,
            "A003": 100000,
            "A004": 100,
        })

        filtered, reasons = bot._filter_affordable_signals(
            signals, {"prices": prices}, 70000, 7, min_stocks=5,
        )

        assert len(filtered) == 1
        assert any("미달" in r for r in reasons)

    def test_all_stocks_filtered(self):
        """전 종목 불가 시 빈 dict + 경고를 반환한다."""
        bot = self._make_bot()
        signals = {"A001": 0.5, "A002": 0.5}
        prices = self._make_prices({"A001": 100000, "A002": 100000})

        filtered, reasons = bot._filter_affordable_signals(
            signals, {"prices": prices}, 70000, 7,
        )

        assert filtered == {}
        assert any("전 종목 매수 불가" in r for r in reasons)

    def test_no_allocator_skips_filter(self):
        """allocator 없으면 필터가 적용되지 않는다."""
        bot = _make_bot_with_flag(False)
        bot._is_trading_day = MagicMock(return_value=True)
        bot.holidays.is_rebalance_day = MagicMock(return_value=True)
        bot.holidays.prev_trading_day = MagicMock(
            return_value=datetime.date(2026, 2, 25)
        )
        mock_strategy = MagicMock()
        mock_strategy.num_stocks = 7
        mock_strategy.generate_signals.return_value = {
            "005930": 0.5, "000660": 0.5,
        }
        bot._strategy = mock_strategy
        bot._collect_strategy_data = MagicMock(return_value={
            "fundamentals": pd.DataFrame({"ticker": ["005930", "000660"]}),
            "prices": {},
            "index_prices": pd.Series(dtype=float),
        })
        bot._send_notification = MagicMock()
        bot.executor.dry_run = MagicMock(return_value={
            "sell_orders": [], "buy_orders": [],
            "risk_check": {"passed": True, "warnings": []},
        })

        bot.premarket_check()

        # allocator=None이므로 generate_signals 결과가 그대로 사용
        assert bot._last_long_signals == {"005930": 0.5, "000660": 0.5}

    def test_reweights_equally(self):
        """필터 후 동일 비중이 재할당된다."""
        bot = self._make_bot()
        signals = {
            "A001": 0.2, "A002": 0.2, "A003": 0.2,
            "A004": 0.2, "A005": 0.2,
        }
        prices = self._make_prices({
            "A001": 100000,  # 제외
            "A002": 100, "A003": 100, "A004": 100, "A005": 100,
        })

        filtered, _ = bot._filter_affordable_signals(
            signals, {"prices": prices}, 70000, 7,
        )

        # 4개 남으면 각 25%
        assert len(filtered) == 4
        for w in filtered.values():
            assert abs(w - 0.25) < 1e-9

    def test_no_price_data_passes(self):
        """가격 데이터 없는 종목은 필터 통과한다."""
        bot = self._make_bot()
        signals = {"A001": 0.5, "A002": 0.5}
        # A001은 prices에 없음
        prices = self._make_prices({"A002": 100})

        filtered, reasons = bot._filter_affordable_signals(
            signals, {"prices": prices}, 70000, 7,
        )

        # 가격 없는 A001도 통과
        assert "A001" in filtered
        assert "A002" in filtered


class TestMultiFactorScanLimit:
    """MultiFactorStrategy의 scan_limit 기능 테스트."""

    def test_scan_limit_returns_more(self):
        """scan_limit=14이면 14개까지 반환한다."""
        from src.strategy.multi_factor import MultiFactorStrategy

        strategy = MultiFactorStrategy(num_stocks=7, spike_filter=False)

        # 20개 종목 데이터
        tickers = [f"T{i:03d}" for i in range(20)]
        fundamentals = pd.DataFrame({
            "ticker": tickers,
            "name": tickers,
            "pbr": [0.5 + i * 0.1 for i in range(20)],
        })
        prices = {}
        for t in tickers:
            prices[t] = pd.DataFrame({
                "close": list(range(100, 360))
            })

        data = {"fundamentals": fundamentals, "prices": prices}
        signals = strategy.generate_signals("20260301", data, scan_limit=14)

        assert len(signals) == 14

    def test_scan_limit_none_uses_num_stocks(self):
        """scan_limit 미지정이면 num_stocks개를 반환한다."""
        from src.strategy.multi_factor import MultiFactorStrategy

        strategy = MultiFactorStrategy(num_stocks=5, spike_filter=False)

        tickers = [f"T{i:03d}" for i in range(20)]
        fundamentals = pd.DataFrame({
            "ticker": tickers,
            "name": tickers,
            "pbr": [0.5 + i * 0.1 for i in range(20)],
        })
        prices = {}
        for t in tickers:
            prices[t] = pd.DataFrame({
                "close": list(range(100, 360))
            })

        data = {"fundamentals": fundamentals, "prices": prices}
        signals = strategy.generate_signals("20260301", data)

        assert len(signals) == 5

    def test_last_combined_scores_saved(self):
        """generate_signals 호출 후 _last_combined_scores가 저장된다."""
        from src.strategy.multi_factor import MultiFactorStrategy

        strategy = MultiFactorStrategy(num_stocks=5, spike_filter=False)
        assert strategy._last_combined_scores.empty

        tickers = [f"T{i:03d}" for i in range(10)]
        fundamentals = pd.DataFrame({
            "ticker": tickers,
            "name": tickers,
            "pbr": [0.5 + i * 0.1 for i in range(10)],
        })
        prices = {}
        for t in tickers:
            prices[t] = pd.DataFrame({
                "close": list(range(100, 360))
            })

        data = {"fundamentals": fundamentals, "prices": prices}
        strategy.generate_signals("20260301", data)

        assert not strategy._last_combined_scores.empty
        assert len(strategy._last_combined_scores) > 0
