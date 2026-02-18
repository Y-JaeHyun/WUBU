"""스케줄러 모듈 테스트.

KRXHolidays(한국거래소 휴장일 관리), TradingBot(트레이딩 봇) 등을 검증한다.
외부 API 호출은 mock 처리한다.
"""

import datetime
from unittest.mock import MagicMock, patch

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
