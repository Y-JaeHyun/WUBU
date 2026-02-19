"""한국 장 휴일 관리 모듈.

KRX 휴장일(공휴일, 주말)을 관리하고,
거래일 판별, 다음/이전 거래일 계산, 리밸런싱일 판별 기능을 제공한다.
"""

from __future__ import annotations

import datetime
from typing import Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


class KRXHolidays:
    """KRX 휴장일 관리 클래스.

    고정 공휴일(신정, 삼일절, 어린이날 등)과 주말을 처리한다.
    음력 공휴일(설날, 추석, 부처님오신날)은 연도별로 하드코딩한다.

    Attributes:
        FIXED_HOLIDAYS: 매년 반복되는 고정 공휴일 (MM-DD 형식).
        LUNAR_HOLIDAYS: 음력 기반 변동 공휴일 (연도별 MM-DD 리스트).
    """

    # 고정 공휴일 (MM-DD)
    # 신정, 삼일절, 어린이날, 현충일, 광복절, 개천절, 한글날, 성탄절
    FIXED_HOLIDAYS: list[str] = [
        "01-01",  # 신정
        "03-01",  # 삼일절
        "05-05",  # 어린이날
        "06-06",  # 현충일
        "08-15",  # 광복절
        "10-03",  # 개천절
        "10-09",  # 한글날
        "12-25",  # 성탄절
    ]

    # 음력 공휴일 (연도별 MM-DD 리스트)
    # 설날 연휴, 부처님오신날, 추석 연휴 (대체공휴일 포함)
    LUNAR_HOLIDAYS: dict[int, list[str]] = {
        2024: [
            "02-09", "02-10", "02-11", "02-12",  # 설날 연휴 + 대체
            "04-10",                                # 부처님오신날
            "09-16", "09-17", "09-18",              # 추석 연휴
        ],
        2025: [
            "01-28", "01-29", "01-30", "01-31",  # 설날 연휴 + 대체
            "05-05",                                # 부처님오신날 (어린이날과 겹침)
            "10-05", "10-06", "10-07",              # 추석 연휴
        ],
        2026: [
            "02-16", "02-17", "02-18",              # 설날 연휴
            "05-24",                                # 부처님오신날
            "09-24", "09-25", "09-26", "09-28",  # 추석 연휴 + 대체(토→월)
        ],
        2027: [
            "02-05", "02-06", "02-07", "02-08",  # 설날 연휴 + 대체
            "05-13",                                # 부처님오신날
            "10-14", "10-15", "10-16",              # 추석 연휴
        ],
        2028: [
            "01-26", "01-27", "01-28",              # 설날 연휴
            "05-02",                                # 부처님오신날
            "10-02", "10-03", "10-04", "10-05",  # 추석 연휴 + 대체(개천절 겹침)
        ],
        2029: [
            "02-12", "02-13", "02-14",              # 설날 연휴
            "05-20", "05-21",                        # 부처님오신날 + 대체(일→월)
            "09-21", "09-22", "09-23", "09-24",  # 추석 연휴 + 대체(일→월)
        ],
        2030: [
            "02-02", "02-03", "02-04", "02-05",  # 설날 연휴 + 대체(일→화)
            "05-09",                                # 부처님오신날
            "09-11", "09-12", "09-13",              # 추석 연휴
        ],
    }

    # 연말 휴장일 (12-31은 KRX 정규 마감일이 아닐 수 있음)
    # 필요 시 여기에 추가

    def __init__(self) -> None:
        """KRXHolidays를 초기화한다."""
        # 빠른 조회를 위해 set으로 변환
        self._holiday_cache: dict[int, set[datetime.date]] = {}
        logger.debug("KRXHolidays 초기화 완료")

    def _build_holiday_set(self, year: int) -> set[datetime.date]:
        """특정 연도의 전체 휴장일 set을 구축한다.

        Args:
            year: 연도.

        Returns:
            해당 연도의 휴장일 date 집합.
        """
        if year in self._holiday_cache:
            return self._holiday_cache[year]

        holidays: set[datetime.date] = set()

        # 고정 공휴일 추가
        for md in self.FIXED_HOLIDAYS:
            try:
                month, day = md.split("-")
                holidays.add(datetime.date(year, int(month), int(day)))
            except ValueError:
                logger.warning("잘못된 고정 공휴일 형식: %s", md)

        # 음력 공휴일 추가
        lunar_dates = self.LUNAR_HOLIDAYS.get(year, [])
        for md in lunar_dates:
            try:
                month, day = md.split("-")
                holidays.add(datetime.date(year, int(month), int(day)))
            except ValueError:
                logger.warning("잘못된 음력 공휴일 형식: %s (연도: %d)", md, year)

        self._holiday_cache[year] = holidays
        return holidays

    @staticmethod
    def _to_date(date: Union[datetime.date, datetime.datetime, str]) -> datetime.date:
        """다양한 날짜 형식을 datetime.date로 변환한다.

        Args:
            date: 변환할 날짜. date, datetime, 또는 문자열 (YYYY-MM-DD, YYYYMMDD).

        Returns:
            datetime.date 객체.

        Raises:
            ValueError: 파싱할 수 없는 형식인 경우.
        """
        if isinstance(date, datetime.datetime):
            return date.date()
        if isinstance(date, datetime.date):
            return date
        if isinstance(date, str):
            clean = date.replace("-", "")
            if len(clean) == 8:
                return datetime.date(
                    int(clean[:4]), int(clean[4:6]), int(clean[6:8])
                )
            raise ValueError(f"날짜 형식을 파싱할 수 없습니다: {date}")
        raise TypeError(f"지원하지 않는 날짜 타입: {type(date)}")

    def is_holiday(self, date: Union[datetime.date, datetime.datetime, str]) -> bool:
        """해당 날짜가 공휴일인지 확인한다.

        주말은 포함하지 않는다. 순수 공휴일만 확인한다.

        Args:
            date: 확인할 날짜.

        Returns:
            공휴일이면 True.
        """
        d = self._to_date(date)
        holiday_set = self._build_holiday_set(d.year)
        return d in holiday_set

    def is_weekend(self, date: Union[datetime.date, datetime.datetime, str]) -> bool:
        """해당 날짜가 주말(토/일)인지 확인한다.

        Args:
            date: 확인할 날짜.

        Returns:
            주말이면 True.
        """
        d = self._to_date(date)
        return d.weekday() >= 5  # 5=토요일, 6=일요일

    def is_trading_day(self, date: Union[datetime.date, datetime.datetime, str]) -> bool:
        """해당 날짜가 거래일인지 확인한다.

        주말도 아니고 공휴일도 아니면 거래일이다.

        Args:
            date: 확인할 날짜.

        Returns:
            거래일이면 True.
        """
        d = self._to_date(date)
        if self.is_weekend(d):
            return False
        if self.is_holiday(d):
            return False
        return True

    def next_trading_day(
        self, date: Union[datetime.date, datetime.datetime, str]
    ) -> datetime.date:
        """다음 거래일을 반환한다.

        Args:
            date: 기준 날짜.

        Returns:
            기준 날짜 다음의 첫 거래일.
        """
        d = self._to_date(date)
        current = d + datetime.timedelta(days=1)

        # 무한루프 방지: 최대 30일까지 탐색
        max_attempts = 30
        for _ in range(max_attempts):
            if self.is_trading_day(current):
                return current
            current += datetime.timedelta(days=1)

        logger.warning(
            "다음 거래일을 %d일 이내에 찾지 못했습니다: %s", max_attempts, d
        )
        return current

    def prev_trading_day(
        self, date: Union[datetime.date, datetime.datetime, str]
    ) -> datetime.date:
        """이전 거래일을 반환한다.

        Args:
            date: 기준 날짜.

        Returns:
            기준 날짜 이전의 마지막 거래일.
        """
        d = self._to_date(date)
        current = d - datetime.timedelta(days=1)

        max_attempts = 30
        for _ in range(max_attempts):
            if self.is_trading_day(current):
                return current
            current -= datetime.timedelta(days=1)

        logger.warning(
            "이전 거래일을 %d일 이내에 찾지 못했습니다: %s", max_attempts, d
        )
        return current

    def trading_days_between(
        self,
        start: Union[datetime.date, datetime.datetime, str],
        end: Union[datetime.date, datetime.datetime, str],
    ) -> list[datetime.date]:
        """두 날짜 사이의 거래일 리스트를 반환한다.

        시작일과 종료일을 모두 포함한다.

        Args:
            start: 시작 날짜 (포함).
            end: 종료 날짜 (포함).

        Returns:
            거래일 리스트 (오름차순 정렬).
        """
        start_d = self._to_date(start)
        end_d = self._to_date(end)

        if start_d > end_d:
            logger.warning("시작일이 종료일보다 큽니다: %s > %s", start_d, end_d)
            return []

        trading_days: list[datetime.date] = []
        current = start_d

        while current <= end_d:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += datetime.timedelta(days=1)

        return trading_days

    def is_rebalance_day(
        self,
        date: Union[datetime.date, datetime.datetime, str],
        freq: str = "monthly",
    ) -> bool:
        """리밸런싱일인지 확인한다.

        매월 첫 거래일 또는 분기 첫 거래일인지 확인한다.

        Args:
            date: 확인할 날짜.
            freq: 리밸런싱 주기. "monthly" 또는 "quarterly".

        Returns:
            리밸런싱일이면 True.
        """
        d = self._to_date(date)

        if not self.is_trading_day(d):
            return False

        if freq == "monthly":
            # 해당 월의 첫 거래일인지 확인
            first_of_month = datetime.date(d.year, d.month, 1)
            first_trading = first_of_month

            # 1일부터 거래일 탐색
            max_attempts = 15
            for _ in range(max_attempts):
                if self.is_trading_day(first_trading):
                    break
                first_trading += datetime.timedelta(days=1)
            else:
                return False

            return d == first_trading

        elif freq == "quarterly":
            # 분기 시작월 (1, 4, 7, 10)의 첫 거래일인지 확인
            quarter_months = {1, 4, 7, 10}
            if d.month not in quarter_months:
                return False

            first_of_month = datetime.date(d.year, d.month, 1)
            first_trading = first_of_month

            max_attempts = 15
            for _ in range(max_attempts):
                if self.is_trading_day(first_trading):
                    break
                first_trading += datetime.timedelta(days=1)
            else:
                return False

            return d == first_trading

        else:
            logger.warning("지원하지 않는 리밸런싱 주기: %s", freq)
            return False

    def get_next_rebalance_day(
        self,
        date: Union[datetime.date, datetime.datetime, str],
        freq: str = "monthly",
    ) -> datetime.date:
        """다음 리밸런싱일을 반환한다.

        Args:
            date: 기준 날짜.
            freq: 리밸런싱 주기. "monthly" 또는 "quarterly".

        Returns:
            다음 리밸런싱일.
        """
        d = self._to_date(date)
        current = d + datetime.timedelta(days=1)

        # 최대 4개월(약 120일) 탐색
        max_attempts = 120
        for _ in range(max_attempts):
            if self.is_rebalance_day(current, freq):
                return current
            current += datetime.timedelta(days=1)

        logger.warning("다음 리밸런싱일을 찾지 못했습니다: %s (주기: %s)", d, freq)
        return current

    def days_to_next_rebalance(
        self,
        date: Union[datetime.date, datetime.datetime, str],
        freq: str = "monthly",
    ) -> int:
        """다음 리밸런싱일까지 남은 날수를 반환한다.

        Args:
            date: 기준 날짜.
            freq: 리밸런싱 주기.

        Returns:
            남은 날수. 오늘이 리밸런싱일이면 0.
        """
        d = self._to_date(date)

        if self.is_rebalance_day(d, freq):
            return 0

        next_rebal = self.get_next_rebalance_day(d, freq)
        return (next_rebal - d).days
