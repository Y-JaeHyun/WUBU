"""단기 트레이딩 전용 리스크 관리 모듈."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataHealthStatus(Enum):
    """데이터 건강 상태."""
    OK = "ok"
    WARNING = "warning"        # 3분 이상 틱 없음
    EMERGENCY = "emergency"    # 10분 이상 틱 없음


@dataclass
class ShortTermRiskConfig:
    """단기 리스크 설정."""
    stop_loss_pct: float = -0.05       # -5% 손절
    take_profit_pct: float = 0.10      # +10% 익절
    max_concurrent_positions: int = 3   # 최대 동시 포지션
    max_daily_loss_pct: float = -0.03  # 일일 최대 손실 -3%
    max_single_position_pct: float = 0.50  # 단기 풀 대비 단일 종목 최대 50%
    daytrading_close_time: str = "15:20"  # 데이트레이딩 강제 청산 시각
    data_warning_seconds: int = 180    # 3분 경고
    data_emergency_seconds: int = 600  # 10분 긴급


class ShortTermRiskManager:
    """단기 풀 전용 리스크 관리자.

    손절/익절, 일일 손실 한도, 동시 포지션 제한, 데이터 헬스 모니터링 등.
    """

    def __init__(self, config: Optional[ShortTermRiskConfig] = None):
        self._config = config or ShortTermRiskConfig()
        self._daily_pnl: float = 0.0        # 당일 실현 손익 (비율)
        self._daily_trade_count: int = 0
        self._trade_date: Optional[str] = None  # "YYYY-MM-DD" 형태로 날짜 변경 감지

    @property
    def config(self) -> ShortTermRiskConfig:
        return self._config

    def reset_daily(self) -> None:
        """일일 통계 리셋 (매일 장 시작 전)."""
        self._daily_pnl = 0.0
        self._daily_trade_count = 0
        self._trade_date = datetime.now().strftime("%Y-%m-%d")

    def _ensure_today(self) -> None:
        """날짜가 변경되면 자동 리셋."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._trade_date != today:
            self.reset_daily()

    def record_trade_pnl(self, pnl_pct: float) -> None:
        """거래 결과(손익비율)를 기록한다.

        Args:
            pnl_pct: 손익 비율 (예: 0.05 = +5%, -0.03 = -3%)
        """
        self._ensure_today()
        self._daily_pnl += pnl_pct
        self._daily_trade_count += 1

    def check_can_open(self, current_position_count: int,
                       order_amount: int, short_term_budget: int) -> tuple[bool, str]:
        """신규 진입 가능 여부를 체크한다.

        Args:
            current_position_count: 현재 단기 포지션 수.
            order_amount: 주문 금액 (원).
            short_term_budget: 단기 풀 총 예산 (원).

        Returns:
            (가능 여부, 사유)
        """
        self._ensure_today()

        # 1. 최대 포지션 수 체크
        if current_position_count >= self._config.max_concurrent_positions:
            return False, f"최대 동시 포지션 초과: {current_position_count}/{self._config.max_concurrent_positions}"

        # 2. 일일 손실 한도 체크
        if self._daily_pnl <= self._config.max_daily_loss_pct:
            return False, f"일일 손실 한도 도달: {self._daily_pnl:.2%} (한도: {self._config.max_daily_loss_pct:.2%})"

        # 3. 단일 포지션 비중 체크
        if short_term_budget > 0:
            position_pct = order_amount / short_term_budget
            if position_pct > self._config.max_single_position_pct:
                return False, f"단일 포지션 비중 초과: {position_pct:.1%} > {self._config.max_single_position_pct:.1%}"

        return True, ""

    def check_stop_loss(self, entry_price: float, current_price: float) -> tuple[bool, str]:
        """손절 여부를 체크한다.

        Args:
            entry_price: 진입 가격.
            current_price: 현재 가격.

        Returns:
            (손절 필요 여부, 사유). True이면 즉시 매도 필요.
        """
        if entry_price <= 0:
            return False, ""

        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct <= self._config.stop_loss_pct:
            return True, f"손절 트리거: {pnl_pct:.2%} (한도: {self._config.stop_loss_pct:.2%})"

        return False, ""

    def check_take_profit(self, entry_price: float, current_price: float) -> tuple[bool, str]:
        """익절 여부를 체크한다.

        Returns:
            (익절 필요 여부, 사유). True이면 매도 고려.
        """
        if entry_price <= 0:
            return False, ""

        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct >= self._config.take_profit_pct:
            return True, f"익절 트리거: {pnl_pct:.2%} (목표: {self._config.take_profit_pct:.2%})"

        return False, ""

    def check_time_stop(self, entry_date: str, max_holding_days: int = 5) -> tuple[bool, str]:
        """시간 손절을 체크한다 (스윙 전용).

        Args:
            entry_date: 진입일 "YYYY-MM-DD".
            max_holding_days: 최대 보유일 (기본 5영업일).

        Returns:
            (청산 필요 여부, 사유)
        """
        try:
            entry = datetime.strptime(entry_date, "%Y-%m-%d")
            elapsed = (datetime.now() - entry).days
            if elapsed >= max_holding_days:
                return True, f"시간 손절: {elapsed}일 경과 (한도: {max_holding_days}일)"
        except ValueError:
            pass
        return False, ""

    def check_data_health(self, last_tick_time: Optional[datetime]) -> DataHealthStatus:
        """데이터 수신 상태를 체크한다.

        Args:
            last_tick_time: 마지막 틱 수신 시각. None이면 EMERGENCY.

        Returns:
            DataHealthStatus 열거값.
        """
        if last_tick_time is None:
            return DataHealthStatus.EMERGENCY

        elapsed = (datetime.now() - last_tick_time).total_seconds()

        if elapsed >= self._config.data_emergency_seconds:
            return DataHealthStatus.EMERGENCY
        elif elapsed >= self._config.data_warning_seconds:
            return DataHealthStatus.WARNING
        else:
            return DataHealthStatus.OK

    def should_force_close_daytrading(self) -> bool:
        """데이트레이딩 강제 청산 시각 여부를 판단한다.

        Returns:
            True이면 모든 데이트레이딩 포지션 즉시 청산 필요.
        """
        now = datetime.now()
        close_time_str = self._config.daytrading_close_time
        try:
            hour, minute = map(int, close_time_str.split(":"))
            close_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return now >= close_time
        except (ValueError, AttributeError):
            return False

    def check_position(self, entry_price: float, current_price: float,
                       entry_date: str, mode: str = "swing",
                       max_holding_days: int = 5,
                       last_tick_time: Optional[datetime] = None) -> dict:
        """포지션 종합 리스크 체크.

        모든 리스크 조건을 한번에 체크하고 결과를 반환한다.

        Args:
            entry_price: 진입가.
            current_price: 현재가.
            entry_date: 진입일 "YYYY-MM-DD".
            mode: "swing" 또는 "daytrading".
            max_holding_days: 최대 보유일.
            last_tick_time: 마지막 틱 수신 시각.

        Returns:
            {
                "should_close": bool,
                "reasons": [str],
                "pnl_pct": float,
                "data_health": str,
                "checks": {
                    "stop_loss": bool,
                    "take_profit": bool,
                    "time_stop": bool,
                    "data_emergency": bool,
                    "daytrading_close": bool,
                }
            }
        """
        result = {
            "should_close": False,
            "reasons": [],
            "pnl_pct": 0.0,
            "data_health": "ok",
            "checks": {
                "stop_loss": False,
                "take_profit": False,
                "time_stop": False,
                "data_emergency": False,
                "daytrading_close": False,
            }
        }

        if entry_price > 0:
            result["pnl_pct"] = (current_price - entry_price) / entry_price

        # 1. 손절
        triggered, reason = self.check_stop_loss(entry_price, current_price)
        if triggered:
            result["should_close"] = True
            result["reasons"].append(reason)
            result["checks"]["stop_loss"] = True

        # 2. 익절
        triggered, reason = self.check_take_profit(entry_price, current_price)
        if triggered:
            result["should_close"] = True
            result["reasons"].append(reason)
            result["checks"]["take_profit"] = True

        # 3. 시간 손절 (스윙만)
        if mode == "swing":
            triggered, reason = self.check_time_stop(entry_date, max_holding_days)
            if triggered:
                result["should_close"] = True
                result["reasons"].append(reason)
                result["checks"]["time_stop"] = True

        # 4. 데이터 헬스
        health = self.check_data_health(last_tick_time)
        result["data_health"] = health.value
        if health == DataHealthStatus.EMERGENCY:
            result["should_close"] = True
            result["reasons"].append("데이터 수신 긴급: 장시간 틱 미수신")
            result["checks"]["data_emergency"] = True

        # 5. 데이트레이딩 시간 청산
        if mode == "daytrading" and self.should_force_close_daytrading():
            result["should_close"] = True
            result["reasons"].append(f"데이트레이딩 청산 시각: {self._config.daytrading_close_time}")
            result["checks"]["daytrading_close"] = True

        return result

    def get_daily_summary(self) -> dict:
        """당일 거래 요약을 반환한다."""
        self._ensure_today()
        return {
            "date": self._trade_date,
            "daily_pnl_pct": round(self._daily_pnl, 4),
            "trade_count": self._daily_trade_count,
            "daily_loss_limit": self._config.max_daily_loss_pct,
            "daily_loss_remaining": round(self._config.max_daily_loss_pct - self._daily_pnl, 4) if self._daily_pnl < 0 else self._config.max_daily_loss_pct,
            "can_trade": self._daily_pnl > self._config.max_daily_loss_pct,
        }
