"""알림 조건 정의 모듈.

포트폴리오 상태를 평가하여 알림이 필요한 조건을 판별한다.
각 조건 클래스는 AlertCondition ABC를 상속하며,
check(), format_message(), level, cooldown_hours, name을 구현해야 한다.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlertCondition(ABC):
    """알림 조건의 추상 기본 클래스.

    모든 알림 조건은 이 클래스를 상속하여
    check, format_message, level, cooldown_hours, name을 구현해야 한다.
    """

    @abstractmethod
    def check(self, state: dict) -> bool:
        """주어진 상태에서 알림 조건이 충족되는지 검사한다.

        Args:
            state: 포트폴리오/시스템 상태 딕셔너리.

        Returns:
            조건이 충족되면 True, 아니면 False.
        """

    @abstractmethod
    def format_message(self, state: dict) -> str:
        """알림 메시지를 생성한다.

        Args:
            state: 포트폴리오/시스템 상태 딕셔너리.

        Returns:
            발송할 알림 메시지 문자열.
        """

    @property
    @abstractmethod
    def level(self) -> str:
        """알림 심각도 수준을 반환한다.

        Returns:
            "INFO", "WARNING", "CRITICAL" 중 하나.
        """

    @property
    @abstractmethod
    def cooldown_hours(self) -> int:
        """동일 조건의 재알림 대기 시간(시간 단위)을 반환한다.

        Returns:
            쿨다운 시간 (시간 단위 정수).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """조건의 이름을 반환한다.

        Returns:
            조건 이름 문자열.
        """


class MddThresholdCondition(AlertCondition):
    """최대 낙폭(MDD) 임계값 초과 조건.

    포트폴리오의 현재 MDD가 설정한 임계값보다 낮으면
    (즉, 더 큰 손실이 발생하면) 알림을 발동한다.

    Attributes:
        threshold: MDD 임계값 (음수, 예: -0.15 = -15%).
    """

    def __init__(self, threshold: float = -0.15) -> None:
        """MddThresholdCondition을 초기화한다.

        Args:
            threshold: MDD 임계값. 기본값 -0.15 (-15%).
        """
        self.threshold: float = threshold

    @property
    def name(self) -> str:
        """조건 이름을 반환한다."""
        return "MDD 임계값 초과"

    @property
    def level(self) -> str:
        """심각도: CRITICAL."""
        return "CRITICAL"

    @property
    def cooldown_hours(self) -> int:
        """쿨다운: 24시간."""
        return 24

    def check(self, state: dict) -> bool:
        """현재 MDD가 임계값 이하인지 검사한다.

        Args:
            state: 'current_mdd' 키를 포함하는 상태 딕셔너리.

        Returns:
            current_mdd < threshold이면 True.
        """
        current_mdd: float = state.get("current_mdd", 0.0)
        return current_mdd < self.threshold

    def format_message(self, state: dict) -> str:
        """MDD 경고 메시지를 생성한다.

        Args:
            state: 'current_mdd' 키를 포함하는 상태 딕셔너리.

        Returns:
            MDD 경고 메시지 문자열.
        """
        current_mdd: float = state.get("current_mdd", 0.0)
        return (
            f"[MDD 경고] 현재 MDD: {current_mdd:.2%} "
            f"(임계값: {self.threshold:.2%})"
        )


class DailyMoveCondition(AlertCondition):
    """일일 급등락 조건.

    보유 종목 중 일일 수익률의 절대값이 임계값을 초과하는 종목이 있으면
    알림을 발동한다.

    Attributes:
        threshold: 일일 변동 임계값 (양수, 예: 0.05 = 5%).
    """

    def __init__(self, threshold: float = 0.05) -> None:
        """DailyMoveCondition을 초기화한다.

        Args:
            threshold: 일일 변동 임계값. 기본값 0.05 (5%).
        """
        self.threshold: float = threshold

    @property
    def name(self) -> str:
        """조건 이름을 반환한다."""
        return "일일 급등락"

    @property
    def level(self) -> str:
        """심각도: WARNING."""
        return "WARNING"

    @property
    def cooldown_hours(self) -> int:
        """쿨다운: 4시간."""
        return 4

    def check(self, state: dict) -> bool:
        """보유 종목 중 급등락 종목이 있는지 검사한다.

        Args:
            state: 'holdings_daily_returns' 키를 포함하는 상태 딕셔너리.
                holdings_daily_returns는 {종목코드: 수익률} 형태의 dict.

        Returns:
            abs(수익률) > threshold인 종목이 하나라도 있으면 True.
        """
        daily_returns: Dict[str, float] = state.get(
            "holdings_daily_returns", {}
        )
        return any(
            abs(ret) > self.threshold for ret in daily_returns.values()
        )

    def format_message(self, state: dict) -> str:
        """급등락 종목 알림 메시지를 생성한다.

        Args:
            state: 'holdings_daily_returns' 키를 포함하는 상태 딕셔너리.

        Returns:
            급등락 종목 목록이 포함된 메시지 문자열.
        """
        daily_returns: Dict[str, float] = state.get(
            "holdings_daily_returns", {}
        )
        triggered: List[str] = []
        for ticker, ret in daily_returns.items():
            if abs(ret) > self.threshold:
                direction = "급등" if ret > 0 else "급락"
                triggered.append(f"  - {ticker}: {ret:+.2%} ({direction})")

        lines = "\n".join(triggered)
        return f"[일일 급등락 알림]\n{lines}"


class UnderperformCondition(AlertCondition):
    """장기 언더퍼폼 조건.

    벤치마크 대비 연속 저조한 성과가 지정된 개월 수 이상이면
    알림을 발동한다.

    Attributes:
        months: 언더퍼폼 허용 기간 (개월).
    """

    def __init__(self, months: int = 3) -> None:
        """UnderperformCondition을 초기화한다.

        Args:
            months: 연속 언더퍼폼 허용 기간. 기본값 3개월.
        """
        self.months: int = months

    @property
    def name(self) -> str:
        """조건 이름을 반환한다."""
        return "장기 언더퍼폼"

    @property
    def level(self) -> str:
        """심각도: WARNING."""
        return "WARNING"

    @property
    def cooldown_hours(self) -> int:
        """쿨다운: 168시간 (1주일)."""
        return 168

    def check(self, state: dict) -> bool:
        """연속 언더퍼폼 기간이 임계값 이상인지 검사한다.

        Args:
            state: 'underperform_months' 키를 포함하는 상태 딕셔너리.

        Returns:
            underperform_months >= months이면 True.
        """
        underperform_months: int = state.get("underperform_months", 0)
        return underperform_months >= self.months

    def format_message(self, state: dict) -> str:
        """언더퍼폼 경고 메시지를 생성한다.

        Args:
            state: 'underperform_months' 키를 포함하는 상태 딕셔너리.

        Returns:
            언더퍼폼 경고 메시지 문자열.
        """
        underperform_months: int = state.get("underperform_months", 0)
        return (
            f"[언더퍼폼 경고] 벤치마크 대비 {underperform_months}개월 연속 저조한 성과. "
            f"전략 점검이 필요합니다."
        )


class SystemErrorCondition(AlertCondition):
    """시스템 오류 조건.

    시스템 오류 리스트가 비어 있지 않으면 알림을 발동한다.
    """

    @property
    def name(self) -> str:
        """조건 이름을 반환한다."""
        return "시스템 오류"

    @property
    def level(self) -> str:
        """심각도: CRITICAL."""
        return "CRITICAL"

    @property
    def cooldown_hours(self) -> int:
        """쿨다운: 1시간."""
        return 1

    def check(self, state: dict) -> bool:
        """시스템 오류가 존재하는지 검사한다.

        Args:
            state: 'system_errors' 키를 포함하는 상태 딕셔너리.
                system_errors는 오류 메시지 문자열의 리스트.

        Returns:
            system_errors 리스트가 비어있지 않으면 True.
        """
        errors: List[str] = state.get("system_errors", [])
        return len(errors) > 0

    def format_message(self, state: dict) -> str:
        """시스템 오류 알림 메시지를 생성한다.

        Args:
            state: 'system_errors' 키를 포함하는 상태 딕셔너리.

        Returns:
            시스템 오류 목록이 포함된 메시지 문자열.
        """
        errors: List[str] = state.get("system_errors", [])
        error_lines = "\n".join(f"  - {err}" for err in errors)
        return f"[시스템 오류] {len(errors)}건 발생:\n{error_lines}"


class RebalanceAlertCondition(AlertCondition):
    """리밸런싱 예정 알림 조건.

    리밸런싱까지 남은 일수가 지정된 기간 이내이면 알림을 발동한다.

    Attributes:
        days_before: 리밸런싱 전 알림 시작 일수.
    """

    def __init__(self, days_before: int = 3) -> None:
        """RebalanceAlertCondition을 초기화한다.

        Args:
            days_before: 리밸런싱 전 알림 시작 일수. 기본값 3일.
        """
        self.days_before: int = days_before

    @property
    def name(self) -> str:
        """조건 이름을 반환한다."""
        return "리밸런싱 예정"

    @property
    def level(self) -> str:
        """심각도: INFO."""
        return "INFO"

    @property
    def cooldown_hours(self) -> int:
        """쿨다운: 24시간."""
        return 24

    def check(self, state: dict) -> bool:
        """리밸런싱까지 남은 일수가 기준 이내인지 검사한다.

        Args:
            state: 'days_to_rebalance' 키를 포함하는 상태 딕셔너리.

        Returns:
            days_to_rebalance <= days_before이면 True.
        """
        days_to_rebalance: int = state.get("days_to_rebalance", 999)
        return days_to_rebalance <= self.days_before

    def format_message(self, state: dict) -> str:
        """리밸런싱 예정 알림 메시지를 생성한다.

        Args:
            state: 'days_to_rebalance' 키를 포함하는 상태 딕셔너리.

        Returns:
            리밸런싱 예정 알림 메시지 문자열.
        """
        days_to_rebalance: int = state.get("days_to_rebalance", 0)
        if days_to_rebalance <= 0:
            return "[리밸런싱 알림] 오늘이 리밸런싱 예정일입니다."
        return (
            f"[리밸런싱 알림] 리밸런싱까지 {days_to_rebalance}일 남았습니다."
        )


class MarketTimingChangeCondition(AlertCondition):
    """마켓 타이밍 신호 변경 조건.

    마켓 타이밍 신호가 변경되었을 때 알림을 발동한다.
    """

    @property
    def name(self) -> str:
        """조건 이름을 반환한다."""
        return "마켓 타이밍 변경"

    @property
    def level(self) -> str:
        """심각도: WARNING."""
        return "WARNING"

    @property
    def cooldown_hours(self) -> int:
        """쿨다운: 24시간."""
        return 24

    def check(self, state: dict) -> bool:
        """마켓 타이밍 신호가 변경되었는지 검사한다.

        Args:
            state: 'timing_signal_changed' 키를 포함하는 상태 딕셔너리.

        Returns:
            timing_signal_changed가 True이면 True.
        """
        return bool(state.get("timing_signal_changed", False))

    def format_message(self, state: dict) -> str:
        """마켓 타이밍 변경 알림 메시지를 생성한다.

        Args:
            state: 'timing_signal', 'timing_signal_previous' 등의
                키를 포함할 수 있는 상태 딕셔너리.

        Returns:
            마켓 타이밍 변경 알림 메시지 문자열.
        """
        previous = state.get("timing_signal_previous", "N/A")
        current = state.get("timing_signal", "N/A")
        return (
            f"[마켓 타이밍 변경] 신호가 변경되었습니다: "
            f"{previous} -> {current}"
        )
