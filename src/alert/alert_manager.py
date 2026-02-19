"""알림 통합 관리 모듈.

모든 알림 조건을 검사하고, 쿨다운을 관리하며,
등록된 모든 알림 채널(notifier)로 메시지를 발송한다.
알림 이력은 JSON 파일로 영속화하여 재시작 후에도 유지된다.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.alert.conditions import AlertCondition
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_HISTORY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "alert_history.json"
)
_MAX_HISTORY = 200


class AlertManager:
    """알림 조건 검사 및 발송을 통합 관리하는 클래스.

    조건(AlertCondition)과 알림 채널(notifier)을 등록하고,
    상태(state)를 기반으로 조건을 검사하여 알림을 발송한다.
    동일 조건에 대한 중복 알림을 방지하기 위해 쿨다운을 추적한다.

    Attributes:
        notifiers: 등록된 알림 채널 목록.
        conditions: 등록된 알림 조건 목록.
    """

    def __init__(self, history_path: Optional[str] = None) -> None:
        """AlertManager를 초기화한다.

        Args:
            history_path: 알림 이력 JSON 파일 경로. None이면 기본 경로 사용.
        """
        self.notifiers: List[Any] = []
        self.conditions: List[AlertCondition] = []
        self._cooldown_tracker: Dict[str, datetime] = {}
        self._history_path = Path(history_path) if history_path else _DEFAULT_HISTORY_PATH
        self._alert_history: List[Dict[str, Any]] = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """저장된 알림 이력을 로드한다."""
        try:
            if self._history_path.exists():
                data = json.loads(self._history_path.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def _save_history(self) -> None:
        """알림 이력을 파일에 저장한다."""
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            # 최근 _MAX_HISTORY건만 유지
            trimmed = self._alert_history[-_MAX_HISTORY:]
            self._history_path.write_text(
                json.dumps(trimmed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("알림 이력 저장 실패: %s", e)

    def add_notifier(self, notifier: Any) -> None:
        """알림 채널을 추가한다.

        TelegramNotifier 등 send_message 메서드를 가진 객체를 등록한다.

        Args:
            notifier: send_message(text) 메서드를 구현한 알림 채널 객체.
        """
        self.notifiers.append(notifier)
        logger.info("알림 채널 추가됨: %s", type(notifier).__name__)

    def add_condition(self, condition: AlertCondition) -> None:
        """알림 조건을 추가한다.

        Args:
            condition: AlertCondition을 상속한 조건 객체.
        """
        self.conditions.append(condition)
        logger.info("알림 조건 추가됨: %s", condition.name)

    def _is_on_cooldown(self, condition: AlertCondition) -> bool:
        """해당 조건이 쿨다운 중인지 확인한다.

        Args:
            condition: 확인할 알림 조건.

        Returns:
            쿨다운 중이면 True, 아니면 False.
        """
        last_sent: Optional[datetime] = self._cooldown_tracker.get(
            condition.name
        )
        if last_sent is None:
            return False

        cooldown_delta = timedelta(hours=condition.cooldown_hours)
        return datetime.now() - last_sent < cooldown_delta

    def _update_cooldown(self, condition: AlertCondition) -> None:
        """해당 조건의 쿨다운 타이머를 갱신한다.

        Args:
            condition: 쿨다운을 갱신할 알림 조건.
        """
        self._cooldown_tracker[condition.name] = datetime.now()

    def check_and_alert(self, state: dict) -> List[str]:
        """모든 조건을 검사하고 발동된 알림 메시지를 반환한다.

        각 조건의 check()를 호출하여 조건이 충족되면
        format_message()로 메시지를 생성하고, 쿨다운을 확인한 뒤
        알림을 발송한다. 쿨다운 중인 조건은 건너뛴다.

        Args:
            state: 포트폴리오/시스템 상태 딕셔너리.

        Returns:
            발동된 알림 메시지 문자열의 리스트.
        """
        triggered_messages: List[str] = []

        for condition in self.conditions:
            try:
                if not condition.check(state):
                    continue

                if self._is_on_cooldown(condition):
                    logger.debug(
                        "조건 '%s' 쿨다운 중 - 알림 스킵.", condition.name
                    )
                    continue

                message = condition.format_message(state)
                success = self.send(message, level=condition.level)

                if success:
                    self._update_cooldown(condition)
                    triggered_messages.append(message)

                    # 발송 이력 기록
                    self._alert_history.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "condition_name": condition.name,
                            "level": condition.level,
                            "message": message,
                        }
                    )
                    logger.info(
                        "알림 발동: [%s] %s", condition.level, condition.name
                    )

            except Exception as e:
                logger.error(
                    "조건 '%s' 검사 중 오류 발생: %s", condition.name, e
                )

        if triggered_messages:
            self._save_history()

        return triggered_messages

    def send(self, message: str, level: str = "INFO") -> bool:
        """등록된 모든 알림 채널로 메시지를 발송한다.

        CRITICAL 레벨이면 메시지 앞에 경고 이모지를 추가한다.

        Args:
            message: 발송할 메시지 텍스트.
            level: 알림 심각도 ("INFO", "WARNING", "CRITICAL").

        Returns:
            하나 이상의 채널에서 발송 성공하면 True, 아니면 False.
        """
        if level == "CRITICAL":
            message = f"\u26a0\ufe0f\u26a0\ufe0f CRITICAL \u26a0\ufe0f\u26a0\ufe0f\n{message}"
        elif level == "WARNING":
            message = f"\u26a0\ufe0f WARNING\n{message}"

        if not self.notifiers:
            logger.warning(
                "등록된 알림 채널이 없습니다. 메시지를 로그로만 기록합니다."
            )
            logger.info("[%s] %s", level, message)
            return True  # 채널이 없어도 로그 기록은 성공으로 간주

        any_success = False
        for notifier in self.notifiers:
            try:
                # notifier가 is_configured를 가지고 있으면 확인
                if hasattr(notifier, "is_configured") and not notifier.is_configured():
                    logger.debug(
                        "알림 채널 '%s' 미설정 상태 - 스킵.",
                        type(notifier).__name__,
                    )
                    continue

                result = notifier.send_message(message, parse_mode="")
                if result:
                    any_success = True
            except Exception as e:
                logger.error(
                    "알림 채널 '%s' 발송 실패: %s",
                    type(notifier).__name__,
                    e,
                )

        return any_success

    def get_alert_history(self) -> List[Dict[str, Any]]:
        """발송된 알림 이력을 반환한다.

        Returns:
            알림 이력 딕셔너리의 리스트.
            각 항목은 timestamp, condition_name, level, message를 포함한다.
        """
        return list(self._alert_history)
