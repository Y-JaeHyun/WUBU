"""Telegram 양방향 커맨드 핸들러.

requests 기반 getUpdates 롱폴링으로 사용자 커맨드를 수신하고,
FeatureFlags 및 시스템 상태를 제어한다.
"""

import threading
import time
from typing import Any, Callable, Optional

import requests

from src.alert.telegram_bot import TelegramNotifier
from src.utils.feature_flags import FeatureFlags
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TelegramCommander:
    """Telegram 커맨드 핸들러.

    getUpdates API로 메시지를 폴링하고,
    등록된 커맨드 핸들러를 호출한다.

    Attributes:
        notifier: 메시지 발송용 TelegramNotifier.
        feature_flags: FeatureFlags 인스턴스.
    """

    POLLING_TIMEOUT = 30
    DEFAULT_POLLING_INTERVAL = 2

    def __init__(
        self,
        notifier: TelegramNotifier,
        feature_flags: FeatureFlags,
        polling_interval: float = DEFAULT_POLLING_INTERVAL,
    ) -> None:
        """TelegramCommander를 초기화한다.

        Args:
            notifier: 메시지 발송용 TelegramNotifier.
            feature_flags: FeatureFlags 인스턴스.
            polling_interval: 폴링 간격 (초).
        """
        self.notifier = notifier
        self.feature_flags = feature_flags
        self.polling_interval = polling_interval
        self._last_update_id: int = 0
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._commands: dict[str, Callable] = {}
        self._register_default_commands()

    def _register_default_commands(self) -> None:
        """기본 커맨드를 등록한다."""
        self._commands["/help"] = self._cmd_help
        self._commands["/features"] = self._cmd_features
        self._commands["/toggle"] = self._cmd_toggle
        self._commands["/config"] = self._cmd_config
        self._commands["/status"] = self._cmd_status
        self._commands["/portfolio"] = self._cmd_portfolio
        self._commands["/reload"] = self._cmd_reload

    def register_command(
        self, command: str, handler: Callable[[str], str]
    ) -> None:
        """외부에서 커맨드를 추가 등록한다.

        Args:
            command: 커맨드 문자열 (예: "/portfolio").
            handler: args를 받아 응답 문자열을 반환하는 함수.
        """
        self._commands[command] = handler

    def _get_updates(self) -> list[dict]:
        """Telegram getUpdates API를 호출한다."""
        if not self.notifier.is_configured():
            return []
        url = self.notifier._build_url("getUpdates")
        params = {
            "offset": self._last_update_id + 1,
            "timeout": self.POLLING_TIMEOUT,
            "allowed_updates": ["message"],
        }
        try:
            resp = requests.get(
                url, params=params, timeout=self.POLLING_TIMEOUT + 5
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                return data.get("result", [])
        except (requests.RequestException, ValueError) as e:
            logger.debug("getUpdates 실패: %s", e)
        return []

    def _process_update(self, update: dict) -> None:
        """개별 update를 처리한다."""
        self._last_update_id = max(
            self._last_update_id, update.get("update_id", 0)
        )
        message = update.get("message", {})
        text = message.get("text", "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))

        # 보안: 등록된 chat_id만 허용
        if chat_id != self.notifier.chat_id:
            logger.warning("미등록 chat_id에서 메시지 수신: %s", chat_id)
            return

        if not text.startswith("/"):
            return

        # @botname 제거 (그룹 채팅 대응)
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower().split("@")[0]
        args = parts[1] if len(parts) > 1 else ""

        handler = self._commands.get(cmd)
        if handler:
            try:
                response = handler(args)
                if response:
                    self.notifier.send_message(response, parse_mode="")
            except Exception as e:
                logger.error("커맨드 '%s' 처리 오류: %s", cmd, e)
                self.notifier.send_message(f"오류: {e}", parse_mode="")
        else:
            self.notifier.send_message(
                f"알 수 없는 커맨드: {cmd}\n/help 로 확인하세요.",
                parse_mode="",
            )

    # ────────────────────────────────────────────
    # 기본 커맨드 핸들러
    # ────────────────────────────────────────────

    def _cmd_help(self, args: str) -> str:
        """사용 가능한 커맨드 목록."""
        lines = [
            "[사용 가능한 커맨드]",
            "/features - 피처 플래그 목록",
            "/toggle <name> - 피처 on/off 토글",
            "/config <name> [key=val] - 설정 조회/변경",
            "/status - 봇 상태 요약",
            "/portfolio - 포트폴리오 현황",
            "/reload - 설정 파일 리로드",
            "/help - 이 도움말",
        ]
        return "\n".join(lines)

    def _cmd_features(self, args: str) -> str:
        """피처 플래그 목록."""
        return self.feature_flags.get_summary()

    def _cmd_toggle(self, args: str) -> str:
        """피처 토글."""
        feature_name = args.strip()
        if not feature_name:
            return (
                "사용법: /toggle <feature_name>\n\n"
                + self.feature_flags.get_summary()
            )
        success = self.feature_flags.toggle(feature_name)
        if success:
            status = (
                "ON" if self.feature_flags.is_enabled(feature_name) else "OFF"
            )
            return f"'{feature_name}' -> {status}"
        return f"알 수 없는 피처: {feature_name}"

    def _cmd_config(self, args: str) -> str:
        """설정 조회/변경."""
        parts = args.strip().split(maxsplit=1)
        if not parts or not parts[0]:
            return "사용법: /config <feature> [key=value]"

        feature_name = parts[0]
        if feature_name not in self.feature_flags.get_all_status():
            return f"알 수 없는 피처: {feature_name}"

        config = self.feature_flags.get_config(feature_name)

        # 조회 모드
        if len(parts) == 1:
            if not config:
                return f"{feature_name}: 설정 없음"
            lines = [f"[{feature_name} 설정]"]
            for k, v in config.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        # 변경 모드
        kv = parts[1]
        if "=" not in kv:
            return "사용법: /config <feature> key=value"
        key, value_str = kv.split("=", 1)
        value = self._parse_value(value_str.strip())
        success = self.feature_flags.set_config(
            feature_name, key.strip(), value
        )
        if success:
            return f"{feature_name}.{key.strip()} = {value}"
        return "설정 변경 실패"

    def _cmd_status(self, args: str) -> str:
        """봇 상태 요약."""
        return self.feature_flags.get_summary()

    def _cmd_portfolio(self, args: str) -> str:
        """포트폴리오 현황 (TradingBot에서 콜백으로 교체)."""
        return "포트폴리오 조회는 봇 통합 후 사용 가능합니다."

    def _cmd_reload(self, args: str) -> str:
        """설정 파일 리로드."""
        self.feature_flags.reload()
        return "설정 리로드 완료.\n" + self.feature_flags.get_summary()

    @staticmethod
    def _parse_value(value_str: str) -> Any:
        """문자열을 적절한 타입으로 변환한다."""
        if value_str.lower() in ("true", "yes"):
            return True
        if value_str.lower() in ("false", "no"):
            return False
        try:
            return int(value_str)
        except ValueError:
            pass
        try:
            return float(value_str)
        except ValueError:
            pass
        return value_str

    # ────────────────────────────────────────────
    # 폴링 스레드 관리
    # ────────────────────────────────────────────

    def start_polling(self) -> None:
        """백그라운드 폴링 스레드를 시작한다."""
        if self._running:
            return
        if not self.notifier.is_configured():
            logger.warning(
                "Telegram 미설정 - 커맨드 폴링을 시작하지 않습니다."
            )
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._polling_loop, daemon=True, name="telegram-commander"
        )
        self._thread.start()
        logger.info("Telegram 커맨드 폴링 시작")

    def stop_polling(self) -> None:
        """폴링을 중지한다."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Telegram 커맨드 폴링 종료")

    def _polling_loop(self) -> None:
        """폴링 루프 (데몬 스레드에서 실행)."""
        while self._running:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._process_update(update)
            except Exception as e:
                logger.error("폴링 루프 오류: %s", e)
            time.sleep(self.polling_interval)
