"""텔레그램 봇 알림 발송 모듈.

Telegram Bot API를 HTTP 요청으로 직접 호출하여
메시지와 이미지를 전송한다. python-telegram-bot 패키지 없이
순수 requests 기반으로 동작한다.
"""

import os
import re
from typing import Optional

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# MarkdownV2에서 이스케이프가 필요한 특수 문자들
_MARKDOWNV2_ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!"


def escape_markdownv2(text: str) -> str:
    """MarkdownV2 형식에서 특수 문자를 이스케이프한다.

    텔레그램 MarkdownV2 파싱 모드는 특수 문자 앞에
    백슬래시(\\)를 붙여야 한다.

    Args:
        text: 이스케이프할 원본 문자열.

    Returns:
        이스케이프 처리된 문자열.
    """
    pattern = f"([{re.escape(_MARKDOWNV2_ESCAPE_CHARS)}])"
    return re.sub(pattern, r"\\\1", text)


class TelegramNotifier:
    """텔레그램 봇을 통해 알림을 발송하는 클래스.

    Telegram Bot API의 sendMessage, sendPhoto 엔드포인트를
    HTTP POST로 호출한다. 토큰이나 chat_id가 미설정이면
    에러 없이 로그만 남긴다 (graceful degradation).

    Attributes:
        bot_token: 텔레그램 봇 토큰.
        chat_id: 메시지를 보낼 채팅 ID.
    """

    BASE_URL = "https://api.telegram.org/bot{token}/{method}"
    REQUEST_TIMEOUT = 10  # seconds

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        """TelegramNotifier를 초기화한다.

        Args:
            bot_token: 텔레그램 봇 토큰.
                None이면 환경변수 TELEGRAM_BOT_TOKEN에서 로드한다.
            chat_id: 메시지를 보낼 채팅 ID.
                None이면 환경변수 TELEGRAM_CHAT_ID에서 로드한다.
        """
        self.bot_token: str = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id: str = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

        if not self.is_configured():
            logger.warning(
                "텔레그램 봇 설정이 불완전합니다. "
                "TELEGRAM_BOT_TOKEN과 TELEGRAM_CHAT_ID를 확인하세요."
            )

    def is_configured(self) -> bool:
        """봇 토큰과 chat_id가 모두 설정되어 있는지 확인한다.

        Returns:
            설정이 완료되었으면 True, 아니면 False.
        """
        return bool(self.bot_token) and bool(self.chat_id)

    def _build_url(self, method: str) -> str:
        """Telegram Bot API URL을 생성한다.

        Args:
            method: API 메서드 이름 (예: sendMessage, sendPhoto).

        Returns:
            완성된 API URL 문자열.
        """
        return self.BASE_URL.format(token=self.bot_token, method=method)

    def send_message(
        self, text: str, parse_mode: str = "MarkdownV2"
    ) -> bool:
        """텔레그램 메시지를 발송한다.

        Args:
            text: 전송할 메시지 텍스트.
            parse_mode: 메시지 파싱 모드. 기본값 "MarkdownV2".

        Returns:
            발송 성공 시 True, 실패 시 False.
        """
        if not self.is_configured():
            logger.warning("텔레그램 미설정 상태로 메시지를 발송하지 않습니다.")
            return False

        url = self._build_url("sendMessage")
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            response = requests.post(
                url, json=payload, timeout=self.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("ok"):
                logger.error(
                    "텔레그램 메시지 발송 실패: %s",
                    result.get("description", "알 수 없는 오류"),
                )
                return False

            logger.info("텔레그램 메시지 발송 성공.")
            return True

        except requests.exceptions.Timeout:
            logger.error("텔레그램 메시지 발송 타임아웃.")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("텔레그램 서버 연결 실패.")
            return False
        except requests.exceptions.RequestException as e:
            logger.error("텔레그램 메시지 발송 중 오류: %s", e)
            return False

    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """텔레그램으로 이미지(차트 등)를 전송한다.

        Args:
            photo_path: 전송할 이미지 파일의 경로.
            caption: 이미지에 첨부할 캡션 텍스트.

        Returns:
            발송 성공 시 True, 실패 시 False.
        """
        if not self.is_configured():
            logger.warning("텔레그램 미설정 상태로 사진을 발송하지 않습니다.")
            return False

        url = self._build_url("sendPhoto")
        data = {
            "chat_id": self.chat_id,
        }
        if caption:
            data["caption"] = caption

        try:
            with open(photo_path, "rb") as photo_file:
                files = {"photo": photo_file}
                response = requests.post(
                    url, data=data, files=files, timeout=self.REQUEST_TIMEOUT
                )
            response.raise_for_status()
            result = response.json()

            if not result.get("ok"):
                logger.error(
                    "텔레그램 사진 발송 실패: %s",
                    result.get("description", "알 수 없는 오류"),
                )
                return False

            logger.info("텔레그램 사진 발송 성공: %s", photo_path)
            return True

        except FileNotFoundError:
            logger.error("사진 파일을 찾을 수 없습니다: %s", photo_path)
            return False
        except requests.exceptions.Timeout:
            logger.error("텔레그램 사진 발송 타임아웃.")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("텔레그램 서버 연결 실패.")
            return False
        except requests.exceptions.RequestException as e:
            logger.error("텔레그램 사진 발송 중 오류: %s", e)
            return False
