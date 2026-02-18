"""SMTP 기반 이메일 알림 모듈.

AlertManager와 호환되는 인터페이스(send_message)를 제공하며,
HTML 리포트 및 이미지 첨부 발송 기능을 지원한다.
환경변수 미설정 시 에러 없이 로그만 남긴다 (graceful degradation).
"""

import os
import smtplib
import ssl
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmailNotifier:
    """SMTP 이메일 알림기.

    AlertManager와 호환되는 인터페이스(send_message 메서드)를 제공한다.
    TelegramNotifier와 동일한 패턴으로 is_configured(), send_message()를 구현한다.

    환경변수:
        SMTP_HOST: SMTP 서버 주소 (예: smtp.gmail.com).
        SMTP_PORT: SMTP 서버 포트 (기본 587).
        SMTP_SENDER: 발신 이메일 주소.
        SMTP_PASSWORD: 발신 이메일 비밀번호 (앱 비밀번호 권장).
        SMTP_RECIPIENTS: 수신자 이메일 (쉼표 구분, 복수 가능).

    Attributes:
        smtp_host: SMTP 서버 주소.
        smtp_port: SMTP 서버 포트.
        sender_email: 발신 이메일 주소.
        recipients: 수신자 이메일 리스트.
    """

    DEFAULT_PORT = 587
    REQUEST_TIMEOUT = 30  # seconds

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
        recipients: Optional[list[str]] = None,
    ) -> None:
        """EmailNotifier를 초기화한다.

        Args:
            smtp_host: SMTP 서버 주소.
                None이면 환경변수 SMTP_HOST에서 로드한다.
            smtp_port: SMTP 서버 포트.
                None이면 환경변수 SMTP_PORT에서 로드한다 (기본 587).
            sender_email: 발신 이메일 주소.
                None이면 환경변수 SMTP_SENDER에서 로드한다.
            sender_password: 발신 이메일 비밀번호.
                None이면 환경변수 SMTP_PASSWORD에서 로드한다.
            recipients: 수신자 이메일 리스트.
                None이면 환경변수 SMTP_RECIPIENTS에서 로드한다 (쉼표 구분).
        """
        self.smtp_host: str = smtp_host or os.getenv("SMTP_HOST", "")
        self.smtp_port: int = smtp_port or int(os.getenv("SMTP_PORT", str(self.DEFAULT_PORT)))
        self.sender_email: str = sender_email or os.getenv("SMTP_SENDER", "")
        self._sender_password: str = sender_password or os.getenv("SMTP_PASSWORD", "")

        if recipients is not None:
            self.recipients: list[str] = recipients
        else:
            raw = os.getenv("SMTP_RECIPIENTS", "")
            self.recipients = [
                r.strip() for r in raw.split(",") if r.strip()
            ]

        if not self.is_configured():
            logger.warning(
                "이메일 알림 설정이 불완전합니다. "
                "SMTP_HOST, SMTP_SENDER, SMTP_PASSWORD, SMTP_RECIPIENTS를 확인하세요."
            )

    def is_configured(self) -> bool:
        """이메일 설정이 완료되어 있는지 확인한다.

        Returns:
            설정이 완료되었으면 True, 아니면 False.
        """
        return bool(
            self.smtp_host
            and self.sender_email
            and self._sender_password
            and self.recipients
        )

    # ------------------------------------------------------------------
    # AlertManager 호환 인터페이스
    # ------------------------------------------------------------------

    def send_message(
        self,
        text: str,
        subject: str = "[Quant Bot] 알림",
        parse_mode: str = "",
    ) -> bool:
        """텍스트 메시지를 이메일로 발송한다.

        AlertManager.send()에서 호출하는 호환 인터페이스이다.
        parse_mode 매개변수는 Telegram 호환을 위해 존재하며 이메일에서는 무시된다.

        Args:
            text: 전송할 메시지 텍스트.
            subject: 이메일 제목 (기본 "[Quant Bot] 알림").
            parse_mode: 무시됨 (AlertManager 호환).

        Returns:
            발송 성공 시 True, 실패 시 False.
        """
        if not self.is_configured():
            logger.warning("이메일 미설정 상태로 메시지를 발송하지 않습니다.")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = ", ".join(self.recipients)

        # 순수 텍스트 본문
        text_part = MIMEText(text, "plain", "utf-8")
        msg.attach(text_part)

        # HTML 본문 (줄바꿈 → <br>)
        html_body = self._text_to_html(text)
        html_part = MIMEText(html_body, "html", "utf-8")
        msg.attach(html_part)

        return self._send_email(msg)

    # ------------------------------------------------------------------
    # HTML 리포트 발송
    # ------------------------------------------------------------------

    def send_html_report(
        self,
        html: str,
        subject: str,
        attachments: Optional[list[str]] = None,
    ) -> bool:
        """HTML 리포트를 이메일로 발송한다.

        이미지 파일을 첨부하여 인라인 표시할 수 있다.

        Args:
            html: HTML 본문 문자열.
            subject: 이메일 제목.
            attachments: 첨부할 이미지 파일 경로 리스트 (선택).
                각 이미지는 Content-ID로 인라인 참조 가능.

        Returns:
            발송 성공 시 True, 실패 시 False.
        """
        if not self.is_configured():
            logger.warning("이메일 미설정 상태로 HTML 리포트를 발송하지 않습니다.")
            return False

        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = ", ".join(self.recipients)

        # HTML 본문
        html_part = MIMEText(html, "html", "utf-8")
        msg.attach(html_part)

        # 이미지 첨부
        if attachments:
            for file_path in attachments:
                image_part = self._attach_image(file_path)
                if image_part is not None:
                    msg.attach(image_part)

        return self._send_email(msg)

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _send_email(self, msg: MIMEMultipart) -> bool:
        """SMTP로 이메일을 전송한다.

        TLS(STARTTLS)를 사용하여 보안 연결을 수립한다.

        Args:
            msg: 구성된 MIMEMultipart 이메일 객체.

        Returns:
            전송 성공 시 True, 실패 시 False.
        """
        try:
            context = ssl.create_default_context()

            with smtplib.SMTP(
                self.smtp_host,
                self.smtp_port,
                timeout=self.REQUEST_TIMEOUT,
            ) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(self.sender_email, self._sender_password)
                server.sendmail(
                    self.sender_email,
                    self.recipients,
                    msg.as_string(),
                )

            logger.info(
                "이메일 발송 성공: %s → %s",
                msg["Subject"],
                ", ".join(self.recipients),
            )
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error(
                "이메일 인증 실패. SMTP_SENDER와 SMTP_PASSWORD를 확인하세요."
            )
            return False
        except smtplib.SMTPConnectError:
            logger.error(
                "SMTP 서버 연결 실패: %s:%d", self.smtp_host, self.smtp_port
            )
            return False
        except smtplib.SMTPException as e:
            logger.error("이메일 발송 중 SMTP 오류: %s", e)
            return False
        except TimeoutError:
            logger.error("이메일 발송 타임아웃.")
            return False
        except OSError as e:
            logger.error("이메일 발송 중 네트워크 오류: %s", e)
            return False

    @staticmethod
    def _attach_image(file_path: str) -> Optional[MIMEImage]:
        """이미지 파일을 MIME 첨부로 변환한다.

        Content-ID를 파일명으로 설정하여 HTML에서
        <img src="cid:filename.png"> 형식으로 참조할 수 있다.

        Args:
            file_path: 이미지 파일 경로.

        Returns:
            MIMEImage 객체. 파일이 없거나 오류 시 None.
        """
        path = Path(file_path)

        if not path.is_file():
            logger.warning("첨부 이미지 파일을 찾을 수 없습니다: %s", file_path)
            return None

        # 확장자로 이미지 서브타입 결정
        ext = path.suffix.lower().lstrip(".")
        subtype_map = {
            "png": "png",
            "jpg": "jpeg",
            "jpeg": "jpeg",
            "gif": "gif",
            "bmp": "bmp",
            "webp": "webp",
        }
        subtype = subtype_map.get(ext, "png")

        try:
            with open(path, "rb") as f:
                image_data = f.read()

            image_part = MIMEImage(image_data, _subtype=subtype)
            image_part.add_header(
                "Content-ID", f"<{path.name}>"
            )
            image_part.add_header(
                "Content-Disposition",
                "inline",
                filename=path.name,
            )
            return image_part

        except OSError as e:
            logger.error("이미지 첨부 실패 (%s): %s", file_path, e)
            return None

    @staticmethod
    def _text_to_html(text: str) -> str:
        """순수 텍스트를 기본 HTML로 변환한다.

        줄바꿈을 <br> 태그로 변환하고, 기본 스타일을 적용한다.

        Args:
            text: 변환할 텍스트.

        Returns:
            HTML 문자열.
        """
        # 특수 문자 이스케이프
        escaped = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        # 줄바꿈 변환
        body = escaped.replace("\n", "<br>\n")

        return (
            "<!DOCTYPE html>"
            "<html><head><meta charset='utf-8'></head>"
            "<body style='font-family: monospace; font-size: 14px; "
            "line-height: 1.6; color: #333; padding: 20px;'>"
            f"{body}"
            "</body></html>"
        )
