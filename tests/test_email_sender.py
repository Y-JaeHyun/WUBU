"""이메일 알림 모듈(src/alert/email_sender.py) 테스트.

EmailNotifier의 환경변수 초기화, 설정 확인, SMTP mock 전송을 검증한다.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _try_import_email_notifier():
    """EmailNotifier가 있으면 임포트한다."""
    try:
        from src.alert.email_sender import EmailNotifier
        return EmailNotifier
    except ImportError:
        return None


# ===================================================================
# EmailNotifier 검증
# ===================================================================

class TestEmailNotifier:
    """EmailNotifier 검증."""

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_PORT": "587",
        "SMTP_SENDER": "test@gmail.com",
        "SMTP_PASSWORD": "test_password",
        "SMTP_RECIPIENTS": "recipient@gmail.com",
    })
    def test_init_from_env(self):
        """환경변수로부터 정상 초기화된다."""
        EmailNotifier = _try_import_email_notifier()
        if EmailNotifier is None:
            pytest.skip("EmailNotifier가 아직 구현되지 않았습니다.")

        notifier = EmailNotifier()
        assert notifier is not None, "EmailNotifier가 초기화되어야 합니다."
        assert notifier.smtp_host == "smtp.gmail.com"
        assert notifier.sender_email == "test@gmail.com"

    def test_is_configured_missing(self):
        """필수 환경변수가 없으면 is_configured()가 False이다."""
        EmailNotifier = _try_import_email_notifier()
        if EmailNotifier is None:
            pytest.skip("EmailNotifier가 아직 구현되지 않았습니다.")

        # 명시적으로 빈 값 전달
        notifier = EmailNotifier(
            smtp_host="", sender_email="", sender_password="", recipients=[],
        )

        assert notifier.is_configured() is False, (
            "환경변수 미설정 시 is_configured()가 False여야 합니다."
        )

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_PORT": "587",
        "SMTP_SENDER": "test@gmail.com",
        "SMTP_PASSWORD": "test_password",
        "SMTP_RECIPIENTS": "recipient@gmail.com",
    })
    def test_is_configured_complete(self):
        """모든 설정이 완료되면 is_configured()가 True이다."""
        EmailNotifier = _try_import_email_notifier()
        if EmailNotifier is None:
            pytest.skip("EmailNotifier가 아직 구현되지 않았습니다.")

        notifier = EmailNotifier()

        assert notifier.is_configured() is True, (
            "모든 설정 완료 시 is_configured()가 True여야 합니다."
        )

    def test_send_message_unconfigured(self):
        """미설정 상태에서 전송 시 False를 반환한다."""
        EmailNotifier = _try_import_email_notifier()
        if EmailNotifier is None:
            pytest.skip("EmailNotifier가 아직 구현되지 않았습니다.")

        notifier = EmailNotifier(
            smtp_host="", sender_email="", sender_password="", recipients=[],
        )

        result = notifier.send_message(text="테스트 메시지")

        assert result is False, (
            "미설정 상태에서 send_message는 False를 반환해야 합니다."
        )

    @patch("src.alert.email_sender.smtplib.SMTP")
    def test_send_message_success(self, mock_smtp_class):
        """SMTP mock으로 이메일 전송이 성공한다."""
        EmailNotifier = _try_import_email_notifier()
        if EmailNotifier is None:
            pytest.skip("EmailNotifier가 아직 구현되지 않았습니다.")

        # SMTP context manager mock 설정
        mock_smtp_instance = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

        notifier = EmailNotifier(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            sender_email="test@gmail.com",
            sender_password="test_password",
            recipients=["recipient@gmail.com"],
        )

        result = notifier.send_message(
            text="테스트 본문입니다.",
            subject="테스트 제목",
        )

        assert result is True, "SMTP mock으로 전송이 성공해야 합니다."
        mock_smtp_instance.sendmail.assert_called_once()

    @patch("src.alert.email_sender.smtplib.SMTP")
    def test_send_html_report(self, mock_smtp_class):
        """HTML 리포트 이메일이 전송된다."""
        EmailNotifier = _try_import_email_notifier()
        if EmailNotifier is None:
            pytest.skip("EmailNotifier가 아직 구현되지 않았습니다.")

        mock_smtp_instance = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

        notifier = EmailNotifier(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            sender_email="test@gmail.com",
            sender_password="test_password",
            recipients=["recipient@gmail.com"],
        )

        html_body = "<html><body><h1>일간 리포트</h1><p>수익률: +2.5%</p></body></html>"

        result = notifier.send_html_report(
            html=html_body,
            subject="일간 리포트",
        )

        assert result is True, "HTML 리포트 전송이 성공해야 합니다."
        mock_smtp_instance.sendmail.assert_called_once()
