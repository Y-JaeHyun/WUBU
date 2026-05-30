"""scripts/email_weekly_report.py 헬퍼 테스트.

마크다운→HTML 변환, 이메일 제목 형식, 다중 파일 연결,
SMTP 미설정 시 graceful degradation을 검증한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.email_weekly_report import (
    extract_report_date,
    render_html,
    send_weekly_report,
)


SAMPLE_MD = """# 주간 밸류체인 분석

## 1. 시장 개요
- KOSPI: 2,500
- KOSDAQ: 800

## 2. 핵심 섹터
**Physical AI**가 주도했다.
"""


class TestRenderHtml:
    def test_renders_headings_and_bold(self):
        html = render_html(SAMPLE_MD)
        assert "<h1>" in html
        assert "<h2>" in html
        assert "<strong>Physical AI</strong>" in html

    def test_wraps_with_html_skeleton(self):
        html = render_html("# Hi")
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_utf8_meta(self):
        html = render_html("# 한글")
        assert "charset" in html.lower()
        assert "한글" in html


class TestExtractReportDate:
    def test_parses_filename(self):
        path = Path("/tmp/2026-05-30-weekly-valuechain-report.md")
        assert extract_report_date(path) == "2026-05-30"

    def test_returns_none_when_no_date(self):
        path = Path("/tmp/some-other-file.md")
        assert extract_report_date(path) is None


class TestSendWeeklyReport:
    def test_graceful_degradation_when_not_configured(self, tmp_path):
        report = tmp_path / "2026-05-30-weekly-valuechain-report.md"
        report.write_text(SAMPLE_MD, encoding="utf-8")

        with patch.dict(os.environ, {
            "SMTP_HOST": "",
            "SMTP_SENDER": "",
            "SMTP_PASSWORD": "",
            "SMTP_RECIPIENTS": "",
        }, clear=False):
            result = send_weekly_report(report)
            assert result is False

    def test_subject_format_uses_filename_date(self, tmp_path):
        report = tmp_path / "2026-05-30-weekly-valuechain-report.md"
        report.write_text(SAMPLE_MD, encoding="utf-8")

        fake_notifier = MagicMock()
        fake_notifier.is_configured.return_value = True
        fake_notifier.send_html_report.return_value = True

        with patch(
            "scripts.email_weekly_report.EmailNotifier",
            return_value=fake_notifier,
        ):
            result = send_weekly_report(report)

        assert result is True
        fake_notifier.send_html_report.assert_called_once()
        kwargs = fake_notifier.send_html_report.call_args.kwargs
        args = fake_notifier.send_html_report.call_args.args
        # subject may be passed positionally or as kwarg
        subject = kwargs.get("subject") or (args[1] if len(args) > 1 else None)
        assert subject == "[주간 리포트] 2026-05-30 밸류체인 분석"

    def test_fallback_subject_uses_today_when_no_date_in_filename(
        self, tmp_path, monkeypatch
    ):
        report = tmp_path / "weekly-report.md"
        report.write_text("# 보고서", encoding="utf-8")

        fake_notifier = MagicMock()
        fake_notifier.is_configured.return_value = True
        fake_notifier.send_html_report.return_value = True

        with patch(
            "scripts.email_weekly_report.EmailNotifier",
            return_value=fake_notifier,
        ):
            result = send_weekly_report(report)

        assert result is True
        kwargs = fake_notifier.send_html_report.call_args.kwargs
        args = fake_notifier.send_html_report.call_args.args
        subject = kwargs.get("subject") or args[1]
        # YYYY-MM-DD pattern
        import re
        assert re.match(
            r"^\[주간 리포트\] \d{4}-\d{2}-\d{2} 밸류체인 분석$", subject
        )

    def test_html_body_passed_to_notifier(self, tmp_path):
        report = tmp_path / "2026-05-30-weekly-valuechain-report.md"
        report.write_text(SAMPLE_MD, encoding="utf-8")

        fake_notifier = MagicMock()
        fake_notifier.is_configured.return_value = True
        fake_notifier.send_html_report.return_value = True

        with patch(
            "scripts.email_weekly_report.EmailNotifier",
            return_value=fake_notifier,
        ):
            send_weekly_report(report)

        args = fake_notifier.send_html_report.call_args.args
        kwargs = fake_notifier.send_html_report.call_args.kwargs
        html_body = kwargs.get("html") or args[0]
        assert "<h1>" in html_body
        assert "주간 밸류체인 분석" in html_body

    def test_returns_false_when_file_missing(self, tmp_path):
        missing = tmp_path / "missing.md"
        result = send_weekly_report(missing)
        assert result is False


class TestMultiFile:
    def test_concatenates_files_in_input_order_with_separator(self, tmp_path):
        first = tmp_path / "2026-05-30-weekly-sector-summary.md"
        second = tmp_path / "2026-05-30-weekly-sector-ai.md"
        third = tmp_path / "2026-05-30-weekly-valuechain-report.md"
        first.write_text("# 섹터 요약\n\n핵심 메시지", encoding="utf-8")
        second.write_text("# AI 섹터\n\nNVDA 기반", encoding="utf-8")
        third.write_text("# 밸류체인\n\n종합 분석", encoding="utf-8")

        fake_notifier = MagicMock()
        fake_notifier.is_configured.return_value = True
        fake_notifier.send_html_report.return_value = True

        with patch(
            "scripts.email_weekly_report.EmailNotifier",
            return_value=fake_notifier,
        ):
            result = send_weekly_report([first, second, third])

        assert result is True
        args = fake_notifier.send_html_report.call_args.args
        kwargs = fake_notifier.send_html_report.call_args.kwargs
        html_body = kwargs.get("html") or args[0]
        # 세 본문이 모두 결합됐고 입력 순서대로 등장한다
        assert html_body.index("섹터 요약") < html_body.index("AI 섹터")
        assert html_body.index("AI 섹터") < html_body.index("밸류체인")
        # `---` 마크다운 구분선은 HTML <hr> 로 렌더링된다
        assert "<hr" in html_body

    def test_subject_uses_first_existing_file_date(self, tmp_path):
        first = tmp_path / "2026-05-30-weekly-sector-summary.md"
        second = tmp_path / "2026-05-30-weekly-sector-ai.md"
        first.write_text("# A", encoding="utf-8")
        second.write_text("# B", encoding="utf-8")

        fake_notifier = MagicMock()
        fake_notifier.is_configured.return_value = True
        fake_notifier.send_html_report.return_value = True

        with patch(
            "scripts.email_weekly_report.EmailNotifier",
            return_value=fake_notifier,
        ):
            send_weekly_report([first, second])

        args = fake_notifier.send_html_report.call_args.args
        kwargs = fake_notifier.send_html_report.call_args.kwargs
        subject = kwargs.get("subject") or args[1]
        assert subject == "[주간 리포트] 2026-05-30 밸류체인 분석"

    def test_skips_missing_files_but_sends_remaining(self, tmp_path):
        present = tmp_path / "2026-05-30-weekly-sector-ai.md"
        present.write_text("# 있는 파일", encoding="utf-8")
        missing = tmp_path / "2026-05-30-weekly-sector-missing.md"

        fake_notifier = MagicMock()
        fake_notifier.is_configured.return_value = True
        fake_notifier.send_html_report.return_value = True

        with patch(
            "scripts.email_weekly_report.EmailNotifier",
            return_value=fake_notifier,
        ):
            result = send_weekly_report([present, missing])

        assert result is True
        args = fake_notifier.send_html_report.call_args.args
        kwargs = fake_notifier.send_html_report.call_args.kwargs
        html_body = kwargs.get("html") or args[0]
        assert "있는 파일" in html_body

    def test_returns_false_when_all_files_missing(self, tmp_path):
        result = send_weekly_report(
            [tmp_path / "a.md", tmp_path / "b.md"]
        )
        assert result is False
