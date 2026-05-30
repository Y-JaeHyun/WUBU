"""주간 리포트 마크다운 파일을 HTML로 변환해 이메일로 발송한다.

사용:
    python scripts/email_weekly_report.py <markdown_file_path>

요건:
- SMTP_HOST, SMTP_SENDER, SMTP_PASSWORD, SMTP_RECIPIENTS 환경변수 설정 (.env)
- requirements.txt 의 `markdown` 패키지

미설정 시 graceful degradation: 에러 없이 경고 로그만 남기고 종료한다.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import markdown as md_lib  # type: ignore[import-not-found]

# scripts/ 하위에서 src 임포트를 위해 프로젝트 루트를 path 에 추가
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.alert.email_sender import EmailNotifier  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_KST = ZoneInfo("Asia/Seoul")

_HTML_STYLE = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         font-size: 15px; line-height: 1.7; color: #2c3e50;
         max-width: 820px; margin: 0 auto; padding: 24px; }
  h1 { color: #1a365d; border-bottom: 3px solid #2b6cb0; padding-bottom: 8px; }
  h2 { color: #2b6cb0; border-bottom: 1px solid #cbd5e0; padding-bottom: 4px;
       margin-top: 28px; }
  h3 { color: #2c5282; margin-top: 20px; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  th, td { border: 1px solid #cbd5e0; padding: 8px 12px; text-align: left; }
  th { background-color: #ebf4ff; font-weight: 600; }
  code { background-color: #edf2f7; padding: 2px 6px; border-radius: 3px;
         font-family: 'SF Mono', Monaco, monospace; font-size: 0.9em; }
  pre { background-color: #2d3748; color: #e2e8f0; padding: 12px;
        border-radius: 6px; overflow-x: auto; }
  pre code { background-color: transparent; padding: 0; color: inherit; }
  blockquote { border-left: 4px solid #2b6cb0; margin: 12px 0;
               padding: 8px 16px; background-color: #ebf8ff; color: #2c5282; }
  ul, ol { padding-left: 28px; }
  li { margin: 4px 0; }
  strong { color: #1a365d; }
  hr { border: none; border-top: 1px solid #cbd5e0; margin: 24px 0; }
</style>
"""


def extract_report_date(path: Path) -> Optional[str]:
    """파일명에서 YYYY-MM-DD 날짜를 추출한다.

    Args:
        path: 마크다운 파일 경로.

    Returns:
        날짜 문자열 (`YYYY-MM-DD`) 또는 매칭 실패 시 None.
    """
    match = _DATE_RE.search(path.name)
    return match.group(1) if match else None


def render_html(markdown_text: str) -> str:
    """마크다운 텍스트를 메일용 HTML 문자열로 변환한다.

    Args:
        markdown_text: 마크다운 원문.

    Returns:
        `<!DOCTYPE html>` 로 시작하는 완전한 HTML 문서.
    """
    body = md_lib.markdown(
        markdown_text,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
    )
    return (
        "<!DOCTYPE html>"
        "<html><head><meta charset='utf-8'>"
        f"{_HTML_STYLE}"
        "</head><body>"
        f"{body}"
        "</body></html>"
    )


def _build_subject(report_date: Optional[str]) -> str:
    """`[주간 리포트] YYYY-MM-DD 밸류체인 분석` 형식 제목을 만든다."""
    date_str = report_date or datetime.now(_KST).strftime("%Y-%m-%d")
    return f"[주간 리포트] {date_str} 밸류체인 분석"


def send_weekly_report(report_path: Path) -> bool:
    """주간 리포트 마크다운 파일을 읽어 이메일로 발송한다.

    Args:
        report_path: 마크다운 리포트 파일 경로.

    Returns:
        발송 성공 시 True, 미설정/실패/파일 부재 시 False.
        예외는 던지지 않는다 (graceful degradation).
    """
    path = Path(report_path)
    if not path.is_file():
        logger.error("리포트 파일을 찾을 수 없습니다: %s", path)
        return False

    markdown_text = path.read_text(encoding="utf-8")
    html_body = render_html(markdown_text)
    subject = _build_subject(extract_report_date(path))

    notifier = EmailNotifier()
    if not notifier.is_configured():
        logger.warning(
            "SMTP 미설정 — 이메일 발송을 건너뜁니다. "
            ".env 에 SMTP_HOST/SMTP_SENDER/SMTP_PASSWORD/SMTP_RECIPIENTS 를 설정하세요."
        )
        return False

    return notifier.send_html_report(html=html_body, subject=subject)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="주간 리포트 마크다운 파일을 이메일로 발송한다."
    )
    parser.add_argument("report_path", help="마크다운 리포트 파일 경로")
    args = parser.parse_args(argv)

    success = send_weekly_report(Path(args.report_path))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
