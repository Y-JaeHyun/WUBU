"""주간 리포트 마크다운을 별도 GitHub 리서치 레포에 자동 push 한다.

사용:
    python scripts/push_weekly_report_to_github.py <markdown_file_path>

요건 (.env):
- GITHUB_TOKEN          : Personal Access Token (repo write 권한)
- GITHUB_RESEARCH_REPO  : `owner/name` 형식 (예: wogus3314/quant-research)
선택:
- GITHUB_RESEARCH_BRANCH: 기본 main
- GITHUB_RESEARCH_CACHE : 로컬 클론 경로 (기본 ~/.cache/quant-research-repo)
- GIT_AUTHOR_NAME / GIT_AUTHOR_EMAIL: 커밋 author (없으면 git 전역 설정 사용)

미설정·실패 시 graceful degradation: 예외를 던지지 않고 경고 로그만 남긴 뒤 False 반환.

레포 구조:
    reports/YYYY/YYYY-MM-DD-weekly-report.md
    README.md  (날짜 역순 인덱스 자동 갱신)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
_KST = ZoneInfo("Asia/Seoul")
_DEFAULT_CACHE = Path.home() / ".cache" / "quant-research-repo"
_DEFAULT_BRANCH = "main"
_INDEX_BEGIN = "<!-- WEEKLY-INDEX:BEGIN -->"
_INDEX_END = "<!-- WEEKLY-INDEX:END -->"


class GithubPushError(RuntimeError):
    """내부 흐름 제어용 — 외부에는 노출되지 않는다."""


def _extract_report_date(path: Path) -> str:
    match = _DATE_RE.search(path.name)
    if match:
        return match.group(1)
    return datetime.now(_KST).strftime("%Y-%m-%d")


def _run_git(
    args: Sequence[str],
    cwd: Path,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """git 서브프로세스 실행. 토큰이 args 에 들어가지 않도록 호출자가 책임진다."""
    cmd = ["git", *args]
    logger.debug("git %s (cwd=%s)", " ".join(args), cwd)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        # stderr 만 노출. URL 에 토큰이 박혀있을 수 있어 args 는 로깅하지 않는다.
        raise GithubPushError(
            f"git {args[0]} failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result


def _build_authenticated_url(repo: str, token: str) -> str:
    """`owner/name` + token → HTTPS clone URL.

    Token 은 URL 에 박혀 들어가므로 stdout/stderr 출력 시 함께 노출되지 않도록 주의.
    """
    return f"https://x-access-token:{token}@github.com/{repo}.git"


def _sync_repo(cache_dir: Path, repo: str, token: str, branch: str) -> None:
    """캐시에 클론이 없으면 클론, 있으면 fetch 후 hard reset."""
    auth_url = _build_authenticated_url(repo, token)
    if not (cache_dir / ".git").is_dir():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        _run_git(
            ["clone", "--depth", "50", "--branch", branch, auth_url, str(cache_dir)],
            cwd=cache_dir.parent,
        )
        return

    _run_git(["remote", "set-url", "origin", auth_url], cwd=cache_dir)
    _run_git(["fetch", "--depth", "50", "origin", branch], cwd=cache_dir)
    _run_git(["checkout", branch], cwd=cache_dir, check=False)
    _run_git(["reset", "--hard", f"origin/{branch}"], cwd=cache_dir)
    _run_git(["clean", "-fd"], cwd=cache_dir)


def _copy_report(report_path: Path, repo_root: Path, report_date: str) -> Path:
    year = report_date.split("-")[0]
    dest_dir = repo_root / "reports" / year
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{report_date}-weekly-report.md"
    shutil.copyfile(report_path, dest)
    return dest


def _collect_existing_reports(repo_root: Path) -> list[tuple[str, Path]]:
    """`reports/YYYY/YYYY-MM-DD-weekly-report.md` 패턴을 모두 수집한다."""
    reports: list[tuple[str, Path]] = []
    base = repo_root / "reports"
    if not base.is_dir():
        return reports
    for path in base.rglob("*-weekly-report.md"):
        m = _DATE_RE.search(path.name)
        if not m:
            continue
        reports.append((m.group(1), path.relative_to(repo_root)))
    reports.sort(key=lambda item: item[0], reverse=True)
    return reports


def _render_index_section(reports: Sequence[tuple[str, Path]]) -> str:
    if not reports:
        return f"{_INDEX_BEGIN}\n_아직 업로드된 리포트가 없습니다._\n{_INDEX_END}"
    lines = [_INDEX_BEGIN, ""]
    current_year: Optional[str] = None
    for date_str, rel_path in reports:
        year = date_str.split("-")[0]
        if year != current_year:
            if current_year is not None:
                lines.append("")
            lines.append(f"### {year}")
            current_year = year
        link = rel_path.as_posix()
        lines.append(f"- [{date_str}]({link})")
    lines.append("")
    lines.append(_INDEX_END)
    return "\n".join(lines)


def _update_readme(repo_root: Path, reports: Sequence[tuple[str, Path]]) -> Path:
    readme = repo_root / "README.md"
    new_section = _render_index_section(reports)

    if readme.is_file():
        body = readme.read_text(encoding="utf-8")
        if _INDEX_BEGIN in body and _INDEX_END in body:
            pre, _, rest = body.partition(_INDEX_BEGIN)
            _, _, post = rest.partition(_INDEX_END)
            new_body = f"{pre}{new_section}{post}"
        else:
            new_body = body.rstrip() + "\n\n## 주간 리포트 인덱스\n\n" + new_section + "\n"
    else:
        new_body = (
            "# Quant Research Archive\n\n"
            "WUBU Quant 시스템이 자동 push 하는 주간 리서치 리포트 아카이브.\n\n"
            "## 주간 리포트 인덱스\n\n"
            f"{new_section}\n"
        )

    readme.write_text(new_body, encoding="utf-8")
    return readme


def _has_changes(repo_root: Path) -> bool:
    result = _run_git(["status", "--porcelain"], cwd=repo_root)
    return bool(result.stdout.strip())


def _commit_and_push(
    repo_root: Path, branch: str, report_date: str
) -> bool:
    if not _has_changes(repo_root):
        logger.info("변경 사항 없음 — push 생략 (date=%s)", report_date)
        return True

    env = os.environ.copy()
    author_name = env.get("GIT_AUTHOR_NAME") or env.get("GIT_COMMITTER_NAME")
    author_email = env.get("GIT_AUTHOR_EMAIL") or env.get("GIT_COMMITTER_EMAIL")
    if author_name:
        env.setdefault("GIT_AUTHOR_NAME", author_name)
        env.setdefault("GIT_COMMITTER_NAME", author_name)
    if author_email:
        env.setdefault("GIT_AUTHOR_EMAIL", author_email)
        env.setdefault("GIT_COMMITTER_EMAIL", author_email)

    _run_git(["add", "-A"], cwd=repo_root, env=env)
    _run_git(
        ["commit", "-m", f"chore(weekly): {report_date} 주간 리포트 업로드"],
        cwd=repo_root,
        env=env,
    )
    _run_git(["push", "origin", branch], cwd=repo_root, env=env)
    return True


def push_weekly_report(report_path: Path) -> bool:
    """주간 리포트 마크다운을 GitHub 리서치 레포에 업로드한다.

    Args:
        report_path: 업로드할 마크다운 파일 경로.

    Returns:
        성공 시 True, 미설정/파일 부재/실패 시 False.
        예외를 던지지 않는다.
    """
    path = Path(report_path)
    if not path.is_file():
        logger.error("리포트 파일을 찾을 수 없습니다: %s", path)
        return False

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_RESEARCH_REPO", "").strip()
    if not token or not repo:
        logger.warning(
            "GITHUB_TOKEN/GITHUB_RESEARCH_REPO 미설정 — GitHub 업로드를 건너뜁니다. "
            ".env 에 GITHUB_TOKEN, GITHUB_RESEARCH_REPO(owner/name) 를 설정하세요."
        )
        return False
    if not _REPO_RE.match(repo):
        logger.error("GITHUB_RESEARCH_REPO 형식이 잘못되었습니다 (expect owner/name): %s", repo)
        return False

    branch = os.environ.get("GITHUB_RESEARCH_BRANCH", _DEFAULT_BRANCH).strip() or _DEFAULT_BRANCH
    cache_env = os.environ.get("GITHUB_RESEARCH_CACHE", "").strip()
    cache_dir = Path(cache_env).expanduser() if cache_env else _DEFAULT_CACHE

    report_date = _extract_report_date(path)

    try:
        _sync_repo(cache_dir, repo, token, branch)
        _copy_report(path, cache_dir, report_date)
        reports = _collect_existing_reports(cache_dir)
        _update_readme(cache_dir, reports)
        _commit_and_push(cache_dir, branch, report_date)
    except GithubPushError as exc:
        logger.error("GitHub push 실패: %s", exc)
        return False
    except OSError as exc:
        logger.error("파일/네트워크 오류로 GitHub push 실패: %s", exc)
        return False

    logger.info(
        "주간 리포트 GitHub 업로드 완료: repo=%s branch=%s date=%s",
        repo,
        branch,
        report_date,
    )
    return True


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="주간 리포트 마크다운을 GitHub 리서치 레포에 push 한다."
    )
    parser.add_argument("report_path", help="마크다운 리포트 파일 경로")
    args = parser.parse_args(argv)

    success = push_weekly_report(Path(args.report_path))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
