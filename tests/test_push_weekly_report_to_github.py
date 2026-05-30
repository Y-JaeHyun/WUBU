"""scripts/push_weekly_report_to_github.py 헬퍼 테스트.

GitHub 토큰/레포 환경변수 검증, README 인덱스 렌더링, graceful degradation,
git 커맨드 시퀀스(클론/리셋/커밋/푸시) 및 다중 파일 업로드를 mock 으로 검증한다.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import pytest

from scripts.push_weekly_report_to_github import (
    _collect_existing_reports,
    _extract_report_date,
    _render_index_section,
    _update_readme,
    push_weekly_report,
)


SAMPLE_MD = """# 주간 밸류체인 분석

## 1. 시장 개요
- KOSPI: 2,500
"""


@pytest.fixture
def report_md(tmp_path: Path) -> Path:
    path = tmp_path / "2026-05-30-weekly-valuechain-report.md"
    path.write_text(SAMPLE_MD, encoding="utf-8")
    return path


@pytest.fixture
def sector_reports(tmp_path: Path) -> list[Path]:
    """JAE-83 표준 섹터 파일 셋트 — summary 먼저, valuechain 마지막."""
    names = [
        "2026-05-30-weekly-sector-summary.md",
        "2026-05-30-weekly-sector-ai.md",
        "2026-05-30-weekly-sector-physical-ai.md",
        "2026-05-30-weekly-sector-energy.md",
        "2026-05-30-weekly-sector-game.md",
        "2026-05-30-weekly-sector-other.md",
        "2026-05-30-weekly-valuechain-report.md",
    ]
    paths: list[Path] = []
    for name in names:
        p = tmp_path / name
        p.write_text(f"# {name}", encoding="utf-8")
        paths.append(p)
    return paths


class TestExtractReportDate:
    def test_uses_filename_date(self, tmp_path: Path) -> None:
        path = tmp_path / "2026-05-30-weekly-report.md"
        path.touch()
        assert _extract_report_date(path) == "2026-05-30"

    def test_falls_back_to_today_kst(self, tmp_path: Path) -> None:
        path = tmp_path / "no-date.md"
        path.touch()
        result = _extract_report_date(path)
        # YYYY-MM-DD 형식만 검증 — 실제 KST 일자는 실행 시점 의존
        assert len(result) == 10 and result.count("-") == 2


class TestReadmeRendering:
    def test_render_index_empty(self) -> None:
        out = _render_index_section([])
        assert "WEEKLY-INDEX:BEGIN" in out
        assert "WEEKLY-INDEX:END" in out
        assert "아직 업로드된 리포트가 없습니다" in out

    def test_render_index_groups_by_year_desc(self) -> None:
        reports = [
            ("2026-05-30", Path("reports/2026/2026-05-30-weekly-report.md")),
            ("2026-05-23", Path("reports/2026/2026-05-23-weekly-report.md")),
            ("2025-12-29", Path("reports/2025/2025-12-29-weekly-report.md")),
        ]
        out = _render_index_section(reports)
        assert "### 2026" in out
        assert "### 2025" in out
        assert out.index("### 2026") < out.index("### 2025"), "최신 연도가 먼저 와야 한다"
        assert out.index("2026-05-30") < out.index("2026-05-23")
        assert "reports/2026/2026-05-30-weekly-report.md" in out

    def test_render_index_groups_multiple_files_per_date(self) -> None:
        reports = [
            ("2026-05-30", Path("reports/2026/2026-05-30-weekly-sector-ai.md")),
            ("2026-05-30", Path("reports/2026/2026-05-30-weekly-sector-energy.md")),
            ("2026-05-30", Path("reports/2026/2026-05-30-weekly-valuechain-report.md")),
            ("2026-05-23", Path("reports/2026/2026-05-23-weekly-valuechain-report.md")),
        ]
        out = _render_index_section(reports)
        # 같은 날짜의 여러 파일은 sublist 로 표시
        assert "- 2026-05-30" in out
        assert "[sector-ai](reports/2026/2026-05-30-weekly-sector-ai.md)" in out
        assert "[sector-energy](reports/2026/2026-05-30-weekly-sector-energy.md)" in out
        assert "[종합](reports/2026/2026-05-30-weekly-valuechain-report.md)" in out
        # 종합이 같은 날 sector 들 뒤에 위치
        assert out.index("[sector-ai]") < out.index("[종합]")
        # 단일 파일 날짜는 기존처럼 한 라인
        assert "- [2026-05-23](reports/2026/2026-05-23-weekly-valuechain-report.md)" in out

    def test_update_readme_creates_new_when_missing(self, tmp_path: Path) -> None:
        reports = [("2026-05-30", Path("reports/2026/2026-05-30-weekly-report.md"))]
        _update_readme(tmp_path, reports)
        body = (tmp_path / "README.md").read_text(encoding="utf-8")
        assert "Quant Research Archive" in body
        assert "2026-05-30" in body
        assert "WEEKLY-INDEX:BEGIN" in body

    def test_update_readme_replaces_index_block(self, tmp_path: Path) -> None:
        existing = (
            "# Custom Header\n\nintro paragraph.\n\n"
            "## 주간 리포트 인덱스\n\n"
            "<!-- WEEKLY-INDEX:BEGIN -->\n"
            "_old_\n"
            "<!-- WEEKLY-INDEX:END -->\n\n"
            "footer text.\n"
        )
        readme = tmp_path / "README.md"
        readme.write_text(existing, encoding="utf-8")

        reports = [("2026-05-30", Path("reports/2026/2026-05-30-weekly-report.md"))]
        _update_readme(tmp_path, reports)

        body = readme.read_text(encoding="utf-8")
        assert "# Custom Header" in body
        assert "footer text." in body
        assert "_old_" not in body
        assert "2026-05-30" in body


class TestCollectExistingReports:
    def test_returns_empty_when_no_reports_dir(self, tmp_path: Path) -> None:
        assert _collect_existing_reports(tmp_path) == []

    def test_collects_sorted_descending(self, tmp_path: Path) -> None:
        (tmp_path / "reports" / "2026").mkdir(parents=True)
        (tmp_path / "reports" / "2025").mkdir(parents=True)
        old = tmp_path / "reports" / "2025" / "2025-12-29-weekly-report.md"
        new = tmp_path / "reports" / "2026" / "2026-05-30-weekly-report.md"
        old.write_text("a", encoding="utf-8")
        new.write_text("b", encoding="utf-8")

        out = _collect_existing_reports(tmp_path)
        assert [d for d, _ in out] == ["2026-05-30", "2025-12-29"]

    def test_collects_sector_files_with_weekly_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "reports" / "2026").mkdir(parents=True)
        for name in [
            "2026-05-30-weekly-sector-ai.md",
            "2026-05-30-weekly-sector-energy.md",
            "2026-05-30-weekly-valuechain-report.md",
        ]:
            (tmp_path / "reports" / "2026" / name).write_text("x", encoding="utf-8")

        out = _collect_existing_reports(tmp_path)
        names = [path.name for _, path in out]
        assert "2026-05-30-weekly-sector-ai.md" in names
        assert "2026-05-30-weekly-sector-energy.md" in names
        assert "2026-05-30-weekly-valuechain-report.md" in names


class TestGracefulDegradation:
    def test_returns_false_when_token_missing(
        self, report_md: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_RESEARCH_REPO", "owner/repo")
        assert push_weekly_report(report_md) is False

    def test_returns_false_when_repo_missing(
        self, report_md: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
        monkeypatch.delenv("GITHUB_RESEARCH_REPO", raising=False)
        assert push_weekly_report(report_md) is False

    def test_returns_false_when_repo_format_invalid(
        self, report_md: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
        monkeypatch.setenv("GITHUB_RESEARCH_REPO", "invalid-repo-name")
        assert push_weekly_report(report_md) is False

    def test_returns_false_when_file_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
        monkeypatch.setenv("GITHUB_RESEARCH_REPO", "owner/repo")
        assert push_weekly_report(tmp_path / "missing.md") is False

    def test_returns_false_when_any_file_in_list_missing(
        self, report_md: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
        monkeypatch.setenv("GITHUB_RESEARCH_REPO", "owner/repo")
        assert (
            push_weekly_report([report_md, tmp_path / "missing.md"]) is False
        )


def _make_fake_git_runner(cache_dir: Path):
    """clone 호출 시 실제 디렉토리를 만들고 .git 디렉토리를 흉내낸다."""
    history: list[list[str]] = []

    def fake_run(cmd, cwd=None, env=None, capture_output=True, text=True, check=False):
        history.append(list(cmd))
        assert cmd[0] == "git"
        sub = cmd[1] if len(cmd) > 1 else ""
        if sub == "clone":
            target = Path(cmd[-1])
            target.mkdir(parents=True, exist_ok=True)
            (target / ".git").mkdir(exist_ok=True)
            stdout = ""
        elif sub == "status":
            stdout = " M README.md\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    return history, fake_run


class TestFullFlow:
    def test_uploads_creates_report_and_pushes(
        self,
        report_md: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache" / "quant-research-repo"
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.setenv("GITHUB_RESEARCH_REPO", "wogus3314/quant-research")
        monkeypatch.setenv("GITHUB_RESEARCH_CACHE", str(cache_dir))

        history, fake_run = _make_fake_git_runner(cache_dir)
        with patch("scripts.push_weekly_report_to_github.subprocess.run", side_effect=fake_run):
            result = push_weekly_report(report_md)

        assert result is True
        # 원본 파일명을 그대로 유지
        copied = cache_dir / "reports" / "2026" / "2026-05-30-weekly-valuechain-report.md"
        assert copied.is_file()
        assert copied.read_text(encoding="utf-8") == SAMPLE_MD

        readme = cache_dir / "README.md"
        assert readme.is_file()
        body = readme.read_text(encoding="utf-8")
        assert "2026-05-30" in body

        # 시퀀스 확인: clone → add → commit → push
        subs = [cmd[1] for cmd in history]
        assert "clone" in subs
        assert subs.index("clone") < subs.index("add")
        assert subs.index("add") < subs.index("commit")
        assert subs.index("commit") < subs.index("push")

        # 토큰은 clone URL 에만 박혀있어야 한다 (다른 명령에는 없음)
        for cmd in history:
            if "clone" in cmd:
                continue
            joined = " ".join(cmd)
            assert "ghp_secret" not in joined

    def test_skips_commit_when_no_changes(
        self,
        report_md: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache" / "quant-research-repo"
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.setenv("GITHUB_RESEARCH_REPO", "wogus3314/quant-research")
        monkeypatch.setenv("GITHUB_RESEARCH_CACHE", str(cache_dir))

        history: list[list[str]] = []

        def fake_run(cmd, cwd=None, env=None, capture_output=True, text=True, check=False):
            history.append(list(cmd))
            sub = cmd[1] if len(cmd) > 1 else ""
            if sub == "clone":
                target = Path(cmd[-1])
                target.mkdir(parents=True, exist_ok=True)
                (target / ".git").mkdir(exist_ok=True)
            stdout = "" if sub != "status" else ""  # 변경 없음
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

        with patch("scripts.push_weekly_report_to_github.subprocess.run", side_effect=fake_run):
            result = push_weekly_report(report_md)

        assert result is True
        subs = [cmd[1] for cmd in history]
        assert "commit" not in subs
        assert "push" not in subs

    def test_existing_clone_resets_to_origin(
        self,
        report_md: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache" / "quant-research-repo"
        cache_dir.mkdir(parents=True)
        (cache_dir / ".git").mkdir()

        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.setenv("GITHUB_RESEARCH_REPO", "wogus3314/quant-research")
        monkeypatch.setenv("GITHUB_RESEARCH_CACHE", str(cache_dir))

        history, fake_run = _make_fake_git_runner(cache_dir)
        with patch("scripts.push_weekly_report_to_github.subprocess.run", side_effect=fake_run):
            result = push_weekly_report(report_md)

        assert result is True
        subs = [cmd[1] for cmd in history]
        assert "clone" not in subs
        assert "fetch" in subs
        assert "reset" in subs

    def test_returns_false_when_git_command_fails(
        self,
        report_md: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache" / "quant-research-repo"
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.setenv("GITHUB_RESEARCH_REPO", "wogus3314/quant-research")
        monkeypatch.setenv("GITHUB_RESEARCH_CACHE", str(cache_dir))

        def fake_run(cmd, cwd=None, env=None, capture_output=True, text=True, check=False):
            return subprocess.CompletedProcess(
                args=cmd, returncode=128, stdout="", stderr="fatal: remote not found"
            )

        with patch("scripts.push_weekly_report_to_github.subprocess.run", side_effect=fake_run):
            result = push_weekly_report(report_md)

        assert result is False


class TestMultiFileUpload:
    def test_copies_all_files_with_original_names(
        self,
        sector_reports: list[Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache" / "quant-research-repo"
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.setenv("GITHUB_RESEARCH_REPO", "wogus3314/quant-research")
        monkeypatch.setenv("GITHUB_RESEARCH_CACHE", str(cache_dir))

        history, fake_run = _make_fake_git_runner(cache_dir)
        with patch("scripts.push_weekly_report_to_github.subprocess.run", side_effect=fake_run):
            result = push_weekly_report(sector_reports)

        assert result is True
        for src in sector_reports:
            copied = cache_dir / "reports" / "2026" / src.name
            assert copied.is_file(), f"missing: {copied}"
            assert copied.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")

        # README 에 sublist 인덱스가 들어갔는지
        readme = (cache_dir / "README.md").read_text(encoding="utf-8")
        assert "- 2026-05-30" in readme
        assert "[종합]" in readme
        assert "[sector-ai]" in readme

    def test_commit_message_includes_file_count_when_multi(
        self,
        sector_reports: list[Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache" / "quant-research-repo"
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.setenv("GITHUB_RESEARCH_REPO", "wogus3314/quant-research")
        monkeypatch.setenv("GITHUB_RESEARCH_CACHE", str(cache_dir))

        history, fake_run = _make_fake_git_runner(cache_dir)
        with patch("scripts.push_weekly_report_to_github.subprocess.run", side_effect=fake_run):
            push_weekly_report(sector_reports)

        commits = [cmd for cmd in history if cmd[1] == "commit"]
        assert commits, "commit 호출이 있어야 한다"
        message = commits[0][commits[0].index("-m") + 1]
        assert "2026-05-30" in message
        assert f"{len(sector_reports)}개 파일" in message

    def test_single_file_commit_message_unchanged(
        self,
        report_md: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache" / "quant-research-repo"
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.setenv("GITHUB_RESEARCH_REPO", "wogus3314/quant-research")
        monkeypatch.setenv("GITHUB_RESEARCH_CACHE", str(cache_dir))

        history, fake_run = _make_fake_git_runner(cache_dir)
        with patch("scripts.push_weekly_report_to_github.subprocess.run", side_effect=fake_run):
            push_weekly_report(report_md)

        commits = [cmd for cmd in history if cmd[1] == "commit"]
        message = commits[0][commits[0].index("-m") + 1]
        # 단일 파일은 (n개 파일) suffix 없이 기존 메시지 유지
        assert "개 파일" not in message
        assert message == "chore(weekly): 2026-05-30 주간 리포트 업로드"
