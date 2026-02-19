"""PortfolioTracker 테스트."""

import json
import os
import tempfile

import pytest

from src.report.portfolio_tracker import PortfolioTracker


class TestPortfolioTracker:
    """포트폴리오 성과 추적기 검증."""

    def test_init_empty(self, tmp_path):
        """이력 없이 초기화된다."""
        path = str(tmp_path / "test_history.json")
        tracker = PortfolioTracker(history_path=path)
        assert tracker.peak == 0.0
        assert tracker.current_mdd == 0.0
        assert tracker.get_history_count() == 0

    def test_update_sets_peak(self, tmp_path):
        """첫 업데이트에서 고점이 설정된다."""
        path = str(tmp_path / "test_history.json")
        tracker = PortfolioTracker(history_path=path)

        tracker.update(1_000_000)
        assert tracker.peak == 1_000_000
        assert tracker.current_mdd == 0.0

    def test_mdd_calculation(self, tmp_path):
        """고점 후 하락 시 MDD가 올바르게 계산된다."""
        path = str(tmp_path / "test_history.json")
        tracker = PortfolioTracker(history_path=path)

        tracker.update(1_000_000)  # 고점
        tracker.update(1_200_000)  # 새 고점
        mdd = tracker.update(900_000)  # 하락

        assert tracker.peak == 1_200_000
        # (900000 - 1200000) / 1200000 = -0.25
        assert abs(mdd - (-0.25)) < 0.001

    def test_mdd_recovers(self, tmp_path):
        """고점 회복 시 MDD가 0으로 돌아온다."""
        path = str(tmp_path / "test_history.json")
        tracker = PortfolioTracker(history_path=path)

        tracker.update(1_000_000)
        tracker.update(800_000)  # MDD -20%
        mdd = tracker.update(1_100_000)  # 새 고점

        assert tracker.peak == 1_100_000
        assert mdd == 0.0

    def test_persistence(self, tmp_path):
        """재시작 후 이력이 복원된다."""
        path = str(tmp_path / "test_history.json")

        # 첫 번째 인스턴스
        t1 = PortfolioTracker(history_path=path)
        t1.update(1_500_000)
        t1.update(1_200_000)

        # 두 번째 인스턴스 (재시작 시뮬레이션)
        t2 = PortfolioTracker(history_path=path)
        assert t2.peak == 1_500_000
        assert abs(t2.current_mdd - (-0.2)) < 0.001
        assert t2.get_history_count() == 2

    def test_max_records(self, tmp_path):
        """max_records를 초과하면 오래된 레코드가 정리된다."""
        path = str(tmp_path / "test_history.json")
        tracker = PortfolioTracker(history_path=path, max_records=5)

        for i in range(10):
            tracker.update(1_000_000 + i * 10_000)

        assert tracker.get_history_count() == 5

    def test_zero_eval_ignored(self, tmp_path):
        """평가금액 0은 무시된다."""
        path = str(tmp_path / "test_history.json")
        tracker = PortfolioTracker(history_path=path)

        tracker.update(1_000_000)
        mdd = tracker.update(0)

        assert tracker.peak == 1_000_000
        assert tracker.get_history_count() == 1  # 0은 기록 안됨

    def test_corrupted_file(self, tmp_path):
        """손상된 파일에서도 정상 초기화된다."""
        path = tmp_path / "test_history.json"
        path.write_text("NOT VALID JSON", encoding="utf-8")

        tracker = PortfolioTracker(history_path=str(path))
        assert tracker.peak == 0.0
        assert tracker.get_history_count() == 0
