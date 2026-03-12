"""배치 백테스트 러너(scripts/batch_backtest.py) 테스트.

전략 레지스트리, 체크포인트 관리, 태스크 생성 등 인프라를 검증한다.
실제 백테스트 실행은 mock 처리한다.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# batch_backtest는 scripts/ 아래에 있으므로 직접 import
import importlib
import sys
sys.path.insert(0, "/mnt/data/quant-dev/scripts")


@pytest.fixture
def batch_module():
    """batch_backtest 모듈을 import한다."""
    # 환경 변수 설정을 건너뛰기 위해 이미 로드된 모듈 사용
    spec = importlib.util.spec_from_file_location(
        "batch_backtest",
        "/mnt/data/quant-dev/scripts/batch_backtest.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # dotenv load_dotenv를 패치하여 실행 시 부작용 방지
    with patch("dotenv.load_dotenv"):
        spec.loader.exec_module(mod)
    return mod


class TestStrategyRegistry:
    """전략 레지스트리 테스트."""

    def test_all_groups_present(self, batch_module):
        """A~D 모든 그룹이 레지스트리에 존재한다."""
        groups = {v["group"] for v in batch_module.STRATEGY_REGISTRY.values()}
        assert groups == {"A", "B", "C", "D"}

    def test_each_strategy_has_configs(self, batch_module):
        """모든 전략이 최소 1개의 설정을 가진다."""
        for name, reg in batch_module.STRATEGY_REGISTRY.items():
            assert len(reg["configs"]) >= 1, f"{name}에 configs 없음"

    def test_multi_factor_has_live_config(self, batch_module):
        """multi_factor 전략에 is_live=True인 설정이 있다."""
        mf = batch_module.STRATEGY_REGISTRY["multi_factor"]
        live_configs = [c for c in mf["configs"] if c.get("is_live")]
        assert len(live_configs) == 1

    def test_registry_has_all_expected_strategies(self, batch_module):
        """예상되는 주요 전략이 모두 등록되어 있다."""
        expected = {
            "value", "momentum", "quality", "three_factor", "multi_factor",
            "etf_rotation", "risk_parity", "dual_momentum",
            "low_volatility", "shareholder_yield",
            "bb_squeeze", "high_breakout",
        }
        actual = set(batch_module.STRATEGY_REGISTRY.keys())
        assert expected.issubset(actual)


class TestCheckpoint:
    """체크포인트 관리 테스트."""

    def test_save_and_load_progress(self, batch_module, tmp_path):
        """체크포인트 저장 및 로드가 정상 동작한다."""
        # 임시 경로로 교체
        orig_dir = batch_module.RESULTS_DIR
        orig_file = batch_module.PROGRESS_FILE
        batch_module.RESULTS_DIR = tmp_path
        batch_module.PROGRESS_FILE = tmp_path / "progress.json"
        try:
            progress = {
                "done": ["multi_factor_live"],
                "pending": ["value_default"],
                "failed": [],
            }
            batch_module.save_progress(progress)

            loaded = batch_module.load_progress()
            assert loaded["done"] == ["multi_factor_live"]
            assert loaded["pending"] == ["value_default"]
            assert "updated_at" in loaded
        finally:
            batch_module.RESULTS_DIR = orig_dir
            batch_module.PROGRESS_FILE = orig_file

    def test_load_missing_file_returns_empty(self, batch_module, tmp_path):
        """체크포인트 파일이 없으면 빈 상태를 반환한다."""
        orig_file = batch_module.PROGRESS_FILE
        batch_module.PROGRESS_FILE = tmp_path / "nonexistent.json"
        try:
            progress = batch_module.load_progress()
            assert progress["done"] == []
            assert progress["pending"] == []
        finally:
            batch_module.PROGRESS_FILE = orig_file


class TestTaskGeneration:
    """태스크 생성 테스트."""

    def test_get_all_tasks(self, batch_module):
        """전체 태스크 목록이 레지스트리 기반으로 생성된다."""
        tasks = batch_module.get_tasks()
        total_configs = sum(
            len(reg["configs"])
            for reg in batch_module.STRATEGY_REGISTRY.values()
        )
        assert len(tasks) == total_configs

    def test_get_tasks_by_group(self, batch_module):
        """그룹 필터링이 정상 동작한다."""
        tasks_a = batch_module.get_tasks(group="A")
        for name, _ in tasks_a:
            assert batch_module.STRATEGY_REGISTRY[name]["group"] == "A"

    def test_get_tasks_by_strategy(self, batch_module):
        """전략명 필터링이 정상 동작한다."""
        tasks = batch_module.get_tasks(strategy="multi_factor")
        for name, _ in tasks:
            assert name == "multi_factor"

    def test_task_id_generation(self, batch_module):
        """태스크 ID가 올바르게 생성된다."""
        tid = batch_module._task_id("multi_factor", "live")
        assert tid == "multi_factor_live"


class TestConfigHash:
    """설정 해시 테스트."""

    def test_same_input_same_hash(self, batch_module):
        """동일 입력은 동일 해시를 생성한다."""
        h1 = batch_module._config_hash("multi_factor", "live")
        h2 = batch_module._config_hash("multi_factor", "live")
        assert h1 == h2

    def test_different_input_different_hash(self, batch_module):
        """다른 입력은 다른 해시를 생성한다."""
        h1 = batch_module._config_hash("multi_factor", "live")
        h2 = batch_module._config_hash("multi_factor", "backtest")
        assert h1 != h2


class TestSaveResult:
    """결과 저장 테스트."""

    def test_save_creates_json_file(self, batch_module, tmp_path):
        """결과가 JSON 파일로 저장된다."""
        orig_dir = batch_module.RESULTS_DIR
        batch_module.RESULTS_DIR = tmp_path
        try:
            result = {
                "strategy": "value",
                "summary": {"total_return": 10.5},
            }
            filepath = batch_module.save_result("value", "default", result)
            assert filepath.exists()

            saved = json.loads(filepath.read_text(encoding="utf-8"))
            assert saved["strategy"] == "value"
            assert "saved_at" in saved
        finally:
            batch_module.RESULTS_DIR = orig_dir
