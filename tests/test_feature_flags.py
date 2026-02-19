"""FeatureFlags 시스템 테스트.

JSON 파일 기반 Feature Flag 관리자를 검증한다.
초기화, 토글, 설정, 영속화, 스레드 안전성, 에러 처리를 포함한다.
외부 서비스 의존 없음 - tmp_path 픽스처를 사용하여 격리된 환경에서 테스트.
"""

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.feature_flags import FeatureFlags


# ===================================================================
# 헬퍼
# ===================================================================


def _flags_path(tmp_path: Path) -> str:
    """테스트용 JSON 파일 경로를 반환한다."""
    return str(tmp_path / "test_flags.json")


def _read_json(path: str) -> dict:
    """JSON 파일을 읽어 딕셔너리를 반환한다."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, data: dict) -> None:
    """딕셔너리를 JSON 파일로 저장한다."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ===================================================================
# 1. 기본 초기화 테스트
# ===================================================================


class TestDefaultInitialization:
    """FeatureFlags 초기화 시 기본 동작을 검증한다."""

    def test_creates_json_file_with_defaults(self, tmp_path):
        """JSON 파일이 없으면 기본 플래그로 파일을 생성한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        assert Path(path).exists(), "초기화 시 JSON 파일이 생성되어야 합니다."

        data = _read_json(path)
        assert "version" in data
        assert "features" in data
        assert data["version"] == 1

    def test_default_flags_populated(self, tmp_path):
        """기본 플래그가 모두 포함되어 있다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        data = _read_json(path)
        features = data["features"]

        for key in FeatureFlags.DEFAULT_FLAGS:
            assert key in features, f"기본 플래그 '{key}'가 누락되었습니다."

    def test_default_enabled_states(self, tmp_path):
        """기본 활성화 상태가 DEFAULT_FLAGS와 일치한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        for key, default in FeatureFlags.DEFAULT_FLAGS.items():
            expected = default["enabled"]
            actual = ff.is_enabled(key)
            assert actual == expected, (
                f"'{key}' 기본 상태가 {expected}이어야 하는데 {actual}입니다."
            )

    def test_default_descriptions_preserved(self, tmp_path):
        """기본 플래그 description이 보존된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        data = _read_json(path)
        for key, default in FeatureFlags.DEFAULT_FLAGS.items():
            assert data["features"][key]["description"] == default["description"]

    def test_updated_at_present(self, tmp_path):
        """JSON 파일에 updated_at 필드가 포함된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        data = _read_json(path)
        assert "updated_at" in data, "updated_at 필드가 있어야 합니다."

    def test_loads_existing_file(self, tmp_path):
        """기존 JSON 파일이 있으면 해당 파일을 로드한다."""
        path = _flags_path(tmp_path)
        custom_data = {
            "version": 1,
            "updated_at": "2025-01-01T00:00:00",
            "features": {
                "data_cache": {
                    "enabled": False,
                    "description": "custom desc",
                    "config": {},
                },
            },
        }
        _write_json(path, custom_data)

        ff = FeatureFlags(flags_path=path)

        # 기존 파일의 data_cache는 False로 저장됨
        assert ff.is_enabled("data_cache") is False
        # 누락된 기본 플래그가 자동 보충됨
        assert ff.is_enabled("global_monitor") is not None

    def test_missing_flags_supplemented_from_defaults(self, tmp_path):
        """기존 파일에 누락된 플래그가 DEFAULT_FLAGS에서 보충된다."""
        path = _flags_path(tmp_path)
        # data_cache만 있는 파일 생성
        partial_data = {
            "version": 1,
            "updated_at": "2025-01-01T00:00:00",
            "features": {
                "data_cache": {
                    "enabled": False,
                    "description": "캐시",
                    "config": {},
                },
            },
        }
        _write_json(path, partial_data)

        ff = FeatureFlags(flags_path=path)

        # 모든 DEFAULT_FLAGS 키가 존재해야 함
        status = ff.get_all_status()
        for key in FeatureFlags.DEFAULT_FLAGS:
            assert key in status, f"보충된 플래그 '{key}'가 없습니다."

    def test_nested_subdirectory_creation(self, tmp_path):
        """중첩 디렉토리가 자동 생성된다."""
        path = str(tmp_path / "deep" / "nested" / "dir" / "flags.json")
        ff = FeatureFlags(flags_path=path)

        assert Path(path).exists(), "중첩 디렉토리가 생성되어야 합니다."


# ===================================================================
# 2. is_enabled() 테스트
# ===================================================================


class TestIsEnabled:
    """is_enabled() 메서드를 검증한다."""

    def test_enabled_flag_returns_true(self, tmp_path):
        """활성화된 플래그는 True를 반환한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        # data_cache는 기본 True
        assert ff.is_enabled("data_cache") is True

    def test_disabled_flag_returns_false(self, tmp_path):
        """비활성화된 플래그는 False를 반환한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        # global_monitor는 기본 False
        assert ff.is_enabled("global_monitor") is False

    def test_unknown_feature_returns_false(self, tmp_path):
        """존재하지 않는 피처는 False를 반환한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        assert ff.is_enabled("nonexistent_feature") is False

    def test_empty_string_feature_returns_false(self, tmp_path):
        """빈 문자열 피처명은 False를 반환한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        assert ff.is_enabled("") is False


# ===================================================================
# 3. toggle() 테스트
# ===================================================================


class TestToggle:
    """toggle() 메서드를 검증한다."""

    def test_toggle_on(self, tmp_path):
        """비활성 플래그를 토글하면 활성화된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        assert ff.is_enabled("global_monitor") is False
        result = ff.toggle("global_monitor")

        assert result is True, "toggle()은 성공 시 True를 반환해야 합니다."
        assert ff.is_enabled("global_monitor") is True

    def test_toggle_off(self, tmp_path):
        """활성 플래그를 토글하면 비활성화된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        assert ff.is_enabled("data_cache") is True
        result = ff.toggle("data_cache")

        assert result is True
        assert ff.is_enabled("data_cache") is False

    def test_toggle_explicit_true(self, tmp_path):
        """enabled=True로 명시적 설정한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        ff.toggle("global_monitor", enabled=True)
        assert ff.is_enabled("global_monitor") is True

    def test_toggle_explicit_false(self, tmp_path):
        """enabled=False로 명시적 설정한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        ff.toggle("data_cache", enabled=False)
        assert ff.is_enabled("data_cache") is False

    def test_toggle_explicit_same_state(self, tmp_path):
        """이미 활성인 플래그에 enabled=True를 설정해도 성공한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        assert ff.is_enabled("data_cache") is True
        result = ff.toggle("data_cache", enabled=True)
        assert result is True
        assert ff.is_enabled("data_cache") is True

    def test_toggle_unknown_feature_returns_false(self, tmp_path):
        """존재하지 않는 피처 토글은 False를 반환한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        result = ff.toggle("nonexistent_feature")
        assert result is False

    def test_toggle_unknown_with_explicit_value(self, tmp_path):
        """존재하지 않는 피처에 명시적 값 설정도 False를 반환한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        result = ff.toggle("nonexistent_feature", enabled=True)
        assert result is False

    def test_double_toggle_restores_state(self, tmp_path):
        """두 번 토글하면 원래 상태로 복원된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        original = ff.is_enabled("data_cache")
        ff.toggle("data_cache")
        ff.toggle("data_cache")
        assert ff.is_enabled("data_cache") == original

    def test_toggle_saves_to_disk(self, tmp_path):
        """toggle() 후 JSON 파일에 변경이 반영된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        ff.toggle("global_monitor", enabled=True)

        data = _read_json(path)
        assert data["features"]["global_monitor"]["enabled"] is True


# ===================================================================
# 4. get_config() / set_config() 테스트
# ===================================================================


class TestConfig:
    """get_config()와 set_config()를 검증한다."""

    def test_get_config_returns_copy(self, tmp_path):
        """get_config()는 원본이 아닌 복사본을 반환한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        config = ff.get_config("data_cache")
        config["cache_ttl_hours"] = 9999

        # 원본은 변경되지 않아야 함
        assert ff.get_config("data_cache")["cache_ttl_hours"] == 24

    def test_get_config_default_values(self, tmp_path):
        """기본 config 값이 올바르게 반환된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        config = ff.get_config("data_cache")
        assert config["cache_ttl_hours"] == 24
        assert config["max_cache_size_mb"] == 500

    def test_get_config_unknown_feature(self, tmp_path):
        """존재하지 않는 피처의 config는 빈 딕셔너리를 반환한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        assert ff.get_config("nonexistent") == {}

    def test_get_config_no_config_key(self, tmp_path):
        """config 키가 없는 피처는 빈 딕셔너리를 반환한다."""
        path = _flags_path(tmp_path)
        custom_data = {
            "version": 1,
            "updated_at": "2025-01-01T00:00:00",
            "features": {
                "data_cache": {
                    "enabled": True,
                    "description": "no config key here",
                    # config 키 의도적 누락
                },
            },
        }
        _write_json(path, custom_data)
        ff = FeatureFlags(flags_path=path)

        assert ff.get_config("data_cache") == {}

    def test_set_config_updates_value(self, tmp_path):
        """set_config()로 설정값을 변경한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        result = ff.set_config("data_cache", "cache_ttl_hours", 48)

        assert result is True
        assert ff.get_config("data_cache")["cache_ttl_hours"] == 48

    def test_set_config_adds_new_key(self, tmp_path):
        """set_config()로 새로운 설정 키를 추가한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        # 기존 config 키 목록 기록
        original_config = ff.get_config("data_cache")
        original_keys = set(original_config.keys())

        result = ff.set_config("data_cache", "new_setting", "value123")

        assert result is True
        config = ff.get_config("data_cache")
        assert config["new_setting"] == "value123"
        # 기존 설정 키가 모두 유지됨
        assert original_keys.issubset(set(config.keys())), (
            "기존 설정 키가 유지되어야 합니다."
        )

    def test_set_config_unknown_feature(self, tmp_path):
        """존재하지 않는 피처의 설정 변경은 False를 반환한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        result = ff.set_config("nonexistent", "key", "value")
        assert result is False

    def test_set_config_saves_to_disk(self, tmp_path):
        """set_config() 후 JSON 파일에 반영된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        ff.set_config("data_cache", "cache_ttl_hours", 72)

        data = _read_json(path)
        assert data["features"]["data_cache"]["config"]["cache_ttl_hours"] == 72

    def test_set_config_creates_config_if_missing(self, tmp_path):
        """config 키가 없는 피처에 set_config()하면 config 딕셔너리를 생성한다."""
        path = _flags_path(tmp_path)
        custom_data = {
            "version": 1,
            "updated_at": "2025-01-01T00:00:00",
            "features": {
                "data_cache": {
                    "enabled": True,
                    "description": "no config",
                },
            },
        }
        _write_json(path, custom_data)
        ff = FeatureFlags(flags_path=path)

        result = ff.set_config("data_cache", "new_key", 42)
        assert result is True
        assert ff.get_config("data_cache")["new_key"] == 42

    def test_set_config_various_types(self, tmp_path):
        """다양한 타입의 값을 설정할 수 있다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        ff.set_config("data_cache", "int_val", 42)
        ff.set_config("data_cache", "float_val", 3.14)
        ff.set_config("data_cache", "str_val", "hello")
        ff.set_config("data_cache", "bool_val", True)
        ff.set_config("data_cache", "list_val", [1, 2, 3])

        config = ff.get_config("data_cache")
        assert config["int_val"] == 42
        assert config["float_val"] == 3.14
        assert config["str_val"] == "hello"
        assert config["bool_val"] is True
        assert config["list_val"] == [1, 2, 3]


# ===================================================================
# 5. get_all_status() / get_summary() 테스트
# ===================================================================


class TestStatusAndSummary:
    """get_all_status()와 get_summary()를 검증한다."""

    def test_get_all_status_returns_all_flags(self, tmp_path):
        """get_all_status()는 모든 플래그의 활성 상태를 반환한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        status = ff.get_all_status()

        assert isinstance(status, dict)
        for key in FeatureFlags.DEFAULT_FLAGS:
            assert key in status
            assert isinstance(status[key], bool)

    def test_get_all_status_reflects_toggles(self, tmp_path):
        """toggle 후 get_all_status()에 반영된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        ff.toggle("global_monitor", enabled=True)
        ff.toggle("data_cache", enabled=False)

        status = ff.get_all_status()
        assert status["global_monitor"] is True
        assert status["data_cache"] is False

    def test_get_all_status_values_match_is_enabled(self, tmp_path):
        """get_all_status() 값이 is_enabled()와 일치한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        status = ff.get_all_status()
        for key, val in status.items():
            assert val == ff.is_enabled(key)

    def test_get_summary_returns_string(self, tmp_path):
        """get_summary()는 문자열을 반환한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        summary = ff.get_summary()
        assert isinstance(summary, str)

    def test_get_summary_contains_header(self, tmp_path):
        """get_summary()에 헤더가 포함된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        summary = ff.get_summary()
        assert "[Feature Flags]" in summary

    def test_get_summary_contains_all_features(self, tmp_path):
        """get_summary()에 모든 피처 이름이 포함된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        summary = ff.get_summary()
        for key in FeatureFlags.DEFAULT_FLAGS:
            assert key in summary, f"'{key}'가 summary에 없습니다."

    def test_get_summary_shows_on_off(self, tmp_path):
        """get_summary()에 ON/OFF 상태가 표시된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        summary = ff.get_summary()
        # data_cache는 기본 True → ON
        assert "ON" in summary
        # global_monitor는 기본 False → OFF
        assert "OFF" in summary

    def test_get_summary_reflects_toggle(self, tmp_path):
        """toggle 후 get_summary()에 변경이 반영된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        ff.toggle("global_monitor", enabled=True)
        summary = ff.get_summary()

        # global_monitor 라인에서 ON이 나타나야 함
        lines = summary.split("\n")
        gm_line = [l for l in lines if "global_monitor" in l][0]
        assert "ON" in gm_line

    def test_get_summary_contains_descriptions(self, tmp_path):
        """get_summary()에 피처 설명이 포함된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        summary = ff.get_summary()
        for key, default in FeatureFlags.DEFAULT_FLAGS.items():
            desc = default["description"]
            assert desc in summary, f"설명 '{desc}'가 summary에 없습니다."


# ===================================================================
# 6. reload() 테스트
# ===================================================================


class TestReload:
    """reload() 메서드를 검증한다."""

    def test_reload_picks_up_external_changes(self, tmp_path):
        """외부에서 파일을 변경한 후 reload()하면 반영된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        # data_cache 기본값은 True
        assert ff.is_enabled("data_cache") is True

        # 파일을 외부에서 직접 수정
        data = _read_json(path)
        data["features"]["data_cache"]["enabled"] = False
        _write_json(path, data)

        # reload 전에는 메모리 값 유지
        assert ff.is_enabled("data_cache") is True

        # reload 후 반영
        ff.reload()
        assert ff.is_enabled("data_cache") is False

    def test_reload_adds_new_flags_from_defaults(self, tmp_path):
        """reload 시 파일에 없는 플래그가 기본값으로 보충된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        # 파일에서 일부 플래그 제거
        data = _read_json(path)
        del data["features"]["auto_backtest"]
        _write_json(path, data)

        ff.reload()

        # 기본값으로 보충됨
        assert "auto_backtest" in ff.get_all_status()

    def test_reload_preserves_existing_custom_values(self, tmp_path):
        """reload 시 파일에 있는 기존 값은 유지된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        ff.toggle("global_monitor", enabled=True)
        ff.set_config("data_cache", "cache_ttl_hours", 48)

        ff.reload()

        assert ff.is_enabled("global_monitor") is True
        assert ff.get_config("data_cache")["cache_ttl_hours"] == 48


# ===================================================================
# 7. 영속화 (Persistence) 테스트
# ===================================================================


class TestPersistence:
    """toggle/set_config이 디스크에 저장되고, 재시작 후 올바르게 로드되는지 검증한다."""

    def test_toggle_persists_across_instances(self, tmp_path):
        """toggle 후 새 인스턴스에서 상태가 유지된다."""
        path = _flags_path(tmp_path)

        # 인스턴스 1: toggle
        ff1 = FeatureFlags(flags_path=path)
        ff1.toggle("global_monitor", enabled=True)
        ff1.toggle("data_cache", enabled=False)

        # 인스턴스 2: 같은 파일로 새로 로드
        ff2 = FeatureFlags(flags_path=path)
        assert ff2.is_enabled("global_monitor") is True
        assert ff2.is_enabled("data_cache") is False

    def test_set_config_persists_across_instances(self, tmp_path):
        """set_config 후 새 인스턴스에서 설정이 유지된다."""
        path = _flags_path(tmp_path)

        ff1 = FeatureFlags(flags_path=path)
        ff1.set_config("data_cache", "cache_ttl_hours", 100)
        ff1.set_config("stock_review", "max_stocks", 50)

        ff2 = FeatureFlags(flags_path=path)
        assert ff2.get_config("data_cache")["cache_ttl_hours"] == 100
        assert ff2.get_config("stock_review")["max_stocks"] == 50

    def test_multiple_toggles_persist(self, tmp_path):
        """여러 번 toggle 후 최종 상태가 유지된다."""
        path = _flags_path(tmp_path)

        ff1 = FeatureFlags(flags_path=path)
        ff1.toggle("data_cache")  # True -> False
        ff1.toggle("data_cache")  # False -> True
        ff1.toggle("data_cache")  # True -> False

        ff2 = FeatureFlags(flags_path=path)
        assert ff2.is_enabled("data_cache") is False

    def test_json_file_structure_valid(self, tmp_path):
        """저장된 JSON 파일의 구조가 올바르다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)
        ff.toggle("night_research", enabled=True)

        data = _read_json(path)

        assert "version" in data
        assert "updated_at" in data
        assert "features" in data
        assert isinstance(data["features"], dict)

        for name, info in data["features"].items():
            assert "enabled" in info, f"'{name}'에 enabled 키가 없습니다."
            assert "description" in info, f"'{name}'에 description 키가 없습니다."


# ===================================================================
# 8. 스레드 안전성 테스트
# ===================================================================


class TestThreadSafety:
    """기본적인 동시성 안전을 검증한다."""

    def test_concurrent_toggles_no_crash(self, tmp_path):
        """다수의 스레드에서 동시에 toggle해도 예외가 발생하지 않는다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)
        errors = []

        def toggle_many(feature_name, count):
            try:
                for _ in range(count):
                    ff.toggle(feature_name)
            except Exception as e:
                errors.append(e)

        threads = []
        features = list(FeatureFlags.DEFAULT_FLAGS.keys())
        for feature in features:
            t = threading.Thread(target=toggle_many, args=(feature, 50))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"스레드 에러 발생: {errors}"

    def test_concurrent_toggle_final_state_consistent(self, tmp_path):
        """짝수 횟수 토글 후 원래 상태로 복원된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        original = ff.is_enabled("data_cache")
        toggle_count = 100  # 짝수

        def toggle_n():
            for _ in range(toggle_count):
                ff.toggle("data_cache")

        # 단일 스레드에서 짝수 토글 → 원래 상태
        toggle_n()
        assert ff.is_enabled("data_cache") == original

    def test_concurrent_reads_and_writes(self, tmp_path):
        """읽기/쓰기가 동시에 일어나도 예외가 발생하지 않는다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)
        errors = []

        def reader():
            try:
                for _ in range(100):
                    ff.is_enabled("data_cache")
                    ff.get_config("data_cache")
                    ff.get_all_status()
                    ff.get_summary()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for _ in range(100):
                    ff.toggle("data_cache")
                    ff.set_config("data_cache", "cache_ttl_hours", 48)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=writer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"동시성 에러: {errors}"


# ===================================================================
# 9. 손상/누락 파일 처리 테스트
# ===================================================================


class TestCorruptedAndMissingFile:
    """손상된 파일이나 누락된 파일 처리를 검증한다."""

    def test_corrupted_json_falls_back_to_defaults(self, tmp_path):
        """손상된 JSON 파일이면 기본값으로 폴백한다."""
        path = _flags_path(tmp_path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("{{{INVALID JSON!!!", encoding="utf-8")

        ff = FeatureFlags(flags_path=path)

        # 기본값으로 동작해야 함
        assert ff.is_enabled("data_cache") is True
        assert ff.is_enabled("global_monitor") is False
        status = ff.get_all_status()
        for key in FeatureFlags.DEFAULT_FLAGS:
            assert key in status

    def test_empty_file_falls_back_to_defaults(self, tmp_path):
        """빈 파일이면 기본값으로 폴백한다."""
        path = _flags_path(tmp_path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("", encoding="utf-8")

        ff = FeatureFlags(flags_path=path)

        assert ff.is_enabled("data_cache") is True
        status = ff.get_all_status()
        assert len(status) == len(FeatureFlags.DEFAULT_FLAGS)

    def test_missing_features_key_falls_back(self, tmp_path):
        """JSON에 'features' 키가 없으면 빈 딕셔너리로 처리, 기본값 보충."""
        path = _flags_path(tmp_path)
        _write_json(path, {"version": 1, "updated_at": "2025-01-01"})

        ff = FeatureFlags(flags_path=path)

        # features가 빈 {} → 모든 기본 플래그 보충
        status = ff.get_all_status()
        for key in FeatureFlags.DEFAULT_FLAGS:
            assert key in status

    def test_file_deleted_after_init_reload_uses_defaults(self, tmp_path):
        """초기화 후 파일이 삭제되면 reload()에서 기본값으로 재초기화한다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        # 파일 삭제
        Path(path).unlink()
        assert not Path(path).exists()

        # reload → 파일이 없으므로 기본값으로 재생성
        ff.reload()

        assert ff.is_enabled("data_cache") is True
        assert Path(path).exists(), "reload 후 파일이 재생성되어야 합니다."

    def test_partial_feature_data_handled(self, tmp_path):
        """피처 데이터에 enabled 키가 누락되어도 False로 처리된다."""
        path = _flags_path(tmp_path)
        _write_json(path, {
            "version": 1,
            "updated_at": "2025-01-01",
            "features": {
                "data_cache": {
                    "description": "enabled 키 없음",
                    "config": {},
                },
            },
        })

        ff = FeatureFlags(flags_path=path)

        # enabled 키가 없으면 False로 처리
        assert ff.is_enabled("data_cache") is False

    def test_non_boolean_enabled_handled(self, tmp_path):
        """enabled가 bool이 아닌 값이면 bool()로 변환된다."""
        path = _flags_path(tmp_path)
        _write_json(path, {
            "version": 1,
            "updated_at": "2025-01-01",
            "features": {
                "data_cache": {
                    "enabled": 1,  # int
                    "description": "int enabled",
                    "config": {},
                },
                "global_monitor": {
                    "enabled": 0,  # int
                    "description": "int disabled",
                    "config": {},
                },
                "stock_review": {
                    "enabled": "yes",  # truthy string
                    "description": "string enabled",
                    "config": {},
                },
            },
        })

        ff = FeatureFlags(flags_path=path)

        assert ff.is_enabled("data_cache") is True  # bool(1) == True
        assert ff.is_enabled("global_monitor") is False  # bool(0) == False
        assert ff.is_enabled("stock_review") is True  # bool("yes") == True

    def test_read_only_directory_save_fails_gracefully(self, tmp_path):
        """저장 실패 시 예외가 발생하지 않는다 (OSError 처리)."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        # _save에서 OSError가 발생하도록 mock
        with patch.object(Path, "write_text", side_effect=OSError("Permission denied")):
            # toggle 자체는 메모리에서 성공, save만 실패
            result = ff.toggle("data_cache")
            # toggle은 True 반환 (메모리 업데이트 후 save 실패)
            assert result is True

    def test_unicode_in_json_file(self, tmp_path):
        """한국어 등 유니코드 설명이 올바르게 처리된다."""
        path = _flags_path(tmp_path)
        ff = FeatureFlags(flags_path=path)

        data = _read_json(path)
        # 기본 플래그의 한국어 설명이 보존되는지 확인
        assert "데이터 캐싱" in data["features"]["data_cache"]["description"]
        assert "글로벌" in data["features"]["global_monitor"]["description"]
