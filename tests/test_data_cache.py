"""데이터 캐시 모듈(src/data/cache.py) 테스트.

DataCache 클래스와 cached 데코레이터를 검증한다.
tmp_path 픽스처를 사용하여 격리된 캐시 디렉토리에서 테스트한다.
"""

import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.cache import DataCache, cached


# ===================================================================
# 공통 픽스처
# ===================================================================


@pytest.fixture
def cache(tmp_path):
    """기본 DataCache 인스턴스 (TTL 24시간, 최대 500MB)."""
    return DataCache(cache_dir=str(tmp_path), ttl_hours=24, max_size_mb=500)


@pytest.fixture
def sample_df():
    """테스트용 간단한 DataFrame.

    parquet 라운드트립 시 freq 메타데이터가 소실되므로
    freq 없는 DatetimeIndex를 사용한다.
    """
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {"close": [100, 200, 300], "volume": [1000, 2000, 3000]},
        index=idx,
    )


@pytest.fixture
def feature_flags_enabled():
    """data_cache가 활성화된 mock FeatureFlags."""
    flags = MagicMock()
    flags.is_enabled.return_value = True
    return flags


@pytest.fixture
def feature_flags_disabled():
    """data_cache가 비활성화된 mock FeatureFlags."""
    flags = MagicMock()
    flags.is_enabled.return_value = False
    return flags


# ===================================================================
# DataCache.make_key() 테스트
# ===================================================================


class TestMakeKey:
    """make_key() 캐시 키 생성 테스트."""

    def test_consistent_hash(self):
        """동일 인자는 항상 같은 키를 생성한다."""
        key1 = DataCache.make_key("get_price", "005930", date="20240101")
        key2 = DataCache.make_key("get_price", "005930", date="20240101")
        assert key1 == key2

    def test_different_args_different_key(self):
        """다른 인자는 다른 키를 생성한다."""
        key1 = DataCache.make_key("get_price", "005930")
        key2 = DataCache.make_key("get_price", "000660")
        assert key1 != key2

    def test_different_func_name_different_key(self):
        """다른 함수명은 다른 키를 생성한다."""
        key1 = DataCache.make_key("get_price", "005930")
        key2 = DataCache.make_key("get_volume", "005930")
        assert key1 != key2

    def test_kwargs_order_independent(self):
        """kwargs 순서가 달라도 같은 키를 생성한다."""
        key1 = DataCache.make_key("func", a=1, b=2)
        key2 = DataCache.make_key("func", b=2, a=1)
        assert key1 == key2

    def test_key_is_hex_string(self):
        """생성된 키는 32자리 16진수 문자열(MD5)이다."""
        key = DataCache.make_key("test_func", "arg1")
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)


# ===================================================================
# DataCache.put() + get() 라운드트립 테스트
# ===================================================================


class TestPutGet:
    """put/get 라운드트립 테스트."""

    def test_roundtrip_simple_dataframe(self, cache, sample_df):
        """저장 후 조회하면 동일한 DataFrame이 반환된다."""
        key = "test_roundtrip"
        cache.put(key, sample_df)
        result = cache.get(key)

        assert result is not None
        pd.testing.assert_frame_equal(result, sample_df)

    def test_roundtrip_preserves_dtypes(self, cache):
        """다양한 dtype이 보존된다."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
            }
        )
        key = "test_dtypes"
        cache.put(key, df)
        result = cache.get(key)

        assert result is not None
        pd.testing.assert_frame_equal(result, df)

    def test_get_missing_key_returns_none(self, cache):
        """존재하지 않는 키 조회 시 None을 반환한다."""
        result = cache.get("nonexistent_key")
        assert result is None

    def test_put_skips_empty_dataframe(self, cache):
        """빈 DataFrame은 저장하지 않는다."""
        empty_df = pd.DataFrame()
        key = "test_empty"
        cache.put(key, empty_df)

        result = cache.get(key)
        assert result is None

    def test_put_skips_empty_dataframe_with_columns(self, cache):
        """컬럼만 있고 행이 없는 DataFrame도 저장하지 않는다."""
        empty_df = pd.DataFrame(columns=["a", "b", "c"])
        key = "test_empty_cols"
        cache.put(key, empty_df)

        result = cache.get(key)
        assert result is None


# ===================================================================
# TTL 만료 테스트
# ===================================================================


class TestTTLExpiry:
    """TTL(Time-To-Live) 만료 테스트."""

    def test_expired_cache_returns_none(self, tmp_path, sample_df):
        """TTL이 지난 캐시는 None을 반환한다."""
        # TTL 1초로 설정
        cache = DataCache(
            cache_dir=str(tmp_path), ttl_hours=0, max_size_mb=500
        )
        # ttl_hours=0 이면 _ttl_seconds=0이므로 즉시 만료
        key = "test_ttl"
        cache.put(key, sample_df)

        # st_mtime 기준으로 age > 0 이면 만료
        # 아주 짧은 대기로 확실히 만료시킴
        time.sleep(0.1)

        result = cache.get(key)
        assert result is None

    def test_valid_ttl_returns_data(self, tmp_path, sample_df):
        """TTL 이내의 캐시는 정상 반환한다."""
        cache = DataCache(
            cache_dir=str(tmp_path), ttl_hours=1, max_size_mb=500
        )
        key = "test_valid_ttl"
        cache.put(key, sample_df)

        result = cache.get(key)
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_df)

    def test_expired_file_is_deleted(self, tmp_path, sample_df):
        """만료된 캐시 파일은 디스크에서 삭제된다."""
        cache = DataCache(
            cache_dir=str(tmp_path), ttl_hours=0, max_size_mb=500
        )
        key = "test_delete"
        cache.put(key, sample_df)

        cache_path = tmp_path / f"{key}.parquet"
        assert cache_path.exists()

        time.sleep(0.1)
        cache.get(key)

        assert not cache_path.exists()


# ===================================================================
# clear() 테스트
# ===================================================================


class TestClear:
    """clear() 전체 캐시 삭제 테스트."""

    def test_clear_removes_all_files(self, cache, sample_df):
        """clear()는 모든 캐시 파일을 삭제한다."""
        cache.put("key1", sample_df)
        cache.put("key2", sample_df)

        count = cache.clear()

        assert count == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_clear_returns_zero_when_empty(self, cache):
        """캐시가 비어 있으면 0을 반환한다."""
        count = cache.clear()
        assert count == 0

    def test_clear_only_removes_parquet_files(self, tmp_path, sample_df):
        """clear()는 .parquet 파일만 삭제한다."""
        cache = DataCache(cache_dir=str(tmp_path))
        cache.put("key1", sample_df)

        # non-parquet 파일 생성
        other_file = tmp_path / "notes.txt"
        other_file.write_text("keep me")

        cache.clear()

        assert other_file.exists()


# ===================================================================
# get_stats() 테스트
# ===================================================================


class TestGetStats:
    """get_stats() 캐시 통계 테스트."""

    def test_empty_cache_stats(self, cache):
        """빈 캐시의 통계는 모두 0이다."""
        stats = cache.get_stats()
        assert stats["file_count"] == 0
        assert stats["total_size_mb"] == 0.0
        assert stats["max_size_mb"] == 500.0

    def test_stats_after_put(self, cache, sample_df):
        """put() 후 file_count와 total_size_mb가 갱신된다."""
        cache.put("key1", sample_df)
        cache.put("key2", sample_df)

        stats = cache.get_stats()
        assert stats["file_count"] == 2
        assert stats["total_size_mb"] > 0

    def test_stats_after_clear(self, cache, sample_df):
        """clear() 후 통계가 0으로 리셋된다."""
        cache.put("key1", sample_df)
        cache.clear()

        stats = cache.get_stats()
        assert stats["file_count"] == 0
        assert stats["total_size_mb"] == 0.0

    def test_stats_max_size_reflects_config(self, tmp_path):
        """max_size_mb는 생성자에서 지정한 값을 반영한다."""
        cache = DataCache(cache_dir=str(tmp_path), max_size_mb=100)
        stats = cache.get_stats()
        assert stats["max_size_mb"] == 100.0


# ===================================================================
# LRU Eviction 테스트
# ===================================================================


class TestLRUEviction:
    """LRU 방식 캐시 제거 테스트."""

    def test_evicts_oldest_when_size_exceeded(self, tmp_path):
        """max_size_mb 초과 시 가장 오래된 파일부터 삭제한다."""
        # 매우 작은 max_size로 설정 (예: 0.001 MB = ~1KB)
        # 실제 parquet 파일은 보통 수 KB이므로 넘침
        cache = DataCache(
            cache_dir=str(tmp_path), ttl_hours=24, max_size_mb=0
        )
        # max_size_mb=0 => _max_size_bytes=0, 모든 파일이 evict 대상

        df = pd.DataFrame({"a": range(100)})

        cache.put("oldest", df)
        time.sleep(0.05)
        cache.put("middle", df)
        time.sleep(0.05)
        cache.put("newest", df)

        # max_size=0이므로 put()할 때마다 eviction 발생
        # 최종적으로 newest만 남아있을 수도 있고, 모두 삭제될 수 있음
        # _evict_if_needed는 put() 저장 후 실행되므로 마지막 파일은 삭제됨
        files = list(tmp_path.glob("*.parquet"))
        # 총 크기 > 0이므로 오래된 것부터 삭제
        assert len(files) == 0

    def test_no_eviction_when_under_limit(self, tmp_path):
        """캐시 크기가 제한 이내이면 eviction이 발생하지 않는다."""
        cache = DataCache(
            cache_dir=str(tmp_path), ttl_hours=24, max_size_mb=100
        )
        df = pd.DataFrame({"a": range(10)})

        cache.put("key1", df)
        cache.put("key2", df)
        cache.put("key3", df)

        stats = cache.get_stats()
        assert stats["file_count"] == 3

    def test_eviction_preserves_newest_files(self, tmp_path):
        """eviction 시 최신 파일이 가능한 한 보존된다."""
        # 1파일이 들어갈 수 있는 크기로 설정
        # 먼저 1개 파일의 대략적 크기를 확인
        probe_cache = DataCache(
            cache_dir=str(tmp_path / "probe"), ttl_hours=24, max_size_mb=500
        )
        df = pd.DataFrame({"a": range(100)})
        probe_cache.put("probe", df)
        one_file_bytes = (tmp_path / "probe" / "probe.parquet").stat().st_size

        # 2개 파일이 들어갈 크기로 설정 (약간의 여유)
        max_mb = (one_file_bytes * 2.5) / (1024 * 1024)
        actual_cache_dir = tmp_path / "actual"
        cache = DataCache(
            cache_dir=str(actual_cache_dir), ttl_hours=24, max_size_mb=max_mb
        )

        cache.put("file1", df)
        time.sleep(0.05)
        cache.put("file2", df)
        time.sleep(0.05)
        cache.put("file3", df)

        # 3개 중 가장 오래된 file1은 삭제되고 file2, file3은 보존
        files = sorted(
            actual_cache_dir.glob("*.parquet"),
            key=lambda f: f.stat().st_mtime,
        )
        assert len(files) == 2
        # 가장 오래된 file1이 삭제되었는지 확인
        assert not (actual_cache_dir / "file1.parquet").exists()
        assert (actual_cache_dir / "file3.parquet").exists()


# ===================================================================
# cached 데코레이터 테스트
# ===================================================================


class TestCachedDecorator:
    """cached() 데코레이터 테스트."""

    def test_cache_miss_calls_function_and_stores(
        self, tmp_path, sample_df, feature_flags_enabled
    ):
        """캐시 미스 시 원본 함수를 호출하고 결과를 저장한다."""
        cache = DataCache(cache_dir=str(tmp_path))
        call_count = 0

        @cached(cache, feature_flags_enabled)
        def get_data(ticker):
            nonlocal call_count
            call_count += 1
            return sample_df

        result = get_data("005930")

        assert call_count == 1
        pd.testing.assert_frame_equal(result, sample_df)
        # 캐시에 저장되었는지 확인
        key = DataCache.make_key("get_data", "005930")
        cached_result = cache.get(key)
        assert cached_result is not None
        pd.testing.assert_frame_equal(cached_result, sample_df)

    def test_cache_hit_returns_cached_without_calling(
        self, tmp_path, sample_df, feature_flags_enabled
    ):
        """캐시 히트 시 함수를 호출하지 않고 캐시된 결과를 반환한다."""
        cache = DataCache(cache_dir=str(tmp_path))
        call_count = 0

        @cached(cache, feature_flags_enabled)
        def get_data(ticker):
            nonlocal call_count
            call_count += 1
            return sample_df

        # 첫 번째 호출: 캐시 미스
        get_data("005930")
        assert call_count == 1

        # 두 번째 호출: 캐시 히트
        result = get_data("005930")
        assert call_count == 1  # 함수가 다시 호출되지 않음
        pd.testing.assert_frame_equal(result, sample_df)

    def test_disabled_flag_always_calls_function(
        self, tmp_path, sample_df, feature_flags_disabled
    ):
        """data_cache 플래그가 비활성화되면 항상 함수를 호출한다."""
        cache = DataCache(cache_dir=str(tmp_path))
        call_count = 0

        @cached(cache, feature_flags_disabled)
        def get_data(ticker):
            nonlocal call_count
            call_count += 1
            return sample_df

        get_data("005930")
        get_data("005930")
        get_data("005930")

        assert call_count == 3
        feature_flags_disabled.is_enabled.assert_called_with("data_cache")

    def test_disabled_flag_does_not_store_in_cache(
        self, tmp_path, sample_df, feature_flags_disabled
    ):
        """data_cache 비활성화 시 캐시에 저장하지 않는다."""
        cache = DataCache(cache_dir=str(tmp_path))

        @cached(cache, feature_flags_disabled)
        def get_data(ticker):
            return sample_df

        get_data("005930")

        stats = cache.get_stats()
        assert stats["file_count"] == 0

    def test_non_dataframe_result_not_cached(
        self, tmp_path, feature_flags_enabled
    ):
        """반환값이 DataFrame이 아니면 캐시에 저장하지 않는다."""
        cache = DataCache(cache_dir=str(tmp_path))

        @cached(cache, feature_flags_enabled)
        def get_list():
            return [1, 2, 3]

        result = get_list()
        assert result == [1, 2, 3]

        stats = cache.get_stats()
        assert stats["file_count"] == 0

    def test_different_args_cached_separately(
        self, tmp_path, feature_flags_enabled
    ):
        """다른 인자로 호출하면 별도로 캐시된다."""
        cache = DataCache(cache_dir=str(tmp_path))

        @cached(cache, feature_flags_enabled)
        def get_data(ticker):
            return pd.DataFrame({"ticker": [ticker], "price": [100]})

        result1 = get_data("005930")
        result2 = get_data("000660")

        assert result1.iloc[0]["ticker"] == "005930"
        assert result2.iloc[0]["ticker"] == "000660"

        stats = cache.get_stats()
        assert stats["file_count"] == 2

    def test_decorator_preserves_function_name(
        self, tmp_path, feature_flags_enabled
    ):
        """데코레이터가 원본 함수의 이름을 보존한다 (functools.wraps)."""
        cache = DataCache(cache_dir=str(tmp_path))

        @cached(cache, feature_flags_enabled)
        def my_special_function():
            return pd.DataFrame()

        assert my_special_function.__name__ == "my_special_function"
