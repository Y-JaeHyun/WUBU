"""PriceStore 단위 테스트.

영구 가격 캐시의 캐시 히트/미스, 증분 fetch, 펀더멘탈, 지수,
삭제, 통계, 손상 복구 등을 검증한다.
"""

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.price_store import PriceStore, get_price_store, reset_price_store


@pytest.fixture
def tmp_store(tmp_path):
    """임시 디렉토리에 PriceStore를 생성한다."""
    return PriceStore(store_dir=str(tmp_path / "price_store"))


@pytest.fixture
def sample_ohlcv():
    """20 영업일 분량의 OHLCV 데이터를 생성하는 팩토리."""
    def _make(start="20240102", periods=20, base=70000):
        dates = pd.bdate_range(start, periods=periods)
        np.random.seed(42)
        close = base + np.cumsum(np.random.randn(periods) * 500).astype(int)
        close = np.maximum(close, 1000)
        df = pd.DataFrame(
            {
                "open": close - 100,
                "high": close + 200,
                "low": close - 200,
                "close": close,
                "volume": np.random.randint(100_000, 10_000_000, periods),
            },
            index=dates,
        )
        df.index.name = "date"
        return df
    return _make


@pytest.fixture
def sample_fundamentals():
    """전종목 펀더멘탈 샘플."""
    np.random.seed(42)
    n = 10
    return pd.DataFrame({
        "ticker": [f"{i:06d}" for i in range(1, n + 1)],
        "name": [f"종목{i}" for i in range(1, n + 1)],
        "per": np.random.rand(n) * 20,
        "pbr": np.random.rand(n) * 3,
        "close": np.random.randint(5000, 100000, n),
        "market_cap": np.random.randint(1_000_000_000, 5_000_000_000_000, n),
    })


# ─── 캐시 히트/미스 ───────────────────────────────────────────

class TestCacheHitMiss:
    def test_cache_miss_fetches_and_stores(self, tmp_store, sample_ohlcv):
        """캐시 미스 → fetch → parquet 저장."""
        data = sample_ohlcv()
        call_count = 0

        def fetcher(ticker, start, end):
            nonlocal call_count
            call_count += 1
            return data

        result = tmp_store.get_price("005930", "20240102", "20240129", fetcher=fetcher)
        assert call_count == 1
        assert not result.empty
        assert (tmp_store._prices_dir / "005930.parquet").exists()

    def test_cache_hit_no_fetch(self, tmp_store, sample_ohlcv):
        """캐시 히트 → fetch 0회."""
        data = sample_ohlcv()
        call_count = 0

        def fetcher(ticker, start, end):
            nonlocal call_count
            call_count += 1
            return data

        # 1차: 캐시 미스
        tmp_store.get_price("005930", "20240102", "20240129", fetcher=fetcher)
        assert call_count == 1

        # 2차: 캐시 히트
        result = tmp_store.get_price("005930", "20240102", "20240129", fetcher=fetcher)
        assert call_count == 1  # 추가 fetch 없음
        assert not result.empty

    def test_returns_only_requested_range(self, tmp_store, sample_ohlcv):
        """캐시에 넓은 범위가 있어도 요청 범위만 반환."""
        data = sample_ohlcv(start="20240102", periods=60)

        def fetcher(ticker, start, end):
            return data

        # 넓은 범위 저장
        tmp_store.get_price("005930", "20240102", "20240326", fetcher=fetcher)

        # 좁은 범위 요청
        result = tmp_store.get_price("005930", "20240115", "20240215", fetcher=fetcher)
        assert result.index.min() >= pd.Timestamp("20240115")
        assert result.index.max() <= pd.Timestamp("20240215")


# ─── 증분 Fetch ──────────────────────────────────────────────

class TestIncrementalFetch:
    def test_extend_right(self, tmp_store, sample_ohlcv):
        """캐시 끝 이후 요청 → 뒤쪽만 증분 fetch."""
        data_initial = sample_ohlcv(start="20240102", periods=20)
        data_extension = sample_ohlcv(start="20240130", periods=20, base=72000)
        fetch_ranges = []

        def fetcher(ticker, start, end):
            fetch_ranges.append((start, end))
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            combined = pd.concat([data_initial, data_extension])
            return combined[(combined.index >= s) & (combined.index <= e)]

        # 초기 로드
        tmp_store.get_price("005930", "20240102", "20240129", fetcher=fetcher)
        assert len(fetch_ranges) == 1

        # 확장 요청 (20240102~20240228)
        result = tmp_store.get_price("005930", "20240102", "20240228", fetcher=fetcher)
        assert len(fetch_ranges) == 2
        # 두 번째 fetch는 캐시 끝 이후부터
        assert fetch_ranges[1][0] >= "20240129"
        assert not result.empty

    def test_extend_left(self, tmp_store, sample_ohlcv):
        """캐시 시작 이전 요청 → 앞쪽만 증분 fetch."""
        data_initial = sample_ohlcv(start="20240201", periods=20)
        data_prefix = sample_ohlcv(start="20240102", periods=22, base=68000)
        fetch_ranges = []

        def fetcher(ticker, start, end):
            fetch_ranges.append((start, end))
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            combined = pd.concat([data_prefix, data_initial])
            combined = combined[~combined.index.duplicated(keep="last")]
            return combined[(combined.index >= s) & (combined.index <= e)]

        # 초기 로드
        tmp_store.get_price("005930", "20240201", "20240228", fetcher=fetcher)
        assert len(fetch_ranges) == 1

        # 앞쪽 확장 요청
        result = tmp_store.get_price("005930", "20240102", "20240228", fetcher=fetcher)
        assert len(fetch_ranges) == 2
        # 두 번째 fetch는 캐시 시작 이전까지
        assert fetch_ranges[1][1] <= "20240201"
        assert not result.empty

    def test_extend_both_sides(self, tmp_store, sample_ohlcv):
        """양쪽 확장 → 2번 증분 fetch."""
        data_mid = sample_ohlcv(start="20240201", periods=10)
        full_data = sample_ohlcv(start="20240102", periods=60, base=68000)
        fetch_count = 0

        def fetcher(ticker, start, end):
            nonlocal fetch_count
            fetch_count += 1
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            return full_data[(full_data.index >= s) & (full_data.index <= e)]

        # 중간 범위 저장
        tmp_store.get_price("005930", "20240201", "20240214", fetcher=fetcher)
        assert fetch_count == 1

        # 양쪽 확장
        result = tmp_store.get_price("005930", "20240102", "20240328", fetcher=fetcher)
        assert fetch_count == 3  # 앞쪽 + 뒤쪽
        assert not result.empty


# ─── 펀더멘탈 ─────────────────────────────────────────────────

class TestFundamentals:
    def test_fundamentals_cache_hit(self, tmp_store, sample_fundamentals):
        """펀더멘탈 캐시 히트."""
        call_count = 0

        def fetcher(date):
            nonlocal call_count
            call_count += 1
            return sample_fundamentals

        tmp_store.get_fundamentals("20240102", fetcher=fetcher)
        assert call_count == 1

        result = tmp_store.get_fundamentals("20240102", fetcher=fetcher)
        assert call_count == 1  # 추가 fetch 없음
        assert len(result) == 10

    def test_fundamentals_different_dates(self, tmp_store, sample_fundamentals):
        """다른 날짜는 별도 캐시."""
        call_count = 0

        def fetcher(date):
            nonlocal call_count
            call_count += 1
            return sample_fundamentals

        tmp_store.get_fundamentals("20240102", fetcher=fetcher)
        tmp_store.get_fundamentals("20240202", fetcher=fetcher)
        assert call_count == 2


# ─── 지수 ─────────────────────────────────────────────────────

class TestIndex:
    def test_index_cache(self, tmp_store, sample_ohlcv):
        """지수 데이터 캐싱."""
        data = sample_ohlcv()
        call_count = 0

        def fetcher(index_name, start, end):
            nonlocal call_count
            call_count += 1
            return data

        tmp_store.get_index("KOSPI", "20240102", "20240129", fetcher=fetcher)
        assert call_count == 1

        result = tmp_store.get_index("KOSPI", "20240102", "20240129", fetcher=fetcher)
        assert call_count == 1
        assert not result.empty

    def test_index_incremental(self, tmp_store, sample_ohlcv):
        """지수 증분 fetch."""
        full_data = sample_ohlcv(start="20240102", periods=60)
        call_count = 0

        def fetcher(index_name, start, end):
            nonlocal call_count
            call_count += 1
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            return full_data[(full_data.index >= s) & (full_data.index <= e)]

        tmp_store.get_index("KOSPI", "20240102", "20240129", fetcher=fetcher)
        assert call_count == 1

        tmp_store.get_index("KOSPI", "20240102", "20240228", fetcher=fetcher)
        assert call_count == 2  # 뒤쪽만 추가 fetch


# ─── 삭제 ─────────────────────────────────────────────────────

class TestClear:
    def test_clear_all(self, tmp_store, sample_ohlcv, sample_fundamentals):
        """전체 삭제."""
        data = sample_ohlcv()
        tmp_store.get_price("005930", "20240102", "20240129", fetcher=lambda t, s, e: data)
        tmp_store.get_fundamentals("20240102", fetcher=lambda d: sample_fundamentals)

        tmp_store.clear()

        stats = tmp_store.get_stats()
        assert stats["total_items"] == 0

    def test_clear_by_type(self, tmp_store, sample_ohlcv):
        """타입별 삭제."""
        data = sample_ohlcv()
        tmp_store.get_price("005930", "20240102", "20240129", fetcher=lambda t, s, e: data)
        tmp_store.get_index("KOSPI", "20240102", "20240129", fetcher=lambda n, s, e: data)

        tmp_store.clear(data_type="price")

        stats = tmp_store.get_stats()
        assert stats.get("price", {}).get("items", 0) == 0
        assert stats.get("index", {}).get("items", 0) == 1


# ─── 통계 ─────────────────────────────────────────────────────

class TestStats:
    def test_empty_stats(self, tmp_store):
        """빈 저장소 통계."""
        stats = tmp_store.get_stats()
        assert stats["total_items"] == 0
        assert stats["total_rows"] == 0
        assert stats["disk_usage_mb"] >= 0

    def test_stats_after_data(self, tmp_store, sample_ohlcv):
        """데이터 저장 후 통계."""
        data = sample_ohlcv()
        tmp_store.get_price("005930", "20240102", "20240129", fetcher=lambda t, s, e: data)
        tmp_store.get_price("000660", "20240102", "20240129", fetcher=lambda t, s, e: data)

        stats = tmp_store.get_stats()
        assert stats["total_items"] == 2
        assert stats["price"]["items"] == 2
        assert stats["disk_usage_mb"] > 0


# ─── 손상 복구 ────────────────────────────────────────────────

class TestCorruptionRecovery:
    def test_corrupted_parquet_triggers_redownload(self, tmp_store, sample_ohlcv):
        """parquet 손상 시 메타 삭제 후 재다운로드."""
        data = sample_ohlcv()
        call_count = 0

        def fetcher(ticker, start, end):
            nonlocal call_count
            call_count += 1
            return data

        # 초기 저장
        tmp_store.get_price("005930", "20240102", "20240129", fetcher=fetcher)
        assert call_count == 1

        # parquet 손상 시뮬레이션
        corrupt_path = tmp_store._prices_dir / "005930.parquet"
        corrupt_path.write_bytes(b"corrupted data")

        # 재요청 → 재다운로드
        result = tmp_store.get_price("005930", "20240102", "20240129", fetcher=fetcher)
        assert call_count == 2
        assert not result.empty


# ─── 싱글톤 ──────────────────────────────────────────────────

class TestSingleton:
    def test_get_price_store_returns_same_instance(self):
        """get_price_store()는 같은 인스턴스를 반환한다."""
        reset_price_store()
        s1 = get_price_store()
        s2 = get_price_store()
        assert s1 is s2
        reset_price_store()

    def test_reset_clears_singleton(self):
        """reset_price_store()는 싱글톤을 초기화한다."""
        reset_price_store()
        s1 = get_price_store()
        reset_price_store()
        s2 = get_price_store()
        assert s1 is not s2
        reset_price_store()


# ─── 엣지 케이스 ──────────────────────────────────────────────

class TestEdgeCases:
    def test_fetch_returns_empty(self, tmp_store):
        """fetcher가 빈 DataFrame을 반환해도 에러 없음."""
        result = tmp_store.get_price(
            "999999", "20240102", "20240129",
            fetcher=lambda t, s, e: pd.DataFrame()
        )
        assert result.empty

    def test_date_format_with_dashes(self, tmp_store, sample_ohlcv):
        """하이픈 포함 날짜도 처리."""
        data = sample_ohlcv()
        result = tmp_store.get_price(
            "005930", "2024-01-02", "2024-01-29",
            fetcher=lambda t, s, e: data
        )
        assert not result.empty
