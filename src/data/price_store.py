"""PriceStore: 백테스트용 영구 가격 캐시.

티커별 parquet 파일 + SQLite 메타데이터를 이용하여 가격/펀더멘탈/지수 데이터를
영구 캐싱한다. 최초 1회 다운로드 후 로컬에서 재활용하며, 요청 범위가 캐시를
초과하는 부분만 증분 fetch한다.

구조:
    data/price_store/
      prices/005930.parquet    # 티커당 1파일, OHLCV
      fundamentals/20240102.parquet  # 날짜별 전종목 펀더멘탈
      index/KOSPI.parquet      # 지수 데이터
      metadata.db              # SQLite: 커버리지 추적
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "price_store",
)


class PriceStore:
    """백테스트용 영구 가격 캐시.

    Args:
        store_dir: 저장소 루트 디렉토리 (기본: data/price_store/)
    """

    def __init__(self, store_dir: Optional[str] = None):
        self.store_dir = Path(store_dir or _DEFAULT_STORE_DIR)
        self._prices_dir = self.store_dir / "prices"
        self._fundamentals_dir = self.store_dir / "fundamentals"
        self._index_dir = self.store_dir / "index"
        self._db_path = self.store_dir / "metadata.db"

        # 디렉토리 생성
        for d in [self._prices_dir, self._fundamentals_dir, self._index_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # SQLite 메타데이터 초기화
        self._init_db()

    def _init_db(self) -> None:
        """SQLite 메타데이터 테이블을 초기화한다."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_metadata (
                    ticker TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    min_date TEXT NOT NULL,
                    max_date TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (ticker, data_type)
                )
            """)
            conn.commit()

    def _get_meta(self, ticker: str, data_type: str) -> Optional[dict]:
        """메타데이터 조회."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT min_date, max_date, row_count, updated_at "
                "FROM price_metadata WHERE ticker = ? AND data_type = ?",
                (ticker, data_type),
            ).fetchone()
        if row is None:
            return None
        return {
            "min_date": row[0],
            "max_date": row[1],
            "row_count": row[2],
            "updated_at": row[3],
        }

    def _set_meta(
        self, ticker: str, data_type: str, min_date: str, max_date: str, row_count: int
    ) -> None:
        """메타데이터 저장/갱신."""
        now = datetime.now().isoformat()
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO price_metadata "
                "(ticker, data_type, min_date, max_date, row_count, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ticker, data_type, min_date, max_date, row_count, now),
            )
            conn.commit()

    def _delete_meta(self, ticker: str, data_type: str) -> None:
        """메타데이터 삭제."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "DELETE FROM price_metadata WHERE ticker = ? AND data_type = ?",
                (ticker, data_type),
            )
            conn.commit()

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """날짜를 YYYYMMDD 형식으로 정규화."""
        return date_str.replace("-", "")

    @staticmethod
    def _prev_bday(date_str: str) -> str:
        """주어진 날짜의 전 영업일을 반환한다."""
        ts = pd.Timestamp(date_str) - pd.tseries.offsets.BDay(1)
        return ts.strftime("%Y%m%d")

    @staticmethod
    def _next_bday(date_str: str) -> str:
        """주어진 날짜의 다음 영업일을 반환한다."""
        ts = pd.Timestamp(date_str) + pd.tseries.offsets.BDay(1)
        return ts.strftime("%Y%m%d")

    def _compute_missing_ranges(
        self, start_date: str, end_date: str, meta: Optional[dict]
    ) -> list[tuple[str, str]]:
        """캐시 대비 부족한 날짜 범위를 계산한다.

        Returns:
            fetch가 필요한 (start, end) 튜플 리스트. 비어 있으면 캐시 100% 히트.
        """
        if meta is None:
            return [(start_date, end_date)]

        cache_min = meta["min_date"]
        cache_max = meta["max_date"]
        ranges: list[tuple[str, str]] = []

        # 요청 시작 < 캐시 시작 → 앞쪽 부족
        if start_date < cache_min:
            ranges.append((start_date, self._prev_bday(cache_min)))

        # 요청 끝 > 캐시 끝 → 뒤쪽 부족
        if end_date > cache_max:
            ranges.append((self._next_bday(cache_max), end_date))

        return ranges

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """parquet 파일 읽기. 손상 시 빈 DataFrame 반환."""
        try:
            if path.exists():
                return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"parquet 읽기 실패 (삭제 후 재다운로드): {path} - {e}")
            try:
                path.unlink()
            except OSError:
                pass
        return pd.DataFrame()

    def _merge_and_save(
        self, existing: pd.DataFrame, new_data: pd.DataFrame, path: Path
    ) -> pd.DataFrame:
        """기존 데이터와 신규 데이터를 병합하여 parquet 저장."""
        if existing.empty:
            merged = new_data
        elif new_data.empty:
            merged = existing
        else:
            merged = pd.concat([existing, new_data])
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()

        if not merged.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(path)

        return merged

    # ─── Public API ────────────────────────────────────────────────

    def get_price(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        fetcher=None,
    ) -> pd.DataFrame:
        """종목 가격 데이터를 캐시에서 조회하거나 fetch하여 반환한다.

        Args:
            ticker: 종목코드 (예: '005930')
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            fetcher: 가격 데이터 fetch 함수. None이면 collector.get_price_data 사용.
                     시그니처: (ticker, start, end) -> DataFrame

        Returns:
            요청 범위 전체를 커버하는 OHLCV DataFrame
        """
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        parquet_path = self._prices_dir / f"{ticker}.parquet"
        meta = self._get_meta(ticker, "price")

        missing = self._compute_missing_ranges(start_date, end_date, meta)
        if not missing:
            # 100% 캐시 히트
            df = self._read_parquet(parquet_path)
            if df.empty:
                # parquet 손상 — 메타 삭제 후 전체 재다운로드
                self._delete_meta(ticker, "price")
                missing = [(start_date, end_date)]
            else:
                ts_start = pd.Timestamp(start_date)
                ts_end = pd.Timestamp(end_date)
                return df[(df.index >= ts_start) & (df.index <= ts_end)]

        # fetch 필요
        if fetcher is None:
            from src.data.collector import get_price_data
            fetcher = get_price_data

        existing = self._read_parquet(parquet_path)
        for fetch_start, fetch_end in missing:
            try:
                new_data = fetcher(ticker, fetch_start, fetch_end)
                if not new_data.empty:
                    existing = self._merge_and_save(existing, new_data, parquet_path)
            except Exception as e:
                logger.warning(f"PriceStore fetch 실패: {ticker} ({fetch_start}~{fetch_end}) - {e}")

        # 메타 갱신
        if not existing.empty:
            min_d = existing.index.min().strftime("%Y%m%d")
            max_d = existing.index.max().strftime("%Y%m%d")
            self._set_meta(ticker, "price", min_d, max_d, len(existing))

        # 요청 범위만 반환
        if existing.empty:
            return existing
        ts_start = pd.Timestamp(start_date)
        ts_end = pd.Timestamp(end_date)
        return existing[(existing.index >= ts_start) & (existing.index <= ts_end)]

    def get_fundamentals(
        self,
        date: str,
        fetcher=None,
    ) -> pd.DataFrame:
        """특정 날짜의 전종목 펀더멘탈을 캐시에서 조회하거나 fetch한다.

        Args:
            date: 조회일 (YYYYMMDD)
            fetcher: fetch 함수. None이면 collector.get_all_fundamentals 사용.
                     시그니처: (date) -> DataFrame

        Returns:
            전종목 펀더멘탈 DataFrame
        """
        date = self._normalize_date(date)
        parquet_path = self._fundamentals_dir / f"{date}.parquet"

        if parquet_path.exists():
            df = self._read_parquet(parquet_path)
            if not df.empty:
                return df

        # fetch
        if fetcher is None:
            from src.data.collector import get_all_fundamentals
            fetcher = get_all_fundamentals

        try:
            df = fetcher(date)
        except Exception as e:
            logger.warning(f"PriceStore fundamentals fetch 실패: {date} - {e}")
            return pd.DataFrame()

        if not df.empty:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path)
            self._set_meta(date, "fundamentals", date, date, len(df))

        return df

    def get_index(
        self,
        index_name: str,
        start_date: str,
        end_date: str,
        fetcher=None,
    ) -> pd.DataFrame:
        """지수 데이터를 캐시에서 조회하거나 fetch한다.

        Args:
            index_name: 지수명 (예: 'KOSPI')
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            fetcher: fetch 함수. None이면 index_collector.get_index_data 사용.

        Returns:
            지수 OHLCV DataFrame
        """
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        parquet_path = self._index_dir / f"{index_name}.parquet"
        meta = self._get_meta(index_name, "index")

        missing = self._compute_missing_ranges(start_date, end_date, meta)
        if not missing:
            df = self._read_parquet(parquet_path)
            if df.empty:
                self._delete_meta(index_name, "index")
                missing = [(start_date, end_date)]
            else:
                ts_start = pd.Timestamp(start_date)
                ts_end = pd.Timestamp(end_date)
                return df[(df.index >= ts_start) & (df.index <= ts_end)]

        if fetcher is None:
            from src.data.index_collector import get_index_data
            fetcher = get_index_data

        existing = self._read_parquet(parquet_path)
        for fetch_start, fetch_end in missing:
            try:
                new_data = fetcher(index_name, fetch_start, fetch_end)
                if not new_data.empty:
                    existing = self._merge_and_save(existing, new_data, parquet_path)
            except Exception as e:
                logger.warning(
                    f"PriceStore index fetch 실패: {index_name} ({fetch_start}~{fetch_end}) - {e}"
                )

        if not existing.empty:
            min_d = existing.index.min().strftime("%Y%m%d")
            max_d = existing.index.max().strftime("%Y%m%d")
            self._set_meta(index_name, "index", min_d, max_d, len(existing))

        if existing.empty:
            return existing
        ts_start = pd.Timestamp(start_date)
        ts_end = pd.Timestamp(end_date)
        return existing[(existing.index >= ts_start) & (existing.index <= ts_end)]

    def clear(self, data_type: Optional[str] = None) -> None:
        """캐시를 삭제한다.

        Args:
            data_type: 'price', 'fundamentals', 'index' 중 하나.
                       None이면 전체 삭제.
        """
        import shutil

        dirs_map = {
            "price": self._prices_dir,
            "fundamentals": self._fundamentals_dir,
            "index": self._index_dir,
        }

        if data_type is None:
            # 전체 삭제
            for d in dirs_map.values():
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
            # DB 초기화
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("DELETE FROM price_metadata")
                conn.commit()
        elif data_type in dirs_map:
            d = dirs_map[data_type]
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "DELETE FROM price_metadata WHERE data_type = ?",
                    (data_type if data_type != "price" else "price",),
                )
                # fundamentals 메타는 data_type='fundamentals'로 저장
                conn.commit()
        else:
            raise ValueError(f"알 수 없는 data_type: {data_type}")

        logger.info(f"PriceStore 캐시 삭제 완료: {data_type or '전체'}")

    def get_stats(self) -> dict:
        """저장소 통계를 반환한다."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT data_type, COUNT(*), SUM(row_count) FROM price_metadata GROUP BY data_type"
            ).fetchall()

        stats: dict = {"store_dir": str(self.store_dir)}
        total_items = 0
        total_rows = 0
        for data_type, count, row_sum in rows:
            stats[data_type] = {"items": count, "total_rows": row_sum or 0}
            total_items += count
            total_rows += row_sum or 0

        stats["total_items"] = total_items
        stats["total_rows"] = total_rows

        # 디스크 사용량
        total_size = 0
        for path in self.store_dir.rglob("*.parquet"):
            total_size += path.stat().st_size
        if self._db_path.exists():
            total_size += self._db_path.stat().st_size
        stats["disk_usage_mb"] = round(total_size / (1024 * 1024), 2)

        return stats


# ─── 싱글톤 ──────────────────────────────────────────────────────

_price_store_instance: Optional[PriceStore] = None


def get_price_store(store_dir: Optional[str] = None) -> PriceStore:
    """PriceStore 싱글톤 인스턴스를 반환한다."""
    global _price_store_instance
    if _price_store_instance is None:
        _price_store_instance = PriceStore(store_dir=store_dir)
    return _price_store_instance


def reset_price_store() -> None:
    """싱글톤 인스턴스를 리셋한다 (테스트용)."""
    global _price_store_instance
    _price_store_instance = None
