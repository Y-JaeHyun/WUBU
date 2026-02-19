"""데이터 캐싱 모듈.

pykrx API 응답을 로컬 parquet 파일로 캐싱하여 반복 호출을 줄인다.
Feature Flag 'data_cache'로 on/off 제어.
"""

import functools
import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


class DataCache:
    """파일 기반 데이터 캐시.

    API 응답 DataFrame을 parquet로 캐싱한다.

    Args:
        cache_dir: 캐시 디렉토리. None이면 기본 경로.
        ttl_hours: 캐시 유효 시간 (시간). 기본 24.
        max_size_mb: 최대 캐시 크기 (MB). 기본 500.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl_hours: int = 24,
        max_size_mb: int = 500,
    ) -> None:
        self._dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_hours * 3600
        self._max_size_bytes = max_size_mb * 1024 * 1024

    @staticmethod
    def make_key(func_name: str, *args: Any, **kwargs: Any) -> str:
        """함수명+인자로 캐시 키를 생성한다.

        Args:
            func_name: 함수 이름.
            *args: 위치 인자.
            **kwargs: 키워드 인자.

        Returns:
            MD5 해시 캐시 키.
        """
        raw = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._dir / f"{key}.parquet"

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """캐시에서 DataFrame을 조회한다.

        Args:
            key: 캐시 키.

        Returns:
            캐시된 DataFrame. 만료 또는 미존재 시 None.
        """
        path = self._cache_path(key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self._ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        """DataFrame을 캐시에 저장한다.

        Args:
            key: 캐시 키.
            df: 저장할 DataFrame. 빈 경우 저장하지 않음.
        """
        if df.empty:
            return
        path = self._cache_path(key)
        try:
            df.to_parquet(path)
        except Exception as e:
            logger.warning("캐시 저장 실패: %s", e)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """캐시 크기 초과 시 오래된 파일부터 삭제."""
        files = sorted(self._dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime)
        total = sum(f.stat().st_size for f in files)
        while total > self._max_size_bytes and files:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)

    def clear(self) -> int:
        """전체 캐시를 삭제한다.

        Returns:
            삭제된 파일 수.
        """
        count = 0
        for f in self._dir.glob("*.parquet"):
            f.unlink(missing_ok=True)
            count += 1
        return count

    def get_stats(self) -> dict:
        """캐시 통계를 반환한다.

        Returns:
            file_count, total_size_mb, max_size_mb 딕셔너리.
        """
        files = list(self._dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "file_count": len(files),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "max_size_mb": self._max_size_bytes / 1024 / 1024,
        }


def cached(cache: DataCache, feature_flags: Any) -> Callable:
    """캐시 데코레이터.

    feature_flags.is_enabled('data_cache')가 True일 때만 캐시한다.

    Args:
        cache: DataCache 인스턴스.
        feature_flags: FeatureFlags 인스턴스.

    Returns:
        데코레이터 함수.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not feature_flags.is_enabled("data_cache"):
                return func(*args, **kwargs)
            key = DataCache.make_key(func.__name__, *args, **kwargs)
            result = cache.get(key)
            if result is not None:
                logger.debug("캐시 히트: %s", func.__name__)
                return result
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                cache.put(key, result)
            return result

        return wrapper

    return decorator
