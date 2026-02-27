"""Feature Flag 시스템.

JSON 파일 기반 런타임 토글을 지원한다.
Telegram 커맨드로 제어 가능, 재시작 없이 즉시 반영.
"""

import copy
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_FLAGS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "feature_flags.json"
)


class FeatureFlags:
    """Feature Flag 관리자.

    JSON 파일에서 피처 플래그를 읽고, 런타임에 토글하며,
    변경사항을 디스크에 영속화한다. Thread-safe.

    Attributes:
        DEFAULT_FLAGS: 기본 플래그 정의.
    """

    DEFAULT_FLAGS: dict[str, dict] = {
        "data_cache": {
            "enabled": True,
            "description": "데이터 캐싱 (pykrx 응답 로컬 저장)",
            "config": {"cache_ttl_hours": 24, "max_cache_size_mb": 500},
        },
        "global_monitor": {
            "enabled": False,
            "description": "글로벌 시장 모니터 (S&P500, NASDAQ, VIX 등)",
            "config": {},
        },
        "stock_review": {
            "enabled": True,
            "description": "보유 종목 일일 리뷰",
            "config": {"max_stocks": 10},
        },
        "auto_backtest": {
            "enabled": False,
            "description": "주간 자동 백테스트",
            "config": {
                "lookback_months": 6,
                "strategies": ["value", "momentum", "multi_factor"],
            },
        },
        "night_research": {
            "enabled": False,
            "description": "야간 리서치 (글로벌 동향 + 시사점)",
            "config": {"include_global": True},
        },
        "short_term_trading": {
            "enabled": False,
            "description": "단기 트레이딩 (스윙 + 데이트레이딩)",
            "config": {
                "long_term_pct": 0.95,
                "short_term_pct": 0.05,
                "stop_loss_pct": -0.05,
                "take_profit_pct": 0.10,
                "max_concurrent_positions": 3,
                "max_daily_loss_pct": -0.03,
                "confirm_timeout_minutes": 30,
                "mode": "swing",
                "strategy": "high_breakout",
            },
        },
        "etf_rotation": {
            "enabled": True,
            "description": "ETF 로테이션 전략 (확장 유니버스)",
            "config": {
                "lookback_months": 12,
                "n_select": 3,
                "rebalance_freq": "monthly",
                "volatility_target": 0.0,
                "etf_rotation_pct": 0.30,
                "max_same_sector": 1,
                "momentum_cap": 3.0,
            },
        },
        "daily_simulation": {
            "enabled": True,
            "description": "일일 리밸런싱 시뮬레이션 (가상 포트폴리오 히스토리)",
            "config": {
                "strategies": ["multi_factor", "three_factor", "etf_rotation"],
                "report_time": "16:00",
                "primary_strategy": "multi_factor",
            },
        },
        "news_collector": {
            "enabled": True,
            "description": "DART 공시/뉴스 자동 수집 + 뉴스레터",
            "config": {
                "check_interval_hours": 1,
                "important_only": True,
            },
        },
        "macro_monitor": {
            "enabled": True,
            "description": "매크로 데이터 수집 (ECOS, FRED, VIX 등)",
            "config": {
                "include_us_treasury": True,
                "include_vix": True,
            },
        },
    }

    def __init__(self, flags_path: Optional[str] = None) -> None:
        """FeatureFlags를 초기화한다.

        Args:
            flags_path: JSON 파일 경로. None이면 기본 경로 사용.
        """
        self._path = Path(flags_path) if flags_path else _DEFAULT_FLAGS_PATH
        self._lock = threading.RLock()
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """JSON 파일에서 플래그를 로드한다."""
        with self._lock:
            try:
                if self._path.exists():
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                    self._data = raw.get("features", {})
                    # 누락된 신규 플래그는 기본값으로 보충
                    for key, default in self.DEFAULT_FLAGS.items():
                        if key not in self._data:
                            self._data[key] = copy.deepcopy(default)
                    logger.info("Feature flags 로드 완료: %d개", len(self._data))
                else:
                    self._data = copy.deepcopy(self.DEFAULT_FLAGS)
                    self._save()
                    logger.info("Feature flags 기본값으로 초기화")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Feature flags 로드 실패: %s. 기본값 사용.", e)
                self._data = copy.deepcopy(self.DEFAULT_FLAGS)

    def _save(self) -> None:
        """현재 플래그 상태를 JSON 파일에 저장한다."""
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "version": 1,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "features": self._data,
                }
                self._path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError as e:
                logger.error("Feature flags 저장 실패: %s", e)

    def is_enabled(self, feature_name: str) -> bool:
        """특정 피처가 활성화되어 있는지 확인한다.

        Args:
            feature_name: 피처 이름.

        Returns:
            활성화되어 있으면 True.
        """
        with self._lock:
            flag = self._data.get(feature_name, {})
            return bool(flag.get("enabled", False))

    def toggle(self, feature_name: str, enabled: Optional[bool] = None) -> bool:
        """피처를 토글한다.

        Args:
            feature_name: 피처 이름.
            enabled: True/False 지정. None이면 현재의 반대로 전환.

        Returns:
            성공 시 True, 알 수 없는 피처면 False.
        """
        with self._lock:
            if feature_name not in self._data:
                logger.warning("알 수 없는 피처: %s", feature_name)
                return False
            if enabled is None:
                enabled = not self._data[feature_name].get("enabled", False)
            self._data[feature_name]["enabled"] = enabled
            self._save()
            state = "활성화" if enabled else "비활성화"
            logger.info("피처 '%s' %s", feature_name, state)
            return True

    def get_config(self, feature_name: str) -> dict:
        """특정 피처의 설정값을 반환한다.

        Args:
            feature_name: 피처 이름.

        Returns:
            설정 딕셔너리 (복사본).
        """
        with self._lock:
            return dict(self._data.get(feature_name, {}).get("config", {}))

    def set_config(self, feature_name: str, key: str, value: Any) -> bool:
        """특정 피처의 설정값을 변경한다.

        Args:
            feature_name: 피처 이름.
            key: 설정 키.
            value: 설정 값.

        Returns:
            성공 시 True.
        """
        with self._lock:
            if feature_name not in self._data:
                return False
            if "config" not in self._data[feature_name]:
                self._data[feature_name]["config"] = {}
            self._data[feature_name]["config"][key] = value
            self._save()
            return True

    def get_all_status(self) -> dict[str, bool]:
        """모든 피처의 활성 상태를 반환한다.

        Returns:
            {피처이름: 활성여부} 딕셔너리.
        """
        with self._lock:
            return {k: v.get("enabled", False) for k, v in self._data.items()}

    def get_summary(self) -> str:
        """모든 피처 상태를 텍스트로 반환한다.

        Returns:
            Telegram 표시용 텍스트.
        """
        with self._lock:
            lines = ["[Feature Flags]", "-" * 30]
            for name, info in self._data.items():
                status = "ON " if info.get("enabled") else "OFF"
                desc = info.get("description", "")
                lines.append(f"  {status} | {name}: {desc}")
            return "\n".join(lines)

    def reload(self) -> None:
        """디스크에서 플래그를 다시 로드한다."""
        self._load()
