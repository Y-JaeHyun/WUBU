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
        FEATURE_HELP: 피처별 상세 도움말.
    """

    FEATURE_HELP: dict[str, str] = {
        "data_cache": (
            "pykrx API 응답을 로컬에 캐싱하여 반복 호출을 줄인다.\n"
            "끄면 매번 API를 호출해서 느려지므로, 특별한 이유 없으면 켜둘 것.\n"
            "설정: cache_ttl_hours(캐시 유효시간), max_cache_size_mb(최대 용량)"
        ),
        "global_monitor": (
            "S&P500, NASDAQ, VIX 등 글로벌 시장 지표를 수집한다.\n"
            "해외 시장 동향을 리포트에 포함하고 싶을 때 켜면 된다.\n"
            "설정: include_global(글로벌 데이터 포함 여부)"
        ),
        "stock_review": (
            "보유 종목의 일일 리뷰를 수행한다. (기업 실적, 공시, 뉴스 요약)\n"
            "매일 장 마감 후 보유 종목 현황 점검에 유용.\n"
            "설정: max_stocks(리뷰 대상 최대 종목 수)"
        ),
        "auto_backtest": (
            "매주 자동으로 백테스트를 돌려 전략 성과를 점검한다.\n"
            "전략이 최근 시장에서 잘 작동하는지 모니터링할 때 유용.\n"
            "설정: lookback_months(백테스트 기간), strategies(대상 전략)"
        ),
        "night_research": (
            "야간에 글로벌 시장 동향을 수집하고 시사점을 정리한다.\n"
            "다음날 장 시작 전 해외 이슈를 파악하고 싶을 때 유용.\n"
            "설정: include_global(글로벌 동향 포함)"
        ),
        "short_term_trading": (
            "단기 트레이딩(스윙/데이) 기능을 활성화한다.\n"
            "장기 전략과 별도로 단기 매매를 병행하고 싶을 때 켠다.\n"
            "설정: long_term_pct(장기비중), short_term_pct(단기비중), "
            "stop_loss_pct, take_profit_pct 등"
        ),
        "etf_rotation": (
            "모멘텀 기반 ETF 로테이션 전략. 상위 N개 ETF에 투자한다.\n"
            "주식 팩터 전략과 별도로 ETF 자산배분을 원할 때 사용.\n"
            "설정: lookback_months(모멘텀 기간), n_select(ETF 수), "
            "etf_rotation_pct(전체 자산 중 ETF 비중)"
        ),
        "daily_simulation": (
            "일일 가상 리밸런싱을 시뮬레이션하여 포트폴리오 히스토리를 쌓는다.\n"
            "실제 매매 없이 전략 성과를 추적하고 싶을 때 유용.\n"
            "설정: strategies(시뮬 대상), primary_strategy(주력 전략)"
        ),
        "news_collector": (
            "DART 공시와 뉴스를 주기적으로 수집하여 알림한다.\n"
            "보유 종목 관련 공시를 놓치지 않으려면 켜둘 것.\n"
            "설정: check_interval_hours(수집 주기), important_only(중요 공시만),\n"
            "page_count(조회 건수), scope_filter(보유/관심 종목만 알림)"
        ),
        "macro_monitor": (
            "ECOS, FRED, VIX 등 매크로 데이터를 수집한다.\n"
            "금리/환율/유동성 등 거시 환경을 모니터링할 때 유용.\n"
            "설정: include_us_treasury(미국 국채), include_vix(변동성 지수)"
        ),
        "emergency_monitor": (
            "종목 급등락, 시장 급변, 긴급 공시를 실시간 모니터링한다.\n"
            "장중 이상 움직임을 즉시 알림 받고 싶을 때 켜둔다.\n"
            "설정: price_shock_pct(급등락 기준%), market_crash_pct(시장급변%), "
            "auto_exit_enabled(자동 청산 여부)"
        ),
        "walk_forward_backtest": (
            "Walk-Forward 방식으로 백테스트하여 과적합을 검증한다.\n"
            "전략이 진짜 유효한지 Out-of-Sample 성과로 확인할 때 사용.\n"
            "설정: train_years(훈련기간), test_years(검증기간), "
            "step_months(슬라이딩 간격)"
        ),
        "low_volatility_factor": (
            "기존 3팩터(밸류/모멘텀/퀄리티)에 저변동성 팩터를 추가한다.\n"
            "변동성 낮은 종목이 장기적으로 위험 대비 수익이 좋다는 이상현상 활용.\n"
            "설정: vol_period(변동성 계산 기간), weight(팩터 가중치)"
        ),
        "drawdown_overlay": (
            "포트폴리오가 고점 대비 하락하면 비중을 자동 축소한다.\n"
            "-10%→75%, -15%→50%, -20%→25%로 단계적 디레버리징.\n"
            "하락장 손실 제한에 효과적. vol_targeting과 겹칠 수 있으니 주의.\n"
            "설정: thresholds(하락폭별 비중), recovery_buffer(회복 버퍼)"
        ),
        "sector_neutral": (
            "팩터 스코어링 시 섹터별 균등 배분으로 쏠림을 방지한다.\n"
            "IT/반도체 등 특정 섹터에 과도하게 집중되는 걸 막고 싶을 때.\n"
            "설정: max_sector_pct(단일 섹터 최대 비중, 기본 25%)"
        ),
        "vol_targeting": (
            "포트폴리오 변동성이 목표치를 초과하면 비중을 줄인다.\n"
            "변동성이 높아지는 구간에서 자동으로 리스크를 낮춰준다.\n"
            "drawdown_overlay와 동시 사용 시 비중이 과도하게 줄 수 있음.\n"
            "설정: target_vol(목표 변동성), lookback_days(측정 기간)"
        ),
        "turnover_reduction": (
            "리밸런싱 시 불필요한 종목 교체를 줄여 거래비용을 절감한다.\n"
            "한국은 매도 시 증권거래세가 붙어서 실전 매매라면 필수.\n"
            "설정: buffer_size(버퍼존 크기), holding_bonus(보유 종목 가산점)"
        ),
        "regime_meta_model": (
            "시장 레짐(상승/하락/횡보)에 따라 팩터 가중치를 동적 조절.\n"
            "상승장→모멘텀 비중 증가, 하락장→퀄리티 비중 증가 등.\n"
            "과적합 리스크가 있으므로 충분한 백테스트 후 사용 권장.\n"
            "설정: level(rule_based/ml), update_freq(갱신 주기)"
        ),
        "enhanced_etf_rotation": (
            "기존 ETF 로테이션을 복합모멘텀+레짐필터+추세필터로 강화.\n"
            "KOSPI 200MA 기반 시장 레짐 판단, 추세 전환 확인 등 추가.\n"
            "충분한 OOS 검증 전에는 꺼두는 것을 권장.\n"
            "설정: cash_ratio_risk_off, use_vol_weight, use_market_filter 등"
        ),
    }

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
                "strategies": ["value", "momentum", "multi_factor", "three_factor", "quality"],
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
                "page_count": 300,
                "scope_filter": True,
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
        "emergency_monitor": {
            "enabled": True,
            "description": "긴급 리밸런싱 모니터 (급등락/시장급변/공시)",
            "config": {
                "price_shock_pct": 5.0,
                "market_crash_pct": 3.0,
                "auto_exit_enabled": False,
                "check_interval_minutes": 30,
            },
        },
        "walk_forward_backtest": {
            "enabled": False,
            "description": "Walk-Forward 백테스트 (OOS Sharpe 계산)",
            "config": {
                "train_years": 5,
                "test_years": 1,
                "step_months": 12,
            },
        },
        "low_volatility_factor": {
            "enabled": False,
            "description": "저변동성 팩터 (N팩터 전략 활성화)",
            "config": {
                "vol_period": 60,
                "weight": 0.15,
            },
        },
        "drawdown_overlay": {
            "enabled": False,
            "description": "드로다운 기반 디레버리징 오버레이",
            "config": {
                "thresholds": [[-0.10, 0.75], [-0.15, 0.50], [-0.20, 0.25]],
                "recovery_buffer": 0.02,
            },
        },
        "sector_neutral": {
            "enabled": False,
            "description": "섹터 중립 스코어링 (섹터 편향 제거)",
            "config": {
                "max_sector_pct": 0.25,
            },
        },
        "vol_targeting": {
            "enabled": False,
            "description": "변동성 타겟팅 (하방 변동성 기반)",
            "config": {
                "target_vol": 0.15,
                "lookback_days": 20,
                "use_downside_only": True,
            },
        },
        "turnover_reduction": {
            "enabled": False,
            "description": "회전율 감소 (버퍼 존 + 보유 보너스)",
            "config": {
                "buffer_size": 5,
                "holding_bonus": 0.1,
            },
        },
        "regime_meta_model": {
            "enabled": False,
            "description": "레짐 기반 팩터 가중치 동적 조절",
            "config": {
                "level": "rule_based",
                "update_freq": "monthly",
            },
        },
        "krx_openapi": {
            "enabled": False,
            "description": "KRX Open API 사용 (pykrx 대체)",
            "config": {
                "rate_limit_per_second": 5,
                "daily_quota": 10000,
            },
        },
        "enhanced_etf_rotation": {
            "enabled": False,
            "description": "Enhanced ETF 로테이션 (복합모멘텀+레짐필터+추세필터)",
            "config": {
                "cash_ratio_risk_off": 0.7,
                "use_vol_weight": True,
                "use_market_filter": True,
                "use_trend_filter": True,
                "use_max_drawdown_filter": True,
                "max_drawdown_filter": 0.15,
                "vol_lookback": 60,
                "trend_short_ma": 20,
                "trend_long_ma": 60,
                "market_ma_period": 200,
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

    def get_help(self, feature_name: str) -> str:
        """특정 피처의 상세 도움말을 반환한다.

        Args:
            feature_name: 피처 이름.

        Returns:
            상세 도움말 텍스트. 없으면 안내 메시지.
        """
        with self._lock:
            if feature_name not in self._data:
                return f"알 수 없는 피처: {feature_name}"

            info = self._data[feature_name]
            status = "ON" if info.get("enabled") else "OFF"
            desc = info.get("description", "")
            help_text = self.FEATURE_HELP.get(feature_name, "상세 도움말 없음")
            config = info.get("config", {})

            lines = [
                f"[{feature_name}] ({status})",
                desc,
                "",
                help_text,
            ]
            if config:
                lines.append("")
                lines.append("[현재 설정]")
                for k, v in config.items():
                    lines.append(f"  {k}: {v}")
            return "\n".join(lines)

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
            lines.append("")
            lines.append("상세: /help <피처명>")
            return "\n".join(lines)

    def reload(self) -> None:
        """디스크에서 플래그를 다시 로드한다."""
        self._load()
