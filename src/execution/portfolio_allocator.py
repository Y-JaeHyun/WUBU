"""포트폴리오 할당 관리 모듈.

KIS는 서브계좌를 지원하지 않으므로 JSON 기반 포지션 태깅으로
장기(long_term)와 단기(short_term) 풀을 논리적으로 분리한다.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_ALLOCATION_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "portfolio_allocation.json"
)


class PortfolioAllocator:
    """포트폴리오 풀 할당 관리자.

    하나의 KIS 계좌를 장기(long_term)와 단기(short_term)로 논리 분리한다.
    포지션 태깅 정보를 JSON 파일에 영속화하며, thread-safe 하게 동작한다.

    Attributes:
        ALLOCATION_PATH: 기본 JSON 저장 경로.
    """

    ALLOCATION_PATH = "data/portfolio_allocation.json"

    def __init__(
        self,
        kis_client,
        long_term_pct: float = 0.90,
        short_term_pct: float = 0.10,
        allocation_path: Optional[str] = None,
    ) -> None:
        """PortfolioAllocator를 초기화한다.

        Args:
            kis_client: KIS OpenAPI 클라이언트 (get_balance, get_current_price 지원).
            long_term_pct: 장기 풀 비율 (0~1). 기본 0.90.
            short_term_pct: 단기 풀 비율 (0~1). 기본 0.10.
            allocation_path: JSON 파일 경로. None이면 기본 경로 사용.
        """
        self._kis_client = kis_client
        self._long_term_pct = long_term_pct
        self._short_term_pct = short_term_pct
        self._path = Path(allocation_path) if allocation_path else _DEFAULT_ALLOCATION_PATH
        self._lock = threading.RLock()
        self._data: dict = {}
        self._cached_balance: Optional[dict] = None
        self._load()

    # ──────────────────────────────────────────────────────────
    # 예산 조회
    # ──────────────────────────────────────────────────────────

    def get_total_portfolio_value(self) -> int:
        """KIS 잔고에서 총 포트폴리오 가치를 조회한다.

        KIS 점검시간(01:00~05:00) 등 조회 실패 시 캐시 값을 사용한다.

        Returns:
            총 평가금액 (원).
        """
        try:
            balance = self._kis_client.get_balance()
            total = balance.get("total_eval", 0)
            if total > 0:
                self._cached_balance = balance
            return total
        except Exception as e:
            logger.warning("KIS 잔고 조회 실패, 캐시 사용: %s", e)
            if self._cached_balance:
                return self._cached_balance.get("total_eval", 0)
            return 0

    def get_long_term_budget(self) -> int:
        """장기 풀 예산을 반환한다.

        Returns:
            총 포트폴리오 가치 * 장기 비율 (원).
        """
        return int(self.get_total_portfolio_value() * self._long_term_pct)

    def get_short_term_budget(self) -> int:
        """단기 풀 예산을 반환한다.

        Returns:
            총 포트폴리오 가치 * 단기 비율 (원).
        """
        return int(self.get_total_portfolio_value() * self._short_term_pct)

    def get_short_term_cash(self) -> int:
        """단기 풀의 사용 가능 현금을 반환한다.

        단기 예산에서 단기 포지션들의 평가금액 합계를 뺀 값.

        Returns:
            단기 풀 가용 현금 (원). 음수면 0.
        """
        budget = self.get_short_term_budget()
        positions = self.get_positions_by_pool("short_term")
        used = sum(p.get("eval_amount", 0) for p in positions)
        return max(budget - used, 0)

    def get_long_term_cash(self) -> int:
        """장기 풀의 사용 가능 현금을 반환한다.

        장기 예산에서 장기 포지션들의 평가금액 합계를 뺀 값.

        Returns:
            장기 풀 가용 현금 (원). 음수면 0.
        """
        budget = self.get_long_term_budget()
        positions = self.get_positions_by_pool("long_term")
        used = sum(p.get("eval_amount", 0) for p in positions)
        return max(budget - used, 0)

    # ──────────────────────────────────────────────────────────
    # 포지션 태깅
    # ──────────────────────────────────────────────────────────

    def tag_position(
        self,
        ticker: str,
        pool: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """종목을 특정 풀에 태깅한다.

        Args:
            ticker: 종목코드 (6자리).
            pool: 'long_term' 또는 'short_term'.
            metadata: 추가 메타데이터 (entry_price, strategy 등).

        Raises:
            ValueError: pool이 'long_term' 또는 'short_term'이 아닌 경우.
        """
        if pool not in ("long_term", "short_term"):
            raise ValueError(f"pool은 'long_term' 또는 'short_term'이어야 합니다: {pool}")

        with self._lock:
            positions = self._data.get("positions", {})
            entry = {
                "pool": pool,
                "entry_date": datetime.now().strftime("%Y-%m-%d"),
            }
            if metadata:
                entry.update(metadata)
            positions[ticker] = entry
            self._data["positions"] = positions
            self._save()
            logger.info("포지션 태깅: %s -> %s", ticker, pool)

    def untag_position(self, ticker: str) -> None:
        """종목의 풀 태그를 제거한다.

        Args:
            ticker: 종목코드.
        """
        with self._lock:
            positions = self._data.get("positions", {})
            if ticker in positions:
                del positions[ticker]
                self._data["positions"] = positions
                self._save()
                logger.info("포지션 태그 제거: %s", ticker)

    def get_positions_by_pool(self, pool: str) -> list[dict]:
        """특정 풀에 속한 포지션 목록을 반환한다.

        KIS 실시간 평가 데이터를 포함한다.

        Args:
            pool: 'long_term' 또는 'short_term'.

        Returns:
            포지션 딕셔너리 리스트. 각 항목은 ticker, pool, eval_amount 등을 포함.
        """
        with self._lock:
            positions = self._data.get("positions", {})
            result = []

            # KIS 잔고 조회 (한 번만)
            try:
                balance = self._kis_client.get_balance()
                holdings_map = {
                    h["ticker"]: h for h in balance.get("holdings", [])
                }
            except Exception as e:
                logger.warning("잔고 조회 실패: %s", e)
                holdings_map = {}

            for ticker, info in positions.items():
                if info.get("pool") != pool:
                    continue
                entry = dict(info)
                entry["ticker"] = ticker
                # KIS 실시간 데이터 병합
                holding = holdings_map.get(ticker, {})
                entry["eval_amount"] = holding.get("eval_amount", 0)
                entry["current_price"] = holding.get("current_price", 0)
                entry["qty"] = holding.get("qty", 0)
                entry["pnl"] = holding.get("pnl", 0)
                entry["pnl_pct"] = holding.get("pnl_pct", 0.0)
                result.append(entry)

            return result

    def get_position_pool(self, ticker: str) -> Optional[str]:
        """종목이 속한 풀을 반환한다.

        Args:
            ticker: 종목코드.

        Returns:
            'long_term', 'short_term', 또는 태깅되지 않았으면 None.
        """
        with self._lock:
            positions = self._data.get("positions", {})
            info = positions.get(ticker)
            if info:
                return info.get("pool")
            return None

    # ──────────────────────────────────────────────────────────
    # 리밸런싱 통합
    # ──────────────────────────────────────────────────────────

    def filter_long_term_weights(
        self, target_weights: dict[str, float]
    ) -> dict[str, float]:
        """장기 전략 weights를 장기 풀 비율에 맞게 스케일링한다.

        예: 10개 종목 각 10% -> 장기 90%이면 각 9%로 조정.

        Args:
            target_weights: {ticker: weight} 형태의 원래 목표 비중 (합계 ~1.0).

        Returns:
            장기 풀 비율로 스케일된 {ticker: weight} 딕셔너리.
        """
        if not target_weights:
            return {}

        scaled = {}
        for ticker, weight in target_weights.items():
            scaled[ticker] = round(weight * self._long_term_pct, 6)
        return scaled

    def rebalance_allocation(self) -> dict:
        """월간 리밸런싱 시 풀 비율을 재조정한다.

        soft cap 모드: 초과 성장은 허용하되, 신규 투자 시 목표 비율을 적용.

        Returns:
            재조정 결과 딕셔너리:
            {
                "long_term_target": float,
                "short_term_target": float,
                "long_term_actual": float,
                "short_term_actual": float,
                "long_term_eval": int,
                "short_term_eval": int,
                "total_eval": int,
                "rebalance_needed": bool,
                "drift_pct": float,
            }
        """
        total = self.get_total_portfolio_value()
        if total <= 0:
            return {
                "long_term_target": self._long_term_pct,
                "short_term_target": self._short_term_pct,
                "long_term_actual": 0.0,
                "short_term_actual": 0.0,
                "long_term_eval": 0,
                "short_term_eval": 0,
                "total_eval": 0,
                "rebalance_needed": False,
                "drift_pct": 0.0,
            }

        long_positions = self.get_positions_by_pool("long_term")
        short_positions = self.get_positions_by_pool("short_term")

        long_eval = sum(p.get("eval_amount", 0) for p in long_positions)
        short_eval = sum(p.get("eval_amount", 0) for p in short_positions)

        long_actual = long_eval / total if total > 0 else 0.0
        short_actual = short_eval / total if total > 0 else 0.0

        # drift: 실제 비율과 목표 비율의 차이 (절대값)
        drift = abs(long_actual - self._long_term_pct)

        # 5%p 이상 drift면 리밸런싱 필요
        rebalance_needed = drift >= 0.05

        result = {
            "long_term_target": self._long_term_pct,
            "short_term_target": self._short_term_pct,
            "long_term_actual": round(long_actual, 4),
            "short_term_actual": round(short_actual, 4),
            "long_term_eval": long_eval,
            "short_term_eval": short_eval,
            "total_eval": total,
            "rebalance_needed": rebalance_needed,
            "drift_pct": round(drift * 100, 2),
        }

        logger.info(
            "풀 할당 현황: 장기=%.1f%% (목표 %.0f%%), 단기=%.1f%% (목표 %.0f%%), drift=%.2f%%p",
            long_actual * 100,
            self._long_term_pct * 100,
            short_actual * 100,
            self._short_term_pct * 100,
            drift * 100,
        )

        return result

    # ──────────────────────────────────────────────────────────
    # 영속화
    # ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """JSON 파일에서 할당 데이터를 로드한다."""
        with self._lock:
            try:
                if self._path.exists():
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                    self._data = raw
                    # config에서 비율 복원
                    config = raw.get("config", {})
                    if "long_term_pct" in config:
                        self._long_term_pct = config["long_term_pct"]
                    if "short_term_pct" in config:
                        self._short_term_pct = config["short_term_pct"]
                    logger.info(
                        "포트폴리오 할당 로드: %d개 포지션",
                        len(self._data.get("positions", {})),
                    )
                else:
                    self._data = {
                        "version": 1,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "config": {
                            "long_term_pct": self._long_term_pct,
                            "short_term_pct": self._short_term_pct,
                            "soft_cap_mode": True,
                        },
                        "positions": {},
                    }
                    self._save()
                    logger.info("포트폴리오 할당 기본값으로 초기화")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("포트폴리오 할당 로드 실패: %s. 기본값 사용.", e)
                self._data = {
                    "version": 1,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "config": {
                        "long_term_pct": self._long_term_pct,
                        "short_term_pct": self._short_term_pct,
                        "soft_cap_mode": True,
                    },
                    "positions": {},
                }

    def _save(self) -> None:
        """현재 할당 데이터를 JSON 파일에 저장한다."""
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._data["version"] = 1
                self._data["updated_at"] = datetime.now().isoformat(timespec="seconds")
                self._data["config"] = {
                    "long_term_pct": self._long_term_pct,
                    "short_term_pct": self._short_term_pct,
                    "soft_cap_mode": True,
                }
                self._path.write_text(
                    json.dumps(self._data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError as e:
                logger.error("포트폴리오 할당 저장 실패: %s", e)
