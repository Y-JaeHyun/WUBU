"""단기 트레이딩 실행 관리 모듈."""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_SIGNALS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "short_term_signals.json"
)


class ShortTermTrader:
    """단기 매매 라이프사이클 관리자.

    시그널 스캔 -> 텔레그램 알림 -> 사용자 확인 -> 주문 실행 -> 포지션 모니터링.
    """

    def __init__(
        self,
        allocator=None,           # PortfolioAllocator
        risk_manager=None,        # ShortTermRiskManager
        order_manager=None,       # OrderManager (KIS)
        strategies=None,          # list[ShortTermStrategy]
        signals_path=None,        # JSON 경로
        confirm_timeout_minutes=30,
        mode="swing",             # "swing", "daytrading", "multi"
    ):
        self._allocator = allocator
        self._risk_manager = risk_manager
        self._order_manager = order_manager
        self._strategies = strategies or []
        self._path = Path(signals_path) if signals_path else _DEFAULT_SIGNALS_PATH
        self._confirm_timeout = confirm_timeout_minutes
        self._mode = mode
        self._signals: list = []  # ShortTermSignal.to_dict() 리스트
        self._load()

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("swing", "daytrading", "multi"):
            raise ValueError(f"Invalid mode: {value}. Must be 'swing', 'daytrading', or 'multi'")
        self._mode = value
        logger.info("단기 트레이딩 모드 변경: %s", value)

    def register_strategy(self, strategy) -> None:
        """전략을 등록한다."""
        self._strategies.append(strategy)
        logger.info("단기 전략 등록: %s (mode=%s)", strategy.name, strategy.mode)

    def scan_for_signals(self, market_data=None) -> list:
        """등록된 전략들로 시그널을 스캔한다.

        Returns:
            새로 생성된 ShortTermSignal 리스트
        """
        new_signals = []
        market_data = market_data or {}

        for strategy in self._strategies:
            # Mode filtering: only scan strategies that match current mode
            if self._mode != "multi" and strategy.mode != self._mode:
                logger.debug("전략 %s 스킵 (mode=%s, 현재=%s)", strategy.name, strategy.mode, self._mode)
                continue
            try:
                signals = strategy.scan_signals(market_data)
                for sig in signals:
                    # ID 자동 생성
                    if not sig.id:
                        sig.id = f"sig_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
                    if not sig.created_at:
                        sig.created_at = datetime.now().isoformat(timespec="seconds")
                    if not sig.expires_at:
                        expires = datetime.now() + timedelta(minutes=self._confirm_timeout)
                        sig.expires_at = expires.isoformat(timespec="seconds")

                    self._signals.append(sig.to_dict())
                    new_signals.append(sig)
                    logger.info("시그널 생성: %s %s %s (conf=%.2f)",
                               sig.side, sig.ticker, sig.strategy, sig.confidence)
            except Exception as e:
                logger.error("전략 %s 스캔 실패: %s", strategy.name, e)

        if new_signals:
            self._save()

        return new_signals

    def get_active_strategies(self) -> list:
        """현재 모드에 맞는 활성 전략 목록 반환."""
        if self._mode == "multi":
            return list(self._strategies)
        return [s for s in self._strategies if s.mode == self._mode]

    def get_pending_signals(self) -> list[dict]:
        """대기 중인 시그널 목록 반환."""
        self._expire_old_signals()
        return [s for s in self._signals if s.get("state") == "pending"]

    def get_signal_by_id(self, signal_id: str) -> Optional[dict]:
        """ID로 시그널을 조회한다."""
        for s in self._signals:
            if s.get("id") == signal_id:
                return s
        return None

    def confirm_signal(self, signal_id: str) -> tuple[bool, str]:
        """사용자가 시그널을 확인(승인)한다.

        Args:
            signal_id: 시그널 ID

        Returns:
            (성공 여부, 메시지)
        """
        signal = self.get_signal_by_id(signal_id)
        if not signal:
            return False, f"시그널을 찾을 수 없습니다: {signal_id}"

        if signal.get("state") != "pending":
            return False, f"확인 불가 상태: {signal.get('state')}"

        # 만료 체크
        expires_at = signal.get("expires_at", "")
        if expires_at:
            try:
                if datetime.now() > datetime.fromisoformat(expires_at):
                    signal["state"] = "expired"
                    self._save()
                    return False, "시그널이 만료되었습니다."
            except ValueError:
                pass

        # 리스크 체크
        if self._risk_manager and self._allocator:
            current_count = len([
                s for s in self._signals
                if s.get("state") in ("confirmed", "executing")
            ])
            budget = self._allocator.get_short_term_budget()
            # 간이 주문 금액 추정 (예산의 1/max_positions)
            estimated_amount = budget // max(self._risk_manager.config.max_concurrent_positions, 1)

            can_open, reason = self._risk_manager.check_can_open(
                current_count, estimated_amount, budget
            )
            if not can_open:
                return False, f"리스크 체크 실패: {reason}"

        signal["state"] = "confirmed"
        signal["confirmed_at"] = datetime.now().isoformat(timespec="seconds")
        self._save()
        logger.info("시그널 확인: %s", signal_id)
        return True, "시그널이 확인되었습니다."

    def reject_signal(self, signal_id: str) -> tuple[bool, str]:
        """시그널을 거절한다."""
        signal = self.get_signal_by_id(signal_id)
        if not signal:
            return False, f"시그널을 찾을 수 없습니다: {signal_id}"

        if signal.get("state") != "pending":
            return False, f"거절 불가 상태: {signal.get('state')}"

        signal["state"] = "rejected"
        self._save()
        logger.info("시그널 거절: %s", signal_id)
        return True, "시그널이 거절되었습니다."

    def execute_confirmed_signals(self) -> list[dict]:
        """확인된 시그널들을 실행한다.

        Returns:
            실행 결과 리스트 [{signal_id, success, order, error}, ...]
        """
        results = []
        confirmed = [s for s in self._signals if s.get("state") == "confirmed"]

        for signal in confirmed:
            signal["state"] = "executing"
            self._save()

            result = {"signal_id": signal["id"], "success": False, "order": None, "error": ""}

            try:
                if self._order_manager is None:
                    result["error"] = "OrderManager가 설정되지 않았습니다."
                    signal["state"] = "confirmed"  # 롤백
                    results.append(result)
                    continue

                # 주문 수량 계산
                ticker = signal.get("ticker", "")
                side = signal.get("side", "buy")

                if side == "buy" and self._allocator:
                    budget = self._allocator.get_short_term_cash()
                    max_positions = 3
                    if self._risk_manager:
                        max_positions = self._risk_manager.config.max_concurrent_positions
                    per_position = budget // max(max_positions, 1)

                    # 현재가 조회
                    target_price = signal.get("target_price", 0)
                    if target_price > 0:
                        qty = int(per_position / target_price)
                    else:
                        qty = 0  # 시장가인 경우 order_manager가 처리

                    if qty <= 0:
                        result["error"] = f"주문 수량 0: 예산 부족 (가용={per_position:,}원)"
                        signal["state"] = "done"
                        results.append(result)
                        continue
                else:
                    qty = signal.get("metadata", {}).get("qty", 0)

                # 주문 실행
                order = self._order_manager.submit_order(
                    ticker=ticker,
                    side=side,
                    qty=qty,
                    order_type="시장가",
                )

                result["order"] = order.to_dict() if hasattr(order, 'to_dict') else str(order)
                result["success"] = True
                signal["state"] = "done"
                signal["executed_at"] = datetime.now().isoformat(timespec="seconds")

                # 포지션 태깅
                if self._allocator and side == "buy":
                    self._allocator.tag_position(
                        ticker, "short_term",
                        metadata={
                            "strategy": signal.get("strategy", ""),
                            "mode": signal.get("mode", "swing"),
                            "entry_price": signal.get("target_price", 0),
                            "signal_id": signal["id"],
                        }
                    )

                logger.info("시그널 실행 완료: %s %s %s", side, ticker, signal["id"])

            except Exception as e:
                logger.error("시그널 실행 실패: %s - %s", signal["id"], e)
                result["error"] = str(e)
                signal["state"] = "confirmed"  # 재시도 가능하도록

            results.append(result)

        self._save()
        return results

    def _expire_old_signals(self) -> None:
        """만료된 시그널 상태를 업데이트한다."""
        now = datetime.now()
        changed = False
        for signal in self._signals:
            if signal.get("state") != "pending":
                continue
            expires_at = signal.get("expires_at", "")
            if expires_at:
                try:
                    if now > datetime.fromisoformat(expires_at):
                        signal["state"] = "expired"
                        changed = True
                except ValueError:
                    pass
        if changed:
            self._save()

    def get_all_signals(self) -> list[dict]:
        """모든 시그널 반환."""
        return list(self._signals)

    def get_active_signals(self) -> list[dict]:
        """활성 시그널 (pending, confirmed, executing) 반환."""
        return [s for s in self._signals if s.get("state") in ("pending", "confirmed", "executing")]

    def cleanup_old_signals(self, max_age_days: int = 7) -> int:
        """오래된 완료/만료 시그널을 정리한다.

        Returns:
            삭제된 시그널 수
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        original_count = len(self._signals)

        self._signals = [
            s for s in self._signals
            if s.get("state") in ("pending", "confirmed", "executing")
            or self._signal_created_after(s, cutoff)
        ]

        removed = original_count - len(self._signals)
        if removed > 0:
            self._save()
            logger.info("오래된 시그널 %d개 정리", removed)
        return removed

    def _signal_created_after(self, signal: dict, cutoff: datetime) -> bool:
        """시그널이 cutoff 이후에 생성되었는지 확인."""
        created = signal.get("created_at", "")
        if created:
            try:
                return datetime.fromisoformat(created) > cutoff
            except ValueError:
                pass
        return True  # 파싱 실패 시 보존

    # -- 영속화 --

    def _load(self) -> None:
        """JSON에서 시그널 로드."""
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._signals = raw.get("signals", [])
                logger.info("단기 시그널 %d개 로드", len(self._signals))
            else:
                self._signals = []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("시그널 로드 실패: %s", e)
            self._signals = []

    def _save(self) -> None:
        """시그널을 JSON에 저장."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "signals": self._signals,
            }
            self._path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error("시그널 저장 실패: %s", e)
