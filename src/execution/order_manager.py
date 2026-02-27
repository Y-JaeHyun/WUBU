"""주문 추적 및 관리 모듈.

주문 생성, 상태 추적, 이력 관리 기능을 제공한다.
KISClient를 통해 실제 주문을 실행하고 상태를 동기화한다.
소규모 자본 최적화(SmallCapitalConfig)를 지원한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pytz

from src.execution.kis_client import KISClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

KST = pytz.timezone("Asia/Seoul")


@dataclass
class SmallCapitalConfig:
    """소규모 자본 최적화 설정.

    Attributes:
        min_order_amount: 최소 주문 금액 (원). 이 미만의 주문은 스킵.
        max_stocks: 최대 보유 종목 수. 집중 투자.
        rebalance_band: 리밸런싱 밴드 (0~1). 목표 비중 대비 이 범위 이내면 리밸런싱 스킵.
        min_alpha_cost_ratio: 최소 알파/비용 비율. 예상 알파 > 왕복 비용 × 이 비율이어야 매매.
        round_trip_cost_pct: 왕복 거래비용 비율 (매수+매도). KIS 기본 0.25%.
    """

    min_order_amount: int = 70_000
    max_stocks: int = 10
    rebalance_band: float = 0.20
    min_alpha_cost_ratio: float = 2.0
    round_trip_cost_pct: float = 0.0025


@dataclass
class Order:
    """단일 주문 정보.

    Attributes:
        ticker: 종목코드 (6자리).
        side: 매매 구분 ("buy" 또는 "sell").
        qty: 주문 수량.
        price: 주문 가격. 시장가는 0.
        order_type: 주문 유형 ("시장가" 또는 "지정가").
        status: 주문 상태 ("pending", "submitted", "filled", "partial",
                "cancelled", "failed").
        order_no: 한국투자증권 주문번호.
        filled_qty: 체결 수량.
        filled_price: 체결 평균 가격.
        created_at: 주문 생성 시각 (KST).
        updated_at: 마지막 상태 갱신 시각 (KST).
        error_msg: 오류 발생 시 메시지.
    """

    ticker: str
    side: str
    qty: int
    price: int = 0
    order_type: str = "시장가"
    status: str = "pending"
    order_no: str = ""
    filled_qty: int = 0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(KST))
    updated_at: datetime = field(default_factory=lambda: datetime.now(KST))
    error_msg: str = ""

    @property
    def is_complete(self) -> bool:
        """주문이 최종 상태인지 확인한다.

        Returns:
            체결/취소/실패 상태이면 True.
        """
        return self.status in ("filled", "cancelled", "failed")

    @property
    def is_market_order(self) -> bool:
        """시장가 주문인지 확인한다.

        Returns:
            시장가 주문이면 True.
        """
        return self.order_type == "시장가"

    @property
    def filled_amount(self) -> float:
        """체결 금액을 반환한다.

        Returns:
            체결 수량 * 체결 가격.
        """
        return self.filled_qty * self.filled_price

    def to_dict(self) -> dict:
        """주문 정보를 딕셔너리로 반환한다.

        Returns:
            주문 정보 딕셔너리.
        """
        return {
            "ticker": self.ticker,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "order_type": self.order_type,
            "status": self.status,
            "order_no": self.order_no,
            "filled_qty": self.filled_qty,
            "filled_price": self.filled_price,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error_msg": self.error_msg,
        }


class OrderManager:
    """주문 생성, 추적, 이력 관리 클래스.

    KISClient를 통해 실제 주문을 실행하고,
    각 주문의 상태를 추적하며, 이력을 보관한다.
    SmallCapitalConfig를 통해 소규모 자본 최적화를 지원한다.

    Attributes:
        client: KISClient 인스턴스.
        orders: 현재 세션의 전체 주문 리스트.
        capital_config: 소규모 자본 최적화 설정.
    """

    def __init__(
        self,
        kis_client: KISClient,
        capital_config: Optional[SmallCapitalConfig] = None,
    ) -> None:
        """OrderManager를 초기화한다.

        Args:
            kis_client: 한국투자증권 API 클라이언트.
            capital_config: 소규모 자본 최적화 설정. None이면 기본값 사용.
        """
        self.client: KISClient = kis_client
        self.orders: list[Order] = []
        self.capital_config: SmallCapitalConfig = capital_config or SmallCapitalConfig()
        logger.info("OrderManager 초기화 완료 (min_order=%d원)", self.capital_config.min_order_amount)

    def validate_order_amount(self, qty: int, price: int) -> bool:
        """주문 금액이 최소 금액 이상인지 검증한다.

        Args:
            qty: 주문 수량.
            price: 주문 가격 (시장가인 경우 예상 가격).

        Returns:
            최소 금액 이상이면 True.
        """
        if price <= 0:
            return True  # 시장가는 사전 검증 불가
        amount = qty * price
        if amount < self.capital_config.min_order_amount:
            logger.warning(
                "최소 주문 금액 미달: %d원 < %d원 (수량=%d, 가격=%d)",
                amount,
                self.capital_config.min_order_amount,
                qty,
                price,
            )
            return False
        return True

    def check_rebalance_needed(
        self, current_weight: float, target_weight: float
    ) -> bool:
        """리밸런싱이 필요한지 밴드 기준으로 확인한다.

        현재 비중과 목표 비중의 차이가 리밸런싱 밴드를 초과하면 True.

        Args:
            current_weight: 현재 비중 (0~1).
            target_weight: 목표 비중 (0~1).

        Returns:
            리밸런싱이 필요하면 True.
        """
        if target_weight <= 0:
            return current_weight > 0  # 목표 0인데 보유 중이면 매도 필요
        drift = abs(current_weight - target_weight) / target_weight
        needed = drift > self.capital_config.rebalance_band
        if not needed:
            logger.debug(
                "리밸런싱 스킵: drift=%.1f%% <= band=%.0f%%",
                drift * 100,
                self.capital_config.rebalance_band * 100,
            )
        return needed

    def check_alpha_exceeds_cost(self, expected_alpha: float) -> bool:
        """예상 알파가 왕복 거래비용의 N배를 초과하는지 확인한다.

        Args:
            expected_alpha: 예상 초과수익률 (0.01 = 1%).

        Returns:
            알파가 충분하면 True.
        """
        threshold = self.capital_config.round_trip_cost_pct * self.capital_config.min_alpha_cost_ratio
        if expected_alpha < threshold:
            logger.debug(
                "알파 부족: %.4f < %.4f (비용 %.4f × %.1f배)",
                expected_alpha,
                threshold,
                self.capital_config.round_trip_cost_pct,
                self.capital_config.min_alpha_cost_ratio,
            )
            return False
        return True

    def filter_weights_by_max_stocks(
        self, target_weights: dict[str, float]
    ) -> dict[str, float]:
        """최대 종목 수를 초과하는 경우 비중 상위 종목만 유지한다.

        Args:
            target_weights: {ticker: weight} 원래 목표 비중.

        Returns:
            최대 종목 수로 제한된 {ticker: weight} (비중 재정규화).
        """
        max_stocks = self.capital_config.max_stocks
        if len(target_weights) <= max_stocks:
            return dict(target_weights)

        # 비중 상위 max_stocks개 선택
        sorted_items = sorted(target_weights.items(), key=lambda x: x[1], reverse=True)
        top = sorted_items[:max_stocks]

        # 비중 재정규화
        total = sum(w for _, w in top)
        if total > 0:
            result = {t: w / total for t, w in top}
        else:
            result = {t: 1.0 / max_stocks for t, _ in top}

        dropped = len(target_weights) - max_stocks
        logger.info(
            "종목 수 제한: %d → %d종목 (하위 %d종목 제외)",
            len(target_weights),
            max_stocks,
            dropped,
        )
        return result

    def submit_order(
        self,
        ticker: str,
        side: str,
        qty: int,
        price: int = 0,
        order_type: str = "시장가",
    ) -> Order:
        """주문을 제출한다.

        주문을 생성하고 KIS API를 통해 실제 주문을 실행한다.
        성공 시 submitted, 실패 시 failed 상태로 기록된다.

        Args:
            ticker: 종목코드 (6자리).
            side: "buy" 또는 "sell".
            qty: 주문 수량.
            price: 주문 가격. 시장가는 0.
            order_type: "시장가" 또는 "지정가".

        Returns:
            생성된 Order 객체.
        """
        order = Order(
            ticker=ticker,
            side=side,
            qty=qty,
            price=price,
            order_type=order_type,
            status="pending",
        )

        side_kr = "매수" if side == "buy" else "매도"
        logger.info(
            "주문 제출: %s %s %d주 (%s, %s원)",
            side_kr,
            ticker,
            qty,
            order_type,
            f"{price:,}" if price > 0 else "시장가",
        )

        try:
            if side == "buy":
                result = self.client.place_buy_order(
                    ticker=ticker,
                    qty=qty,
                    price=price,
                    order_type=order_type,
                )
            elif side == "sell":
                result = self.client.place_sell_order(
                    ticker=ticker,
                    qty=qty,
                    price=price,
                    order_type=order_type,
                )
            else:
                logger.error("잘못된 매매 구분: %s", side)
                order.status = "failed"
                order.error_msg = f"잘못된 매매 구분: {side}"
                order.updated_at = datetime.now(KST)
                self.orders.append(order)
                return order

            if result.get("success"):
                order.status = "submitted"
                order.order_no = result.get("order_no", "")
                logger.info(
                    "주문 접수 성공: %s %s, 주문번호=%s",
                    side_kr,
                    ticker,
                    order.order_no,
                )
            else:
                order.status = "failed"
                order.error_msg = result.get("error", "알 수 없는 오류")
                logger.error(
                    "주문 접수 실패: %s %s - %s",
                    side_kr,
                    ticker,
                    order.error_msg,
                )

        except Exception as e:
            order.status = "failed"
            order.error_msg = str(e)
            logger.error("주문 제출 중 예외 발생: %s %s - %s", side_kr, ticker, e)

        order.updated_at = datetime.now(KST)
        self.orders.append(order)
        return order

    def cancel_order(self, order: Order) -> bool:
        """주문을 취소한다.

        Args:
            order: 취소할 Order 객체.

        Returns:
            취소 성공 여부.
        """
        if order.is_complete:
            logger.warning(
                "이미 완료된 주문은 취소할 수 없습니다: order_no=%s, status=%s",
                order.order_no,
                order.status,
            )
            return False

        if not order.order_no:
            logger.warning("주문번호가 없는 주문은 취소할 수 없습니다.")
            return False

        logger.info("주문 취소 요청: order_no=%s, ticker=%s", order.order_no, order.ticker)

        try:
            result = self.client.cancel_order(
                order_no=order.order_no,
                ticker=order.ticker,
                qty=order.qty,
            )

            if result.get("success"):
                order.status = "cancelled"
                order.updated_at = datetime.now(KST)
                logger.info("주문 취소 성공: order_no=%s", order.order_no)
                return True
            else:
                error_msg = result.get("error", "알 수 없는 오류")
                logger.error(
                    "주문 취소 실패: order_no=%s - %s",
                    order.order_no,
                    error_msg,
                )
                return False

        except Exception as e:
            logger.error(
                "주문 취소 중 예외 발생: order_no=%s - %s",
                order.order_no,
                e,
            )
            return False

    def get_pending_orders(self) -> list[Order]:
        """미체결 주문 리스트를 반환한다.

        Returns:
            pending 또는 submitted 상태인 주문 리스트.
        """
        return [
            order
            for order in self.orders
            if order.status in ("pending", "submitted", "partial")
        ]

    def get_order_history(self, date: Optional[datetime] = None) -> list[Order]:
        """주문 이력을 반환한다.

        Args:
            date: 특정 날짜의 주문만 필터링. None이면 전체 이력.

        Returns:
            주문 이력 리스트.
        """
        if date is None:
            return list(self.orders)

        target_date = date.date() if isinstance(date, datetime) else date
        return [
            order
            for order in self.orders
            if order.created_at.date() == target_date
        ]

    def sync_order_status(self) -> None:
        """미체결 주문의 상태를 KIS API에서 동기화한다.

        submitted, partial 상태인 주문의 최신 상태를 조회하여 갱신한다.
        """
        pending_orders = [
            order
            for order in self.orders
            if order.status in ("submitted", "partial")
        ]

        if not pending_orders:
            logger.debug("동기화할 미체결 주문이 없습니다.")
            return

        logger.info("미체결 주문 %d건 상태 동기화 시작", len(pending_orders))

        for order in pending_orders:
            try:
                status_info = self.client.get_order_status(order.order_no)
                new_status = status_info.get("status", "unknown")

                if new_status == "unknown":
                    logger.debug(
                        "주문 상태 확인 불가: order_no=%s", order.order_no
                    )
                    continue

                old_status = order.status
                order.status = new_status
                order.filled_qty = status_info.get("filled_qty", order.filled_qty)
                order.filled_price = status_info.get(
                    "filled_price", order.filled_price
                )
                order.updated_at = datetime.now(KST)

                if old_status != new_status:
                    logger.info(
                        "주문 상태 변경: order_no=%s, %s -> %s "
                        "(체결: %d/%d주, 체결가: %s)",
                        order.order_no,
                        old_status,
                        new_status,
                        order.filled_qty,
                        order.qty,
                        f"{order.filled_price:,.0f}" if order.filled_price > 0 else "-",
                    )

            except Exception as e:
                logger.error(
                    "주문 상태 동기화 실패: order_no=%s - %s",
                    order.order_no,
                    e,
                )

        logger.info("주문 상태 동기화 완료")

    def get_summary(self) -> dict:
        """현재 세션의 주문 요약을 반환한다.

        Returns:
            주문 요약 딕셔너리.
        """
        total = len(self.orders)
        by_status: dict[str, int] = {}
        for order in self.orders:
            by_status[order.status] = by_status.get(order.status, 0) + 1

        total_buy_amount = sum(
            order.filled_amount
            for order in self.orders
            if order.side == "buy" and order.status == "filled"
        )
        total_sell_amount = sum(
            order.filled_amount
            for order in self.orders
            if order.side == "sell" and order.status == "filled"
        )

        return {
            "total_orders": total,
            "by_status": by_status,
            "total_buy_amount": total_buy_amount,
            "total_sell_amount": total_sell_amount,
        }
