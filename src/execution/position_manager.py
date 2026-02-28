"""포지션 조회 및 동기화 모듈.

현재 보유 포지션과 목표 포지션을 비교하여
리밸런싱에 필요한 매매 주문을 계산한다.
"""

from __future__ import annotations

import math
from typing import Optional

import pytz

from src.execution.kis_client import KISClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

KST = pytz.timezone("Asia/Seoul")


class PositionManager:
    """현재 포지션과 목표 포지션 비교/관리 클래스.

    KISClient를 통해 실제 보유 포지션을 조회하고,
    목표 비중과 비교하여 리밸런싱 주문을 생성한다.

    Attributes:
        client: KISClient 인스턴스.
    """

    # 최소 주문 금액 (이 금액 이하의 차이는 무시)
    MIN_ORDER_AMOUNT = 50_000  # 5만원

    def __init__(self, kis_client: KISClient) -> None:
        """PositionManager를 초기화한다.

        Args:
            kis_client: 한국투자증권 API 클라이언트.
        """
        self.client: KISClient = kis_client
        logger.info("PositionManager 초기화 완료")

    def get_current_positions(self) -> dict[str, int]:
        """현재 보유 종목의 {종목코드: 수량} 딕셔너리를 반환한다.

        Returns:
            {ticker: qty} 형태의 딕셔너리.
            보유 종목이 없으면 빈 딕셔너리.
        """
        try:
            positions_df = self.client.get_positions()

            if positions_df.empty:
                logger.info("현재 보유 종목 없음")
                return {}

            result: dict[str, int] = {}
            for _, row in positions_df.iterrows():
                ticker = str(row["ticker"])
                qty = int(row["qty"])
                if qty > 0:
                    result[ticker] = qty

            logger.info("현재 보유 종목: %d개", len(result))
            return result

        except Exception as e:
            logger.error("포지션 조회 실패: %s", e)
            return {}

    def get_portfolio_value(self) -> int:
        """총 포트폴리오 가치(보유종목 + 현금)를 반환한다.

        Returns:
            총 포트폴리오 가치 (원).
        """
        try:
            balance = self.client.get_balance()

            total_eval = balance.get("total_eval", 0)
            cash = balance.get("cash", 0)

            # total_eval이 이미 현금 포함인 경우도 있으므로 큰 쪽 사용
            portfolio_value = max(total_eval, cash)

            # total_eval이 0이면 (보유 종목이 없으면) 현금만
            if total_eval == 0:
                portfolio_value = cash

            logger.info("포트폴리오 가치: %s원", f"{portfolio_value:,}")
            return portfolio_value

        except Exception as e:
            logger.error("포트폴리오 가치 조회 실패: %s", e)
            return 0

    def _get_current_prices(self, tickers: list[str]) -> dict[str, int]:
        """종목 리스트의 현재가를 조회한다.

        Args:
            tickers: 종목코드 리스트.

        Returns:
            {ticker: price} 딕셔너리.
        """
        prices: dict[str, int] = {}

        for ticker in tickers:
            try:
                price_info = self.client.get_current_price(ticker)
                price = price_info.get("price", 0)
                if price > 0:
                    prices[ticker] = price
                else:
                    logger.warning("현재가 0원: ticker=%s", ticker)
            except Exception as e:
                logger.error("현재가 조회 실패: ticker=%s - %s", ticker, e)

        return prices

    def calculate_target_quantities(
        self,
        target_weights: dict[str, float],
        total_value: int,
    ) -> dict[str, int]:
        """목표 비중을 목표 수량으로 변환한다.

        각 종목의 현재가를 조회하여 비중을 수량으로 변환한다.
        소수점 이하는 내림 처리한다.

        Args:
            target_weights: {ticker: weight} 형태. weight는 0~1 사이 비율.
            total_value: 총 포트폴리오 가치 (원).

        Returns:
            {ticker: qty} 형태의 목표 수량 딕셔너리.
        """
        if not target_weights or total_value <= 0:
            logger.warning(
                "목표 비중 또는 포트폴리오 가치가 유효하지 않습니다. "
                "weights=%d개, total_value=%s",
                len(target_weights),
                f"{total_value:,}",
            )
            return {}

        # 비중 합 검증
        total_weight = sum(target_weights.values())
        if total_weight > 1.01:
            logger.warning(
                "목표 비중 합이 1을 초과합니다: %.4f. 정규화합니다.",
                total_weight,
            )
            target_weights = {
                t: w / total_weight for t, w in target_weights.items()
            }

        # 현재가 조회
        tickers = list(target_weights.keys())
        prices = self._get_current_prices(tickers)

        target_quantities: dict[str, int] = {}

        for ticker, weight in target_weights.items():
            if weight <= 0:
                continue

            price = prices.get(ticker, 0)
            if price <= 0:
                logger.warning(
                    "현재가를 조회할 수 없어 종목을 제외합니다: %s", ticker
                )
                continue

            target_amount = total_value * weight
            qty = math.floor(target_amount / price)

            if qty > 0:
                target_quantities[ticker] = qty
                logger.debug(
                    "목표 수량 계산: %s = %d주 "
                    "(비중=%.2f%%, 금액=%s원, 가격=%s원)",
                    ticker,
                    qty,
                    weight * 100,
                    f"{target_amount:,.0f}",
                    f"{price:,}",
                )

        logger.info(
            "목표 수량 계산 완료: %d개 종목 (총 투자 비중: %.1f%%)",
            len(target_quantities),
            sum(target_weights.get(t, 0) for t in target_quantities) * 100,
        )

        return target_quantities

    def calculate_rebalance_orders(
        self,
        target_weights: dict[str, float],
        allocator=None,
        pool: str | None = None,
        integrated: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """현재 포지션과 목표 비중을 비교하여 리밸런싱 주문을 생성한다.

        매도 주문과 매수 주문을 분리하여 반환한다.
        매도를 먼저 실행해야 매수 자금이 확보되므로 분리한다.

        Args:
            target_weights: {ticker: weight} 형태. weight는 0~1 사이 비율.
            allocator: PortfolioAllocator 인스턴스 (선택).
                있으면 다른 풀의 포지션을 리밸런싱 대상에서 제외한다.
            pool: 리밸런싱 대상 풀. None이면 "long_term"으로 동작.
            integrated: True이면 통합 모드로 풀 제외 로직을 바이패스한다.

        Returns:
            (sell_orders, buy_orders) 튜플.
            각 주문은 {"ticker": str, "side": str, "qty": int} 형태.
        """
        # 1. 현재 포지션 및 포트폴리오 가치 조회
        current_positions = self.get_current_positions()
        portfolio_value = self.get_portfolio_value()

        if portfolio_value <= 0:
            logger.error("포트폴리오 가치가 0원입니다. 리밸런싱을 중단합니다.")
            return [], []

        # 1-1. allocator가 있으면 다른 풀의 포지션을 현재 포지션에서 제외
        # 통합 모드에서는 전체 포트폴리오를 대상으로 하므로 제외하지 않음
        if allocator is not None and not integrated:
            all_pools = {"long_term", "short_term", "etf_rotation"}
            target_pool = pool or "long_term"
            exclude_pools = all_pools - {target_pool}

            exclude_tickers: set[str] = set()
            for excl_pool in exclude_pools:
                tickers = {
                    p["ticker"]
                    for p in allocator.get_positions_by_pool(excl_pool)
                }
                exclude_tickers |= tickers

            if exclude_tickers:
                excluded = {
                    t for t in current_positions if t in exclude_tickers
                }
                current_positions = {
                    t: q for t, q in current_positions.items()
                    if t not in exclude_tickers
                }
                if excluded:
                    logger.info(
                        "allocator: %s 외 풀 포지션 %d개 제외: %s",
                        target_pool,
                        len(excluded),
                        ", ".join(sorted(excluded)),
                    )

        # 2. 목표 수량 계산
        target_quantities = self.calculate_target_quantities(
            target_weights, portfolio_value
        )

        # 3. 현재 보유 종목과 목표를 비교
        all_tickers = set(current_positions.keys()) | set(target_quantities.keys())
        sell_orders: list[dict] = []
        buy_orders: list[dict] = []

        # 현재가 조회 (차이 금액 계산용)
        prices = self._get_current_prices(list(all_tickers))

        for ticker in all_tickers:
            current_qty = current_positions.get(ticker, 0)
            target_qty = target_quantities.get(ticker, 0)
            diff = target_qty - current_qty

            if diff == 0:
                continue

            # 최소 주문 금액 확인
            price = prices.get(ticker, 0)
            order_amount = abs(diff) * price if price > 0 else 0

            if order_amount < self.MIN_ORDER_AMOUNT and price > 0:
                logger.debug(
                    "최소 주문 금액 미달로 스킵: %s (차이=%d주, 금액=%s원)",
                    ticker,
                    diff,
                    f"{order_amount:,}",
                )
                continue

            if diff < 0:
                # 매도: 현재가 목표보다 많음
                sell_qty = abs(diff)
                sell_orders.append({
                    "ticker": ticker,
                    "side": "sell",
                    "qty": sell_qty,
                    "amount": sell_qty * price if price > 0 else 0,
                })
                logger.info(
                    "매도 주문 생성: %s %d주 (현재=%d, 목표=%d)",
                    ticker,
                    sell_qty,
                    current_qty,
                    target_qty,
                )
            else:
                # 매수: 현재가 목표보다 적음
                buy_qty = diff
                buy_orders.append({
                    "ticker": ticker,
                    "side": "buy",
                    "qty": buy_qty,
                    "amount": buy_qty * price if price > 0 else 0,
                })
                logger.info(
                    "매수 주문 생성: %s %d주 (현재=%d, 목표=%d)",
                    ticker,
                    buy_qty,
                    current_qty,
                    target_qty,
                )

        total_sell = sum(o["amount"] for o in sell_orders)
        total_buy = sum(o["amount"] for o in buy_orders)

        logger.info(
            "리밸런싱 주문 계산 완료: 매도 %d건 (%s원), 매수 %d건 (%s원)",
            len(sell_orders),
            f"{total_sell:,}",
            len(buy_orders),
            f"{total_buy:,}",
        )

        return sell_orders, buy_orders
