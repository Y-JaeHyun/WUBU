"""전략 시그널 실행 모듈.

전략이 생성한 target_weights를 받아 실제 주문으로 변환하고 실행한다.
매도 -> 매수 순서로 실행하여 매수 자금을 확보한다.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import pytz

from src.execution.kis_client import KISClient
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.execution.risk_guard import RiskGuard
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.execution.portfolio_allocator import PortfolioAllocator

logger = get_logger(__name__)

KST = pytz.timezone("Asia/Seoul")


class RebalanceExecutor:
    """리밸런싱 실행기.

    전략의 target_weights를 받아 실제 주문을 실행한다.
    매도 -> 매수 순서로 실행하여 자금을 확보한다.

    Attributes:
        kis_client: KIS OpenAPI 클라이언트.
        order_manager: 주문 관리기.
        position_manager: 포지션 관리기.
        risk_guard: 리스크 체크 모듈 (선택).
    """

    # 주문 간 대기 시간 (초)
    ORDER_DELAY = 0.5

    def __init__(
        self,
        kis_client: KISClient,
        risk_guard: Optional[RiskGuard] = None,
        allocator: Optional[PortfolioAllocator] = None,
    ) -> None:
        """RebalanceExecutor를 초기화한다.

        Args:
            kis_client: 한국투자증권 API 클라이언트.
            risk_guard: 사전 리스크 체크 객체. None이면 기본값으로 생성.
            allocator: 포트폴리오 할당 관리자. None이면 기존 동작 (전체 포트폴리오).
        """
        self.kis_client: KISClient = kis_client
        self.order_manager: OrderManager = OrderManager(kis_client)
        self.position_manager: PositionManager = PositionManager(kis_client)
        self.risk_guard: RiskGuard = risk_guard or RiskGuard()
        self.allocator = allocator

        logger.info("RebalanceExecutor 초기화 완료")

    def execute_rebalance(
        self,
        target_weights: dict[str, float],
        pool: str | None = None,
    ) -> dict:
        """리밸런싱을 실행한다.

        1. 현재 포지션 조회
        2. 목표 vs 현재 차이 계산
        3. risk_guard 체크
        4. 매도 먼저 실행
        5. 매수 실행
        6. 결과 리포트 반환

        Args:
            target_weights: {ticker: weight} 형태의 목표 비중.
                weight는 0~1 사이 비율.
            pool: 리밸런싱 대상 풀 ("long_term", "etf_rotation" 등).
                None이면 "long_term"으로 동작.

        Returns:
            실행 결과 딕셔너리.
        """
        start_time = datetime.now(KST)
        result = {
            "success": False,
            "timestamp": start_time.isoformat(),
            "sells": [],
            "buys": [],
            "total_sell_amount": 0,
            "total_buy_amount": 0,
            "errors": [],
            "skipped": [],
        }

        mode_tag = self.kis_client.mode_tag
        logger.info(
            "=" * 60 + "\n%s 리밸런싱 실행 시작: %d개 종목, 시각=%s",
            mode_tag,
            len(target_weights),
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # 0. KIS 클라이언트 설정 확인
        if not self.kis_client.is_configured():
            msg = "KIS API가 설정되지 않았습니다. 리밸런싱을 중단합니다."
            logger.error(msg)
            result["errors"].append(msg)
            return result

        # 0-1. 실전 모드 안전장치
        if not self.kis_client.is_paper:
            confirmed = os.getenv("KIS_LIVE_CONFIRMED", "false").lower()
            if confirmed not in ("true", "1", "yes"):
                msg = (
                    "실전 모드 리밸런싱이 차단되었습니다. "
                    "KIS_LIVE_CONFIRMED=true를 .env에 설정하세요."
                )
                logger.error(msg)
                result["errors"].append(msg)
                return result

        # 1. 리스크 검증 (전체 리밸런싱)
        risk_passed, risk_warnings = self.risk_guard.check_rebalance(target_weights)
        if not risk_passed:
            msg = f"리스크 검증 실패: {risk_warnings}"
            logger.error(msg)
            result["errors"].append(msg)
            return result

        if risk_warnings:
            for w in risk_warnings:
                logger.warning("리스크 경고: %s", w)

        # 2. 리밸런싱 주문 계산 (allocator가 있으면 풀 비중으로 스케일링)
        effective_weights = target_weights
        if self.allocator is not None:
            if pool == "etf_rotation":
                effective_weights = self.allocator.filter_etf_rotation_weights(
                    target_weights
                )
            else:
                effective_weights = self.allocator.filter_long_term_weights(
                    target_weights
                )
            logger.info(
                "allocator 적용 (pool=%s): 원래 비중 합=%.4f -> 스케일 비중 합=%.4f",
                pool or "long_term",
                sum(target_weights.values()),
                sum(effective_weights.values()),
            )

        sell_orders, buy_orders = self.position_manager.calculate_rebalance_orders(
            effective_weights, allocator=self.allocator, pool=pool
        )

        if not sell_orders and not buy_orders:
            logger.info("리밸런싱 필요 없음: 현재 포지션이 목표와 동일합니다.")
            result["success"] = True
            return result

        # 3. 회전율 검증
        portfolio_value = self.position_manager.get_portfolio_value()
        turnover_passed, turnover_reason = self.risk_guard.check_turnover(
            sell_orders, buy_orders, portfolio_value
        )
        if not turnover_passed:
            logger.error("회전율 초과: %s", turnover_reason)
            result["errors"].append(turnover_reason)
            return result

        # 4. 개별 주문 리스크 검증 및 매도 실행
        logger.info("매도 주문 %d건 실행 시작", len(sell_orders))
        for order_spec in sell_orders:
            # 개별 주문 리스크 검증
            order_passed, order_reason = self.risk_guard.check_order(
                order_spec, portfolio_value
            )
            if not order_passed:
                result["skipped"].append(
                    f"매도 {order_spec['ticker']}: {order_reason}"
                )
                logger.warning("매도 주문 스킵: %s", order_reason)
                continue

            order = self.order_manager.submit_order(
                ticker=order_spec["ticker"],
                side="sell",
                qty=order_spec["qty"],
                order_type="시장가",
            )

            result["sells"].append(order.to_dict())

            if order.status == "failed":
                result["errors"].append(
                    f"매도 실패 {order.ticker}: {order.error_msg}"
                )

            time.sleep(self.ORDER_DELAY)

        # 5. 매도 체결 대기 (간략 대기)
        if sell_orders:
            logger.info("매도 체결 대기 중 (3초)...")
            time.sleep(3.0)
            self.order_manager.sync_order_status()

        # 6. 매수 실행
        logger.info("매수 주문 %d건 실행 시작", len(buy_orders))
        for order_spec in buy_orders:
            # 개별 주문 리스크 검증
            order_passed, order_reason = self.risk_guard.check_order(
                order_spec, portfolio_value
            )
            if not order_passed:
                result["skipped"].append(
                    f"매수 {order_spec['ticker']}: {order_reason}"
                )
                logger.warning("매수 주문 스킵: %s", order_reason)
                continue

            order = self.order_manager.submit_order(
                ticker=order_spec["ticker"],
                side="buy",
                qty=order_spec["qty"],
                order_type="시장가",
            )

            result["buys"].append(order.to_dict())

            if order.status == "failed":
                result["errors"].append(
                    f"매수 실패 {order.ticker}: {order.error_msg}"
                )

            time.sleep(self.ORDER_DELAY)

        # 7. 최종 상태 동기화
        time.sleep(2.0)
        self.order_manager.sync_order_status()

        # 8. 결과 집계
        result["total_sell_amount"] = sum(
            o.get("amount", 0) for o in sell_orders
        )
        result["total_buy_amount"] = sum(
            o.get("amount", 0) for o in buy_orders
        )

        submitted_sells = [
            s for s in result["sells"] if s.get("status") != "failed"
        ]
        submitted_buys = [
            b for b in result["buys"] if b.get("status") != "failed"
        ]

        result["success"] = (
            len(result["errors"]) == 0
            or (len(submitted_sells) + len(submitted_buys)) > 0
        )

        elapsed = (datetime.now(KST) - start_time).total_seconds()
        logger.info(
            "리밸런싱 실행 완료: "
            "매도 %d/%d건, 매수 %d/%d건, "
            "오류 %d건, 스킵 %d건, 소요시간 %.1f초",
            len(submitted_sells),
            len(sell_orders),
            len(submitted_buys),
            len(buy_orders),
            len(result["errors"]),
            len(result["skipped"]),
            elapsed,
        )
        logger.info("=" * 60)

        return result

    def dry_run(
        self,
        target_weights: dict[str, float],
        pool: str | None = None,
    ) -> dict:
        """리밸런싱 시뮬레이션을 실행한다 (실제 주문 없음).

        실제 주문을 실행하지 않고 예상 매매 내역만 반환한다.
        포지션과 현재가는 실제로 조회한다.

        Args:
            target_weights: {ticker: weight} 형태의 목표 비중.
            pool: 리밸런싱 대상 풀. None이면 "long_term".

        Returns:
            시뮬레이션 결과 딕셔너리:
            {
                "is_dry_run": True,
                "portfolio_value": int,
                "current_positions": dict,
                "target_weights": dict,
                "sell_orders": [dict, ...],
                "buy_orders": [dict, ...],
                "total_sell_amount": int,
                "total_buy_amount": int,
                "net_cash_flow": int,
                "risk_check": {"passed": bool, "warnings": list},
                "turnover_check": {"passed": bool, "reason": str},
            }
        """
        logger.info(
            "리밸런싱 DRY RUN 시작: %d개 종목", len(target_weights)
        )

        result: dict = {
            "is_dry_run": True,
            "portfolio_value": 0,
            "current_positions": {},
            "target_weights": dict(target_weights),
            "sell_orders": [],
            "buy_orders": [],
            "total_sell_amount": 0,
            "total_buy_amount": 0,
            "net_cash_flow": 0,
            "risk_check": {"passed": True, "warnings": []},
            "turnover_check": {"passed": True, "reason": ""},
        }

        # 현재 포지션 및 포트폴리오 가치
        current_positions = self.position_manager.get_current_positions()
        portfolio_value = self.position_manager.get_portfolio_value()

        result["current_positions"] = current_positions
        result["portfolio_value"] = portfolio_value

        if portfolio_value <= 0:
            logger.warning("DRY RUN: 포트폴리오 가치가 0원입니다.")
            return result

        # 리스크 검증
        risk_passed, risk_warnings = self.risk_guard.check_rebalance(target_weights)
        result["risk_check"] = {
            "passed": risk_passed,
            "warnings": risk_warnings,
        }

        # 리밸런싱 주문 계산 (allocator가 있으면 풀 비중으로 스케일링)
        effective_weights = target_weights
        if self.allocator is not None:
            if pool == "etf_rotation":
                effective_weights = self.allocator.filter_etf_rotation_weights(
                    target_weights
                )
            else:
                effective_weights = self.allocator.filter_long_term_weights(
                    target_weights
                )
            logger.info(
                "DRY RUN allocator 적용 (pool=%s): 원래 비중 합=%.4f -> 스케일 비중 합=%.4f",
                pool or "long_term",
                sum(target_weights.values()),
                sum(effective_weights.values()),
            )

        sell_orders, buy_orders = self.position_manager.calculate_rebalance_orders(
            effective_weights, allocator=self.allocator, pool=pool
        )

        result["sell_orders"] = sell_orders
        result["buy_orders"] = buy_orders
        result["total_sell_amount"] = sum(o.get("amount", 0) for o in sell_orders)
        result["total_buy_amount"] = sum(o.get("amount", 0) for o in buy_orders)
        result["net_cash_flow"] = (
            result["total_sell_amount"] - result["total_buy_amount"]
        )

        # 회전율 검증
        turnover_passed, turnover_reason = self.risk_guard.check_turnover(
            sell_orders, buy_orders, portfolio_value
        )
        result["turnover_check"] = {
            "passed": turnover_passed,
            "reason": turnover_reason,
        }

        logger.info(
            "DRY RUN 결과: "
            "포트폴리오=%s원, 매도=%d건 (%s원), 매수=%d건 (%s원), "
            "리스크=%s, 회전율=%s",
            f"{portfolio_value:,}",
            len(sell_orders),
            f"{result['total_sell_amount']:,}",
            len(buy_orders),
            f"{result['total_buy_amount']:,}",
            "통과" if risk_passed else "실패",
            "통과" if turnover_passed else "실패",
        )

        return result
