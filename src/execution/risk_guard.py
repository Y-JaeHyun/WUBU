"""사전 리스크 체크 모듈.

주문 실행 전에 리스크 관련 사전 검증을 수행한다.
개별 주문 검증과 전체 리밸런싱 검증을 제공한다.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class RiskGuard:
    """주문 전 사전 리스크 체크 클래스.

    개별 주문과 전체 리밸런싱에 대해 사전 위험 검증을 수행한다.
    설정 가능한 한도를 초과하는 주문을 차단한다.

    Config keys (from env or dict):
        max_order_pct: 1건당 최대 비중 (기본 0.10 = 10%).
        max_daily_turnover: 일일 최대 회전율 (기본 0.30 = 30%).
        max_single_stock_pct: 개별종목 최대 비중 (기본 0.15 = 15%).
        blocked_tickers: 매매 금지 종목 리스트 (콤마 구분 문자열).

    Attributes:
        max_order_pct: 1건당 최대 주문 비중.
        max_daily_turnover: 일일 최대 회전율.
        max_single_stock_pct: 개별종목 최대 비중.
        blocked_tickers: 매매 금지 종목 코드 집합.
    """

    # 모의투자 기본 한도
    DEFAULT_MAX_ORDER_PCT = 0.10
    DEFAULT_MAX_DAILY_TURNOVER = 1.50
    DEFAULT_MAX_SINGLE_STOCK_PCT = 0.15

    # 실전투자 기본 한도
    # 월 1회 통합 리밸런싱: (매도+매수)/총액 기준 60~100% 정상 범위
    LIVE_MAX_ORDER_PCT = 0.05
    LIVE_MAX_DAILY_TURNOVER = 1.00
    LIVE_MAX_SINGLE_STOCK_PCT = 0.10

    def __init__(
        self, config: Optional[dict] = None, is_live: bool = False
    ) -> None:
        """RiskGuard를 초기화한다.

        Args:
            config: 리스크 설정 딕셔너리. None이면 환경변수에서 로드.
                keys: max_order_pct, max_daily_turnover,
                      max_single_stock_pct, blocked_tickers
            is_live: 실전 모드 여부. True이면 보수적 기본값 적용.
        """
        cfg = config or {}

        # 실전 모드이면 보수적 기본값 사용
        if is_live:
            default_order = self.LIVE_MAX_ORDER_PCT
            default_turnover = self.LIVE_MAX_DAILY_TURNOVER
            default_single = self.LIVE_MAX_SINGLE_STOCK_PCT
        else:
            default_order = self.DEFAULT_MAX_ORDER_PCT
            default_turnover = self.DEFAULT_MAX_DAILY_TURNOVER
            default_single = self.DEFAULT_MAX_SINGLE_STOCK_PCT

        self.max_order_pct: float = float(
            cfg.get(
                "max_order_pct",
                os.getenv("RISK_MAX_ORDER_PCT", str(default_order)),
            )
        )
        self.max_daily_turnover: float = float(
            cfg.get(
                "max_daily_turnover",
                os.getenv("RISK_MAX_DAILY_TURNOVER", str(default_turnover)),
            )
        )
        self.max_single_stock_pct: float = float(
            cfg.get(
                "max_single_stock_pct",
                os.getenv("RISK_MAX_SINGLE_STOCK_PCT", str(default_single)),
            )
        )

        # 매매 금지 종목
        blocked_str = cfg.get(
            "blocked_tickers",
            os.getenv("RISK_BLOCKED_TICKERS", ""),
        )
        if isinstance(blocked_str, (list, set)):
            self.blocked_tickers: set[str] = set(blocked_str)
        elif isinstance(blocked_str, str) and blocked_str.strip():
            self.blocked_tickers = {
                t.strip() for t in blocked_str.split(",") if t.strip()
            }
        else:
            self.blocked_tickers = set()

        logger.info(
            "RiskGuard 초기화 완료: "
            "max_order=%.1f%%, max_turnover=%.1f%%, "
            "max_single_stock=%.1f%%, blocked=%d개",
            self.max_order_pct * 100,
            self.max_daily_turnover * 100,
            self.max_single_stock_pct * 100,
            len(self.blocked_tickers),
        )

    def check_order(
        self, order: dict, portfolio_value: int
    ) -> tuple[bool, str]:
        """단일 주문을 검증한다.

        Args:
            order: 주문 딕셔너리.
                keys: ticker, side, qty, price (또는 amount).
            portfolio_value: 현재 포트폴리오 총 가치 (원).

        Returns:
            (통과 여부, 사유) 튜플.
            통과 시 (True, ""), 거절 시 (False, 사유 문자열).
        """
        ticker = order.get("ticker", "")

        # 1. 매매 금지 종목 확인
        passed, reason = self._check_blocked_tickers(ticker)
        if not passed:
            return False, reason

        # 2. 단일 주문 비중 확인
        if portfolio_value > 0:
            order_amount = order.get("amount", 0)
            if order_amount == 0:
                qty = order.get("qty", 0)
                price = order.get("price", 0)
                order_amount = qty * price

            if order_amount > 0:
                order_pct = order_amount / portfolio_value
                if order_pct > self.max_order_pct:
                    reason = (
                        f"단일 주문 비중 초과: {ticker} "
                        f"({order_pct:.1%} > {self.max_order_pct:.1%})"
                    )
                    logger.warning("리스크 거절: %s", reason)
                    return False, reason

        return True, ""

    def check_rebalance(
        self, target_weights: dict[str, float]
    ) -> tuple[bool, list[str]]:
        """전체 리밸런싱을 검증한다.

        Args:
            target_weights: {ticker: weight} 형태. weight는 0~1 사이 비율.

        Returns:
            (통과 여부, 경고 목록) 튜플.
            통과 시 (True, []), 거절 시 (False, [경고문자열, ...]).
        """
        warnings: list[str] = []
        is_passed = True

        if not target_weights:
            logger.warning("리밸런싱 검증: 목표 비중이 비어 있습니다.")
            return True, ["목표 비중이 비어 있습니다."]

        # 1. 비중 합 검증
        total_weight = sum(target_weights.values())
        if total_weight > 1.01:
            msg = f"비중 합 초과: {total_weight:.4f} (최대 1.0)"
            warnings.append(msg)
            logger.warning("리스크 경고: %s", msg)
            is_passed = False

        # 2. 개별 종목 비중 검증
        for ticker, weight in target_weights.items():
            # 매매 금지 종목 확인
            passed, reason = self._check_blocked_tickers(ticker)
            if not passed:
                warnings.append(reason)
                is_passed = False
                continue

            # 개별 종목 최대 비중 확인
            passed, reason = self._check_single_stock_limit(ticker, weight)
            if not passed:
                warnings.append(reason)
                is_passed = False

        # 3. 음수 비중 검증
        negative_tickers = [
            t for t, w in target_weights.items() if w < 0
        ]
        if negative_tickers:
            msg = f"음수 비중 종목 발견: {negative_tickers}"
            warnings.append(msg)
            logger.warning("리스크 경고: %s", msg)
            is_passed = False

        if is_passed:
            logger.info(
                "리밸런싱 리스크 검증 통과: %d개 종목", len(target_weights)
            )
        else:
            logger.warning(
                "리밸런싱 리스크 검증 실패: %d건 경고", len(warnings)
            )

        return is_passed, warnings

    def check_turnover(
        self,
        sell_orders: list[dict],
        buy_orders: list[dict],
        portfolio_value: int,
    ) -> tuple[bool, str]:
        """일일 회전율을 검증한다.

        Args:
            sell_orders: 매도 주문 리스트.
            buy_orders: 매수 주문 리스트.
            portfolio_value: 포트폴리오 총 가치 (원).

        Returns:
            (통과 여부, 사유) 튜플.
        """
        if portfolio_value <= 0:
            return True, ""

        total_sell = sum(o.get("amount", 0) for o in sell_orders)
        total_buy = sum(o.get("amount", 0) for o in buy_orders)
        total_turnover = (total_sell + total_buy) / portfolio_value

        passed, reason = self._check_daily_turnover(total_turnover)
        return passed, reason

    def _check_single_stock_limit(
        self, ticker: str, weight: float
    ) -> tuple[bool, str]:
        """개별 종목의 비중 한도를 확인한다.

        Args:
            ticker: 종목코드.
            weight: 목표 비중 (0~1).

        Returns:
            (통과 여부, 사유) 튜플.
        """
        if weight > self.max_single_stock_pct:
            reason = (
                f"개별종목 비중 초과: {ticker} "
                f"({weight:.1%} > {self.max_single_stock_pct:.1%})"
            )
            logger.warning("리스크 거절: %s", reason)
            return False, reason
        return True, ""

    def _check_blocked_tickers(self, ticker: str) -> tuple[bool, str]:
        """매매 금지 종목인지 확인한다.

        Args:
            ticker: 종목코드.

        Returns:
            (통과 여부, 사유) 튜플.
        """
        if ticker in self.blocked_tickers:
            reason = f"매매 금지 종목: {ticker}"
            logger.warning("리스크 거절: %s", reason)
            return False, reason
        return True, ""

    def _check_daily_turnover(self, turnover: float) -> tuple[bool, str]:
        """일일 회전율 한도를 확인한다.

        Args:
            turnover: 회전율 (0~1).

        Returns:
            (통과 여부, 사유) 튜플.
        """
        if turnover > self.max_daily_turnover:
            reason = (
                f"일일 회전율 초과: "
                f"{turnover:.1%} > {self.max_daily_turnover:.1%}"
            )
            logger.warning("리스크 거절: %s", reason)
            return False, reason
        return True, ""
