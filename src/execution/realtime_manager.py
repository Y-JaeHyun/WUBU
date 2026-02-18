"""실시간 시세 통합 관리 모듈.

KIS WebSocket 클라이언트를 관리하고 가격 캐시를 제공한다.
이상 변동 감지 콜백을 등록하여 실시간 알림에 활용할 수 있다.
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RealtimeManager:
    """실시간 시세 관리기.

    WebSocket 클라이언트를 관리하고, 수신된 체결 데이터를 캐싱하며,
    이상 변동 감지 콜백을 제공한다.

    Args:
        kis_client: KISClient 인스턴스.
        alert_threshold_pct: 이상 변동 감지 임계치 (%, 기본 5.0).
    """

    DEFAULT_ALERT_THRESHOLD_PCT = 5.0

    def __init__(
        self,
        kis_client: Any,
        alert_threshold_pct: float = DEFAULT_ALERT_THRESHOLD_PCT,
    ) -> None:
        """RealtimeManager를 초기화한다.

        Args:
            kis_client: KISClient 인스턴스.
            alert_threshold_pct: 등락률 기준 이상 변동 감지 임계치 (%).
        """
        self._kis_client = kis_client
        self._alert_threshold_pct = alert_threshold_pct

        # WebSocket 클라이언트는 start()에서 지연 생성
        self._ws_client: Optional[Any] = None
        self._running: bool = False

        # 가격 캐시: {ticker: {price, volume, ...}}
        self._price_cache: dict[str, dict] = {}

        # 이상 변동 콜백 목록
        self._alert_callbacks: list[Callable[[dict], None]] = []

        # 틱 히스토리 (최근 N건, 분석/디버깅용)
        self._tick_history: list[dict] = []
        self._max_history_size: int = 1000

    # ------------------------------------------------------------------
    # 가격 조회
    # ------------------------------------------------------------------

    def get_latest_price(self, ticker: str) -> dict | None:
        """캐시된 최신 tick 데이터를 반환한다.

        Args:
            ticker: 종목 코드 (예: '005930').

        Returns:
            최신 tick 딕셔너리. 데이터가 없으면 None.
            딕셔너리 구조: {ticker, price, volume, acml_volume,
                           open, high, low, change_pct, timestamp}
        """
        return self._price_cache.get(ticker)

    def get_all_prices(self) -> dict[str, dict]:
        """모든 캐시된 가격 데이터를 반환한다.

        Returns:
            {ticker: tick_dict} 형태의 딕셔너리.
        """
        return dict(self._price_cache)

    def get_ticker_price(self, ticker: str) -> int:
        """종목의 현재가만 반환한다.

        Args:
            ticker: 종목 코드.

        Returns:
            현재가 (int). 데이터 없으면 0.
        """
        tick = self._price_cache.get(ticker)
        if tick is None:
            return 0
        return tick.get("price", 0)

    # ------------------------------------------------------------------
    # 콜백 등록
    # ------------------------------------------------------------------

    def register_alert_callback(self, callback: Callable[[dict], None]) -> None:
        """이상 변동 감지 콜백을 등록한다.

        등록된 콜백은 이상 변동 감지 시 tick 데이터 딕셔너리를 인자로 호출된다.

        Args:
            callback: tick dict를 인자로 받는 함수.
        """
        self._alert_callbacks.append(callback)
        logger.info(
            "이상 변동 콜백 등록 (총 %d개).", len(self._alert_callbacks)
        )

    # ------------------------------------------------------------------
    # 시작/종료
    # ------------------------------------------------------------------

    async def start(self, tickers: list[str]) -> None:
        """WebSocket 연결을 시작하고 종목을 구독한다.

        Args:
            tickers: 구독할 종목 코드 리스트.
        """
        if self._running:
            logger.warning("RealtimeManager가 이미 실행 중입니다.")
            return

        if not tickers:
            logger.warning("구독할 종목이 없습니다.")
            return

        # 지연 임포트: 순환 참조 방지
        from src.execution.kis_websocket import KISWebSocketClient

        self._ws_client = KISWebSocketClient(
            kis_client=self._kis_client,
            on_tick=self._on_tick,
        )

        self._running = True
        logger.info("RealtimeManager 시작: %d개 종목 구독 예정.", len(tickers))

        try:
            await self._ws_client.connect()

            # 종목 구독 (최대 41개 제한은 WebSocket 클라이언트에서 처리)
            subscribed = 0
            for ticker in tickers:
                success = await self._ws_client.subscribe(ticker)
                if success:
                    subscribed += 1
                # 종목 간 짧은 지연
                await asyncio.sleep(0.05)

            logger.info(
                "RealtimeManager 구독 완료: %d/%d개 종목.",
                subscribed,
                len(tickers),
            )

        except Exception as e:
            logger.error("RealtimeManager 시작 실패: %s", e)
            self._running = False
            raise

    async def stop(self) -> None:
        """WebSocket 연결을 종료한다."""
        if not self._running:
            logger.debug("RealtimeManager가 실행 중이 아닙니다.")
            return

        logger.info("RealtimeManager 종료 시작...")
        self._running = False

        if self._ws_client is not None:
            try:
                await self._ws_client.disconnect()
            except Exception as e:
                logger.error("WebSocket 종료 중 오류: %s", e)

        logger.info(
            "RealtimeManager 종료 완료. "
            "수신 tick: %d개, 캐시 종목: %d개.",
            len(self._tick_history),
            len(self._price_cache),
        )

    def is_running(self) -> bool:
        """실행 중인지 확인한다.

        Returns:
            실행 상태이면 True, 아니면 False.
        """
        return self._running

    # ------------------------------------------------------------------
    # 내부 콜백
    # ------------------------------------------------------------------

    def _on_tick(self, tick: dict) -> None:
        """WebSocket에서 수신한 tick 데이터를 처리한다.

        가격 캐시를 갱신하고, 이상 변동을 감지하며,
        히스토리에 저장한다.

        Args:
            tick: 파싱된 체결 데이터 딕셔너리.
        """
        ticker = tick.get("ticker", "")
        if not ticker:
            return

        # 캐시 갱신
        self._price_cache[ticker] = tick

        # 히스토리 저장 (크기 제한)
        self._tick_history.append(tick)
        if len(self._tick_history) > self._max_history_size:
            self._tick_history = self._tick_history[-self._max_history_size:]

        # 이상 변동 감지
        change_pct = tick.get("change_pct", 0.0)
        if abs(change_pct) >= self._alert_threshold_pct:
            logger.warning(
                "이상 변동 감지: %s — 등락률 %.2f%%",
                ticker,
                change_pct,
            )
            self._fire_alert_callbacks(tick)

    def _fire_alert_callbacks(self, tick: dict) -> None:
        """등록된 이상 변동 콜백들을 호출한다.

        Args:
            tick: 이상 변동이 감지된 tick 데이터.
        """
        for callback in self._alert_callbacks:
            try:
                callback(tick)
            except Exception as e:
                logger.error("이상 변동 콜백 오류: %s", e)

    # ------------------------------------------------------------------
    # 유틸리티
    # ------------------------------------------------------------------

    def get_tick_history(self, ticker: Optional[str] = None) -> list[dict]:
        """틱 히스토리를 반환한다.

        Args:
            ticker: 특정 종목만 필터링. None이면 전체.

        Returns:
            tick 딕셔너리 리스트.
        """
        if ticker is None:
            return list(self._tick_history)
        return [t for t in self._tick_history if t.get("ticker") == ticker]

    def get_subscribed_tickers(self) -> list[str]:
        """현재 구독 중인 종목 코드 리스트를 반환한다.

        Returns:
            종목 코드 리스트.
        """
        if self._ws_client is not None:
            return sorted(self._ws_client.subscriptions)
        return []
