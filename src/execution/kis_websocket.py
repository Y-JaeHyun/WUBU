"""KIS WebSocket 실시간 시세 클라이언트.

한국투자증권 OpenAPI WebSocket을 통해 실시간 체결가를 수신한다.
비동기(asyncio) 기반으로 동작하며, 자동 재연결과
PINGPONG 핸들링을 내장한다.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Optional

import aiohttp
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


class KISWebSocketClient:
    """KIS OpenAPI WebSocket 실시간 시세 클라이언트.

    KIS WebSocket 프로토콜:
    - URL 실전: ws://ops.koreainvestment.com:21000
    - URL 모의: ws://ops.koreainvestment.com:31000
    - 인증: approval_key (REST /oauth2/Approval 발급)
    - 최대 구독: 41종목/세션
    - 구독 요청: JSON {"header": {"approval_key": "...", "tr_type": "1"},
                        "body": {"input": {"tr_id": "H0STCNT0", "tr_key": "005930"}}}
    - 해제: tr_type = "2"
    - Heartbeat: PINGPONG 응답 필요
    - 데이터: | 구분 텍스트

    Args:
        kis_client: KISClient 인스턴스 (인증용).
        on_tick: 체결 데이터 콜백 함수. dict를 인자로 받는다.
    """

    REAL_WS_URL = "ws://ops.koreainvestment.com:21000"
    PAPER_WS_URL = "ws://ops.koreainvestment.com:31000"
    MAX_SUBSCRIPTIONS = 41

    # 재연결 백오프 설정
    RECONNECT_BASE_DELAY = 1.0  # seconds
    RECONNECT_MAX_DELAY = 60.0  # seconds
    RECONNECT_FACTOR = 2.0

    # H0STCNT0(주식체결) 필드 인덱스 (0-based, | 구분)
    # KIS 실시간 체결 데이터는 | 로 구분된 텍스트로 전달된다.
    _TICK_FIELD_MAP = {
        "ticker": 0,        # MKSC_SHRN_ISCD — 유가증권 단축 종목코드
        "time": 1,          # STCK_CNTG_HOUR — 주식 체결 시간
        "price": 2,         # STCK_PRPR — 주식 현재가
        "change_sign": 3,   # PRDY_VRSS_SIGN — 전일 대비 부호
        "change": 4,        # PRDY_VRSS — 전일 대비
        "change_pct": 5,    # PRDY_CTRT — 전일 대비율
        "weighted_avg": 6,  # WGHN_AVRG_STCK_PRC — 가중 평균 주식 가격
        "open": 7,          # STCK_OPRC — 주식 시가
        "high": 8,          # STCK_HGPR — 주식 최고가
        "low": 9,           # STCK_LWPR — 주식 최저가
        "ask": 10,          # ASKP1 — 매도호가1
        "bid": 11,          # BIDP1 — 매수호가1
        "volume": 12,       # CNTG_VOL — 체결 거래량
        "acml_volume": 13,  # ACML_VOL — 누적 거래량
        "acml_amount": 14,  # ACML_TR_PBMN — 누적 거래 대금
    }

    def __init__(
        self,
        kis_client: Any,
        on_tick: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """KISWebSocketClient를 초기화한다.

        Args:
            kis_client: KISClient 인스턴스 (app_key, app_secret, is_paper 참조).
            on_tick: 체결 데이터 콜백 함수. None이면 내부 로깅만 수행.
        """
        self._kis_client = kis_client
        self._on_tick = on_tick

        self._approval_key: str = ""
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._subscriptions: set[str] = set()
        self._running: bool = False
        self._reconnect_delay: float = self.RECONNECT_BASE_DELAY
        self._message_task: Optional[asyncio.Task] = None

        # 통계
        self._stats = {
            "connected_at": None,
            "messages_received": 0,
            "ticks_parsed": 0,
            "errors": 0,
            "reconnects": 0,
        }

    @property
    def ws_url(self) -> str:
        """현재 모드에 따른 WebSocket URL을 반환한다.

        Returns:
            모의투자 또는 실전 WebSocket URL.
        """
        return self.PAPER_WS_URL if self._kis_client.is_paper else self.REAL_WS_URL

    @property
    def subscriptions(self) -> set[str]:
        """현재 구독 중인 종목 코드 집합을 반환한다."""
        return set(self._subscriptions)

    @property
    def stats(self) -> dict:
        """연결 통계를 반환한다."""
        return dict(self._stats)

    def is_connected(self) -> bool:
        """WebSocket이 연결되어 있는지 확인한다.

        Returns:
            연결 상태이면 True, 아니면 False.
        """
        return self._ws is not None and not self._ws.closed

    # ------------------------------------------------------------------
    # 인증
    # ------------------------------------------------------------------

    async def get_approval_key(self) -> str:
        """REST API로 WebSocket approval_key를 발급받는다.

        POST /oauth2/Approval 엔드포인트를 호출하여
        WebSocket 접속용 인증 키를 발급받는다.

        Returns:
            발급받은 approval_key 문자열.

        Raises:
            RuntimeError: API 호출 실패 시.
        """
        if not self._kis_client.is_configured():
            raise RuntimeError(
                "KIS API 설정이 불완전합니다. "
                "app_key, app_secret을 확인하세요."
            )

        url = f"{self._kis_client.base_url}/oauth2/Approval"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self._kis_client.app_key,
            "secretkey": self._kis_client.app_secret,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self._kis_client.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            self._approval_key = data.get("approval_key", "")

            if not self._approval_key:
                raise RuntimeError(
                    f"approval_key 발급 실패: {data}"
                )

            logger.info("WebSocket approval_key 발급 완료.")
            return self._approval_key

        except requests.exceptions.RequestException as e:
            logger.error("approval_key 발급 중 오류: %s", e)
            raise RuntimeError(f"approval_key 발급 실패: {e}") from e

    # ------------------------------------------------------------------
    # 연결
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """WebSocket 연결을 수립하고 메시지 수신 루프를 시작한다.

        approval_key가 없으면 자동으로 발급받는다.
        연결 후 백그라운드에서 메시지 수신 루프가 동작한다.
        """
        if self._running:
            logger.warning("이미 WebSocket이 실행 중입니다.")
            return

        # approval_key 발급
        if not self._approval_key:
            await self.get_approval_key()

        self._running = True
        self._reconnect_delay = self.RECONNECT_BASE_DELAY

        try:
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(
                self.ws_url,
                heartbeat=30.0,
            )
            self._stats["connected_at"] = datetime.now().isoformat()
            logger.info(
                "WebSocket 연결 완료: %s (모드: %s)",
                self.ws_url,
                "모의투자" if self._kis_client.is_paper else "실전",
            )

            # 메시지 수신 루프 시작
            self._message_task = asyncio.create_task(self._message_loop())

        except Exception as e:
            logger.error("WebSocket 연결 실패: %s", e)
            self._stats["errors"] += 1
            await self._cleanup()
            if self._running:
                await self._handle_reconnect()

    # ------------------------------------------------------------------
    # 구독
    # ------------------------------------------------------------------

    async def subscribe(self, ticker: str) -> bool:
        """종목을 구독한다 (최대 41개).

        Args:
            ticker: 종목 코드 (예: '005930').

        Returns:
            구독 성공 시 True, 실패 시 False.
        """
        if not self.is_connected():
            logger.warning("WebSocket 미연결 상태입니다. 구독 불가.")
            return False

        if ticker in self._subscriptions:
            logger.debug("이미 구독 중인 종목: %s", ticker)
            return True

        if len(self._subscriptions) >= self.MAX_SUBSCRIPTIONS:
            logger.warning(
                "최대 구독 수(%d)에 도달했습니다. "
                "종목 '%s' 구독 불가.",
                self.MAX_SUBSCRIPTIONS,
                ticker,
            )
            return False

        message = self._build_subscribe_message(ticker, subscribe=True)

        try:
            await self._ws.send_json(message)
            self._subscriptions.add(ticker)
            logger.info("종목 구독 완료: %s (총 %d개)", ticker, len(self._subscriptions))
            return True
        except Exception as e:
            logger.error("종목 구독 실패 (%s): %s", ticker, e)
            self._stats["errors"] += 1
            return False

    async def unsubscribe(self, ticker: str) -> bool:
        """종목 구독을 해제한다.

        Args:
            ticker: 종목 코드 (예: '005930').

        Returns:
            해제 성공 시 True, 실패 시 False.
        """
        if not self.is_connected():
            logger.warning("WebSocket 미연결 상태입니다. 해제 불가.")
            return False

        if ticker not in self._subscriptions:
            logger.debug("구독 중이 아닌 종목: %s", ticker)
            return True

        message = self._build_subscribe_message(ticker, subscribe=False)

        try:
            await self._ws.send_json(message)
            self._subscriptions.discard(ticker)
            logger.info("종목 구독 해제: %s (남은 %d개)", ticker, len(self._subscriptions))
            return True
        except Exception as e:
            logger.error("종목 구독 해제 실패 (%s): %s", ticker, e)
            self._stats["errors"] += 1
            return False

    def _build_subscribe_message(self, ticker: str, subscribe: bool = True) -> dict:
        """구독/해제 요청 메시지를 생성한다.

        Args:
            ticker: 종목 코드.
            subscribe: True면 구독, False면 해제.

        Returns:
            KIS WebSocket 프로토콜에 맞는 JSON 딕셔너리.
        """
        return {
            "header": {
                "approval_key": self._approval_key,
                "custtype": "P",
                "tr_type": "1" if subscribe else "2",
                "content-type": "utf-8",
            },
            "body": {
                "input": {
                    "tr_id": "H0STCNT0",
                    "tr_key": ticker,
                },
            },
        }

    # ------------------------------------------------------------------
    # 메시지 수신/파싱
    # ------------------------------------------------------------------

    async def _message_loop(self) -> None:
        """메시지 수신 및 파싱 루프.

        WebSocket으로부터 메시지를 수신하여 파싱하고,
        PINGPONG 핸들링과 체결 데이터 콜백을 수행한다.
        연결 종료 시 자동 재연결을 시도한다.
        """
        try:
            async for msg in self._ws:
                if not self._running:
                    break

                self._stats["messages_received"] += 1

                if msg.type == aiohttp.WSMsgType.TEXT:
                    raw = msg.data

                    # PINGPONG 핸들링
                    if self._is_pingpong(raw):
                        await self._handle_pingpong(raw)
                        continue

                    # JSON 응답 (구독 확인 등)
                    if raw.startswith("{"):
                        self._handle_json_response(raw)
                        continue

                    # 체결 데이터 파싱 (| 구분)
                    tick = self._parse_tick(raw)
                    if tick is not None:
                        self._stats["ticks_parsed"] += 1
                        self._reset_reconnect_delay()

                        if self._on_tick is not None:
                            try:
                                self._on_tick(tick)
                            except Exception as e:
                                logger.error("tick 콜백 오류: %s", e)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        "WebSocket 오류: %s",
                        self._ws.exception() if self._ws else "unknown",
                    )
                    self._stats["errors"] += 1
                    break

                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    logger.warning("WebSocket 연결 종료됨.")
                    break

        except asyncio.CancelledError:
            logger.info("메시지 루프가 취소되었습니다.")
            return
        except Exception as e:
            logger.error("메시지 루프 예외: %s", e)
            self._stats["errors"] += 1

        # 재연결
        if self._running:
            logger.info("연결 유실 — 재연결을 시도합니다.")
            await self._cleanup_ws()
            await self._handle_reconnect()

    def _is_pingpong(self, raw: str) -> bool:
        """PINGPONG 메시지인지 확인한다.

        Args:
            raw: 원시 메시지 문자열.

        Returns:
            PINGPONG 메시지이면 True.
        """
        return "PINGPONG" in raw.upper()

    async def _handle_pingpong(self, raw: str) -> None:
        """PINGPONG에 대한 응답을 전송한다.

        KIS WebSocket은 주기적으로 PINGPONG 메시지를 보내며,
        동일한 메시지로 응답해야 연결이 유지된다.

        Args:
            raw: 수신된 PINGPONG 메시지.
        """
        try:
            if self.is_connected():
                await self._ws.send_str(raw)
                logger.debug("PINGPONG 응답 전송.")
        except Exception as e:
            logger.error("PINGPONG 응답 실패: %s", e)

    def _handle_json_response(self, raw: str) -> None:
        """JSON 응답(구독 확인 등)을 처리한다.

        Args:
            raw: JSON 문자열.
        """
        try:
            data = json.loads(raw)
            header = data.get("header", {})
            tr_id = header.get("tr_id", "")
            msg_cd = header.get("msg_cd", "")
            msg1 = header.get("msg1", "")

            if msg_cd == "OPSP0000":
                logger.debug("구독 확인: tr_id=%s, msg=%s", tr_id, msg1)
            elif msg_cd == "OPSP0002":
                logger.debug("구독 해제 확인: tr_id=%s, msg=%s", tr_id, msg1)
            else:
                logger.debug("JSON 응답: msg_cd=%s, msg=%s", msg_cd, msg1)

        except json.JSONDecodeError:
            logger.debug("JSON 파싱 실패 (비정형 메시지 무시): %s", raw[:100])

    def _parse_tick(self, raw: str) -> Optional[dict]:
        """원시 메시지를 파싱하여 체결 데이터 딕셔너리를 반환한다.

        H0STCNT0 체결가 필드 (| 구분):
        - MKSC_SHRN_ISCD(종목코드), STCK_PRPR(현재가), CNTG_VOL(체결수량),
          ACML_VOL(누적거래량), STCK_OPRC(시가), STCK_HGPR(고가), STCK_LWPR(저가),
          PRDY_CTRT(전일대비등락률)

        Args:
            raw: | 구분 텍스트 메시지.

        Returns:
            파싱된 체결 데이터 딕셔너리. 파싱 실패 시 None.
            반환 딕셔너리 구조:
                - ticker: str — 종목 코드
                - price: int — 현재가
                - volume: int — 체결 수량
                - acml_volume: int — 누적 거래량
                - open: int — 시가
                - high: int — 고가
                - low: int — 저가
                - change_pct: float — 전일 대비 등락률 (%)
                - change: int — 전일 대비 변동
                - timestamp: str — 수신 시각 (ISO 형식)
        """
        try:
            # KIS 체결 메시지: "0|H0STCNT0|NNN|field1^field2^..."
            # 첫 번째 부분은 헤더, | 세 번째 이후가 데이터
            parts = raw.split("|")
            if len(parts) < 4:
                return None

            tr_id = parts[1].strip()
            if tr_id != "H0STCNT0":
                logger.debug("H0STCNT0이 아닌 tr_id 무시: %s", tr_id)
                return None

            # 데이터 부분은 ^ 로 구분
            data_str = parts[3]
            fields = data_str.split("^")

            if len(fields) < 15:
                logger.debug("필드 수 부족 (%d개): %s", len(fields), raw[:100])
                return None

            ticker = fields[self._TICK_FIELD_MAP["ticker"]]
            price = self._safe_int(fields[self._TICK_FIELD_MAP["price"]])
            volume = self._safe_int(fields[self._TICK_FIELD_MAP["volume"]])
            acml_volume = self._safe_int(fields[self._TICK_FIELD_MAP["acml_volume"]])
            open_price = self._safe_int(fields[self._TICK_FIELD_MAP["open"]])
            high_price = self._safe_int(fields[self._TICK_FIELD_MAP["high"]])
            low_price = self._safe_int(fields[self._TICK_FIELD_MAP["low"]])
            change = self._safe_int(fields[self._TICK_FIELD_MAP["change"]])
            change_pct = self._safe_float(fields[self._TICK_FIELD_MAP["change_pct"]])

            tick_data = {
                "ticker": ticker,
                "price": price,
                "volume": volume,
                "acml_volume": acml_volume,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "change": change,
                "change_pct": change_pct,
                "timestamp": datetime.now().isoformat(),
            }

            return tick_data

        except (IndexError, ValueError) as e:
            logger.debug("tick 파싱 실패: %s — 원본: %s", e, raw[:100])
            self._stats["errors"] += 1
            return None

    @staticmethod
    def _safe_int(value: str) -> int:
        """문자열을 int로 안전하게 변환한다.

        Args:
            value: 변환할 문자열.

        Returns:
            정수 값. 변환 실패 시 0.
        """
        try:
            # 부호 제거 후 변환 (KIS는 절대값에 부호를 별도 필드로 제공)
            cleaned = value.strip().lstrip("+").lstrip("-")
            return int(cleaned) if cleaned else 0
        except (ValueError, AttributeError):
            return 0

    @staticmethod
    def _safe_float(value: str) -> float:
        """문자열을 float로 안전하게 변환한다.

        Args:
            value: 변환할 문자열.

        Returns:
            실수 값. 변환 실패 시 0.0.
        """
        try:
            return float(value.strip()) if value.strip() else 0.0
        except (ValueError, AttributeError):
            return 0.0

    # ------------------------------------------------------------------
    # 재연결
    # ------------------------------------------------------------------

    async def _handle_reconnect(self) -> None:
        """지수 백오프로 재연결을 시도한다.

        지연 시간: 1s -> 2s -> 4s -> 8s -> ... -> 60s (최대).
        재연결 성공 시 기존 구독 종목을 복원한다.
        """
        if not self._running:
            return

        self._stats["reconnects"] += 1
        delay = self._reconnect_delay

        logger.info(
            "%.1f초 후 재연결 시도 (시도 횟수: %d)...",
            delay,
            self._stats["reconnects"],
        )
        await asyncio.sleep(delay)

        # 지수 백오프 증가
        self._reconnect_delay = min(
            self._reconnect_delay * self.RECONNECT_FACTOR,
            self.RECONNECT_MAX_DELAY,
        )

        if not self._running:
            return

        # 기존 구독 목록 보존
        prev_subscriptions = set(self._subscriptions)
        self._subscriptions.clear()

        try:
            # approval_key 재발급
            self._approval_key = ""
            await self.get_approval_key()

            # 새 연결
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            self._ws = await self._session.ws_connect(
                self.ws_url,
                heartbeat=30.0,
            )
            self._stats["connected_at"] = datetime.now().isoformat()
            logger.info("WebSocket 재연결 성공.")

            # 기존 구독 복원
            for ticker in prev_subscriptions:
                await self.subscribe(ticker)
                await asyncio.sleep(0.1)  # 구독 간 짧은 지연

            # 메시지 루프 재시작
            self._message_task = asyncio.create_task(self._message_loop())

        except Exception as e:
            logger.error("재연결 실패: %s", e)
            self._stats["errors"] += 1
            if self._running:
                await self._handle_reconnect()

    def _reset_reconnect_delay(self) -> None:
        """재연결 지연 시간을 초기화한다."""
        self._reconnect_delay = self.RECONNECT_BASE_DELAY

    # ------------------------------------------------------------------
    # 종료
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        """WebSocket 연결을 종료한다.

        모든 구독을 해제하고 연결을 정리한다.
        """
        logger.info("WebSocket 연결 종료 시작...")
        self._running = False

        # 메시지 루프 취소
        if self._message_task is not None and not self._message_task.done():
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass

        # 구독 해제
        if self.is_connected():
            for ticker in list(self._subscriptions):
                try:
                    await self.unsubscribe(ticker)
                except Exception as e:
                    logger.debug("종료 시 구독 해제 오류 (무시): %s", e)

        await self._cleanup()
        logger.info("WebSocket 연결 종료 완료.")

    async def _cleanup_ws(self) -> None:
        """WebSocket 연결만 정리한다 (세션은 유지)."""
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _cleanup(self) -> None:
        """WebSocket과 HTTP 세션을 모두 정리한다."""
        await self._cleanup_ws()

        if self._session is not None and not self._session.closed:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
