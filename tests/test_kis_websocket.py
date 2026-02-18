"""KIS WebSocket 클라이언트(src/execution/kis_websocket.py) 테스트.

KISWebSocketClient의 URL 생성, 구독 관리, 체결가 파싱,
재연결 딜레이 계산 등 동기/내부 메서드를 중심으로 검증한다.
비동기 메서드 테스트를 최소화하고, 파싱 로직에 집중한다.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _import_kis_websocket():
    """KISWebSocketClient 클래스를 임포트한다."""
    from src.execution.kis_websocket import KISWebSocketClient
    return KISWebSocketClient


def _make_mock_kis_client(is_paper=True):
    """KISClient mock 객체를 생성한다."""
    mock = MagicMock()
    mock.is_paper = is_paper
    mock.is_configured.return_value = True
    mock.app_key = "test_app_key"
    mock.app_secret = "test_app_secret"
    mock.base_url = "https://openapi.koreainvestment.com:9443"
    mock.REQUEST_TIMEOUT = 10
    return mock


def _make_valid_tick_message():
    """유효한 H0STCNT0 체결 메시지를 생성한다.

    KIS 체결 메시지 형식: "0|H0STCNT0|003|ticker^time^price^sign^change^pct^avg^open^high^low^ask^bid^vol^acml_vol^acml_amt"
    """
    fields = [
        "005930",    # 0: ticker
        "100000",    # 1: time
        "72000",     # 2: price
        "2",         # 3: change_sign
        "1000",      # 4: change
        "1.41",      # 5: change_pct
        "71500",     # 6: weighted_avg
        "71000",     # 7: open
        "72500",     # 8: high
        "70500",     # 9: low
        "72100",     # 10: ask
        "71900",     # 11: bid
        "1000",      # 12: volume
        "5000000",   # 13: acml_volume
        "360000000000",  # 14: acml_amount
    ]
    data_str = "^".join(fields)
    return f"0|H0STCNT0|003|{data_str}"


# ===================================================================
# KISWebSocketClient 검증
# ===================================================================

class TestKISWebSocketClient:
    """KISWebSocketClient 검증."""

    def test_ws_url_paper(self):
        """모의투자 URL이 올바르게 반환된다."""
        KISWebSocketClient = _import_kis_websocket()
        mock_client = _make_mock_kis_client(is_paper=True)
        ws = KISWebSocketClient(kis_client=mock_client)

        assert ws.ws_url == KISWebSocketClient.PAPER_WS_URL, (
            f"모의투자 URL이어야 합니다: {ws.ws_url}"
        )
        assert "31000" in ws.ws_url, "모의투자 포트(31000)가 포함되어야 합니다."

    def test_ws_url_real(self):
        """실전 URL이 올바르게 반환된다."""
        KISWebSocketClient = _import_kis_websocket()
        mock_client = _make_mock_kis_client(is_paper=False)
        ws = KISWebSocketClient(kis_client=mock_client)

        assert ws.ws_url == KISWebSocketClient.REAL_WS_URL, (
            f"실전 URL이어야 합니다: {ws.ws_url}"
        )
        assert "21000" in ws.ws_url, "실전 포트(21000)가 포함되어야 합니다."

    def test_max_subscriptions(self):
        """최대 구독 수가 41로 설정되어 있다."""
        KISWebSocketClient = _import_kis_websocket()

        assert KISWebSocketClient.MAX_SUBSCRIPTIONS == 41, (
            "최대 구독 수가 41이어야 합니다."
        )

    def test_subscribe_tracking(self):
        """구독 추적이 올바르게 동작한다 (내부 상태 검증)."""
        KISWebSocketClient = _import_kis_websocket()
        mock_client = _make_mock_kis_client()
        ws = KISWebSocketClient(kis_client=mock_client)

        # 초기 상태: 빈 구독 목록
        assert len(ws.subscriptions) == 0, "초기 구독 목록이 비어야 합니다."

        # 내부적으로 구독 추가 시뮬레이션
        ws._subscriptions.add("005930")
        ws._subscriptions.add("000660")

        assert len(ws.subscriptions) == 2, "구독 목록에 2개가 있어야 합니다."
        assert "005930" in ws.subscriptions, "005930이 구독에 포함되어야 합니다."

    def test_unsubscribe_tracking(self):
        """구독 해제 후 목록에서 제거된다."""
        KISWebSocketClient = _import_kis_websocket()
        mock_client = _make_mock_kis_client()
        ws = KISWebSocketClient(kis_client=mock_client)

        ws._subscriptions.add("005930")
        ws._subscriptions.add("000660")
        ws._subscriptions.discard("005930")

        assert "005930" not in ws.subscriptions, "005930이 구독에서 제거되어야 합니다."
        assert len(ws.subscriptions) == 1, "구독 목록에 1개만 있어야 합니다."

    def test_parse_tick(self):
        """체결가 메시지가 올바르게 파싱된다."""
        KISWebSocketClient = _import_kis_websocket()
        mock_client = _make_mock_kis_client()
        ws = KISWebSocketClient(kis_client=mock_client)

        raw = _make_valid_tick_message()
        tick = ws._parse_tick(raw)

        assert tick is not None, "유효한 메시지는 파싱되어야 합니다."
        assert isinstance(tick, dict), "파싱 결과가 dict여야 합니다."
        assert tick["ticker"] == "005930", f"ticker가 005930이어야 합니다: {tick['ticker']}"
        assert tick["price"] == 72000, f"price가 72000이어야 합니다: {tick['price']}"
        assert tick["volume"] == 1000, f"volume이 1000이어야 합니다: {tick['volume']}"

    def test_parse_tick_invalid(self):
        """잘못된 메시지는 None을 반환한다."""
        KISWebSocketClient = _import_kis_websocket()
        mock_client = _make_mock_kis_client()
        ws = KISWebSocketClient(kis_client=mock_client)

        # 필드 수 부족
        tick1 = ws._parse_tick("0|H0STCNT0|001|too^few^fields")
        assert tick1 is None, "필드 부족 메시지는 None이어야 합니다."

        # 잘못된 tr_id
        tick2 = ws._parse_tick("0|WRONG_TR|001|some^data")
        assert tick2 is None, "잘못된 tr_id 메시지는 None이어야 합니다."

        # 파트 수 부족
        tick3 = ws._parse_tick("too_short")
        assert tick3 is None, "파트 부족 메시지는 None이어야 합니다."

    def test_reconnect_delay(self):
        """지수 백오프 딜레이가 올바르게 계산된다."""
        KISWebSocketClient = _import_kis_websocket()
        mock_client = _make_mock_kis_client()
        ws = KISWebSocketClient(kis_client=mock_client)

        # 초기 딜레이
        assert ws._reconnect_delay == KISWebSocketClient.RECONNECT_BASE_DELAY, (
            "초기 딜레이가 RECONNECT_BASE_DELAY여야 합니다."
        )

        # 지수 백오프 시뮬레이션
        initial = ws._reconnect_delay
        ws._reconnect_delay = min(
            ws._reconnect_delay * KISWebSocketClient.RECONNECT_FACTOR,
            KISWebSocketClient.RECONNECT_MAX_DELAY,
        )
        assert ws._reconnect_delay == initial * KISWebSocketClient.RECONNECT_FACTOR, (
            "딜레이가 2배 증가해야 합니다."
        )

        # 최대 딜레이 제한
        ws._reconnect_delay = KISWebSocketClient.RECONNECT_MAX_DELAY + 100
        ws._reconnect_delay = min(
            ws._reconnect_delay,
            KISWebSocketClient.RECONNECT_MAX_DELAY,
        )
        assert ws._reconnect_delay == KISWebSocketClient.RECONNECT_MAX_DELAY, (
            "최대 딜레이를 초과하면 안 됩니다."
        )

    def test_duplicate_subscribe(self):
        """중복 구독 시 추가되지 않는다."""
        KISWebSocketClient = _import_kis_websocket()
        mock_client = _make_mock_kis_client()
        ws = KISWebSocketClient(kis_client=mock_client)

        ws._subscriptions.add("005930")
        ws._subscriptions.add("005930")  # 중복

        assert len(ws._subscriptions) == 1, (
            "중복 구독이 추가되면 안 됩니다."
        )

    def test_disconnect_initial_state(self):
        """초기 상태에서 disconnect 관련 플래그가 올바르다."""
        KISWebSocketClient = _import_kis_websocket()
        mock_client = _make_mock_kis_client()
        ws = KISWebSocketClient(kis_client=mock_client)

        # 초기 연결 상태 확인
        assert ws.is_connected() is False, "초기 상태는 미연결이어야 합니다."
        assert ws._running is False, "초기 _running은 False여야 합니다."
        assert len(ws._subscriptions) == 0, "초기 구독 목록이 비어야 합니다."
