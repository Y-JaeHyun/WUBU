"""ShortTermSignal, ShortTermStrategy, ShortTermTrader 테스트."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.strategy.short_term_base import ShortTermSignal, ShortTermStrategy
from src.execution.short_term_trader import ShortTermTrader


# ──────────────────────────────────────────────────────────
# Fixtures & Helpers
# ──────────────────────────────────────────────────────────

class DummyStrategy(ShortTermStrategy):
    """테스트용 더미 단기 전략."""

    name = "dummy_swing"
    mode = "swing"

    def __init__(self, signals=None):
        self._signals = signals or []

    def scan_signals(self, market_data: dict) -> list[ShortTermSignal]:
        return list(self._signals)

    def check_exit(self, position: dict, market_data: dict):
        return None


class FailingStrategy(ShortTermStrategy):
    """스캔 시 예외를 발생시키는 전략."""

    name = "failing"
    mode = "swing"

    def scan_signals(self, market_data: dict) -> list[ShortTermSignal]:
        raise RuntimeError("스캔 실패!")

    def check_exit(self, position: dict, market_data: dict):
        return None


def _make_signal(**kwargs) -> ShortTermSignal:
    """테스트용 시그널 생성 헬퍼."""
    defaults = {
        "id": "",
        "ticker": "005930",
        "strategy": "dummy_swing",
        "side": "buy",
        "mode": "swing",
        "confidence": 0.8,
        "target_price": 70000.0,
        "reason": "테스트 시그널",
    }
    defaults.update(kwargs)
    return ShortTermSignal(**defaults)


@pytest.fixture
def signals_path(tmp_path):
    """임시 시그널 JSON 경로."""
    return tmp_path / "signals.json"


@pytest.fixture
def trader(signals_path):
    """기본 ShortTermTrader (의존성 없음)."""
    return ShortTermTrader(signals_path=str(signals_path))


@pytest.fixture
def trader_with_strategy(signals_path):
    """더미 전략이 등록된 Trader."""
    strategy = DummyStrategy(signals=[_make_signal()])
    return ShortTermTrader(
        strategies=[strategy],
        signals_path=str(signals_path),
    )


# ──────────────────────────────────────────────────────────
# 1. ShortTermSignal 테스트
# ──────────────────────────────────────────────────────────

class TestShortTermSignal:
    """ShortTermSignal 데이터클래스 테스트."""

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict -> from_dict 라운드트립 정합성."""
        original = ShortTermSignal(
            id="sig_test_001",
            ticker="005930",
            strategy="swing_reversion",
            side="buy",
            mode="swing",
            confidence=0.85,
            target_price=70000.0,
            stop_loss_price=66500.0,
            take_profit_price=77000.0,
            reason="RSI 과매도 반등",
            state="pending",
            created_at="2026-02-22T10:00:00",
            expires_at="2026-02-22T10:30:00",
            confirmed_at=None,
            executed_at=None,
            metadata={"rsi": 28.5},
        )
        d = original.to_dict()
        restored = ShortTermSignal.from_dict(d)

        assert restored.id == original.id
        assert restored.ticker == original.ticker
        assert restored.strategy == original.strategy
        assert restored.side == original.side
        assert restored.mode == original.mode
        assert restored.confidence == original.confidence
        assert restored.target_price == original.target_price
        assert restored.stop_loss_price == original.stop_loss_price
        assert restored.take_profit_price == original.take_profit_price
        assert restored.reason == original.reason
        assert restored.state == original.state
        assert restored.created_at == original.created_at
        assert restored.expires_at == original.expires_at
        assert restored.confirmed_at == original.confirmed_at
        assert restored.executed_at == original.executed_at
        assert restored.metadata == original.metadata

    def test_default_values(self):
        """기본값이 올바르게 설정되는지 확인."""
        sig = ShortTermSignal(
            id="sig_001",
            ticker="000660",
            strategy="test",
            side="buy",
            mode="swing",
            confidence=0.5,
        )
        assert sig.target_price == 0.0
        assert sig.stop_loss_price == 0.0
        assert sig.take_profit_price == 0.0
        assert sig.reason == ""
        assert sig.state == "pending"
        assert sig.created_at == ""
        assert sig.expires_at == ""
        assert sig.confirmed_at is None
        assert sig.executed_at is None
        assert sig.metadata == {}

    def test_from_dict_with_missing_keys(self):
        """부분적인 딕셔너리에서도 from_dict가 동작하는지 확인."""
        sig = ShortTermSignal.from_dict({"id": "sig_partial", "ticker": "035720"})
        assert sig.id == "sig_partial"
        assert sig.ticker == "035720"
        assert sig.side == "buy"  # 기본값
        assert sig.confidence == 0.0

    def test_from_dict_empty(self):
        """빈 딕셔너리에서도 생성 가능."""
        sig = ShortTermSignal.from_dict({})
        assert sig.id == ""
        assert sig.ticker == ""

    def test_to_dict_contains_all_fields(self):
        """to_dict 결과에 모든 필드가 포함되는지 확인."""
        sig = _make_signal(id="sig_full")
        d = sig.to_dict()
        expected_keys = {
            "id", "ticker", "strategy", "side", "mode", "confidence",
            "target_price", "stop_loss_price", "take_profit_price",
            "reason", "state", "created_at", "expires_at",
            "confirmed_at", "executed_at", "metadata",
        }
        assert set(d.keys()) == expected_keys


# ──────────────────────────────────────────────────────────
# 2. ShortTermTrader 초기화 테스트
# ──────────────────────────────────────────────────────────

class TestShortTermTraderInit:
    """ShortTermTrader 초기화 테스트."""

    def test_default_init(self, signals_path):
        """모든 파라미터 None으로 초기화."""
        trader = ShortTermTrader(signals_path=str(signals_path))
        assert trader._allocator is None
        assert trader._risk_manager is None
        assert trader._order_manager is None
        assert trader._strategies == []
        assert trader._signals == []
        assert trader._confirm_timeout == 30

    def test_init_with_strategies(self, signals_path):
        """전략 리스트와 함께 초기화."""
        s1 = DummyStrategy()
        s2 = DummyStrategy()
        trader = ShortTermTrader(strategies=[s1, s2], signals_path=str(signals_path))
        assert len(trader._strategies) == 2

    def test_register_strategy(self, trader):
        """전략 등록 테스트."""
        assert len(trader._strategies) == 0
        strategy = DummyStrategy()
        trader.register_strategy(strategy)
        assert len(trader._strategies) == 1
        assert trader._strategies[0].name == "dummy_swing"

    def test_custom_confirm_timeout(self, signals_path):
        """사용자 지정 확인 타임아웃."""
        trader = ShortTermTrader(
            signals_path=str(signals_path),
            confirm_timeout_minutes=60,
        )
        assert trader._confirm_timeout == 60


# ──────────────────────────────────────────────────────────
# 3. 시그널 스캔 테스트
# ──────────────────────────────────────────────────────────

class TestScanSignals:
    """scan_for_signals 테스트."""

    def test_scan_creates_signals(self, signals_path):
        """mock 전략으로 시그널 스캔."""
        strategy = DummyStrategy(signals=[_make_signal()])
        trader = ShortTermTrader(strategies=[strategy], signals_path=str(signals_path))

        results = trader.scan_for_signals()
        assert len(results) == 1
        assert results[0].ticker == "005930"

    def test_auto_id_generation(self, signals_path):
        """ID가 비어있으면 자동 생성."""
        strategy = DummyStrategy(signals=[_make_signal(id="")])
        trader = ShortTermTrader(strategies=[strategy], signals_path=str(signals_path))

        results = trader.scan_for_signals()
        assert results[0].id.startswith("sig_")
        assert len(results[0].id) > 10

    def test_auto_created_at(self, signals_path):
        """created_at 자동 설정."""
        strategy = DummyStrategy(signals=[_make_signal()])
        trader = ShortTermTrader(strategies=[strategy], signals_path=str(signals_path))

        results = trader.scan_for_signals()
        assert results[0].created_at != ""
        # ISO format 파싱 가능 확인
        datetime.fromisoformat(results[0].created_at)

    def test_auto_expires_at(self, signals_path):
        """expires_at 자동 설정 (confirm_timeout 기반)."""
        strategy = DummyStrategy(signals=[_make_signal()])
        trader = ShortTermTrader(
            strategies=[strategy],
            signals_path=str(signals_path),
            confirm_timeout_minutes=15,
        )

        results = trader.scan_for_signals()
        expires = datetime.fromisoformat(results[0].expires_at)
        created = datetime.fromisoformat(results[0].created_at)
        diff = (expires - created).total_seconds()
        # 약 15분 (오차 허용 5초)
        assert abs(diff - 900) < 5

    def test_scan_with_failing_strategy(self, signals_path):
        """스캔 실패 시 에러 처리 (다른 전략은 계속 진행)."""
        good = DummyStrategy(signals=[_make_signal()])
        bad = FailingStrategy()
        trader = ShortTermTrader(
            strategies=[bad, good],
            signals_path=str(signals_path),
        )

        results = trader.scan_for_signals()
        # bad는 실패, good은 성공
        assert len(results) == 1

    def test_scan_multiple_strategies(self, signals_path):
        """여러 전략에서 시그널 수집."""
        s1 = DummyStrategy(signals=[_make_signal(ticker="005930")])
        s2 = DummyStrategy(signals=[_make_signal(ticker="000660"), _make_signal(ticker="035720")])
        trader = ShortTermTrader(strategies=[s1, s2], signals_path=str(signals_path))

        results = trader.scan_for_signals()
        assert len(results) == 3
        tickers = {r.ticker for r in results}
        assert tickers == {"005930", "000660", "035720"}

    def test_scan_persists_to_file(self, signals_path):
        """스캔 결과가 JSON에 저장되는지 확인."""
        strategy = DummyStrategy(signals=[_make_signal()])
        trader = ShortTermTrader(strategies=[strategy], signals_path=str(signals_path))

        trader.scan_for_signals()
        assert signals_path.exists()
        data = json.loads(signals_path.read_text(encoding="utf-8"))
        assert len(data["signals"]) == 1

    def test_scan_no_signals(self, signals_path):
        """시그널이 없을 때 빈 리스트 반환."""
        strategy = DummyStrategy(signals=[])
        trader = ShortTermTrader(strategies=[strategy], signals_path=str(signals_path))

        results = trader.scan_for_signals()
        assert results == []

    def test_scan_no_strategies(self, trader):
        """전략이 없을 때."""
        results = trader.scan_for_signals()
        assert results == []


# ──────────────────────────────────────────────────────────
# 4. 시그널 조회 테스트
# ──────────────────────────────────────────────────────────

class TestSignalQueries:
    """시그널 조회 테스트."""

    def test_get_pending_signals(self, signals_path):
        """pending 상태 시그널 조회."""
        strategy = DummyStrategy(signals=[_make_signal(), _make_signal(ticker="000660")])
        trader = ShortTermTrader(strategies=[strategy], signals_path=str(signals_path))
        trader.scan_for_signals()

        pending = trader.get_pending_signals()
        assert len(pending) == 2
        assert all(s["state"] == "pending" for s in pending)

    def test_get_signal_by_id_found(self, signals_path):
        """ID로 시그널 조회 성공."""
        strategy = DummyStrategy(signals=[_make_signal(id="sig_test_001")])
        trader = ShortTermTrader(strategies=[strategy], signals_path=str(signals_path))
        trader.scan_for_signals()

        result = trader.get_signal_by_id("sig_test_001")
        assert result is not None
        assert result["id"] == "sig_test_001"

    def test_get_signal_by_id_not_found(self, trader):
        """존재하지 않는 ID로 조회."""
        result = trader.get_signal_by_id("nonexistent")
        assert result is None

    def test_get_active_signals(self, signals_path):
        """활성 시그널 조회 (pending, confirmed, executing)."""
        trader = ShortTermTrader(signals_path=str(signals_path))
        # 직접 시그널 삽입
        trader._signals = [
            {"id": "s1", "state": "pending"},
            {"id": "s2", "state": "confirmed"},
            {"id": "s3", "state": "executing"},
            {"id": "s4", "state": "done"},
            {"id": "s5", "state": "expired"},
            {"id": "s6", "state": "rejected"},
        ]

        active = trader.get_active_signals()
        assert len(active) == 3
        active_ids = {s["id"] for s in active}
        assert active_ids == {"s1", "s2", "s3"}

    def test_get_all_signals(self, signals_path):
        """모든 시그널 조회."""
        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [
            {"id": "s1", "state": "pending"},
            {"id": "s2", "state": "done"},
        ]

        all_signals = trader.get_all_signals()
        assert len(all_signals) == 2

    def test_get_all_signals_returns_copy(self, signals_path):
        """get_all_signals가 원본 리스트와 독립적인지 확인."""
        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [{"id": "s1", "state": "pending"}]

        all_signals = trader.get_all_signals()
        all_signals.append({"id": "s_extra"})
        assert len(trader._signals) == 1  # 원본 불변


# ──────────────────────────────────────────────────────────
# 5. confirm / reject 테스트
# ──────────────────────────────────────────────────────────

class TestConfirmReject:
    """시그널 확인/거절 테스트."""

    def test_confirm_success(self, signals_path):
        """시그널 확인 성공."""
        strategy = DummyStrategy(signals=[_make_signal(id="sig_c1")])
        trader = ShortTermTrader(strategies=[strategy], signals_path=str(signals_path))
        trader.scan_for_signals()

        ok, msg = trader.confirm_signal("sig_c1")
        assert ok is True
        assert "확인" in msg

        signal = trader.get_signal_by_id("sig_c1")
        assert signal["state"] == "confirmed"
        assert signal["confirmed_at"] is not None

    def test_confirm_non_pending_fails(self, signals_path):
        """pending이 아닌 상태에서 confirm 실패."""
        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [{"id": "s1", "state": "done"}]

        ok, msg = trader.confirm_signal("s1")
        assert ok is False
        assert "확인 불가" in msg

    def test_confirm_expired_signal(self, signals_path):
        """만료된 시그널 confirm 시 실패."""
        past = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [{"id": "s_exp", "state": "pending", "expires_at": past}]

        ok, msg = trader.confirm_signal("s_exp")
        assert ok is False
        assert "만료" in msg

        signal = trader.get_signal_by_id("s_exp")
        assert signal["state"] == "expired"

    def test_confirm_nonexistent_id(self, trader):
        """존재하지 않는 ID로 confirm."""
        ok, msg = trader.confirm_signal("no_such_id")
        assert ok is False
        assert "찾을 수 없습니다" in msg

    def test_reject_success(self, signals_path):
        """시그널 거절 성공."""
        strategy = DummyStrategy(signals=[_make_signal(id="sig_r1")])
        trader = ShortTermTrader(strategies=[strategy], signals_path=str(signals_path))
        trader.scan_for_signals()

        ok, msg = trader.reject_signal("sig_r1")
        assert ok is True
        assert "거절" in msg

        signal = trader.get_signal_by_id("sig_r1")
        assert signal["state"] == "rejected"

    def test_reject_non_pending_fails(self, signals_path):
        """pending이 아닌 상태에서 reject 실패."""
        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [{"id": "s1", "state": "confirmed"}]

        ok, msg = trader.reject_signal("s1")
        assert ok is False
        assert "거절 불가" in msg

    def test_reject_nonexistent_id(self, trader):
        """존재하지 않는 ID로 reject."""
        ok, msg = trader.reject_signal("no_such")
        assert ok is False
        assert "찾을 수 없습니다" in msg

    def test_confirm_with_risk_check_pass(self, signals_path):
        """리스크 체크 통과 시 confirm 성공."""
        mock_allocator = MagicMock()
        mock_allocator.get_short_term_budget.return_value = 1_000_000

        mock_risk = MagicMock()
        mock_risk.config.max_concurrent_positions = 3
        mock_risk.check_can_open.return_value = (True, "")

        strategy = DummyStrategy(signals=[_make_signal(id="sig_risk_ok")])
        trader = ShortTermTrader(
            allocator=mock_allocator,
            risk_manager=mock_risk,
            strategies=[strategy],
            signals_path=str(signals_path),
        )
        trader.scan_for_signals()

        ok, msg = trader.confirm_signal("sig_risk_ok")
        assert ok is True

    def test_confirm_with_risk_check_fail(self, signals_path):
        """리스크 체크 실패 시 confirm 거부."""
        mock_allocator = MagicMock()
        mock_allocator.get_short_term_budget.return_value = 1_000_000

        mock_risk = MagicMock()
        mock_risk.config.max_concurrent_positions = 3
        mock_risk.check_can_open.return_value = (False, "일일 손실 한도 도달")

        strategy = DummyStrategy(signals=[_make_signal(id="sig_risk_fail")])
        trader = ShortTermTrader(
            allocator=mock_allocator,
            risk_manager=mock_risk,
            strategies=[strategy],
            signals_path=str(signals_path),
        )
        trader.scan_for_signals()

        ok, msg = trader.confirm_signal("sig_risk_fail")
        assert ok is False
        assert "리스크" in msg


# ──────────────────────────────────────────────────────────
# 6. 실행 테스트
# ──────────────────────────────────────────────────────────

class TestExecuteSignals:
    """execute_confirmed_signals 테스트."""

    def test_execute_success(self, signals_path):
        """주문 실행 성공."""
        mock_order_mgr = MagicMock()
        mock_order = MagicMock()
        mock_order.to_dict.return_value = {"order_no": "1234", "status": "filled"}
        mock_order_mgr.submit_order.return_value = mock_order

        mock_allocator = MagicMock()
        mock_allocator.get_short_term_cash.return_value = 300_000

        trader = ShortTermTrader(
            allocator=mock_allocator,
            order_manager=mock_order_mgr,
            signals_path=str(signals_path),
        )
        trader._signals = [{
            "id": "sig_exec_1",
            "ticker": "005930",
            "side": "buy",
            "mode": "swing",
            "strategy": "test",
            "state": "confirmed",
            "target_price": 70000,
            "metadata": {},
        }]

        results = trader.execute_confirmed_signals()
        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["order"]["order_no"] == "1234"

        signal = trader.get_signal_by_id("sig_exec_1")
        assert signal["state"] == "done"
        assert signal["executed_at"] is not None

    def test_execute_no_order_manager(self, signals_path):
        """OrderManager 없을 때 에러."""
        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [{
            "id": "sig_no_om",
            "ticker": "005930",
            "side": "buy",
            "state": "confirmed",
            "target_price": 70000,
            "metadata": {},
        }]

        results = trader.execute_confirmed_signals()
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "OrderManager" in results[0]["error"]

        # 상태 롤백 확인
        signal = trader.get_signal_by_id("sig_no_om")
        assert signal["state"] == "confirmed"

    def test_execute_order_failure_rollback(self, signals_path):
        """주문 실패 시 상태 롤백 (confirmed로 복원)."""
        mock_order_mgr = MagicMock()
        mock_order_mgr.submit_order.side_effect = RuntimeError("네트워크 오류")

        mock_allocator = MagicMock()
        mock_allocator.get_short_term_cash.return_value = 300_000

        trader = ShortTermTrader(
            allocator=mock_allocator,
            order_manager=mock_order_mgr,
            signals_path=str(signals_path),
        )
        trader._signals = [{
            "id": "sig_fail",
            "ticker": "005930",
            "side": "buy",
            "state": "confirmed",
            "target_price": 70000,
            "metadata": {},
        }]

        results = trader.execute_confirmed_signals()
        assert results[0]["success"] is False
        assert "네트워크 오류" in results[0]["error"]

        signal = trader.get_signal_by_id("sig_fail")
        assert signal["state"] == "confirmed"

    def test_execute_zero_qty(self, signals_path):
        """예산 부족으로 수량 0일 때."""
        mock_order_mgr = MagicMock()
        mock_allocator = MagicMock()
        mock_allocator.get_short_term_cash.return_value = 100  # 극히 적은 예산

        trader = ShortTermTrader(
            allocator=mock_allocator,
            order_manager=mock_order_mgr,
            signals_path=str(signals_path),
        )
        trader._signals = [{
            "id": "sig_zero",
            "ticker": "005930",
            "side": "buy",
            "state": "confirmed",
            "target_price": 70000,
            "metadata": {},
        }]

        results = trader.execute_confirmed_signals()
        assert results[0]["success"] is False
        assert "수량 0" in results[0]["error"]

    def test_execute_sell_uses_metadata_qty(self, signals_path):
        """매도 시 metadata에서 수량 사용."""
        mock_order_mgr = MagicMock()
        mock_order = MagicMock()
        mock_order.to_dict.return_value = {"order_no": "5678"}
        mock_order_mgr.submit_order.return_value = mock_order

        trader = ShortTermTrader(
            order_manager=mock_order_mgr,
            signals_path=str(signals_path),
        )
        trader._signals = [{
            "id": "sig_sell",
            "ticker": "005930",
            "side": "sell",
            "state": "confirmed",
            "target_price": 72000,
            "metadata": {"qty": 5},
        }]

        results = trader.execute_confirmed_signals()
        assert results[0]["success"] is True
        mock_order_mgr.submit_order.assert_called_once_with(
            ticker="005930",
            side="sell",
            qty=5,
            order_type="시장가",
        )

    def test_execute_tags_position(self, signals_path):
        """매수 성공 시 포지션 태깅."""
        mock_order_mgr = MagicMock()
        mock_order = MagicMock()
        mock_order.to_dict.return_value = {"order_no": "9999"}
        mock_order_mgr.submit_order.return_value = mock_order

        mock_allocator = MagicMock()
        mock_allocator.get_short_term_cash.return_value = 500_000

        trader = ShortTermTrader(
            allocator=mock_allocator,
            order_manager=mock_order_mgr,
            signals_path=str(signals_path),
        )
        trader._signals = [{
            "id": "sig_tag",
            "ticker": "005930",
            "side": "buy",
            "mode": "swing",
            "strategy": "swing_test",
            "state": "confirmed",
            "target_price": 70000,
            "metadata": {},
        }]

        trader.execute_confirmed_signals()
        mock_allocator.tag_position.assert_called_once()
        call_args = mock_allocator.tag_position.call_args
        assert call_args[0][0] == "005930"
        assert call_args[0][1] == "short_term"

    def test_execute_skips_non_confirmed(self, signals_path):
        """confirmed가 아닌 시그널은 실행하지 않음."""
        mock_order_mgr = MagicMock()
        trader = ShortTermTrader(
            order_manager=mock_order_mgr,
            signals_path=str(signals_path),
        )
        trader._signals = [
            {"id": "s1", "state": "pending"},
            {"id": "s2", "state": "done"},
            {"id": "s3", "state": "expired"},
        ]

        results = trader.execute_confirmed_signals()
        assert len(results) == 0
        mock_order_mgr.submit_order.assert_not_called()


# ──────────────────────────────────────────────────────────
# 7. 만료 테스트
# ──────────────────────────────────────────────────────────

class TestExpiration:
    """만료 및 정리 테스트."""

    def test_expire_old_signals(self, signals_path):
        """만료된 pending 시그널 자동 상태 변경."""
        past = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
        future = (datetime.now() + timedelta(hours=1)).isoformat(timespec="seconds")

        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [
            {"id": "s_old", "state": "pending", "expires_at": past},
            {"id": "s_new", "state": "pending", "expires_at": future},
        ]

        trader._expire_old_signals()
        assert trader.get_signal_by_id("s_old")["state"] == "expired"
        assert trader.get_signal_by_id("s_new")["state"] == "pending"

    def test_expire_does_not_touch_non_pending(self, signals_path):
        """pending이 아닌 시그널은 만료 처리하지 않음."""
        past = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")

        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [
            {"id": "s_done", "state": "done", "expires_at": past},
            {"id": "s_confirmed", "state": "confirmed", "expires_at": past},
        ]

        trader._expire_old_signals()
        assert trader.get_signal_by_id("s_done")["state"] == "done"
        assert trader.get_signal_by_id("s_confirmed")["state"] == "confirmed"

    def test_get_pending_triggers_expire(self, signals_path):
        """get_pending_signals가 만료 처리를 트리거하는지 확인."""
        past = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")

        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [
            {"id": "s_expired", "state": "pending", "expires_at": past},
            {"id": "s_valid", "state": "pending", "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(timespec="seconds")},
        ]

        pending = trader.get_pending_signals()
        assert len(pending) == 1
        assert pending[0]["id"] == "s_valid"

    def test_cleanup_old_signals(self, signals_path):
        """오래된 완료/만료 시그널 정리."""
        old_date = (datetime.now() - timedelta(days=10)).isoformat(timespec="seconds")
        recent_date = (datetime.now() - timedelta(days=1)).isoformat(timespec="seconds")

        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [
            {"id": "s_old_done", "state": "done", "created_at": old_date},
            {"id": "s_old_expired", "state": "expired", "created_at": old_date},
            {"id": "s_recent_done", "state": "done", "created_at": recent_date},
            {"id": "s_active", "state": "pending", "created_at": old_date},  # 활성은 보존
        ]

        removed = trader.cleanup_old_signals(max_age_days=7)
        assert removed == 2
        remaining_ids = {s["id"] for s in trader._signals}
        assert "s_old_done" not in remaining_ids
        assert "s_old_expired" not in remaining_ids
        assert "s_recent_done" in remaining_ids
        assert "s_active" in remaining_ids

    def test_cleanup_no_old_signals(self, signals_path):
        """정리할 시그널이 없을 때."""
        recent = datetime.now().isoformat(timespec="seconds")
        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [
            {"id": "s1", "state": "done", "created_at": recent},
        ]

        removed = trader.cleanup_old_signals(max_age_days=7)
        assert removed == 0

    def test_cleanup_preserves_active_signals(self, signals_path):
        """cleanup이 활성 시그널(pending/confirmed/executing)을 보존하는지 확인."""
        old_date = (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")
        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [
            {"id": "s_p", "state": "pending", "created_at": old_date},
            {"id": "s_c", "state": "confirmed", "created_at": old_date},
            {"id": "s_e", "state": "executing", "created_at": old_date},
        ]

        removed = trader.cleanup_old_signals(max_age_days=1)
        assert removed == 0
        assert len(trader._signals) == 3


# ──────────────────────────────────────────────────────────
# 8. 영속화 테스트
# ──────────────────────────────────────────────────────────

class TestPersistence:
    """JSON 영속화 테스트."""

    def test_save_load_roundtrip(self, signals_path):
        """저장 후 로드 정합성."""
        trader1 = ShortTermTrader(signals_path=str(signals_path))
        trader1._signals = [
            {
                "id": "sig_persist",
                "ticker": "005930",
                "state": "pending",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        ]
        trader1._save()

        # 새 인스턴스에서 로드
        trader2 = ShortTermTrader(signals_path=str(signals_path))
        assert len(trader2._signals) == 1
        assert trader2._signals[0]["id"] == "sig_persist"

    def test_load_corrupted_file(self, signals_path):
        """손상된 JSON 파일 처리."""
        signals_path.write_text("{{not valid json}}", encoding="utf-8")

        trader = ShortTermTrader(signals_path=str(signals_path))
        assert trader._signals == []

    def test_load_nonexistent_file(self, signals_path):
        """존재하지 않는 파일 로드."""
        assert not signals_path.exists()
        trader = ShortTermTrader(signals_path=str(signals_path))
        assert trader._signals == []

    def test_save_creates_directory(self, tmp_path):
        """디렉토리가 없으면 자동 생성."""
        nested = tmp_path / "deep" / "nested" / "signals.json"
        trader = ShortTermTrader(signals_path=str(nested))
        trader._signals = [{"id": "s1", "state": "pending"}]
        trader._save()

        assert nested.exists()
        data = json.loads(nested.read_text(encoding="utf-8"))
        assert len(data["signals"]) == 1

    def test_save_format(self, signals_path):
        """저장된 JSON 포맷 검증."""
        trader = ShortTermTrader(signals_path=str(signals_path))
        trader._signals = [{"id": "s1", "state": "pending"}]
        trader._save()

        data = json.loads(signals_path.read_text(encoding="utf-8"))
        assert "updated_at" in data
        assert "signals" in data
        assert isinstance(data["signals"], list)
        # updated_at 파싱 가능
        datetime.fromisoformat(data["updated_at"])

    def test_load_empty_signals_list(self, signals_path):
        """빈 시그널 리스트 파일 로드."""
        signals_path.write_text(
            json.dumps({"signals": [], "updated_at": "2026-01-01T00:00:00"}),
            encoding="utf-8",
        )
        trader = ShortTermTrader(signals_path=str(signals_path))
        assert trader._signals == []

    def test_multiple_save_load_cycles(self, signals_path):
        """다중 저장/로드 사이클."""
        trader = ShortTermTrader(signals_path=str(signals_path))

        for i in range(5):
            trader._signals.append({"id": f"s_{i}", "state": "pending"})
            trader._save()

        trader2 = ShortTermTrader(signals_path=str(signals_path))
        assert len(trader2._signals) == 5

    def test_scan_then_reload(self, signals_path):
        """스캔 후 새 인스턴스에서 로드하여 상태 유지 확인."""
        strategy = DummyStrategy(signals=[_make_signal(id="sig_reload")])
        trader1 = ShortTermTrader(strategies=[strategy], signals_path=str(signals_path))
        trader1.scan_for_signals()
        trader1.confirm_signal("sig_reload")

        trader2 = ShortTermTrader(signals_path=str(signals_path))
        signal = trader2.get_signal_by_id("sig_reload")
        assert signal is not None
        assert signal["state"] == "confirmed"
