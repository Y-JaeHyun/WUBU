"""통합 리밸런싱 테스트.

RebalanceExecutor의 execute_integrated_rebalance, dry_run_integrated,
_execute_order_batch 메서드를 검증한다.
모든 외부 의존성(KISClient, PositionManager, RiskGuard, PortfolioAllocator)은
mock 처리한다.
"""

import os
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from src.execution.executor import RebalanceExecutor


# ===================================================================
# 헬퍼
# ===================================================================


def _mock_kis(is_paper=True, configured=True):
    """테스트용 KISClient mock을 생성한다."""
    kis = MagicMock()
    kis.is_configured.return_value = configured
    kis.is_paper = is_paper
    kis.mode_tag = "[모의]" if is_paper else "[실전]"
    kis.get_balance.return_value = {
        "total_eval": 10_000_000,
        "cash": 5_000_000,
    }
    kis.get_positions.return_value = pd.DataFrame()
    kis.get_current_price.return_value = {"price": 50_000}
    return kis


def _mock_risk_guard(
    rebalance_pass=True,
    rebalance_warnings=None,
    turnover_pass=True,
    turnover_reason="",
    order_pass=True,
    order_reason="",
):
    """테스트용 RiskGuard mock을 생성한다."""
    rg = MagicMock()
    rg.check_rebalance.return_value = (
        rebalance_pass,
        rebalance_warnings or [],
    )
    rg.check_turnover.return_value = (turnover_pass, turnover_reason)
    rg.check_order.return_value = (order_pass, order_reason)
    return rg


def _mock_allocator(
    long_term_pct=0.70,
    etf_rotation_pct=0.30,
    short_term_pct=0.0,
):
    """테스트용 PortfolioAllocator mock을 생성한다."""
    alloc = MagicMock()
    alloc._long_term_pct = long_term_pct
    alloc._etf_rotation_pct = etf_rotation_pct
    alloc._short_term_pct = short_term_pct

    def _get_pool_pct(pool_name):
        return {
            "long_term": long_term_pct,
            "etf_rotation": etf_rotation_pct,
            "short_term": short_term_pct,
        }.get(pool_name, 0.0)

    alloc.get_pool_pct.side_effect = _get_pool_pct

    def _merge_pool_targets(pool_signals):
        merged = {}
        for pool_name, signals in pool_signals.items():
            pct = _get_pool_pct(pool_name)
            if pct <= 0 or not signals:
                continue
            for ticker, weight in signals.items():
                scaled = round(weight * pct, 6)
                merged[ticker] = round(merged.get(ticker, 0.0) + scaled, 6)
        return merged

    alloc.merge_pool_targets.side_effect = _merge_pool_targets
    alloc.get_positions_by_pool.return_value = []
    alloc.auto_tag_from_pool_signals.return_value = None

    return alloc


def _make_order_mock(ticker, side, status="submitted"):
    """OrderManager.submit_order가 반환하는 Order mock을 생성한다."""
    order = MagicMock()
    order.ticker = ticker
    order.side = side
    order.status = status
    order.error_msg = "" if status != "failed" else f"{ticker} 주문 실패"
    order.to_dict.return_value = {
        "ticker": ticker,
        "side": side,
        "status": status,
        "error_msg": order.error_msg,
    }
    return order


def _build_executor(
    kis=None,
    risk_guard=None,
    allocator=None,
):
    """테스트용 RebalanceExecutor를 생성한다.

    내부적으로 생성되는 PositionManager와 OrderManager를 mock으로 교체한다.
    """
    kis = kis or _mock_kis()
    risk_guard = risk_guard or _mock_risk_guard()

    with patch(
        "src.execution.executor.PositionManager"
    ) as MockPM, patch(
        "src.execution.executor.OrderManager"
    ) as MockOM:
        mock_pm = MagicMock()
        mock_pm.get_current_positions.return_value = {}
        mock_pm.get_portfolio_value.return_value = 10_000_000
        mock_pm.calculate_rebalance_orders.return_value = ([], [])
        MockPM.return_value = mock_pm

        mock_om = MagicMock()
        mock_om.sync_order_status.return_value = None
        MockOM.return_value = mock_om

        executor = RebalanceExecutor(
            kis_client=kis,
            risk_guard=risk_guard,
            allocator=allocator,
        )

    return executor


# ===================================================================
# dry_run_integrated 테스트
# ===================================================================


class TestDryRunIntegrated:
    """dry_run_integrated 메서드를 검증한다."""

    def test_basic_merged_dry_run_returns_correct_structure(self):
        """기본 병합 dry run이 올바른 구조를 반환한다."""
        allocator = _mock_allocator()
        executor = _build_executor(allocator=allocator)

        pool_signals = {
            "long_term": {"005930": 0.5, "000660": 0.5},
            "etf_rotation": {"069500": 0.5, "371460": 0.5},
        }
        result = executor.dry_run_integrated(pool_signals)

        assert result["is_dry_run"] is True
        assert result["integrated"] is True
        assert "merged_weights" in result
        assert "pool_breakdown" in result
        assert "sell_orders" in result
        assert "buy_orders" in result
        assert "total_sell_amount" in result
        assert "total_buy_amount" in result
        assert "net_cash_flow" in result
        assert "risk_check" in result
        assert "turnover_check" in result

    def test_pool_breakdown_includes_original_count_and_scaled_weights(self):
        """풀 분해 정보에 original_count와 scaled_weights가 포함된다."""
        allocator = _mock_allocator(long_term_pct=0.70, etf_rotation_pct=0.30)
        executor = _build_executor(allocator=allocator)

        pool_signals = {
            "long_term": {"005930": 0.5, "000660": 0.5},
            "etf_rotation": {"069500": 1.0},
        }
        result = executor.dry_run_integrated(pool_signals)

        # long_term 분해
        lt_breakdown = result["pool_breakdown"]["long_term"]
        assert lt_breakdown["original_count"] == 2
        assert "scaled_weights" in lt_breakdown
        assert lt_breakdown["scaled_weights"]["005930"] == pytest.approx(
            0.5 * 0.70, abs=0.001
        )
        assert lt_breakdown["scaled_weights"]["000660"] == pytest.approx(
            0.5 * 0.70, abs=0.001
        )
        assert "total_weight" in lt_breakdown

        # etf_rotation 분해
        etf_breakdown = result["pool_breakdown"]["etf_rotation"]
        assert etf_breakdown["original_count"] == 1
        assert etf_breakdown["scaled_weights"]["069500"] == pytest.approx(
            1.0 * 0.30, abs=0.001
        )

    def test_no_allocator_returns_error(self):
        """allocator가 없으면 에러 메시지를 반환한다."""
        executor = _build_executor(allocator=None)

        pool_signals = {"long_term": {"005930": 1.0}}
        result = executor.dry_run_integrated(pool_signals)

        assert "errors" in result
        assert "allocator 미설정" in result["errors"]

    def test_empty_pool_signals_returns_empty_orders(self):
        """빈 pool_signals에 대해 빈 주문을 반환한다."""
        allocator = _mock_allocator()
        executor = _build_executor(allocator=allocator)

        result = executor.dry_run_integrated({})

        assert result["merged_weights"] == {}
        assert result["sell_orders"] == []
        assert result["buy_orders"] == []

    def test_single_pool_works_like_regular(self):
        """단일 풀만 전달해도 정상 동작한다."""
        allocator = _mock_allocator(long_term_pct=0.70, etf_rotation_pct=0.30)
        executor = _build_executor(allocator=allocator)

        pool_signals = {"long_term": {"005930": 1.0}}
        result = executor.dry_run_integrated(pool_signals)

        assert result["is_dry_run"] is True
        assert "long_term" in result["pool_breakdown"]
        assert "etf_rotation" not in result["pool_breakdown"]
        # merged_weights에는 long_term 비율로 스케일된 값
        assert "005930" in result["merged_weights"]
        assert result["merged_weights"]["005930"] == pytest.approx(
            1.0 * 0.70, abs=0.001
        )

    def test_two_pools_no_overlap_both_included(self):
        """겹치지 않는 두 풀이 모두 merged_weights에 포함된다."""
        allocator = _mock_allocator(long_term_pct=0.70, etf_rotation_pct=0.30)
        executor = _build_executor(allocator=allocator)

        pool_signals = {
            "long_term": {"005930": 1.0},
            "etf_rotation": {"069500": 1.0},
        }
        result = executor.dry_run_integrated(pool_signals)

        assert "005930" in result["merged_weights"]
        assert "069500" in result["merged_weights"]
        assert result["merged_weights"]["005930"] == pytest.approx(0.70, abs=0.001)
        assert result["merged_weights"]["069500"] == pytest.approx(0.30, abs=0.001)

    def test_two_pools_overlapping_ticker_weight_summed(self):
        """겹치는 종목의 비중이 합산된다."""
        allocator = _mock_allocator(long_term_pct=0.70, etf_rotation_pct=0.30)
        executor = _build_executor(allocator=allocator)

        # 같은 종목 005930이 두 풀에 포함
        pool_signals = {
            "long_term": {"005930": 0.5},
            "etf_rotation": {"005930": 0.5},
        }
        result = executor.dry_run_integrated(pool_signals)

        expected = round(0.5 * 0.70 + 0.5 * 0.30, 6)
        assert result["merged_weights"]["005930"] == pytest.approx(
            expected, abs=0.0001
        )

    def test_portfolio_value_zero_early_return(self):
        """포트폴리오 가치가 0이면 조기 반환한다."""
        allocator = _mock_allocator()
        executor = _build_executor(allocator=allocator)

        # 포트폴리오 가치를 0으로 설정
        executor.position_manager.get_portfolio_value.return_value = 0

        pool_signals = {
            "long_term": {"005930": 1.0},
        }
        result = executor.dry_run_integrated(pool_signals)

        assert result["portfolio_value"] == 0
        # 주문 계산이 호출되지 않아야 함
        executor.position_manager.calculate_rebalance_orders.assert_not_called()

    def test_risk_check_warnings_included(self):
        """리스크 체크 경고가 결과에 포함된다."""
        risk_guard = _mock_risk_guard(
            rebalance_pass=True,
            rebalance_warnings=["집중도 높음"],
        )
        allocator = _mock_allocator()
        executor = _build_executor(risk_guard=risk_guard, allocator=allocator)

        pool_signals = {"long_term": {"005930": 1.0}}
        result = executor.dry_run_integrated(pool_signals)

        assert result["risk_check"]["passed"] is True
        assert "집중도 높음" in result["risk_check"]["warnings"]

    def test_turnover_check_included(self):
        """회전율 체크 결과가 결과에 포함된다."""
        risk_guard = _mock_risk_guard(
            turnover_pass=False,
            turnover_reason="회전율 30% 초과",
        )
        allocator = _mock_allocator()
        executor = _build_executor(risk_guard=risk_guard, allocator=allocator)

        # 매도/매수 주문이 있어야 회전율 체크가 의미 있음
        executor.position_manager.calculate_rebalance_orders.return_value = (
            [{"ticker": "005930", "side": "sell", "qty": 10, "amount": 500_000}],
            [{"ticker": "000660", "side": "buy", "qty": 5, "amount": 500_000}],
        )

        pool_signals = {"long_term": {"000660": 1.0}}
        result = executor.dry_run_integrated(pool_signals)

        assert result["turnover_check"]["passed"] is False
        assert "회전율" in result["turnover_check"]["reason"]


# ===================================================================
# execute_integrated_rebalance 테스트
# ===================================================================


class TestExecuteIntegratedRebalance:
    """execute_integrated_rebalance 메서드를 검증한다."""

    def test_calls_merge_pool_targets_correctly(self):
        """merge_pool_targets가 올바르게 호출된다."""
        allocator = _mock_allocator()
        executor = _build_executor(allocator=allocator)

        pool_signals = {
            "long_term": {"005930": 0.5, "000660": 0.5},
            "etf_rotation": {"069500": 1.0},
        }

        executor.execute_integrated_rebalance(pool_signals)

        allocator.merge_pool_targets.assert_called_once_with(pool_signals)

    @patch.dict(os.environ, {"KIS_LIVE_CONFIRMED": "true"}, clear=False)
    def test_calls_execute_order_batch(self):
        """_execute_order_batch가 호출된다."""
        allocator = _mock_allocator()
        kis = _mock_kis(is_paper=True)
        risk_guard = _mock_risk_guard()
        executor = _build_executor(
            kis=kis, risk_guard=risk_guard, allocator=allocator
        )

        sell_orders = [
            {"ticker": "005930", "side": "sell", "qty": 5, "amount": 250_000}
        ]
        buy_orders = [
            {"ticker": "069500", "side": "buy", "qty": 10, "amount": 300_000}
        ]
        executor.position_manager.calculate_rebalance_orders.return_value = (
            sell_orders,
            buy_orders,
        )

        with patch.object(
            executor, "_execute_order_batch", return_value={"success": True}
        ) as mock_batch:
            result = executor.execute_integrated_rebalance(
                {"long_term": {"069500": 1.0}}
            )

        mock_batch.assert_called_once_with(sell_orders, buy_orders)
        assert result["success"] is True

    def test_calls_auto_tag_after_execution(self):
        """성공 시 auto_tag_from_pool_signals가 호출된다."""
        allocator = _mock_allocator()
        executor = _build_executor(allocator=allocator)

        pool_signals = {"long_term": {"005930": 1.0}}

        with patch.object(
            executor, "_execute_order_batch", return_value={"success": True}
        ):
            executor.execute_integrated_rebalance(pool_signals)

        allocator.auto_tag_from_pool_signals.assert_called_once_with(
            pool_signals
        )

    def test_no_auto_tag_on_failure(self):
        """실패 시 auto_tag_from_pool_signals가 호출되지 않는다."""
        allocator = _mock_allocator()
        risk_guard = _mock_risk_guard(rebalance_pass=False, rebalance_warnings=["위험"])
        executor = _build_executor(
            risk_guard=risk_guard, allocator=allocator
        )

        pool_signals = {"long_term": {"005930": 1.0}}
        result = executor.execute_integrated_rebalance(pool_signals)

        allocator.auto_tag_from_pool_signals.assert_not_called()
        assert result["success"] is False

    def test_no_allocator_returns_error(self):
        """allocator가 없으면 에러를 반환한다."""
        executor = _build_executor(allocator=None)

        pool_signals = {"long_term": {"005930": 1.0}}
        result = executor.execute_integrated_rebalance(pool_signals)

        assert result["success"] is False
        assert "allocator 미설정" in result["errors"]

    def test_empty_signals_no_execution(self):
        """빈 시그널이면 실행하지 않는다."""
        allocator = _mock_allocator()
        executor = _build_executor(allocator=allocator)

        pool_signals = {}
        result = executor.execute_integrated_rebalance(pool_signals)

        # merge_pool_targets가 빈 딕셔너리를 반환 -> success=True, 실행 없음
        assert result["success"] is True
        assert result["sells"] == []
        assert result["buys"] == []

    def test_risk_check_fails_returns_with_warnings(self):
        """리스크 검증 실패 시 경고와 함께 반환한다."""
        risk_guard = _mock_risk_guard(
            rebalance_pass=False,
            rebalance_warnings=["집중도 초과", "비중 위반"],
        )
        allocator = _mock_allocator()
        executor = _build_executor(
            risk_guard=risk_guard, allocator=allocator
        )

        pool_signals = {"long_term": {"005930": 1.0}}
        result = executor.execute_integrated_rebalance(pool_signals)

        assert result["success"] is False
        assert any("리스크 검증 실패" in e for e in result["errors"])
        assert result["sells"] == []
        assert result["buys"] == []

    def test_success_result_structure(self):
        """성공 결과의 구조가 올바르다."""
        allocator = _mock_allocator()
        executor = _build_executor(allocator=allocator)

        with patch.object(
            executor,
            "_execute_order_batch",
            return_value={
                "success": True,
                "sells": [{"ticker": "005930", "side": "sell"}],
                "buys": [{"ticker": "069500", "side": "buy"}],
                "total_sell_amount": 500_000,
                "total_buy_amount": 300_000,
                "errors": [],
                "skipped": [],
                "timestamp": "2026-02-27T10:00:00",
            },
        ):
            pool_signals = {
                "long_term": {"069500": 0.5},
                "etf_rotation": {"069500": 0.5},
            }
            result = executor.execute_integrated_rebalance(pool_signals)

        assert result["success"] is True
        assert len(result["sells"]) == 1
        assert len(result["buys"]) == 1
        assert result["total_sell_amount"] == 500_000
        assert result["total_buy_amount"] == 300_000

    @patch.dict(os.environ, {"KIS_LIVE_CONFIRMED": "true"}, clear=False)
    def test_turnover_exceeded_skips_execution(self):
        """회전율 초과 시 실행을 건너뛴다."""
        risk_guard = _mock_risk_guard(
            rebalance_pass=True,
            turnover_pass=False,
            turnover_reason="일일 회전율 30% 초과",
        )
        allocator = _mock_allocator()
        kis = _mock_kis(is_paper=True)
        executor = _build_executor(
            kis=kis, risk_guard=risk_guard, allocator=allocator
        )

        # 주문이 있어야 회전율 체크에 도달함
        executor.position_manager.calculate_rebalance_orders.return_value = (
            [{"ticker": "005930", "side": "sell", "qty": 10, "amount": 500_000}],
            [{"ticker": "069500", "side": "buy", "qty": 10, "amount": 500_000}],
        )

        pool_signals = {"long_term": {"069500": 1.0}}
        result = executor.execute_integrated_rebalance(pool_signals)

        # _execute_order_batch 내부에서 회전율 실패 -> success=False
        assert result["success"] is False
        assert any("회전율" in e for e in result["errors"])


# ===================================================================
# _execute_order_batch 테스트
# ===================================================================


class TestExecuteOrderBatch:
    """_execute_order_batch 메서드를 검증한다."""

    @patch("src.execution.executor.time.sleep", return_value=None)
    def test_sells_executed_before_buys(self, mock_sleep):
        """매도가 매수보다 먼저 실행된다."""
        kis = _mock_kis(is_paper=True)
        risk_guard = _mock_risk_guard()
        executor = _build_executor(kis=kis, risk_guard=risk_guard)

        sell_order_mock = _make_order_mock("005930", "sell")
        buy_order_mock = _make_order_mock("069500", "buy")

        call_sequence = []

        def _track_submit(ticker, side, qty, order_type):
            call_sequence.append(side)
            if side == "sell":
                return sell_order_mock
            return buy_order_mock

        executor.order_manager.submit_order.side_effect = _track_submit

        sell_orders = [
            {"ticker": "005930", "side": "sell", "qty": 10, "amount": 500_000}
        ]
        buy_orders = [
            {"ticker": "069500", "side": "buy", "qty": 5, "amount": 250_000}
        ]

        result = executor._execute_order_batch(sell_orders, buy_orders)

        # 매도가 먼저, 매수가 나중에 호출
        assert call_sequence.index("sell") < call_sequence.index("buy")
        assert result["success"] is True

    def test_empty_orders_returns_success(self):
        """빈 주문 리스트면 성공과 함께 반환한다."""
        executor = _build_executor()

        result = executor._execute_order_batch([], [])

        assert result["success"] is True
        assert result["sells"] == []
        assert result["buys"] == []

    def test_kis_not_configured_returns_error(self):
        """KIS가 설정되지 않으면 에러를 반환한다."""
        kis = _mock_kis(configured=False)
        executor = _build_executor(kis=kis)

        sell_orders = [
            {"ticker": "005930", "side": "sell", "qty": 10, "amount": 500_000}
        ]
        result = executor._execute_order_batch(sell_orders, [])

        assert result["success"] is False
        assert any("KIS API" in e for e in result["errors"])

    @patch("src.execution.executor.time.sleep", return_value=None)
    def test_order_errors_collected_in_result(self, mock_sleep):
        """주문 실패가 결과의 errors에 수집된다."""
        kis = _mock_kis(is_paper=True)
        risk_guard = _mock_risk_guard()
        executor = _build_executor(kis=kis, risk_guard=risk_guard)

        failed_order = _make_order_mock("005930", "sell", status="failed")
        executor.order_manager.submit_order.return_value = failed_order

        sell_orders = [
            {"ticker": "005930", "side": "sell", "qty": 10, "amount": 500_000}
        ]
        result = executor._execute_order_batch(sell_orders, [])

        # 실패 주문이 에러에 수집됨
        assert len(result["errors"]) > 0
        assert any("005930" in e for e in result["errors"])
        # sells에는 주문 자체는 기록됨
        assert len(result["sells"]) == 1


# ===================================================================
# 추가 엣지케이스 테스트
# ===================================================================


class TestIntegratedEdgeCases:
    """통합 리밸런싱 관련 엣지 케이스를 검증한다."""

    def test_calculate_rebalance_orders_called_with_integrated_true(self):
        """calculate_rebalance_orders가 integrated=True로 호출된다."""
        allocator = _mock_allocator()
        executor = _build_executor(allocator=allocator)

        pool_signals = {"long_term": {"005930": 1.0}}

        with patch.object(
            executor, "_execute_order_batch", return_value={"success": True}
        ):
            executor.execute_integrated_rebalance(pool_signals)

        # calculate_rebalance_orders 호출 인자 확인
        call_args = (
            executor.position_manager.calculate_rebalance_orders.call_args
        )
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs.get("integrated") is True
        assert kwargs.get("pool") is None
        assert kwargs.get("allocator") is allocator

    def test_dry_run_integrated_calls_calculate_rebalance_with_integrated(self):
        """dry_run_integrated에서도 integrated=True로 호출된다."""
        allocator = _mock_allocator()
        executor = _build_executor(allocator=allocator)

        pool_signals = {"long_term": {"005930": 1.0}}
        executor.dry_run_integrated(pool_signals)

        call_args = (
            executor.position_manager.calculate_rebalance_orders.call_args
        )
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs.get("integrated") is True

    def test_auto_tag_exception_does_not_break_execution(self):
        """auto_tag_from_pool_signals 예외가 실행을 중단시키지 않는다."""
        allocator = _mock_allocator()
        allocator.auto_tag_from_pool_signals.side_effect = RuntimeError(
            "태깅 실패"
        )
        executor = _build_executor(allocator=allocator)

        with patch.object(
            executor, "_execute_order_batch", return_value={"success": True}
        ):
            pool_signals = {"long_term": {"005930": 1.0}}
            result = executor.execute_integrated_rebalance(pool_signals)

        # 예외에도 불구하고 결과는 success
        assert result["success"] is True
        allocator.auto_tag_from_pool_signals.assert_called_once()

    @patch.dict(os.environ, {"KIS_LIVE_CONFIRMED": "false"}, clear=False)
    def test_live_mode_without_confirmation_blocked(self):
        """실전 모드에서 KIS_LIVE_CONFIRMED=false이면 차단된다."""
        kis = _mock_kis(is_paper=False, configured=True)
        allocator = _mock_allocator()
        risk_guard = _mock_risk_guard()
        executor = _build_executor(
            kis=kis, risk_guard=risk_guard, allocator=allocator
        )

        # 주문이 있어야 _execute_order_batch에 도달
        executor.position_manager.calculate_rebalance_orders.return_value = (
            [{"ticker": "005930", "side": "sell", "qty": 5, "amount": 250_000}],
            [],
        )

        pool_signals = {"long_term": {"069500": 1.0}}
        result = executor.execute_integrated_rebalance(pool_signals)

        assert result["success"] is False
        assert any("KIS_LIVE_CONFIRMED" in e for e in result["errors"])
