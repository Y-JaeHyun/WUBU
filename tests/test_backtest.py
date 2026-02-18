"""백테스팅 엔진 모듈(src/backtest/engine.py) 테스트.

Strategy ABC를 상속한 더미 전략으로 백테스트 실행, 거래비용 반영,
성과 지표 계산 정합성 등을 검증한다.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Backtest, Strategy


# ===================================================================
# 더미 전략 클래스
# ===================================================================

class DummyStrategy(Strategy):
    """테스트용 더미 전략. 매 리밸런싱마다 고정된 시그널을 반환한다."""

    def __init__(self, signals: dict | None = None):
        self._signals = signals or {}

    @property
    def name(self) -> str:
        return "DummyStrategy"

    def generate_signals(self, date: str, data: dict) -> dict:
        return self._signals


class EmptyStrategy(Strategy):
    """빈 시그널을 반환하는 전략 (현금 100%)."""

    @property
    def name(self) -> str:
        return "EmptyStrategy"

    def generate_signals(self, date: str, data: dict) -> dict:
        return {}


# ===================================================================
# 헬퍼: mock 가격 DataFrame 생성
# ===================================================================

def _make_price_df(start: str, periods: int, base_price: int = 50000) -> pd.DataFrame:
    """지정 기간의 mock 가격 DataFrame을 생성한다."""
    dates = pd.bdate_range(start, periods=periods)
    np.random.seed(0)
    close_arr = base_price + np.cumsum(np.random.randn(periods) * 200).astype(int)
    close_arr = np.maximum(close_arr, 100)

    df = pd.DataFrame(
        {
            "open": close_arr - 100,
            "high": close_arr + 200,
            "low": close_arr - 200,
            "close": close_arr,
            "volume": [1_000_000] * periods,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ===================================================================
# Strategy ABC 테스트
# ===================================================================

class TestStrategyABC:
    """Strategy 추상 클래스를 제대로 구현하지 않으면 인스턴스화 실패 검증."""

    def test_cannot_instantiate_directly(self):
        """Strategy를 직접 인스턴스화할 수 없다."""
        with pytest.raises(TypeError):
            Strategy()

    def test_subclass_must_implement_name(self):
        """name 프로퍼티를 구현하지 않으면 인스턴스화 실패."""
        class NoName(Strategy):
            def generate_signals(self, date, data):
                return {}

        with pytest.raises(TypeError):
            NoName()

    def test_subclass_must_implement_generate_signals(self):
        """generate_signals를 구현하지 않으면 인스턴스화 실패."""
        class NoSignals(Strategy):
            @property
            def name(self):
                return "NoSignals"

        with pytest.raises(TypeError):
            NoSignals()

    def test_valid_subclass(self):
        """정상 구현 서브클래스는 인스턴스화 성공."""
        s = DummyStrategy({"005930": 0.5})
        assert s.name == "DummyStrategy"
        assert s.generate_signals("20240101", {}) == {"005930": 0.5}


# ===================================================================
# Backtest 기본 동작 테스트
# ===================================================================

class TestBacktestInit:
    """Backtest 초기화 파라미터 검증."""

    def test_default_parameters(self):
        """기본 파라미터가 올바르게 설정된다."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="2024-01-02",
            end_date="2024-03-29",
        )
        assert bt.initial_capital == 100_000_000
        assert bt.rebalance_freq == "monthly"
        assert bt.buy_cost == 0.00015
        assert bt.sell_cost == 0.00245
        assert bt.start_date == "20240102"
        assert bt.end_date == "20240329"

    def test_custom_parameters(self):
        """사용자 정의 파라미터가 올바르게 반영된다."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240102",
            end_date="20240329",
            initial_capital=50_000_000,
            rebalance_freq="quarterly",
            buy_cost=0.001,
            sell_cost=0.003,
        )
        assert bt.initial_capital == 50_000_000
        assert bt.rebalance_freq == "quarterly"
        assert bt.buy_cost == 0.001
        assert bt.sell_cost == 0.003


# ===================================================================
# 리밸런싱 주기 검증
# ===================================================================

class TestRebalanceDates:
    """리밸런싱 날짜 생성 로직 검증."""

    def test_monthly_rebalance(self):
        """월간 리밸런싱 시 매월 첫 영업일이 선택된다."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240101",
            end_date="20240630",
            rebalance_freq="monthly",
        )
        dates = bt._get_rebalance_dates()

        assert len(dates) == 6  # 1월~6월
        # 각 날짜가 해당 월의 첫 영업일인지 확인
        for d in dates:
            ts = pd.Timestamp(d)
            # 영업일이어야 한다 (weekday < 5)
            assert ts.weekday() < 5

    def test_quarterly_rebalance(self):
        """분기별 리밸런싱 시 분기 첫 영업일이 선택된다."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240101",
            end_date="20241231",
            rebalance_freq="quarterly",
        )
        dates = bt._get_rebalance_dates()

        assert len(dates) == 4  # Q1~Q4

    def test_invalid_rebalance_freq(self):
        """지원하지 않는 주기는 ValueError를 발생시킨다."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240101",
            end_date="20240630",
            rebalance_freq="weekly",
        )
        with pytest.raises(ValueError, match="지원하지 않는 리밸런싱 주기"):
            bt._get_rebalance_dates()

    def test_empty_date_range(self):
        """시작일 > 종료일이면 빈 리스트를 반환한다."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240630",
            end_date="20240101",
        )
        dates = bt._get_rebalance_dates()
        assert dates == []


# ===================================================================
# 백테스트 실행 테스트 (mock 사용)
# ===================================================================

class TestBacktestRun:
    """백테스트 실행 및 결과 검증."""

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_run_with_empty_strategy(self, mock_price, mock_fund):
        """빈 전략(현금 100%)으로 실행하면 자본금이 그대로 유지된다."""
        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )

        mock_fund.return_value = pd.DataFrame()

        bt.run()

        history = bt.get_portfolio_history()
        assert not history.empty
        # 현금만 보유하므로 모든 날의 portfolio_value == initial_capital
        assert all(history["portfolio_value"] == bt.initial_capital)

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_run_with_single_stock(self, mock_price, mock_fund):
        """단일 종목 매수/매도를 포함하는 백테스트가 정상 실행된다."""
        # 2024-01-02 ~ 2024-02-29 (약 2개월)
        price_df = _make_price_df("2024-01-02", 40, base_price=50000)

        mock_price.return_value = price_df
        mock_fund.return_value = pd.DataFrame()

        strategy = DummyStrategy(signals={"005930": 1.0})

        bt = Backtest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240229",
            initial_capital=100_000_000,
        )
        bt.run()

        # 거래 내역 확인
        trades = bt.get_trades()
        assert not trades.empty
        assert "buy" in trades["action"].values

        # 리밸런싱 날짜 수만큼 매수 발생
        buy_trades = trades[trades["action"] == "buy"]
        assert len(buy_trades) >= 1

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_transaction_cost_applied(self, mock_price, mock_fund):
        """거래비용이 반영되어 수익률이 감소한다."""
        # 가격이 일정한 주식 (변동 없음)
        dates = pd.bdate_range("2024-01-02", periods=40)
        flat_price = 50000
        flat_df = pd.DataFrame(
            {
                "open": [flat_price] * 40,
                "high": [flat_price] * 40,
                "low": [flat_price] * 40,
                "close": [flat_price] * 40,
                "volume": [1_000_000] * 40,
            },
            index=dates,
        )
        flat_df.index.name = "date"

        mock_price.return_value = flat_df
        mock_fund.return_value = pd.DataFrame()

        strategy = DummyStrategy(signals={"005930": 1.0})

        bt = Backtest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240229",
            initial_capital=100_000_000,
            buy_cost=0.001,
            sell_cost=0.003,
        )
        bt.run()

        results = bt.get_results()
        # 가격 변동 없으나, 리밸런싱마다 거래비용 발생 -> 수익률 음수
        assert results["total_return"] < 0

    def test_get_results_before_run_raises(self):
        """run() 호출 전에 get_results()를 호출하면 RuntimeError."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        with pytest.raises(RuntimeError, match="백테스트가 아직 실행되지 않았습니다"):
            bt.get_results()

    def test_get_portfolio_history_before_run_raises(self):
        """run() 호출 전에 get_portfolio_history()를 호출하면 RuntimeError."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        with pytest.raises(RuntimeError, match="백테스트가 아직 실행되지 않았습니다"):
            bt.get_portfolio_history()

    def test_get_trades_before_run_raises(self):
        """run() 호출 전에 get_trades()를 호출하면 RuntimeError."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        with pytest.raises(RuntimeError, match="백테스트가 아직 실행되지 않았습니다"):
            bt.get_trades()


# ===================================================================
# 성과 지표 계산 정합성
# ===================================================================

class TestPerformanceMetrics:
    """성과 지표 계산 검증."""

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_results_keys(self, mock_price, mock_fund):
        """get_results() 반환값에 필수 키가 포함된다."""
        mock_fund.return_value = pd.DataFrame()
        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        bt.run()

        results = bt.get_results()

        required_keys = [
            "strategy_name",
            "start_date",
            "end_date",
            "initial_capital",
            "final_value",
            "total_return",
            "cagr",
            "sharpe_ratio",
            "mdd",
            "total_trades",
            "rebalance_count",
        ]
        for key in required_keys:
            assert key in results, f"'{key}' 키가 결과에 없습니다."

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_total_return_zero_for_cash(self, mock_price, mock_fund):
        """현금만 보유하면 수익률이 0%이다."""
        mock_fund.return_value = pd.DataFrame()
        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        bt.run()

        results = bt.get_results()
        assert results["total_return"] == 0.0

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_mdd_non_positive(self, mock_price, mock_fund):
        """MDD는 항상 0 이하이다."""
        mock_fund.return_value = pd.DataFrame()

        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240329",
        )
        bt.run()

        results = bt.get_results()
        assert results["mdd"] <= 0

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_final_value_equals_initial_for_empty(self, mock_price, mock_fund):
        """빈 전략이면 최종 가치가 초기 자본금과 동일하다."""
        mock_fund.return_value = pd.DataFrame()
        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
            initial_capital=50_000_000,
        )
        bt.run()

        results = bt.get_results()
        assert results["final_value"] == 50_000_000

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_portfolio_history_columns(self, mock_price, mock_fund):
        """포트폴리오 히스토리 DataFrame의 컬럼이 올바르다."""
        mock_fund.return_value = pd.DataFrame()
        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        bt.run()

        history = bt.get_portfolio_history()

        for col in ["portfolio_value", "cash", "num_holdings"]:
            assert col in history.columns

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_trades_columns(self, mock_price, mock_fund):
        """빈 거래 내역 DataFrame에도 올바른 컬럼이 있다."""
        mock_fund.return_value = pd.DataFrame()
        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        bt.run()

        trades = bt.get_trades()

        for col in ["date", "ticker", "action", "quantity", "price", "cost"]:
            assert col in trades.columns


# ===================================================================
# _get_price_on_date 테스트
# ===================================================================

class TestGetPriceOnDate:
    """_get_price_on_date 메서드 검증."""

    def test_exact_date_match(self):
        """해당 날짜에 데이터가 있으면 정확한 종가를 반환한다."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        price_df = _make_price_df("2024-01-02", 10)

        date_str = pd.Timestamp("2024-01-02").strftime("%Y%m%d")
        price = bt._get_price_on_date(price_df, date_str)

        assert price is not None
        assert price == float(price_df.iloc[0]["close"])

    def test_missing_date_uses_previous(self):
        """해당 날짜에 데이터가 없으면 직전 영업일 종가를 사용한다."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        price_df = _make_price_df("2024-01-02", 5)

        # 주말 날짜 (데이터 없음)
        price = bt._get_price_on_date(price_df, "20240106")  # 토요일

        assert price is not None
        # 1월 5일(금) 종가를 반환해야 함
        assert price == float(price_df.loc[pd.Timestamp("2024-01-05"), "close"])

    def test_empty_dataframe_returns_none(self):
        """빈 DataFrame이면 None을 반환한다."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        price = bt._get_price_on_date(pd.DataFrame(), "20240102")
        assert price is None

    def test_date_before_data_returns_none(self):
        """데이터보다 이전 날짜를 조회하면 None을 반환한다."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        price_df = _make_price_df("2024-01-10", 5)

        price = bt._get_price_on_date(price_df, "20240102")

        assert price is None
