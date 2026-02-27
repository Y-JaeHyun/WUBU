"""백테스팅 엔진 모듈(src/backtest/engine.py) 테스트.

Strategy ABC를 상속한 더미 전략으로 백테스트 실행, 거래비용 반영,
성과 지표 계산 정합성 등을 검증한다.
"""

from typing import Optional
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

    def __init__(self, signals: Optional[dict] = None):
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
            rebalance_freq="daily",
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


# ===================================================================
# Weekly / Biweekly 리밸런싱 테스트
# ===================================================================

class TestWeeklyBiweeklyRebalance:
    """weekly/biweekly 리밸런싱 날짜 생성 검증."""

    def test_weekly_rebalance(self):
        """주간 리밸런싱 시 매주 첫 영업일이 선택된다."""
        bt = Backtest(
            strategy=DummyStrategy(),
            start_date="20240101",
            end_date="20240229",
            rebalance_freq="weekly",
        )
        dates = bt._get_rebalance_dates()

        # 약 9주 (1/1~2/29)
        assert len(dates) >= 8
        # 각 날짜가 영업일
        for d in dates:
            ts = pd.Timestamp(d)
            assert ts.weekday() < 5

    def test_biweekly_rebalance(self):
        """격주 리밸런싱은 주간의 약 절반이다."""
        bt_w = Backtest(
            strategy=DummyStrategy(),
            start_date="20240101",
            end_date="20240630",
            rebalance_freq="weekly",
        )
        bt_bw = Backtest(
            strategy=DummyStrategy(),
            start_date="20240101",
            end_date="20240630",
            rebalance_freq="biweekly",
        )
        weekly_dates = bt_w._get_rebalance_dates()
        biweekly_dates = bt_bw._get_rebalance_dates()

        # biweekly는 weekly의 약 절반
        assert len(biweekly_dates) == len(weekly_dates) // 2 + (
            1 if len(weekly_dates) % 2 else 0
        )

    def test_biweekly_subset_of_weekly(self):
        """biweekly 날짜는 weekly 날짜의 부분집합이다."""
        bt_w = Backtest(
            strategy=DummyStrategy(),
            start_date="20240101",
            end_date="20240630",
            rebalance_freq="weekly",
        )
        bt_bw = Backtest(
            strategy=DummyStrategy(),
            start_date="20240101",
            end_date="20240630",
            rebalance_freq="biweekly",
        )
        weekly_dates = set(bt_w._get_rebalance_dates())
        biweekly_dates = set(bt_bw._get_rebalance_dates())

        assert biweekly_dates.issubset(weekly_dates)


# ===================================================================
# Diff-based / 차등 리밸런싱 테스트
# ===================================================================

class _RotatingStrategy(Strategy):
    """리밸런싱 시점마다 다른 시그널을 반환하는 전략 (diff-based 검증용)."""

    def __init__(self, signals_by_call: list[dict]):
        self._signals_by_call = signals_by_call
        self._call_count = 0

    @property
    def name(self) -> str:
        return "RotatingStrategy"

    def generate_signals(self, date: str, data: dict) -> dict:
        idx = min(self._call_count, len(self._signals_by_call) - 1)
        self._call_count += 1
        return self._signals_by_call[idx]


class DualSignalStrategy(Strategy):
    """리밸런싱마다 번갈아 다른 시그널을 반환하는 전략."""

    def __init__(self, signals_list: list):
        self._signals_list = signals_list
        self._call_count = 0

    @property
    def name(self) -> str:
        return "DualSignalStrategy"

    def generate_signals(self, date: str, data: dict) -> dict:
        idx = self._call_count % len(self._signals_list)
        self._call_count += 1
        return self._signals_list[idx]


class TestDiffBasedRebalancing:
    """리밸런싱이 diff-based로 동작하여 유지 종목을 불필요하게 매매하지 않는지 검증."""

    @staticmethod
    def _make_constant_price(ticker: str, price: int, start: str, periods: int):
        """고정가 DataFrame을 만든다."""
        dates = pd.bdate_range(start, periods=periods)
        df = pd.DataFrame(
            {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 100_000,
            },
            index=dates,
        )
        df.index.name = "date"
        return df

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_retained_stock_not_sold_and_rebought(self, mock_price, mock_fund):
        """유지 종목(A)은 매도+재매수 없이 그대로 보유된다.

        1회차: A=50%, B=50%
        2회차: A=50%, C=50% (B→C 교체, A는 유지)
        → A에 대한 매도 거래가 없어야 함
        """
        mock_fund.return_value = pd.DataFrame()
        period = 45  # ~2개월

        strategy = _RotatingStrategy([
            {"A": 0.5, "B": 0.5},
            {"A": 0.5, "C": 0.5},
        ])

        def price_side_effect(ticker, start, end):
            return self._make_constant_price(ticker, 10_000, "2024-01-02", period)

        mock_price.side_effect = price_side_effect

        bt = Backtest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240229",
            initial_capital=1_000_000,
            buy_cost=0.0,
            sell_cost=0.0,
        )
        bt.run()

        trades = bt.get_trades()
        a_sells = trades[(trades["ticker"] == "A") & (trades["action"] == "sell")]
        assert len(a_sells) == 0, (
            f"유지 종목 A가 매도되었습니다: {len(a_sells)}건. "
            "diff-based 리밸런싱이라면 A는 매도되지 않아야 합니다."
        )

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_diff_based_reduces_trade_count(self, mock_price, mock_fund):
        """동일 시그널이 반복되면 첫 매수 이후 거래가 발생하지 않는다."""
        mock_fund.return_value = pd.DataFrame()
        period = 65

        # 매번 동일한 시그널
        strategy = DummyStrategy({"X": 0.5, "Y": 0.5})

        def price_side_effect(ticker, start, end):
            return self._make_constant_price(ticker, 20_000, "2024-01-02", period)

        mock_price.side_effect = price_side_effect

        bt = Backtest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240331",
            initial_capital=1_000_000,
            buy_cost=0.0,
            sell_cost=0.0,
        )
        bt.run()

        trades = bt.get_trades()
        # 첫 리밸런싱에서 2건 매수, 이후 동일 시그널이므로 추가 거래 없음
        assert len(trades) == 2, (
            f"동일 시그널 반복 시 첫 매수 2건만 있어야 하는데 {len(trades)}건 발생. "
            "diff-based에서는 유지 종목을 재매매하지 않습니다."
        )

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_removed_stock_is_sold(self, mock_price, mock_fund):
        """대상에서 제외된 종목은 매도된다.

        1회차: A=50%, B=50%
        2회차: C=50%, D=50% (A, B 모두 제외)
        → A, B 매도 거래가 있어야 함
        """
        mock_fund.return_value = pd.DataFrame()
        period = 45

        strategy = _RotatingStrategy([
            {"A": 0.5, "B": 0.5},
            {"C": 0.5, "D": 0.5},
        ])

        def price_side_effect(ticker, start, end):
            return self._make_constant_price(ticker, 10_000, "2024-01-02", period)

        mock_price.side_effect = price_side_effect

        bt = Backtest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240229",
            initial_capital=1_000_000,
            buy_cost=0.0,
            sell_cost=0.0,
        )
        bt.run()

        trades = bt.get_trades()
        a_sells = trades[(trades["ticker"] == "A") & (trades["action"] == "sell")]
        b_sells = trades[(trades["ticker"] == "B") & (trades["action"] == "sell")]
        assert len(a_sells) > 0, "제외된 종목 A가 매도되지 않았습니다."
        assert len(b_sells) > 0, "제외된 종목 B가 매도되지 않았습니다."

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_no_cost_for_retained_stock(self, mock_price, mock_fund):
        """유지 종목은 거래비용이 발생하지 않는다."""
        mock_fund.return_value = pd.DataFrame()
        period = 45

        strategy = _RotatingStrategy([
            {"A": 0.5, "B": 0.5},
            {"A": 0.5, "C": 0.5},
        ])

        def price_side_effect(ticker, start, end):
            return self._make_constant_price(ticker, 10_000, "2024-01-02", period)

        mock_price.side_effect = price_side_effect

        bt = Backtest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240229",
            initial_capital=1_000_000,
            buy_cost=0.001,
            sell_cost=0.003,
        )
        bt.run()

        trades = bt.get_trades()
        # A에 대한 매도 거래가 없어야 함 (유지)
        a_trades = trades[trades["ticker"] == "A"]
        a_sells = a_trades[a_trades["action"] == "sell"]
        assert len(a_sells) == 0, "유지 종목 A가 불필요하게 매도되어 비용 발생"


class TestDifferentialRebalancing:
    """차등 리밸런싱 동작 검증 (min_rebalance_threshold 등)."""

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_retained_stock_not_sold(self, mock_price, mock_fund):
        """유지 종목이 매도되지 않는다.

        1차: A=50%, B=50%
        2차: A=50%, B=50% (동일)
        → 2차에서 매도 거래가 발생하지 않아야 한다.
        """
        dates = pd.bdate_range("2024-01-02", periods=45)
        flat_price = 50000
        flat_df = pd.DataFrame(
            {
                "open": [flat_price] * 45,
                "high": [flat_price] * 45,
                "low": [flat_price] * 45,
                "close": [flat_price] * 45,
                "volume": [1_000_000] * 45,
            },
            index=dates,
        )
        flat_df.index.name = "date"

        mock_price.return_value = flat_df
        mock_fund.return_value = pd.DataFrame()

        strategy = DummyStrategy(signals={"005930": 0.5, "000660": 0.5})

        bt = Backtest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240229",
            initial_capital=100_000_000,
        )
        bt.run()

        trades = bt.get_trades()
        # 첫 리밸런싱에서만 매수, 두 번째부터는 비중 동일하므로 거래 없음
        sell_trades = trades[trades["action"] == "sell"]
        assert len(sell_trades) == 0, "유지 종목에 대한 매도가 발생하면 안 됨"

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_min_rebalance_threshold_skips_small_changes(self, mock_price, mock_fund):
        """min_rebalance_threshold 미만의 비중 변화는 거래를 스킵한다."""
        dates = pd.bdate_range("2024-01-02", periods=45)
        flat_price = 50000
        flat_df = pd.DataFrame(
            {
                "open": [flat_price] * 45,
                "high": [flat_price] * 45,
                "low": [flat_price] * 45,
                "close": [flat_price] * 45,
                "volume": [1_000_000] * 45,
            },
            index=dates,
        )
        flat_df.index.name = "date"

        mock_price.return_value = flat_df
        mock_fund.return_value = pd.DataFrame()

        # 두 시그널 간 차이가 0.01 (1%)
        strategy = DualSignalStrategy([
            {"005930": 0.50, "000660": 0.50},
            {"005930": 0.51, "000660": 0.49},
        ])

        bt = Backtest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240229",
            initial_capital=100_000_000,
            min_rebalance_threshold=0.02,  # 2% 이상만 거래
        )
        bt.run()

        trades = bt.get_trades()
        # 첫 리밸런싱에서 매수만, 이후 비중 변화가 2% 미만이라 거래 없음
        buy_trades_after_first = trades[
            (trades["action"] == "buy") &
            (trades["date"] > pd.Timestamp("20240201"))
        ]
        sell_trades = trades[trades["action"] == "sell"]
        assert len(sell_trades) == 0
        assert len(buy_trades_after_first) == 0

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_removed_stock_is_sold(self, mock_price, mock_fund):
        """타겟에서 빠진 종목은 매도된다."""
        dates = pd.bdate_range("2024-01-02", periods=45)
        flat_price = 50000
        flat_df = pd.DataFrame(
            {
                "open": [flat_price] * 45,
                "high": [flat_price] * 45,
                "low": [flat_price] * 45,
                "close": [flat_price] * 45,
                "volume": [1_000_000] * 45,
            },
            index=dates,
        )
        flat_df.index.name = "date"

        mock_price.return_value = flat_df
        mock_fund.return_value = pd.DataFrame()

        # 1차: A+B, 2차: A만 (B는 빠짐)
        strategy = DualSignalStrategy([
            {"005930": 0.5, "000660": 0.5},
            {"005930": 1.0},
        ])

        bt = Backtest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240229",
            initial_capital=100_000_000,
        )
        bt.run()

        trades = bt.get_trades()
        # 000660이 매도되어야 함
        sell_trades = trades[
            (trades["action"] == "sell") & (trades["ticker"] == "000660")
        ]
        assert len(sell_trades) > 0, "타겟에서 빠진 종목이 매도되어야 함"

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_differential_reduces_trade_count(self, mock_price, mock_fund):
        """차등 리밸런싱이 동일 시그널 반복 시 거래 횟수를 줄인다."""
        dates = pd.bdate_range("2024-01-02", periods=65)
        flat_price = 50000
        flat_df = pd.DataFrame(
            {
                "open": [flat_price] * 65,
                "high": [flat_price] * 65,
                "low": [flat_price] * 65,
                "close": [flat_price] * 65,
                "volume": [1_000_000] * 65,
            },
            index=dates,
        )
        flat_df.index.name = "date"

        mock_price.return_value = flat_df
        mock_fund.return_value = pd.DataFrame()

        # 동일 시그널을 3개월 반복
        strategy = DummyStrategy(signals={"005930": 0.5, "000660": 0.5})

        bt = Backtest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240329",
            initial_capital=100_000_000,
        )
        bt.run()

        results = bt.get_results()
        # 동일 시그널이므로 첫 리밸런싱의 매수만 발생 (매도 0건)
        trades = bt.get_trades()
        sell_count = len(trades[trades["action"] == "sell"])
        assert sell_count == 0, (
            f"동일 시그널 반복 시 매도가 발생하면 안 됨 (발생: {sell_count}건)"
        )
