"""Walk-Forward 백테스팅 모듈(src/backtest/walk_forward.py) 테스트.

윈도우 생성 로직, Walk-Forward 실행, OOS 성과 지표 계산을 검증한다.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Strategy
from src.backtest.walk_forward import WalkForwardBacktest


# ===================================================================
# 더미 전략 클래스
# ===================================================================

class DummyStrategy(Strategy):
    """테스트용 더미 전략. 고정된 2종목 균등 배분 시그널을 반환한다."""

    def __init__(self, name: str = "Dummy"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate_signals(self, date: str, data: dict) -> dict:
        return {"000001": 0.5, "000002": 0.5}


class EmptyStrategy(Strategy):
    """빈 시그널을 반환하는 전략 (현금 100%)."""

    @property
    def name(self) -> str:
        return "EmptyStrategy"

    def generate_signals(self, date: str, data: dict) -> dict:
        return {}


# ===================================================================
# 헬퍼 함수
# ===================================================================

def _make_strategy_factory(strategy_cls=None):
    """테스트용 strategy_factory를 생성한다."""
    if strategy_cls is None:
        strategy_cls = DummyStrategy

    def factory(train_start: str, train_end: str) -> Strategy:
        if strategy_cls is DummyStrategy:
            return strategy_cls(name=f"Dummy_{train_start}_{train_end}")
        return strategy_cls()
    return factory


def _make_price_df(start: str, end: str, base_price: int = 50000) -> pd.DataFrame:
    """지정 기간의 mock 가격 DataFrame을 생성한다."""
    dates = pd.bdate_range(start, end)
    n = len(dates)
    if n == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    np.random.seed(42)
    close_arr = base_price + np.cumsum(np.random.randn(n) * 200).astype(int)
    close_arr = np.maximum(close_arr, 1000)

    df = pd.DataFrame(
        {
            "open": close_arr - 100,
            "high": close_arr + 200,
            "low": close_arr - 200,
            "close": close_arr,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _make_flat_price_df(start: str, end: str, price: int = 50000) -> pd.DataFrame:
    """고정 가격의 mock 가격 DataFrame을 생성한다."""
    dates = pd.bdate_range(start, end)
    n = len(dates)
    if n == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        {
            "open": [price] * n,
            "high": [price] * n,
            "low": [price] * n,
            "close": [price] * n,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ===================================================================
# 윈도우 생성 테스트
# ===================================================================

class TestWindowGeneration:
    """_generate_windows() 로직 검증."""

    def test_standard_windows(self):
        """표준 파라미터로 올바른 수의 윈도우가 생성된다.

        2014-01-01 ~ 2024-12-31, train=5년, test=1년, step=12개월
        -> Window 1: train=20140101~20181231, test=20190101~20191231
        -> Window 2: train=20150101~20191231, test=20200101~20201231
        -> Window 3: train=20160101~20201231, test=20210101~20211231
        -> Window 4: train=20170101~20211231, test=20220101~20221231
        -> Window 5: train=20180101~20221231, test=20230101~20231231
        -> Window 6: train=20190101~20231231, test=20240101~20241231
        -> 총 6개 윈도우
        """
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20241231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        windows = wf._generate_windows()

        assert len(windows) == 6

    def test_first_window_dates(self):
        """첫 번째 윈도우의 날짜 범위가 올바르다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20241231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        windows = wf._generate_windows()

        first = windows[0]
        assert first["train_start"] == "20140101"
        assert first["train_end"] == "20181231"
        assert first["test_start"] == "20190101"
        assert first["test_end"] == "20191231"

    def test_last_window_dates(self):
        """마지막 윈도우의 날짜 범위가 올바르다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20241231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        windows = wf._generate_windows()

        last = windows[-1]
        assert last["train_start"] == "20190101"
        assert last["train_end"] == "20231231"
        assert last["test_start"] == "20240101"
        assert last["test_end"] == "20241231"

    def test_window_stepping(self):
        """윈도우가 step_months 간격으로 이동한다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20241231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        windows = wf._generate_windows()

        for i in range(1, len(windows)):
            prev_start = pd.Timestamp(windows[i - 1]["train_start"])
            curr_start = pd.Timestamp(windows[i]["train_start"])
            diff_months = (curr_start.year - prev_start.year) * 12 + (
                curr_start.month - prev_start.month
            )
            assert diff_months == 12

    def test_smaller_step(self):
        """step_months=6인 경우 더 많은 윈도우가 생성된다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20241231",
            train_years=5,
            test_years=1,
            step_months=6,
        )
        windows = wf._generate_windows()

        # step=6 이므로 step=12 대비 약 2배 윈도우
        assert len(windows) > 6

    def test_short_period_no_windows(self):
        """전체 기간이 train + test보다 짧으면 빈 리스트를 반환한다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20200101",
            full_end_date="20230101",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        windows = wf._generate_windows()

        assert len(windows) == 0

    def test_exact_period_one_window(self):
        """전체 기간이 정확히 train + test와 같으면 1개 윈도우가 생성된다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20191231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        windows = wf._generate_windows()

        assert len(windows) == 1

    def test_train_end_before_test_start(self):
        """학습 종료일이 검증 시작일 직전이다 (갭 없음)."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20241231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        windows = wf._generate_windows()

        for w in windows:
            train_end = pd.Timestamp(w["train_end"])
            test_start = pd.Timestamp(w["test_start"])
            # test_start = train_end + 1일
            assert (test_start - train_end).days == 1

    def test_test_period_length(self):
        """각 검증 구간이 test_years에 해당하는 기간이다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20241231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        windows = wf._generate_windows()

        for w in windows:
            test_start = pd.Timestamp(w["test_start"])
            test_end = pd.Timestamp(w["test_end"])
            # 약 365일 (윤년 포함 가능)
            diff_days = (test_end - test_start).days
            assert 364 <= diff_days <= 366

    def test_different_train_test_years(self):
        """train=3, test=2인 경우 윈도우가 올바르게 생성된다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20241231",
            train_years=3,
            test_years=2,
            step_months=12,
        )
        windows = wf._generate_windows()

        # train=3년+test=2년=5년, step=1년
        # 2014~2018, 2015~2019, ..., 2019~2023, 2020~2024 -> 7개
        assert len(windows) == 7

        first = windows[0]
        assert first["train_start"] == "20140101"
        assert first["train_end"] == "20161231"
        assert first["test_start"] == "20170101"
        assert first["test_end"] == "20181231"

    def test_hyphen_format_dates(self):
        """날짜를 YYYY-MM-DD 형식으로 전달해도 정상 작동한다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="2014-01-01",
            full_end_date="2024-12-31",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        windows = wf._generate_windows()

        assert len(windows) == 6


# ===================================================================
# Walk-Forward 실행 테스트
# ===================================================================

class TestWalkForwardRun:
    """Walk-Forward 백테스트 실행 검증."""

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_run_completes_successfully(self, mock_price, mock_fund):
        """Walk-Forward 백테스트가 정상 완료된다."""
        # 전체 기간 가격 데이터 생성
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        assert wf._is_run is True
        assert len(wf._window_results) > 0

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_run_window_count(self, mock_price, mock_fund):
        """실행된 윈도우 수가 예상과 일치한다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        # train=5, test=1, 2014~2021: 3개 윈도우
        # W1: train 2014~2018, test 2019
        # W2: train 2015~2019, test 2020
        # W3: train 2016~2020, test 2021
        assert len(wf._window_results) == 3

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_capital_chaining(self, mock_price, mock_fund):
        """윈도우 간 자본금이 이어진다(capital chaining)."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(EmptyStrategy),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
            initial_capital=100_000_000,
        )
        wf.run()

        # EmptyStrategy는 현금만 보유하므로 자본금이 유지됨
        for result in wf._window_results:
            assert result["initial_capital"] == 100_000_000

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_run_with_empty_strategy(self, mock_price, mock_fund):
        """빈 전략(현금 100%)으로 Walk-Forward를 실행한다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        def empty_factory(train_start, train_end):
            return EmptyStrategy()

        wf = WalkForwardBacktest(
            strategy_factory=empty_factory,
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        results = wf.get_oos_results()
        assert results["total_return"] == 0.0
        assert results["num_windows"] == 3

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_run_stores_window_histories(self, mock_price, mock_fund):
        """각 윈도우의 포트폴리오 히스토리가 저장된다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        assert len(wf._window_histories) == len(wf._window_results)
        for h in wf._window_histories:
            assert isinstance(h, pd.DataFrame)
            assert "portfolio_value" in h.columns

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_run_with_no_windows(self, mock_price, mock_fund):
        """윈도우가 0개인 경우 경고 후 빈 결과를 반환한다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20200101",
            full_end_date="20220101",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        assert wf._is_run is True
        results = wf.get_oos_results()
        assert results["num_windows"] == 0

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_window_results_have_extra_fields(self, mock_price, mock_fund):
        """윈도우 결과에 window_index, train_start, train_end가 포함된다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        for result in wf._window_results:
            assert "window_index" in result
            assert "train_start" in result
            assert "train_end" in result

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_strategy_factory_receives_correct_dates(self, mock_price, mock_fund):
        """strategy_factory가 올바른 학습 구간 날짜를 전달받는다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2020-12-31")
        mock_fund.return_value = pd.DataFrame()

        calls = []

        def tracking_factory(train_start, train_end):
            calls.append((train_start, train_end))
            return DummyStrategy(name=f"T_{train_start}")

        wf = WalkForwardBacktest(
            strategy_factory=tracking_factory,
            full_start_date="20140101",
            full_end_date="20201231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        # 2개 윈도우 예상
        assert len(calls) == 2
        assert calls[0] == ("20140101", "20181231")
        assert calls[1] == ("20150101", "20191231")

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_strategy_factory_failure_skips_window(self, mock_price, mock_fund):
        """strategy_factory가 예외를 발생시키면 해당 윈도우를 건너뛴다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        call_count = [0]

        def failing_factory(train_start, train_end):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("학습 데이터 부족")
            return DummyStrategy()

        wf = WalkForwardBacktest(
            strategy_factory=failing_factory,
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        # 3개 윈도우 중 1개 실패 -> 2개 결과
        assert len(wf._window_results) == 2


# ===================================================================
# OOS 성과 지표 테스트
# ===================================================================

class TestOOSMetrics:
    """OOS(Out-of-Sample) 성과 지표 계산 검증."""

    def test_get_oos_results_before_run_raises(self):
        """run() 호출 전에 get_oos_results()를 호출하면 RuntimeError."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20241231",
        )
        with pytest.raises(RuntimeError, match="Walk-Forward 백테스트가 아직 실행되지 않았습니다"):
            wf.get_oos_results()

    def test_get_oos_portfolio_history_before_run_raises(self):
        """run() 호출 전에 get_oos_portfolio_history()를 호출하면 RuntimeError."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20241231",
        )
        with pytest.raises(RuntimeError, match="Walk-Forward 백테스트가 아직 실행되지 않았습니다"):
            wf.get_oos_portfolio_history()

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_oos_results_keys(self, mock_price, mock_fund):
        """get_oos_results()의 반환값에 필수 키가 포함된다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        results = wf.get_oos_results()

        required_keys = [
            "sharpe_ratio",
            "cagr",
            "mdd",
            "total_return",
            "num_windows",
            "window_results",
        ]
        for key in required_keys:
            assert key in results, f"'{key}' 키가 결과에 없습니다."

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_mdd_non_positive(self, mock_price, mock_fund):
        """MDD는 항상 0 이하이다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        results = wf.get_oos_results()
        assert results["mdd"] <= 0

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_empty_strategy_zero_return(self, mock_price, mock_fund):
        """현금만 보유하면 OOS 총수익률이 0이다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        def empty_factory(train_start, train_end):
            return EmptyStrategy()

        wf = WalkForwardBacktest(
            strategy_factory=empty_factory,
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        results = wf.get_oos_results()
        assert results["total_return"] == 0.0
        assert results["cagr"] == 0.0

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_oos_portfolio_history_not_empty(self, mock_price, mock_fund):
        """OOS 포트폴리오 히스토리가 비어 있지 않다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        history = wf.get_oos_portfolio_history()
        assert not history.empty

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_oos_portfolio_history_columns(self, mock_price, mock_fund):
        """OOS 포트폴리오 히스토리에 올바른 컬럼이 있다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        history = wf.get_oos_portfolio_history()

        for col in ["portfolio_value", "cash", "num_holdings"]:
            assert col in history.columns

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_oos_portfolio_history_sorted(self, mock_price, mock_fund):
        """OOS 포트폴리오 히스토리가 시간순으로 정렬되어 있다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        history = wf.get_oos_portfolio_history()
        dates = history.index.tolist()
        assert dates == sorted(dates)

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_oos_portfolio_history_no_duplicates(self, mock_price, mock_fund):
        """OOS 포트폴리오 히스토리에 중복 날짜가 없다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        history = wf.get_oos_portfolio_history()
        assert not history.index.duplicated().any()

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_window_results_contain_backtest_keys(self, mock_price, mock_fund):
        """각 윈도우 결과에 Backtest.get_results()의 키가 포함된다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        results = wf.get_oos_results()

        backtest_keys = [
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

        for wr in results["window_results"]:
            for key in backtest_keys:
                assert key in wr, f"윈도우 결과에 '{key}' 키가 없습니다."

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_num_windows_in_results(self, mock_price, mock_fund):
        """num_windows가 실행된 윈도우 수와 일치한다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        results = wf.get_oos_results()
        assert results["num_windows"] == 3
        assert len(results["window_results"]) == 3

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_no_windows_returns_zero_metrics(self, mock_price, mock_fund):
        """윈도우가 없으면 모든 지표가 0이다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20200101",
            full_end_date="20220101",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        results = wf.get_oos_results()
        assert results["sharpe_ratio"] == 0.0
        assert results["cagr"] == 0.0
        assert results["mdd"] == 0.0
        assert results["total_return"] == 0.0
        assert results["num_windows"] == 0

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_oos_history_empty_for_no_windows(self, mock_price, mock_fund):
        """윈도우가 없으면 빈 DataFrame을 반환한다."""
        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20200101",
            full_end_date="20220101",
            train_years=5,
            test_years=1,
            step_months=12,
        )
        wf.run()

        history = wf.get_oos_portfolio_history()
        assert history.empty

    @patch("src.backtest.engine.get_all_fundamentals")
    @patch("src.backtest.engine.get_price_data")
    def test_transaction_costs_reduce_returns(self, mock_price, mock_fund):
        """거래비용이 반영되어 OOS 수익률이 감소한다."""
        mock_price.return_value = _make_flat_price_df("2019-01-01", "2021-12-31")
        mock_fund.return_value = pd.DataFrame()

        wf = WalkForwardBacktest(
            strategy_factory=_make_strategy_factory(),
            full_start_date="20140101",
            full_end_date="20211231",
            train_years=5,
            test_years=1,
            step_months=12,
            buy_cost=0.001,
            sell_cost=0.003,
        )
        wf.run()

        results = wf.get_oos_results()
        # 고정 가격이므로 거래비용만큼 손실 발생
        assert results["total_return"] < 0
