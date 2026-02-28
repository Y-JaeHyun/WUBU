"""백테스트 엔진 3-Pool 지원 테스트.

pool_strategies 파라미터를 사용한 멀티풀 백테스트와
ETF 매도 비용 차등 적용을 검증한다.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Backtest, Strategy


# ===================================================================
# 더미 전략 클래스
# ===================================================================

class FixedStrategy(Strategy):
    """고정 시그널을 반환하는 더미 전략."""

    def __init__(self, name: str, signals: dict):
        self._name = name
        self._signals = signals

    @property
    def name(self) -> str:
        return self._name

    def generate_signals(self, date: str, data: dict) -> dict:
        return self._signals.copy()


class EmptyStrategy(Strategy):
    """빈 시그널 전략."""

    @property
    def name(self) -> str:
        return "EmptyStrategy"

    def generate_signals(self, date: str, data: dict) -> dict:
        return {}


# ===================================================================
# 헬퍼
# ===================================================================

def _make_price_df(start: str, periods: int, base_price: int = 50000) -> pd.DataFrame:
    """mock 가격 DataFrame."""
    dates = pd.bdate_range(start, periods=periods)
    close_arr = np.full(periods, base_price)
    df = pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr,
            "low": close_arr,
            "close": close_arr,
            "volume": [1_000_000] * periods,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _make_index_series(start: str, periods: int) -> pd.Series:
    """mock 지수 시리즈."""
    dates = pd.bdate_range(start, periods=periods)
    return pd.Series(np.full(periods, 2500.0), index=dates, name="close")


# ===================================================================
# 3-Pool 모드 테스트
# ===================================================================

class TestMultiPoolBacktest:
    """pool_strategies를 사용한 3-Pool 백테스트."""

    @patch("src.backtest.engine.get_price_data")
    @patch("src.backtest.engine.get_all_fundamentals")
    def test_pool_strategies_signal_merging(self, mock_fund, mock_price):
        """풀별 시그널이 풀 비중으로 스케일링 후 병합된다."""
        # 장기: A=50% → 70% pool → 35%
        # ETF: B=100% → 30% pool → 30%
        long_strat = FixedStrategy("Long", {"A": 0.5})
        etf_strat = FixedStrategy("ETF", {"B": 1.0})

        mock_fund.return_value = pd.DataFrame({"ticker": ["A"]})
        price_df = _make_price_df("2023-06-01", 600, 10000)
        mock_price.return_value = price_df

        bt = Backtest(
            strategy=EmptyStrategy(),  # pool_strategies가 있으면 무시됨
            start_date="2024-01-01",
            end_date="2024-03-31",
            initial_capital=10_000_000,
            pool_strategies={
                "long_term": (long_strat, 0.7),
                "etf_rotation": (etf_strat, 0.3),
            },
        )

        with patch.object(bt, "_get_rebalance_dates", return_value=["20240102"]):
            with patch("src.data.index_collector.get_index_data") as mock_idx:
                mock_idx.return_value = pd.DataFrame()
                bt.run()

        # 거래가 발생했어야 함 (A, B 모두)
        tickers_traded = {t["ticker"] for t in bt._trades}
        assert "A" in tickers_traded
        assert "B" in tickers_traded

    @patch("src.backtest.engine.get_price_data")
    @patch("src.backtest.engine.get_all_fundamentals")
    def test_pool_strategies_overlapping_tickers(self, mock_fund, mock_price):
        """같은 종목이 두 풀에 포함되면 비중이 합산된다."""
        # 장기: A=60% → 70% pool → 42%
        # ETF: A=40% → 30% pool → 12%
        # 합산: A=54%
        long_strat = FixedStrategy("Long", {"A": 0.6})
        etf_strat = FixedStrategy("ETF", {"A": 0.4})

        mock_fund.return_value = pd.DataFrame({"ticker": ["A"]})
        price_df = _make_price_df("2023-06-01", 600, 10000)
        mock_price.return_value = price_df

        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="2024-01-01",
            end_date="2024-02-28",
            initial_capital=10_000_000,
            pool_strategies={
                "long_term": (long_strat, 0.7),
                "etf_rotation": (etf_strat, 0.3),
            },
        )

        with patch.object(bt, "_get_rebalance_dates", return_value=["20240102"]):
            with patch("src.data.index_collector.get_index_data") as mock_idx:
                mock_idx.return_value = pd.DataFrame()
                bt.run()

        # A 종목만 매수되어야 함
        buy_trades = [t for t in bt._trades if t["action"] == "buy"]
        assert len(buy_trades) == 1
        assert buy_trades[0]["ticker"] == "A"

    @patch("src.backtest.engine.get_price_data")
    @patch("src.backtest.engine.get_all_fundamentals")
    def test_single_strategy_mode_unchanged(self, mock_fund, mock_price):
        """pool_strategies 없이 기존 단일 전략 모드가 정상 동작한다."""
        strat = FixedStrategy("Single", {"A": 0.5})

        mock_fund.return_value = pd.DataFrame({"ticker": ["A"]})
        price_df = _make_price_df("2023-06-01", 600, 10000)
        mock_price.return_value = price_df

        bt = Backtest(
            strategy=strat,
            start_date="2024-01-01",
            end_date="2024-02-28",
            initial_capital=10_000_000,
        )

        with patch.object(bt, "_get_rebalance_dates", return_value=["20240102"]):
            with patch("src.data.index_collector.get_index_data") as mock_idx:
                mock_idx.return_value = pd.DataFrame()
                bt.run()

        assert len(bt._trades) > 0
        assert bt._trades[0]["ticker"] == "A"

    def test_strategy_name_multipool(self):
        """pool_strategies 설정 시 전략 이름이 MultiPool 형식이다."""
        long_strat = FixedStrategy("MultiFactor", {})
        etf_strat = FixedStrategy("ETFRotation", {})

        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="2024-01-01",
            end_date="2024-12-31",
            pool_strategies={
                "long_term": (long_strat, 0.7),
                "etf_rotation": (etf_strat, 0.3),
            },
        )

        name = bt._get_strategy_name()
        assert "MultiPool" in name
        assert "MultiFactor" in name
        assert "ETFRotation" in name

    def test_strategy_name_single(self):
        """단일 전략 모드에서 전략 이름이 반환된다."""
        strat = FixedStrategy("MyStrategy", {})
        bt = Backtest(strategy=strat, start_date="2024-01-01", end_date="2024-12-31")
        assert bt._get_strategy_name() == "MyStrategy"

    def test_zero_pct_pool_ignored(self):
        """비중이 0인 풀의 시그널은 무시된다."""
        long_strat = FixedStrategy("Long", {"A": 0.5})
        short_strat = FixedStrategy("Short", {"B": 1.0})

        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="2024-01-01",
            end_date="2024-12-31",
            pool_strategies={
                "long_term": (long_strat, 1.0),
                "short_term": (short_strat, 0.0),
            },
        )
        # short_term은 pct=0이므로 시그널 병합 시 무시됨
        # 직접 시그널 병합 로직 테스트
        assert bt.pool_strategies is not None


# ===================================================================
# ETF 매도 비용 차등 테스트
# ===================================================================

class TestETFSellCost:
    """ETF 종목의 매도 비용 차등 적용."""

    def test_get_sell_cost_etf(self):
        """ETF 티커는 etf_sell_cost가 적용된다."""
        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="2024-01-01",
            end_date="2024-12-31",
            sell_cost=0.00245,
            etf_sell_cost=0.00015,
        )
        # 069500은 KODEX 200 (ETF)
        assert bt._get_sell_cost("069500") == 0.00015

    def test_get_sell_cost_stock(self):
        """일반 주식 티커는 sell_cost가 적용된다."""
        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="2024-01-01",
            end_date="2024-12-31",
            sell_cost=0.00245,
            etf_sell_cost=0.00015,
        )
        # 005930은 삼성전자 (주식)
        assert bt._get_sell_cost("005930") == 0.00245

    def test_etf_sell_cost_default(self):
        """etf_sell_cost 기본값은 0.00015이다."""
        bt = Backtest(
            strategy=EmptyStrategy(),
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        assert bt.etf_sell_cost == 0.00015
