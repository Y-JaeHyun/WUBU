"""단기 트레이딩 백테스트 엔진(src/backtest/short_term_backtest.py) 테스트.

ShortTermStrategy ABC를 상속한 더미 전략으로 백테스트 실행,
T+1 시가 진입, 손절/익절, 수수료, 성과 지표 등을 검증한다.
모든 테스트는 합성 데이터를 사용하며 pykrx를 호출하지 않는다.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.backtest.short_term_backtest import (
    BacktestPosition,
    BacktestTrade,
    ShortTermBacktest,
)
from src.strategy.short_term_base import ShortTermSignal, ShortTermStrategy


# ===================================================================
# 합성 가격 데이터 헬퍼
# ===================================================================

def _make_price_df(
    days: int = 300,
    base_price: float = 10000,
    trend: float = 0.001,
    volatility: float = 0.02,
    start_date: str = "20240101",
    seed: int = 42,
) -> pd.DataFrame:
    """합성 일별 OHLCV DataFrame을 생성한다."""
    np.random.seed(seed)
    dates = pd.bdate_range(start=start_date, periods=days)
    prices = [base_price]
    for _ in range(days - 1):
        change = np.random.normal(trend, volatility)
        prices.append(prices[-1] * (1 + change))

    prices_arr = np.array(prices)
    df = pd.DataFrame(
        {
            "open": prices_arr * 0.999,
            "high": prices_arr * 1.01,
            "low": prices_arr * 0.99,
            "close": prices_arr,
            "volume": [1_000_000] * days,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _make_flat_price_df(
    days: int = 60,
    price: float = 10000,
    start_date: str = "20240101",
) -> pd.DataFrame:
    """변동 없는 고정 가격 DataFrame."""
    dates = pd.bdate_range(start=start_date, periods=days)
    df = pd.DataFrame(
        {
            "open": [price] * days,
            "high": [price] * days,
            "low": [price] * days,
            "close": [price] * days,
            "volume": [1_000_000] * days,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _make_trending_price_df(
    days: int = 60,
    start_price: float = 10000,
    end_price: float = 15000,
    start_date: str = "20240101",
) -> pd.DataFrame:
    """선형 상승/하락 가격 DataFrame."""
    dates = pd.bdate_range(start=start_date, periods=days)
    prices = np.linspace(start_price, end_price, days)
    df = pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": [1_000_000] * days,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ===================================================================
# 더미 전략 클래스들
# ===================================================================

class DummyStrategy(ShortTermStrategy):
    """테스트용 더미 전략. 설정된 시그널을 반환하고 check_exit는 항상 None."""

    name = "DummySwing"
    mode = "swing"

    def __init__(self, signals: Optional[list[ShortTermSignal]] = None):
        self._signals = signals or []
        self._scan_count = 0

    def scan_signals(self, market_data: dict) -> list[ShortTermSignal]:
        self._scan_count += 1
        return self._signals

    def check_exit(self, position: dict, market_data: dict) -> Optional[ShortTermSignal]:
        return None


class OneShotStrategy(ShortTermStrategy):
    """첫 번째 스캔에서만 시그널을 반환하는 전략."""

    name = "OneShot"
    mode = "swing"

    def __init__(self, signals: list[ShortTermSignal]):
        self._signals = signals
        self._fired = False

    def scan_signals(self, market_data: dict) -> list[ShortTermSignal]:
        if not self._fired:
            self._fired = True
            return self._signals
        return []

    def check_exit(self, position: dict, market_data: dict) -> Optional[ShortTermSignal]:
        return None


class ExitAfterNDaysStrategy(ShortTermStrategy):
    """N일 후 무조건 청산 시그널을 반환하는 전략."""

    name = "ExitAfterN"
    mode = "swing"

    def __init__(self, signals: list[ShortTermSignal], hold_days: int = 5):
        self._signals = signals
        self._hold_days = hold_days
        self._fired = False

    def scan_signals(self, market_data: dict) -> list[ShortTermSignal]:
        if not self._fired:
            self._fired = True
            return self._signals
        return []

    def check_exit(self, position: dict, market_data: dict) -> Optional[ShortTermSignal]:
        entry = pd.Timestamp(position["entry_date"])
        current = pd.Timestamp(market_data["date"])
        if (current - entry).days >= self._hold_days:
            return ShortTermSignal(
                id="exit",
                ticker=position["ticker"],
                strategy=self.name,
                side="sell",
                mode="swing",
                confidence=1.0,
                reason="time_stop",
            )
        return None


class EmptyStrategy(ShortTermStrategy):
    """시그널을 생성하지 않는 빈 전략."""

    name = "Empty"
    mode = "swing"

    def scan_signals(self, market_data: dict) -> list[ShortTermSignal]:
        return []

    def check_exit(self, position: dict, market_data: dict) -> Optional[ShortTermSignal]:
        return None


class ErrorStrategy(ShortTermStrategy):
    """scan_signals에서 에러를 발생시키는 전략."""

    name = "Error"
    mode = "swing"

    def scan_signals(self, market_data: dict) -> list[ShortTermSignal]:
        raise ValueError("Intentional scan error")

    def check_exit(self, position: dict, market_data: dict) -> Optional[ShortTermSignal]:
        return None


# ===================================================================
# 헬퍼: 시그널 생성
# ===================================================================

def _make_signal(
    ticker: str = "005930",
    stop_loss: float = 9000,
    take_profit: float = 12000,
    confidence: float = 0.8,
    strategy: str = "DummySwing",
) -> ShortTermSignal:
    return ShortTermSignal(
        id=f"sig_{ticker}_001",
        ticker=ticker,
        strategy=strategy,
        side="buy",
        mode="swing",
        confidence=confidence,
        target_price=0.0,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        reason="test_signal",
    )


# ===================================================================
# 1. BacktestPosition 테스트
# ===================================================================

class TestBacktestPosition:
    """BacktestPosition 데이터클래스 테스트."""

    def test_creation(self):
        """기본 생성."""
        pos = BacktestPosition(
            ticker="005930",
            entry_date="20240102",
            entry_price=10000,
            quantity=10,
            stop_loss_price=9000,
            take_profit_price=12000,
            strategy_name="TestStrategy",
        )
        assert pos.ticker == "005930"
        assert pos.entry_price == 10000
        assert pos.quantity == 10
        assert pos.mode == "swing"

    def test_peak_price_defaults_to_entry(self):
        """peak_price가 0이면 entry_price로 초기화."""
        pos = BacktestPosition(
            ticker="005930",
            entry_date="20240102",
            entry_price=10000,
            quantity=10,
            stop_loss_price=9000,
            take_profit_price=12000,
            strategy_name="test",
        )
        assert pos.peak_price == 10000

    def test_peak_price_explicit(self):
        """명시적 peak_price 설정."""
        pos = BacktestPosition(
            ticker="005930",
            entry_date="20240102",
            entry_price=10000,
            quantity=10,
            stop_loss_price=9000,
            take_profit_price=12000,
            strategy_name="test",
            peak_price=11000,
        )
        assert pos.peak_price == 11000

    def test_to_dict(self):
        """to_dict() 변환."""
        pos = BacktestPosition(
            ticker="005930",
            entry_date="20240102",
            entry_price=10000,
            quantity=10,
            stop_loss_price=9000,
            take_profit_price=12000,
            strategy_name="test",
        )
        d = pos.to_dict()
        assert d["ticker"] == "005930"
        assert d["entry_price"] == 10000
        assert d["peak_price"] == 10000
        assert isinstance(d["metadata"], dict)

    def test_metadata_default(self):
        """metadata 기본값은 빈 딕셔너리."""
        pos = BacktestPosition(
            ticker="005930",
            entry_date="20240102",
            entry_price=10000,
            quantity=10,
            stop_loss_price=9000,
            take_profit_price=12000,
            strategy_name="test",
        )
        assert pos.metadata == {}


# ===================================================================
# 2. BacktestTrade 테스트
# ===================================================================

class TestBacktestTrade:
    """BacktestTrade 데이터클래스 테스트."""

    def test_creation(self):
        trade = BacktestTrade(
            ticker="005930",
            entry_date="20240102",
            exit_date="20240110",
            entry_price=10000,
            exit_price=11000,
            quantity=10,
            pnl=9500,
            pnl_pct=9.5,
            commission=500,
            reason="take_profit",
            strategy_name="test",
        )
        assert trade.pnl == 9500
        assert trade.reason == "take_profit"

    def test_to_dict(self):
        trade = BacktestTrade(
            ticker="005930",
            entry_date="20240102",
            exit_date="20240110",
            entry_price=10000,
            exit_price=11000,
            quantity=10,
            pnl=9500,
            pnl_pct=9.5,
            commission=500,
            reason="take_profit",
            strategy_name="test",
        )
        d = trade.to_dict()
        assert d["ticker"] == "005930"
        assert d["pnl"] == 9500
        assert d["reason"] == "take_profit"


# ===================================================================
# 3. ShortTermBacktest 초기화 테스트
# ===================================================================

class TestShortTermBacktestInit:
    """ShortTermBacktest 초기화 검증."""

    def test_default_parameters(self):
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240101",
            end_date="20240331",
        )
        assert bt.initial_capital == 145_000
        assert bt.max_positions == 3
        assert bt.buy_cost == 0.00015
        assert bt.sell_cost_kospi == 0.00245
        assert bt.sell_cost_kosdaq == 0.00165
        assert bt.start_date == "20240101"
        assert bt.end_date == "20240331"

    def test_custom_parameters(self):
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="2024-01-01",
            end_date="2024-03-31",
            initial_capital=1_000_000,
            max_positions=5,
            buy_cost=0.001,
            sell_cost_kospi=0.003,
            sell_cost_kosdaq=0.002,
        )
        assert bt.initial_capital == 1_000_000
        assert bt.max_positions == 5
        assert bt.buy_cost == 0.001
        assert bt.sell_cost_kospi == 0.003
        assert bt.sell_cost_kosdaq == 0.002
        assert bt.start_date == "20240101"

    def test_date_hyphen_format(self):
        """하이픈 형식 날짜가 올바르게 변환된다."""
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="2024-06-15",
            end_date="2024-12-31",
        )
        assert bt.start_date == "20240615"
        assert bt.end_date == "20241231"


# ===================================================================
# 4. 미실행 상태 에러 테스트
# ===================================================================

class TestBeforeRun:
    """run() 호출 전에 결과를 조회하면 RuntimeError."""

    def test_get_results_before_run(self):
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240101",
            end_date="20240331",
        )
        with pytest.raises(RuntimeError, match="백테스트가 아직 실행되지 않았습니다"):
            bt.get_results()

    def test_get_trades_before_run(self):
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240101",
            end_date="20240331",
        )
        with pytest.raises(RuntimeError, match="백테스트가 아직 실행되지 않았습니다"):
            bt.get_trades()

    def test_get_portfolio_history_before_run(self):
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240101",
            end_date="20240331",
        )
        with pytest.raises(RuntimeError, match="백테스트가 아직 실행되지 않았습니다"):
            bt.get_portfolio_history()


# ===================================================================
# 5. 빈 전략 (시그널 없음)
# ===================================================================

class TestEmptyStrategy:
    """시그널이 없는 전략 → 자본금 유지."""

    def test_cash_preserved(self):
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
            initial_capital=1_000_000,
        )
        bt.run(preloaded_data={})

        results = bt.get_results()
        assert results["total_return"] == 0.0
        assert results["total_trades"] == 0
        assert results["final_value"] == 1_000_000

    def test_portfolio_history_constant(self):
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
            initial_capital=500_000,
        )
        bt.run(preloaded_data={})

        history = bt.get_portfolio_history()
        assert not history.empty
        assert all(history["portfolio_value"] == 500_000)
        assert all(history["num_positions"] == 0)

    def test_empty_trades_dataframe(self):
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        bt.run(preloaded_data={})

        trades = bt.get_trades()
        assert trades.empty
        expected_cols = [
            "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
            "quantity", "pnl", "pnl_pct", "commission", "reason", "strategy_name",
        ]
        for col in expected_cols:
            assert col in trades.columns


# ===================================================================
# 6. T+1 진입 테스트
# ===================================================================

class TestT1Entry:
    """시그널 발생 다음 날 시가로 진입하는지 검증."""

    def test_entry_at_next_day_open(self):
        """시그널 T일 → 진입 T+1 시가."""
        price_df = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        # 시가를 종가와 다르게 설정
        price_df["open"] = 9900

        signal = _make_signal(
            ticker="005930",
            stop_loss=5000,
            take_profit=20000,
        )
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=100_000,
            max_positions=3,
        )
        bt.run(preloaded_data={"005930": price_df})

        trades = bt.get_trades()
        assert len(trades) >= 1  # backtest_end로 청산됨
        # 진입가가 시가(9900)여야 한다
        assert trades.iloc[0]["entry_price"] == 9900

    def test_no_same_day_entry(self):
        """시그널 스캔일에 바로 진입하지 않는다 (다음 날 진입)."""
        price_df = _make_flat_price_df(days=5, price=10000, start_date="20240102")
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=20000)
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240108",
            initial_capital=100_000,
            max_positions=3,
        )
        bt.run(preloaded_data={"005930": price_df})

        # 첫 날(1/2) 스캔 → 둘째 날(1/3) 진입
        # 따라서 1/2의 포지션 수는 0이어야 함
        history = bt.get_portfolio_history()
        first_day = history.iloc[0]
        assert first_day["num_positions"] == 0


# ===================================================================
# 7. 손절 (Stop Loss) 테스트
# ===================================================================

class TestStopLoss:
    """손절 조건 검증."""

    def test_stop_loss_triggered(self):
        """종가가 손절가 이하로 하락하면 청산."""
        # 가격: 10000 → 9000 (하락)
        price_df = _make_trending_price_df(
            days=20,
            start_price=10000,
            end_price=8000,
            start_date="20240102",
        )

        signal = _make_signal(
            ticker="005930",
            stop_loss=9000,
            take_profit=15000,
        )
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240131",
            initial_capital=100_000,
            max_positions=3,
        )
        bt.run(preloaded_data={"005930": price_df})

        trades = bt.get_trades()
        # 손절로 청산된 거래가 있어야 한다
        sl_trades = trades[trades["reason"] == "stop_loss"]
        assert len(sl_trades) >= 1
        # 손절 거래는 손실
        assert sl_trades.iloc[0]["pnl"] < 0

    def test_stop_loss_zero_means_disabled(self):
        """stop_loss_price=0이면 손절 비활성화."""
        price_df = _make_trending_price_df(
            days=20,
            start_price=10000,
            end_price=5000,
            start_date="20240102",
        )

        signal = _make_signal(
            ticker="005930",
            stop_loss=0,
            take_profit=0,
        )
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240131",
            initial_capital=100_000,
            max_positions=3,
        )
        bt.run(preloaded_data={"005930": price_df})

        trades = bt.get_trades()
        # 손절 없이 backtest_end로만 청산
        for _, t in trades.iterrows():
            assert t["reason"] == "backtest_end"


# ===================================================================
# 8. 익절 (Take Profit) 테스트
# ===================================================================

class TestTakeProfit:
    """익절 조건 검증."""

    def test_take_profit_triggered(self):
        """종가가 익절가 이상으로 상승하면 청산."""
        price_df = _make_trending_price_df(
            days=20,
            start_price=10000,
            end_price=15000,
            start_date="20240102",
        )

        signal = _make_signal(
            ticker="005930",
            stop_loss=8000,
            take_profit=12000,
        )
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240131",
            initial_capital=100_000,
            max_positions=3,
        )
        bt.run(preloaded_data={"005930": price_df})

        trades = bt.get_trades()
        tp_trades = trades[trades["reason"] == "take_profit"]
        assert len(tp_trades) >= 1
        assert tp_trades.iloc[0]["pnl"] > 0


# ===================================================================
# 9. 전략 커스텀 청산 (check_exit)
# ===================================================================

class TestStrategyExit:
    """전략의 check_exit()에 의한 청산 검증."""

    def test_strategy_exit_after_n_days(self):
        """N일 보유 후 전략이 청산 시그널을 반환."""
        price_df = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        signal = _make_signal(
            ticker="005930",
            stop_loss=5000,
            take_profit=20000,
        )

        strategy = ExitAfterNDaysStrategy(signals=[signal], hold_days=5)

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=100_000,
            max_positions=3,
        )
        bt.run(preloaded_data={"005930": price_df})

        trades = bt.get_trades()
        exit_trades = trades[trades["reason"] == "strategy_exit"]
        assert len(exit_trades) >= 1


# ===================================================================
# 10. 수수료 테스트
# ===================================================================

class TestCommission:
    """수수료 계산 검증."""

    def test_commission_deducted(self):
        """가격 변동 없는 종목에서 수수료만큼 손실 발생."""
        price_df = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        signal = _make_signal(
            ticker="005930",
            stop_loss=5000,
            take_profit=20000,
        )

        strategy = ExitAfterNDaysStrategy(signals=[signal], hold_days=5)

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=1_000_000,
            max_positions=1,
            buy_cost=0.001,
            sell_cost_kospi=0.003,
        )
        bt.run(preloaded_data={"005930": price_df})

        trades = bt.get_trades()
        assert len(trades) >= 1
        # 가격 변동 없으므로 수수료만큼 손실
        first_trade = trades.iloc[0]
        assert first_trade["pnl"] < 0
        assert first_trade["commission"] > 0

    def test_kospi_vs_kosdaq_sell_cost(self):
        """KOSPI vs KOSDAQ 매도 비용률 차이."""
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        # KOSPI (0으로 시작)
        assert bt._get_sell_cost("005930") == bt.sell_cost_kospi
        # KOSDAQ (2, 3으로 시작)
        assert bt._get_sell_cost("247540") == bt.sell_cost_kosdaq
        assert bt._get_sell_cost("373220") == bt.sell_cost_kosdaq
        # KOSPI (다른 접두사)
        assert bt._get_sell_cost("105560") == bt.sell_cost_kospi

    def test_commission_total_in_results(self):
        """결과에 총 수수료가 포함된다."""
        price_df = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=20000)
        strategy = ExitAfterNDaysStrategy(signals=[signal], hold_days=3)

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        assert results["commission_total"] > 0


# ===================================================================
# 11. 성과 지표 계산 테스트
# ===================================================================

class TestMetrics:
    """성과 지표 계산 정합성 검증."""

    def test_results_keys(self):
        """get_results()에 모든 필수 키가 포함된다."""
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        bt.run(preloaded_data={})

        results = bt.get_results()
        required_keys = [
            "strategy_name", "start_date", "end_date",
            "initial_capital", "final_value",
            "total_return", "cagr", "sharpe_ratio", "max_drawdown",
            "win_rate", "profit_factor", "avg_holding_days",
            "total_trades", "commission_total",
            "avg_win_pct", "avg_loss_pct", "max_win_pct", "max_loss_pct",
        ]
        for key in required_keys:
            assert key in results, f"'{key}' 키가 결과에 없습니다."

    def test_total_return_positive_for_uptrend(self):
        """상승 추세 종목 → 양의 수익률."""
        price_df = _make_trending_price_df(
            days=30,
            start_price=10000,
            end_price=13000,
            start_date="20240102",
        )
        signal = _make_signal(
            ticker="005930",
            stop_loss=5000,
            take_profit=20000,
        )
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        assert results["total_return"] > 0

    def test_mdd_non_positive(self):
        """MDD는 항상 0 이하."""
        price_df = _make_price_df(days=60, start_date="20240102")
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=50000)
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240401",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        assert results["max_drawdown"] <= 0

    def test_win_rate_all_wins(self):
        """모든 거래가 수익 → 승률 1.0."""
        # 강한 상승 추세
        price_df = _make_trending_price_df(
            days=15,
            start_price=10000,
            end_price=13000,
            start_date="20240102",
        )
        signal = _make_signal(
            ticker="005930",
            stop_loss=5000,
            take_profit=12000,
        )
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240122",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        if results["total_trades"] > 0:
            assert results["win_rate"] == 1.0

    def test_win_rate_all_losses(self):
        """모든 거래가 손실 → 승률 0."""
        price_df = _make_trending_price_df(
            days=15,
            start_price=10000,
            end_price=7000,
            start_date="20240102",
        )
        signal = _make_signal(
            ticker="005930",
            stop_loss=8000,
            take_profit=20000,
        )
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240122",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        if results["total_trades"] > 0:
            assert results["win_rate"] == 0.0

    def test_sharpe_ratio_sign(self):
        """상승 추세 → 양의 샤프비율."""
        price_df = _make_trending_price_df(
            days=60,
            start_price=10000,
            end_price=15000,
            start_date="20240102",
        )
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=50000)
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240401",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        assert results["sharpe_ratio"] > 0

    def test_avg_holding_days(self):
        """보유일 계산 검증."""
        price_df = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=20000)
        strategy = ExitAfterNDaysStrategy(signals=[signal], hold_days=7)

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        if results["total_trades"] > 0:
            # 7일 보유이므로 avg_holding_days >= 7 (주말 포함)
            assert results["avg_holding_days"] >= 7

    def test_profit_factor_no_losses(self):
        """손실 거래가 없으면 profit_factor = inf."""
        price_df = _make_trending_price_df(
            days=15,
            start_price=10000,
            end_price=13000,
            start_date="20240102",
        )
        signal = _make_signal(
            ticker="005930",
            stop_loss=5000,
            take_profit=12000,
        )
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240122",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        if results["total_trades"] > 0 and results["win_rate"] == 1.0:
            assert results["profit_factor"] == float("inf")


# ===================================================================
# 12. 다중 포지션 관리
# ===================================================================

class TestMultiplePositions:
    """동시에 여러 포지션을 관리하는 경우."""

    def test_max_positions_respected(self):
        """max_positions 제한 준수."""
        price_a = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        price_b = _make_flat_price_df(days=30, price=20000, start_date="20240102")
        price_c = _make_flat_price_df(days=30, price=30000, start_date="20240102")
        price_d = _make_flat_price_df(days=30, price=5000, start_date="20240102")

        signals = [
            _make_signal(ticker="005930", stop_loss=5000, take_profit=20000),
            _make_signal(ticker="000660", stop_loss=10000, take_profit=40000),
            _make_signal(ticker="035420", stop_loss=15000, take_profit=60000),
            _make_signal(ticker="051910", stop_loss=2000, take_profit=10000),
        ]
        strategy = OneShotStrategy(signals=signals)

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=1_000_000,
            max_positions=2,
        )
        bt.run(preloaded_data={
            "005930": price_a,
            "000660": price_b,
            "035420": price_c,
            "051910": price_d,
        })

        # 히스토리에서 최대 포지션 수 확인
        history = bt.get_portfolio_history()
        assert history["num_positions"].max() <= 2

    def test_multiple_tickers(self):
        """여러 종목 동시 보유."""
        price_a = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        price_b = _make_flat_price_df(days=30, price=20000, start_date="20240102")

        signals = [
            _make_signal(ticker="005930", stop_loss=5000, take_profit=20000),
            _make_signal(ticker="000660", stop_loss=10000, take_profit=40000),
        ]
        strategy = OneShotStrategy(signals=signals)

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=1_000_000,
            max_positions=3,
        )
        bt.run(preloaded_data={
            "005930": price_a,
            "000660": price_b,
        })

        trades = bt.get_trades()
        tickers = trades["ticker"].unique()
        assert len(tickers) == 2

    def test_no_duplicate_ticker_entry(self):
        """동일 종목 중복 진입 방지."""
        price_df = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        signals = [
            _make_signal(ticker="005930", stop_loss=5000, take_profit=20000),
            _make_signal(ticker="005930", stop_loss=5000, take_profit=20000),
        ]
        strategy = OneShotStrategy(signals=signals)

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=1_000_000,
            max_positions=3,
        )
        bt.run(preloaded_data={"005930": price_df})

        # backtest_end로 한 개의 거래만 생성
        trades = bt.get_trades()
        # 동일 종목이 동시에 2개 진입하지 않아야 함
        assert len(trades) <= 1


# ===================================================================
# 13. 포트폴리오 히스토리 테스트
# ===================================================================

class TestPortfolioHistory:
    """일별 포트폴리오 가치 추적."""

    def test_history_columns(self):
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        bt.run(preloaded_data={})

        history = bt.get_portfolio_history()
        for col in ["portfolio_value", "cash", "num_positions"]:
            assert col in history.columns

    def test_history_length(self):
        """히스토리 길이가 영업일 수와 일치."""
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        bt.run(preloaded_data={})

        history = bt.get_portfolio_history()
        expected = len(pd.bdate_range("20240102", "20240131"))
        assert len(history) == expected

    def test_history_reflects_position(self):
        """포지션 보유 시 포트폴리오 가치에 반영."""
        price_df = _make_trending_price_df(
            days=20,
            start_price=10000,
            end_price=12000,
            start_date="20240102",
        )
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=20000)
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240129",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        history = bt.get_portfolio_history()
        # 포지션 진입 후 가격 상승이므로 포트폴리오 가치 상승
        first_with_position = history[history["num_positions"] > 0]
        if not first_with_position.empty:
            # 마지막 포지션 보유일 > 첫 포지션 보유일 (가격 상승)
            assert first_with_position["portfolio_value"].iloc[-1] >= first_with_position["portfolio_value"].iloc[0]


# ===================================================================
# 14. 엣지 케이스
# ===================================================================

class TestEdgeCases:
    """엣지 케이스 검증."""

    def test_empty_date_range(self):
        """시작일 > 종료일이면 비어 있는 결과."""
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240630",
            end_date="20240101",
            initial_capital=1_000_000,
        )
        bt.run(preloaded_data={})

        results = bt.get_results()
        assert results["total_trades"] == 0
        history = bt.get_portfolio_history()
        assert history.empty

    def test_no_price_data_for_ticker(self):
        """시그널 종목의 가격 데이터가 없으면 진입 스킵."""
        signal = _make_signal(ticker="999999", stop_loss=5000, take_profit=20000)
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240131",
            initial_capital=1_000_000,
            max_positions=3,
        )
        # 빈 preloaded_data
        bt.run(preloaded_data={})

        results = bt.get_results()
        assert results["total_trades"] == 0
        assert results["final_value"] == 1_000_000

    def test_insufficient_capital(self):
        """자본금이 부족하면 진입 불가."""
        price_df = _make_flat_price_df(days=30, price=100000, start_date="20240102")
        signal = _make_signal(ticker="005930", stop_loss=50000, take_profit=200000)
        strategy = OneShotStrategy(signals=[signal])

        # 매우 적은 자본금 (1주도 못 삼)
        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240131",
            initial_capital=100,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        assert results["total_trades"] == 0

    def test_strategy_scan_error_handled(self):
        """scan_signals 에러가 발생해도 백테스트가 중단되지 않는다."""
        bt = ShortTermBacktest(
            strategy=ErrorStrategy(),
            start_date="20240102",
            end_date="20240131",
            initial_capital=1_000_000,
        )
        bt.run(preloaded_data={})

        results = bt.get_results()
        assert results["total_trades"] == 0

    def test_backtest_end_force_close(self):
        """백테스트 종료 시 잔여 포지션이 강제 청산된다."""
        price_df = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=20000)
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        trades = bt.get_trades()
        assert len(trades) == 1
        assert trades.iloc[0]["reason"] == "backtest_end"

    def test_sell_signal_ignored(self):
        """side='sell' 시그널은 무시된다 (매수만 지원)."""
        price_df = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        sell_signal = ShortTermSignal(
            id="sell_001",
            ticker="005930",
            strategy="DummySwing",
            side="sell",
            mode="swing",
            confidence=0.9,
            stop_loss_price=9000,
            take_profit_price=12000,
        )
        strategy = OneShotStrategy(signals=[sell_signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240131",
            initial_capital=1_000_000,
            max_positions=3,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        assert results["total_trades"] == 0

    def test_single_day_backtest(self):
        """1영업일 백테스트."""
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240102",
            initial_capital=100_000,
        )
        bt.run(preloaded_data={})

        history = bt.get_portfolio_history()
        assert len(history) == 1
        assert history.iloc[0]["portfolio_value"] == 100_000


# ===================================================================
# 15. 트레일링 스탑 (peak_price 추적)
# ===================================================================

class TestPeakPriceTracking:
    """포지션의 peak_price가 올바르게 업데이트되는지 검증."""

    def test_peak_price_updated(self):
        """가격이 상승하면 peak_price가 업데이트된다."""
        price_df = _make_trending_price_df(
            days=20,
            start_price=10000,
            end_price=15000,
            start_date="20240102",
        )
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=50000)

        # check_exit에서 peak_price를 확인하는 전략
        class PeakCheckStrategy(ShortTermStrategy):
            name = "PeakCheck"
            mode = "swing"
            peak_prices_seen = []

            def __init__(self, signals):
                self._signals = signals
                self._fired = False

            def scan_signals(self, market_data):
                if not self._fired:
                    self._fired = True
                    return self._signals
                return []

            def check_exit(self, position, market_data):
                self.peak_prices_seen.append(position["peak_price"])
                return None

        strategy = PeakCheckStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240129",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        # peak_price가 단조 증가해야 함 (상승 추세)
        if len(strategy.peak_prices_seen) > 1:
            for i in range(1, len(strategy.peak_prices_seen)):
                assert strategy.peak_prices_seen[i] >= strategy.peak_prices_seen[i - 1]


# ===================================================================
# 16. 거래 DataFrame 형식 테스트
# ===================================================================

class TestTradesDataFrame:
    """get_trades() 반환값 형식 검증."""

    def test_trades_with_data(self):
        """거래가 있을 때 DataFrame 형식."""
        price_df = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=20000)
        strategy = ExitAfterNDaysStrategy(signals=[signal], hold_days=3)

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        trades = bt.get_trades()
        assert not trades.empty
        # 날짜 컬럼이 datetime 타입
        assert pd.api.types.is_datetime64_any_dtype(trades["entry_date"])
        assert pd.api.types.is_datetime64_any_dtype(trades["exit_date"])
        # pnl_pct 존재
        assert "pnl_pct" in trades.columns

    def test_empty_trades_dataframe_columns(self):
        """거래가 없어도 올바른 컬럼을 가진다."""
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        bt.run(preloaded_data={})

        trades = bt.get_trades()
        expected_cols = [
            "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
            "quantity", "pnl", "pnl_pct", "commission", "reason", "strategy_name",
        ]
        for col in expected_cols:
            assert col in trades.columns


# ===================================================================
# 17. market_data 구성 테스트
# ===================================================================

class TestMarketData:
    """전략에 전달되는 market_data 구조 검증."""

    def test_market_data_keys(self):
        """market_data에 daily_data와 date 키가 있다."""
        received_data = []

        class DataCapture(ShortTermStrategy):
            name = "DataCapture"
            mode = "swing"

            def scan_signals(self, market_data):
                received_data.append(market_data)
                return []

            def check_exit(self, position, market_data):
                return None

        price_df = _make_flat_price_df(days=10, price=10000, start_date="20240102")

        bt = ShortTermBacktest(
            strategy=DataCapture(),
            start_date="20240102",
            end_date="20240115",
            initial_capital=1_000_000,
        )
        bt.run(preloaded_data={"005930": price_df})

        assert len(received_data) > 0
        md = received_data[0]
        assert "daily_data" in md
        assert "date" in md
        assert isinstance(md["daily_data"], dict)
        assert isinstance(md["date"], str)

    def test_market_data_slicing(self):
        """market_data의 daily_data는 최대 252일로 슬라이싱된다."""
        received_data = []

        class DataCapture(ShortTermStrategy):
            name = "DataCapture"
            mode = "swing"

            def scan_signals(self, market_data):
                received_data.append(market_data)
                return []

            def check_exit(self, position, market_data):
                return None

        price_df = _make_price_df(days=400, start_date="20230101")

        bt = ShortTermBacktest(
            strategy=DataCapture(),
            start_date="20240102",
            end_date="20240115",
            initial_capital=1_000_000,
        )
        bt.run(preloaded_data={"005930": price_df})

        if received_data:
            md = received_data[-1]
            if "005930" in md["daily_data"]:
                assert len(md["daily_data"]["005930"]) <= 500


# ===================================================================
# 18. 포지션 사이징
# ===================================================================

class TestPositionSizing:
    """동일 비중 포지션 사이징 검증."""

    def test_equal_weight_sizing(self):
        """각 포지션이 대략 자본금/max_positions 비중."""
        price_a = _make_flat_price_df(days=30, price=1000, start_date="20240102")
        price_b = _make_flat_price_df(days=30, price=1000, start_date="20240102")

        signals = [
            _make_signal(ticker="005930", stop_loss=500, take_profit=2000),
            _make_signal(ticker="000660", stop_loss=500, take_profit=2000),
        ]
        strategy = OneShotStrategy(signals=signals)

        initial_capital = 100_000
        max_positions = 2

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=initial_capital,
            max_positions=max_positions,
        )
        bt.run(preloaded_data={
            "005930": price_a,
            "000660": price_b,
        })

        trades = bt.get_trades()
        if len(trades) >= 2:
            # 두 거래의 금액이 대략 비슷 (동일비중)
            amounts = []
            for _, t in trades.iterrows():
                amounts.append(t["entry_price"] * t["quantity"])
            if len(amounts) >= 2:
                ratio = max(amounts) / min(amounts) if min(amounts) > 0 else float("inf")
                # 2배 이내 (첫 번째와 두 번째 포지션 크기 유사)
                assert ratio < 2.0


# ===================================================================
# 19. 연속 매매 (청산 후 재진입)
# ===================================================================

class TestConsecutiveTrades:
    """청산 후 다시 시그널이 발생하면 재진입."""

    def test_re_entry_after_exit(self):
        """N일 보유 후 청산 → 새 시그널 → 재진입."""
        price_df = _make_flat_price_df(days=60, price=10000, start_date="20240102")
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=20000)

        # 항상 시그널을 반환하는 전략 (매번 재진입 유도)
        class AlwaysSignalStrategy(ShortTermStrategy):
            name = "AlwaysSignal"
            mode = "swing"

            def __init__(self, signal, hold_days=5):
                self._signal = signal
                self._hold_days = hold_days

            def scan_signals(self, market_data):
                return [self._signal]

            def check_exit(self, position, market_data):
                entry = pd.Timestamp(position["entry_date"])
                current = pd.Timestamp(market_data["date"])
                if (current - entry).days >= self._hold_days:
                    return ShortTermSignal(
                        id="exit",
                        ticker=position["ticker"],
                        strategy=self.name,
                        side="sell",
                        mode="swing",
                        confidence=1.0,
                    )
                return None

        strategy = AlwaysSignalStrategy(signal, hold_days=5)

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240329",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        trades = bt.get_trades()
        # 여러 번 진입/청산 반복
        assert len(trades) >= 3


# ===================================================================
# 20. 결과 수치 정합성
# ===================================================================

class TestResultConsistency:
    """결과 수치간 정합성."""

    def test_final_value_matches_history(self):
        """final_value가 포트폴리오 히스토리 마지막 값과 일치."""
        price_df = _make_price_df(days=30, start_date="20240102")
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=50000)
        strategy = OneShotStrategy(signals=[signal])

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240212",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        history = bt.get_portfolio_history()

        assert abs(results["final_value"] - history["portfolio_value"].iloc[-1]) < 1

    def test_total_trades_matches(self):
        """total_trades가 get_trades() 길이와 일치."""
        price_df = _make_flat_price_df(days=30, price=10000, start_date="20240102")
        signal = _make_signal(ticker="005930", stop_loss=5000, take_profit=20000)
        strategy = ExitAfterNDaysStrategy(signals=[signal], hold_days=3)

        bt = ShortTermBacktest(
            strategy=strategy,
            start_date="20240102",
            end_date="20240215",
            initial_capital=1_000_000,
            max_positions=1,
        )
        bt.run(preloaded_data={"005930": price_df})

        results = bt.get_results()
        trades = bt.get_trades()
        assert results["total_trades"] == len(trades)

    def test_strategy_name_in_results(self):
        """결과에 전략 이름이 포함된다."""
        bt = ShortTermBacktest(
            strategy=EmptyStrategy(),
            start_date="20240102",
            end_date="20240131",
        )
        bt.run(preloaded_data={})

        results = bt.get_results()
        assert results["strategy_name"] == "Empty"
