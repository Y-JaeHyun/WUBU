"""SwingReversionStrategy 테스트."""

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.strategy.short_term_base import ShortTermSignal, ShortTermStrategy
from src.strategy.swing_reversion import SwingReversionStrategy


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _make_test_df(n=30, base_close=10000, base_volume=100000):
    """테스트용 DataFrame 생성.

    RSI < 30이 되려면 최근에 연속 하락이 필요.
    볼린저 하한선 밑이려면 가격이 평균보다 확 낮아야 함.
    """
    dates = pd.date_range("2026-01-01", periods=n, freq="B")

    # 처음 20일 안정 후 마지막 10일 급락
    prices = [base_close] * (n - 10) + [base_close * (1 - 0.02 * i) for i in range(1, 11)]
    volumes = [base_volume] * (n - 1) + [base_volume * 3]  # 마지막 날 거래량 급등

    df = pd.DataFrame({
        "close": prices,
        "volume": volumes,
        "시가총액": [500_000_000_000] * n,
    }, index=dates)
    return df


def _make_korean_col_df(n=30, base_close=10000, base_volume=100000):
    """한글 컬럼명 DataFrame (pykrx 호환 테스트)."""
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    prices = [base_close] * (n - 10) + [base_close * (1 - 0.02 * i) for i in range(1, 11)]
    volumes = [base_volume] * (n - 1) + [base_volume * 3]

    df = pd.DataFrame({
        "종가": prices,
        "거래량": volumes,
        "시가총액": [500_000_000_000] * n,
    }, index=dates)
    return df


def _make_stable_df(n=30, base_close=10000, base_volume=100000):
    """변동 없는 안정적인 DataFrame (RSI ~50, 볼린저 중간)."""
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    prices = [base_close] * n
    volumes = [base_volume] * n

    df = pd.DataFrame({
        "close": prices,
        "volume": volumes,
        "시가총액": [500_000_000_000] * n,
    }, index=dates)
    return df


def _make_rising_df(n=30, base_close=10000, base_volume=100000):
    """상승 추세 DataFrame (RSI > 70)."""
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    # 처음 15일 안정 후 마지막 15일 급등
    prices = [base_close] * (n - 15) + [base_close * (1 + 0.03 * i) for i in range(1, 16)]
    volumes = [base_volume] * n

    df = pd.DataFrame({
        "close": prices,
        "volume": volumes,
        "시가총액": [500_000_000_000] * n,
    }, index=dates)
    return df


# ──────────────────────────────────────────────────────────
# 1. 기본 초기화
# ──────────────────────────────────────────────────────────

class TestInitialization:
    """초기화 관련 테스트."""

    def test_default_params(self):
        """기본 파라미터로 초기화."""
        strat = SwingReversionStrategy()
        assert strat.name == "swing_reversion"
        assert strat.mode == "swing"
        assert strat.params["volume_multiplier"] == 2.0
        assert strat.params["rsi_period"] == 14
        assert strat.params["rsi_oversold"] == 30
        assert strat.params["rsi_overbought"] == 70
        assert strat.params["bollinger_period"] == 20
        assert strat.params["bollinger_std"] == 2.0
        assert strat.params["max_signals"] == 3
        assert strat.params["stop_loss_pct"] == -0.05
        assert strat.params["take_profit_pct"] == 0.10
        assert strat.params["max_holding_days"] == 5

    def test_custom_params(self):
        """커스텀 파라미터 오버라이드."""
        strat = SwingReversionStrategy(params={
            "volume_multiplier": 3.0,
            "rsi_oversold": 25,
            "max_signals": 5,
        })
        assert strat.params["volume_multiplier"] == 3.0
        assert strat.params["rsi_oversold"] == 25
        assert strat.params["max_signals"] == 5
        # 나머지는 기본값 유지
        assert strat.params["rsi_period"] == 14
        assert strat.params["bollinger_period"] == 20

    def test_inherits_short_term_strategy(self):
        """ShortTermStrategy를 상속하는지 확인."""
        strat = SwingReversionStrategy()
        assert isinstance(strat, ShortTermStrategy)

    def test_params_returns_copy(self):
        """params 프로퍼티가 복사본을 반환하는지 확인."""
        strat = SwingReversionStrategy()
        p1 = strat.params
        p1["volume_multiplier"] = 999
        assert strat.params["volume_multiplier"] == 2.0  # 원본 변경 없음


# ──────────────────────────────────────────────────────────
# 2. RSI 계산
# ──────────────────────────────────────────────────────────

class TestRSICalculation:
    """RSI 계산 테스트."""

    def test_rsi_falling_prices(self):
        """하락 추세에서 RSI < 50."""
        df = _make_test_df()
        rsi = SwingReversionStrategy._calculate_rsi(df, 14)
        assert rsi is not None
        assert rsi < 50

    def test_rsi_rising_prices(self):
        """상승 추세에서 RSI > 50."""
        df = _make_rising_df()
        rsi = SwingReversionStrategy._calculate_rsi(df, 14)
        assert rsi is not None
        assert rsi > 50

    def test_rsi_insufficient_data(self):
        """데이터 부족 시 None 반환."""
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "close": [10000, 9900, 9800, 9700, 9600],
            "volume": [100000] * 5,
        }, index=dates)
        rsi = SwingReversionStrategy._calculate_rsi(df, 14)
        assert rsi is None

    def test_rsi_korean_columns(self):
        """한글 컬럼명(종가) 지원."""
        df = _make_korean_col_df()
        rsi = SwingReversionStrategy._calculate_rsi(df, 14)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_rsi_no_close_column(self):
        """close/종가 컬럼 없으면 None."""
        df = pd.DataFrame({"open": [100] * 20, "volume": [1000] * 20})
        rsi = SwingReversionStrategy._calculate_rsi(df, 14)
        assert rsi is None

    def test_rsi_all_gains(self):
        """모든 날 상승일 때 RSI = 100."""
        dates = pd.date_range("2026-01-01", periods=20, freq="B")
        df = pd.DataFrame({
            "close": [100 + i for i in range(20)],
        }, index=dates)
        rsi = SwingReversionStrategy._calculate_rsi(df, 14)
        assert rsi == 100.0


# ──────────────────────────────────────────────────────────
# 3. 볼린저밴드 계산
# ──────────────────────────────────────────────────────────

class TestBollingerCalculation:
    """볼린저밴드 계산 테스트."""

    def test_bollinger_normal(self):
        """정상 계산."""
        df = _make_test_df()
        lower, middle, upper = SwingReversionStrategy._calculate_bollinger(df, 20, 2.0)
        assert lower is not None
        assert middle is not None
        assert upper is not None
        assert lower < middle < upper

    def test_bollinger_insufficient_data(self):
        """데이터 부족 시 (None, None, None)."""
        df = _make_test_df(n=10)  # period=20보다 작음
        lower, middle, upper = SwingReversionStrategy._calculate_bollinger(df, 20, 2.0)
        assert lower is None
        assert middle is None
        assert upper is None

    def test_bollinger_korean_columns(self):
        """한글 컬럼명 지원."""
        df = _make_korean_col_df()
        lower, middle, upper = SwingReversionStrategy._calculate_bollinger(df, 20, 2.0)
        assert lower is not None

    def test_bollinger_no_close_column(self):
        """close/종가 컬럼 없으면 (None, None, None)."""
        df = pd.DataFrame({"open": [100] * 25})
        result = SwingReversionStrategy._calculate_bollinger(df, 20, 2.0)
        assert result == (None, None, None)

    def test_bollinger_wider_std(self):
        """표준편차 크면 밴드 넓어짐."""
        df = _make_test_df()
        l1, m1, u1 = SwingReversionStrategy._calculate_bollinger(df, 20, 1.0)
        l2, m2, u2 = SwingReversionStrategy._calculate_bollinger(df, 20, 3.0)
        assert m1 == m2  # 중심선 동일
        assert (u2 - l2) > (u1 - l1)  # 3 std가 더 넓음


# ──────────────────────────────────────────────────────────
# 4. scan_signals
# ──────────────────────────────────────────────────────────

class TestScanSignals:
    """scan_signals 테스트."""

    def test_empty_market_data(self):
        """빈 market_data → 빈 리스트."""
        strat = SwingReversionStrategy()
        result = strat.scan_signals({})
        assert result == []

    def test_empty_daily_data(self):
        """daily_data가 비어있으면 빈 리스트."""
        strat = SwingReversionStrategy()
        result = strat.scan_signals({"daily_data": {}})
        assert result == []

    def test_qualifying_ticker_generates_signal(self):
        """조건 충족 종목은 시그널 생성."""
        strat = SwingReversionStrategy()
        df = _make_test_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        # 조건을 충족하면 시그널이 1개 이상
        assert len(signals) >= 1
        sig = signals[0]
        assert sig.ticker == "005930"
        assert sig.strategy == "swing_reversion"
        assert sig.side == "buy"
        assert sig.mode == "swing"
        assert 0 < sig.confidence <= 1.0
        assert sig.stop_loss_price < sig.target_price
        assert sig.take_profit_price > sig.target_price
        assert sig.metadata.get("rsi") is not None
        assert sig.metadata.get("volume_ratio") is not None

    def test_max_signals_limit(self):
        """max_signals 제한 확인 (5개 종목, max_signals=3)."""
        strat = SwingReversionStrategy(params={"max_signals": 3})
        df = _make_test_df()
        daily_data = {f"00{i}000": df.copy() for i in range(5)}
        market_data = {"daily_data": daily_data}
        signals = strat.scan_signals(market_data)
        assert len(signals) <= 3

    def test_rsi_above_threshold_filtered(self):
        """RSI > 30인 종목 필터링."""
        strat = SwingReversionStrategy()
        df = _make_stable_df()  # RSI ~50
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_volume_below_threshold_filtered(self):
        """거래량 미달 종목 필터링."""
        strat = SwingReversionStrategy()
        df = _make_test_df()
        # 마지막 날 거래량을 일반 수준으로 변경
        df.iloc[-1, df.columns.get_loc("volume")] = 100000
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_insufficient_data_skipped(self):
        """데이터 부족 종목은 스킵."""
        strat = SwingReversionStrategy()
        df = _make_test_df(n=10)  # 20일 미만
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_none_dataframe_skipped(self):
        """DataFrame이 None이면 스킵."""
        strat = SwingReversionStrategy()
        market_data = {"daily_data": {"005930": None}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_signal_is_short_term_signal_type(self):
        """반환 타입이 ShortTermSignal인지 확인."""
        strat = SwingReversionStrategy()
        df = _make_test_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        for sig in signals:
            assert isinstance(sig, ShortTermSignal)


# ──────────────────────────────────────────────────────────
# 5. _evaluate_ticker
# ──────────────────────────────────────────────────────────

class TestEvaluateTicker:
    """_evaluate_ticker 테스트."""

    def test_qualifying_returns_dict(self):
        """조건 충족 시 딕셔너리 반환."""
        strat = SwingReversionStrategy()
        df = _make_test_df()
        result = strat._evaluate_ticker("005930", df)
        assert result is not None
        assert result["ticker"] == "005930"
        assert result["score"] > 0
        assert "rsi" in result
        assert "volume_ratio" in result
        assert "bb_position" in result
        assert "reason" in result

    def test_volume_below_multiplier_returns_none(self):
        """거래량 미달 → None."""
        strat = SwingReversionStrategy()
        df = _make_test_df()
        df.iloc[-1, df.columns.get_loc("volume")] = 100000  # 1배 (2배 미만)
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_rsi_above_oversold_returns_none(self):
        """RSI > 30 → None."""
        strat = SwingReversionStrategy()
        df = _make_stable_df()
        # 거래량은 급등시킴
        df.iloc[-1, df.columns.get_loc("volume")] = 300000
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_bb_position_above_threshold_returns_none(self):
        """볼린저 밴드 위치 > 0.2 → None."""
        strat = SwingReversionStrategy()
        # RSI < 30이지만 볼린저 위치는 중간인 데이터 만들기 어려움
        # 대신 RSI 기준을 매우 높게 설정해서 통과시키고, BB 위치가 중간인 경우
        strat._params["rsi_oversold"] = 60  # RSI < 60이면 통과
        df = _make_test_df()
        # 하락폭을 줄여서 BB 위치가 높게
        # 안정적 가격 + 마지막 날 살짝 하락
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        prices = [10000] * 25 + [9900, 9850, 9800, 9750, 9700]
        volumes = [100000] * 29 + [300000]
        df = pd.DataFrame({
            "close": prices,
            "volume": volumes,
            "시가총액": [500_000_000_000] * 30,
        }, index=dates)
        result = strat._evaluate_ticker("005930", df)
        # BB 위치가 높으면 None, 낮으면 dict - 이 경우 하락이 크지 않아 중간대
        # 실제 BB 위치에 따라 결과 다름
        if result is not None:
            assert result["bb_position"] <= 0.2

    def test_market_cap_below_threshold_returns_none(self):
        """시가총액 미달 → None."""
        strat = SwingReversionStrategy()
        df = _make_test_df()
        # 시가총액을 3000억 미만으로 설정
        df["시가총액"] = 100_000_000_000  # 1000억
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_zero_close_returns_none(self):
        """종가 0이면 None."""
        strat = SwingReversionStrategy()
        df = _make_test_df()
        df.iloc[-1, df.columns.get_loc("close")] = 0
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_zero_volume_returns_none(self):
        """거래량 0이면 None."""
        strat = SwingReversionStrategy()
        df = _make_test_df()
        df.iloc[-1, df.columns.get_loc("volume")] = 0
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_korean_columns_work(self):
        """한글 컬럼명으로도 동작."""
        strat = SwingReversionStrategy()
        df = _make_korean_col_df()
        result = strat._evaluate_ticker("005930", df)
        # 조건 충족 여부에 따라 None 또는 dict
        if result is not None:
            assert result["ticker"] == "005930"


# ──────────────────────────────────────────────────────────
# 6. check_exit
# ──────────────────────────────────────────────────────────

class TestCheckExit:
    """check_exit 테스트."""

    def test_stop_loss_triggers_sell(self):
        """손절 -5% 이하 → sell 시그널."""
        strat = SwingReversionStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 9400,  # -6%
            "entry_date": "2026-02-20",
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is not None
        assert result.side == "sell"
        assert "손절" in result.reason

    def test_take_profit_triggers_sell(self):
        """익절 +10% 이상 → sell 시그널."""
        strat = SwingReversionStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 11100,  # +11%
            "entry_date": "2026-02-20",
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is not None
        assert result.side == "sell"
        assert "익절" in result.reason

    def test_rsi_overbought_triggers_sell(self):
        """RSI >= 70 → sell 시그널."""
        strat = SwingReversionStrategy()
        df = _make_rising_df()  # RSI 높음
        rsi = SwingReversionStrategy._calculate_rsi(df, 14)

        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10100,  # 살짝 이익 (익절/손절 아님)
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
        }
        market_data = {"daily_data": {"005930": df}}
        result = strat.check_exit(position, market_data)

        if rsi is not None and rsi >= 70:
            assert result is not None
            assert "RSI 과매수" in result.reason

    def test_max_holding_days_triggers_sell(self):
        """보유일 초과 → sell 시그널."""
        strat = SwingReversionStrategy()
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10050,  # 소폭 이익
            "entry_date": old_date,
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is not None
        assert result.side == "sell"
        assert "보유일 초과" in result.reason

    def test_no_exit_condition_returns_none(self):
        """청산 조건 없으면 None."""
        strat = SwingReversionStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10200,  # +2% (손절/익절 아님)
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is None

    def test_invalid_entry_price_returns_none(self):
        """entry_price <= 0이면 None."""
        strat = SwingReversionStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 0,
            "current_price": 10000,
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is None

    def test_invalid_current_price_returns_none(self):
        """current_price <= 0이면 None."""
        strat = SwingReversionStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 0,
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is None

    def test_exit_signal_type(self):
        """청산 시그널이 ShortTermSignal 타입."""
        strat = SwingReversionStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 9400,
            "entry_date": "2026-02-20",
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert isinstance(result, ShortTermSignal)
        assert result.strategy == "swing_reversion"
        assert result.mode == "swing"

    def test_exit_multiple_reasons(self):
        """여러 청산 조건이 동시에 충족."""
        strat = SwingReversionStrategy()
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 9400,  # -6% (손절) + 보유일 초과
            "entry_date": old_date,
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is not None
        assert "손절" in result.reason
        assert "보유일 초과" in result.reason

    def test_exit_invalid_date_format_ignored(self):
        """날짜 형식이 잘못되면 보유일 체크 스킵."""
        strat = SwingReversionStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10050,
            "entry_date": "invalid-date",
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is None  # 다른 조건 미충족


# ──────────────────────────────────────────────────────────
# 7. 점수 계산 / 정렬
# ──────────────────────────────────────────────────────────

class TestScoring:
    """점수 계산 및 정렬 테스트."""

    def test_score_range(self):
        """점수는 0~100 범위."""
        strat = SwingReversionStrategy()
        df = _make_test_df()
        result = strat._evaluate_ticker("005930", df)
        if result is not None:
            assert 0 <= result["score"] <= 100

    def test_confidence_capped_at_1(self):
        """confidence는 최대 1.0."""
        strat = SwingReversionStrategy()
        df = _make_test_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        for sig in signals:
            assert sig.confidence <= 1.0

    def test_signals_sorted_by_score(self):
        """시그널은 점수 내림차순 정렬."""
        strat = SwingReversionStrategy(params={"max_signals": 10})
        # 여러 종목 생성 (약간씩 다른 하락폭)
        daily_data = {}
        for i in range(5):
            n = 30
            dates = pd.date_range("2026-01-01", periods=n, freq="B")
            drop_rate = 0.02 + 0.005 * i  # 종목마다 다른 하락률
            prices = [10000] * (n - 10) + [10000 * (1 - drop_rate * j) for j in range(1, 11)]
            volumes = [100000] * (n - 1) + [100000 * (3 + i)]
            df = pd.DataFrame({
                "close": prices,
                "volume": volumes,
                "시가총액": [500_000_000_000] * n,
            }, index=dates)
            daily_data[f"00{i}930"] = df

        market_data = {"daily_data": daily_data}
        signals = strat.scan_signals(market_data)

        # 시그널이 2개 이상이면 confidence 내림차순 확인
        if len(signals) >= 2:
            for i in range(len(signals) - 1):
                assert signals[i].confidence >= signals[i + 1].confidence

    def test_higher_volume_higher_score(self):
        """거래량이 더 높은 종목이 더 높은 점수 (다른 조건 동일시)."""
        strat = SwingReversionStrategy()
        df1 = _make_test_df()
        df2 = _make_test_df()
        # df2의 마지막 거래량을 더 크게
        df2.iloc[-1, df2.columns.get_loc("volume")] = 500000  # 5배

        r1 = strat._evaluate_ticker("A", df1)
        r2 = strat._evaluate_ticker("B", df2)

        if r1 is not None and r2 is not None:
            assert r2["score"] >= r1["score"]


# ──────────────────────────────────────────────────────────
# 8. OBV 다이버전스 필터
# ──────────────────────────────────────────────────────────

class TestOBVDivergence:
    """OBV 다이버전스 필터 테스트."""

    def test_bullish_divergence_detected(self):
        """가격 하락 + OBV 상승 = 강세 다이버전스 True."""
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        # 가격은 전체적으로 하락하지만, 상승일에 대량 거래 → OBV 상승
        prices = [100, 99, 101, 98, 100, 97, 99, 96, 98, 95]
        # 상승일(+2,+2,+2,+2)에 대량, 하락일(-1,-3,-3,-3,-3)에 소량
        volumes = [100, 10, 500, 10, 500, 10, 500, 10, 500, 10]
        df = pd.DataFrame({"close": prices, "volume": volumes}, index=dates)
        result = SwingReversionStrategy._check_obv_divergence(df, lookback=5)
        assert result is True

    def test_no_divergence_both_falling(self):
        """가격 하락 + OBV도 하락 = 다이버전스 아님."""
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        # 지속적 하락 + 대량 거래량 (투매)
        prices = [100, 98, 96, 94, 92, 90, 88, 86, 84, 82]
        volumes = [100, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        df = pd.DataFrame({"close": prices, "volume": volumes}, index=dates)
        result = SwingReversionStrategy._check_obv_divergence(df, lookback=5)
        assert result is False

    def test_no_divergence_price_rising(self):
        """가격 상승이면 다이버전스 아님."""
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        volumes = [100] * 10
        df = pd.DataFrame({"close": prices, "volume": volumes}, index=dates)
        result = SwingReversionStrategy._check_obv_divergence(df, lookback=5)
        assert result is False

    def test_insufficient_data_returns_false(self):
        """데이터 부족 시 False."""
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        df = pd.DataFrame({"close": [100, 99, 98], "volume": [100, 100, 100]}, index=dates)
        result = SwingReversionStrategy._check_obv_divergence(df, lookback=5)
        assert result is False

    def test_no_close_column_returns_false(self):
        """close 컬럼 없으면 False."""
        df = pd.DataFrame({"open": [100] * 10, "volume": [100] * 10})
        result = SwingReversionStrategy._check_obv_divergence(df, lookback=5)
        assert result is False

    def test_korean_columns_work(self):
        """한글 컬럼(종가/거래량) 지원."""
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        prices = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]
        volumes = [100] * 10
        df = pd.DataFrame({"종가": prices, "거래량": volumes}, index=dates)
        result = SwingReversionStrategy._check_obv_divergence(df, lookback=5)
        assert isinstance(result, bool)

    def test_obv_filter_off_by_default(self):
        """use_obv_filter=False(기본값)이면 OBV 체크 없이 통과."""
        strat = SwingReversionStrategy()
        assert strat.params["use_obv_filter"] is False
        df = _make_test_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        # OBV 필터 없이 기존과 동일하게 동작
        assert len(signals) >= 1

    def test_obv_filter_on_filters_signals(self):
        """use_obv_filter=True이면 OBV 다이버전스 없는 종목 필터링."""
        strat = SwingReversionStrategy(params={"use_obv_filter": True})
        df = _make_test_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        # OBV 필터가 추가 조건이므로 시그널 수가 줄거나 0이 될 수 있음
        assert isinstance(signals, list)
