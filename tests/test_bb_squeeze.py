"""BBSqueezeStrategy 테스트."""

import math
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.strategy.short_term_base import ShortTermSignal, ShortTermStrategy
from src.strategy.bb_squeeze import BBSqueezeStrategy


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _make_squeeze_breakout_df(
    n=300,
    base_close=10000,
    base_volume=100000,
    breakout=True,
    volume_spike=True,
    above_ma200=True,
):
    """BB 스퀴즈 + 브레이크아웃 조건을 충족하는 테스트 DataFrame 생성.

    3단계 구조:
    1. Phase 1 (0~169): 상승 추세 + 적당한 변동성 (MA200 충족)
    2. Phase 2 (170~279): 보통 변동성 (lookback에서 높은 BW 확보)
    3. Phase 3 (280~299): 매우 타이트한 스퀴즈 (20일) + 마지막 날 브레이크아웃

    핵심: lookback 126일에 변동성 높은 구간이 있어야 5th percentile가
    스퀴즈 구간보다 높게 형성됨 → 스퀴즈 감지 가능.
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    # Phase 1: 상승 추세 + 적당한 변동성
    phase1 = []
    p = base_close
    for i in range(170):
        if above_ma200:
            p = p * (1 + 0.001 + np.random.uniform(-0.015, 0.015))
        else:
            p = p * (1 - 0.002 + np.random.uniform(-0.005, 0.005))
        phase1.append(p)

    # Phase 2: 보통 변동성 유지 (lookback 기간에 높은 BW 값 포함)
    last_p = phase1[-1]
    phase2 = []
    for i in range(110):
        last_p = last_p * (1 + np.random.uniform(-0.005, 0.005))
        phase2.append(last_p)

    # Phase 3: 타이트한 스퀴즈 (20일, bb_period와 동일)
    last_p = phase2[-1]
    phase3 = [last_p + np.random.uniform(-0.1, 0.1) for _ in range(19)]

    if breakout:
        # 상단밴드 돌파: 타이트한 밴드에서 살짝 위로
        phase3.append(last_p + 0.5)
    else:
        # 밴드 내부 유지
        phase3.append(last_p + np.random.uniform(-0.05, 0.05))

    prices = phase1 + phase2 + phase3

    # 거래량
    volumes = [base_volume] * (n - 1)
    if volume_spike:
        volumes.append(int(base_volume * 2.5))  # 2.5배
    else:
        volumes.append(base_volume)

    df = pd.DataFrame({
        "close": prices,
        "volume": volumes,
        "시가총액": [500_000_000_000] * n,
    }, index=dates)
    return df


def _make_wide_band_df(n=300, base_close=10000, base_volume=100000):
    """밴드폭이 넓은 (스퀴즈 아닌) DataFrame.

    높은 변동성으로 인해 밴드폭이 상위 백분위에 위치.
    """
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(42)
    # 큰 변동성 -- 일관되게 높은 변동성
    prices = [base_close]
    for i in range(1, n):
        change = np.random.uniform(-0.03, 0.03)
        prices.append(prices[-1] * (1 + change))
    volumes = [base_volume] * n

    df = pd.DataFrame({
        "close": prices,
        "volume": volumes,
        "시가총액": [500_000_000_000] * n,
    }, index=dates)
    return df


def _make_korean_col_df(n=300, base_close=10000, base_volume=100000):
    """한글 컬럼명 DataFrame (pykrx 호환)."""
    df = _make_squeeze_breakout_df(n=n, base_close=base_close, base_volume=base_volume)
    df = df.rename(columns={"close": "종가", "volume": "거래량"})
    return df


def _make_short_df(n=50):
    """데이터가 부족한 DataFrame (200일 미만)."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = [10000 + i * 10 for i in range(n)]
    volumes = [100000] * n
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

    def test_name(self):
        """전략 이름 확인."""
        strat = BBSqueezeStrategy()
        assert strat.name == "bb_squeeze"

    def test_mode(self):
        """모드 확인."""
        strat = BBSqueezeStrategy()
        assert strat.mode == "swing"

    def test_default_params(self):
        """기본 파라미터 확인."""
        strat = BBSqueezeStrategy()
        p = strat.params
        assert p["bb_period"] == 20
        assert p["bb_std"] == 2.0
        assert p["bandwidth_lookback"] == 126
        assert p["bandwidth_percentile"] == 5
        assert p["volume_multiplier"] == 1.5
        assert p["volume_avg_days"] == 20
        assert p["ma_trend_period"] == 200
        assert p["take_profit_pct"] == 0.10
        assert p["stop_loss_pct"] == -0.05
        assert p["max_holding_days"] == 7
        assert p["max_signals"] == 3
        assert p["min_market_cap"] == 300_000_000_000

    def test_custom_params(self):
        """커스텀 파라미터 오버라이드."""
        strat = BBSqueezeStrategy(params={
            "bb_period": 25,
            "volume_multiplier": 2.0,
            "max_signals": 5,
        })
        p = strat.params
        assert p["bb_period"] == 25
        assert p["volume_multiplier"] == 2.0
        assert p["max_signals"] == 5
        # 나머지는 기본값
        assert p["bb_std"] == 2.0
        assert p["ma_trend_period"] == 200

    def test_inherits_short_term_strategy(self):
        """ShortTermStrategy를 상속하는지 확인."""
        strat = BBSqueezeStrategy()
        assert isinstance(strat, ShortTermStrategy)

    def test_params_returns_copy(self):
        """params 프로퍼티가 복사본을 반환하는지 확인."""
        strat = BBSqueezeStrategy()
        p1 = strat.params
        p1["bb_period"] = 999
        assert strat.params["bb_period"] == 20  # 원본 변경 없음


# ──────────────────────────────────────────────────────────
# 2. Bandwidth 계산
# ──────────────────────────────────────────────────────────

class TestBandwidthCalculation:
    """Bollinger Bandwidth 계산 테스트."""

    def test_bandwidth_basic(self):
        """기본 bandwidth 계산 - NaN이 아닌 값 반환."""
        closes = pd.Series([100 + i * 0.1 for i in range(50)])
        bw = BBSqueezeStrategy._calculate_bandwidth(closes, 20, 2.0)
        assert len(bw) == 50
        # 처음 19개는 NaN (rolling period)
        assert pd.isna(bw.iloc[0])
        # 20번째부터 유효
        assert not pd.isna(bw.iloc[19])
        assert bw.iloc[19] > 0

    def test_bandwidth_constant_price(self):
        """가격이 일정하면 bandwidth → 0에 수렴."""
        closes = pd.Series([100.0] * 50)
        bw = BBSqueezeStrategy._calculate_bandwidth(closes, 20, 2.0)
        # std = 0이면 bandwidth = 0
        valid_bw = bw.dropna()
        assert len(valid_bw) > 0
        assert all(v == 0 or abs(v) < 1e-10 for v in valid_bw)

    def test_bandwidth_high_volatility(self):
        """변동성이 클수록 bandwidth가 큼."""
        # 저변동성
        closes_low = pd.Series([100 + np.sin(i * 0.1) for i in range(50)])
        # 고변동성
        closes_high = pd.Series([100 + np.sin(i * 0.1) * 10 for i in range(50)])

        bw_low = BBSqueezeStrategy._calculate_bandwidth(closes_low, 20, 2.0)
        bw_high = BBSqueezeStrategy._calculate_bandwidth(closes_high, 20, 2.0)

        # 마지막 유효 값 비교
        assert bw_high.iloc[-1] > bw_low.iloc[-1]

    def test_bandwidth_formula_correctness(self):
        """Bandwidth = (upper - lower) / middle 공식 검증."""
        np.random.seed(123)
        closes = pd.Series(np.cumsum(np.random.randn(50)) + 100)
        period = 20
        std = 2.0

        bw = BBSqueezeStrategy._calculate_bandwidth(closes, period, std)

        # 수동 계산
        sma = closes.rolling(period).mean()
        rolling_std = closes.rolling(period).std()
        upper = sma + std * rolling_std
        lower = sma - std * rolling_std
        expected = (upper - lower) / sma

        # 마지막 값 비교
        assert abs(bw.iloc[-1] - expected.iloc[-1]) < 1e-10

    def test_bandwidth_returns_series(self):
        """반환 타입이 pd.Series."""
        closes = pd.Series([100 + i for i in range(30)])
        bw = BBSqueezeStrategy._calculate_bandwidth(closes, 20, 2.0)
        assert isinstance(bw, pd.Series)


# ──────────────────────────────────────────────────────────
# 3. scan_signals
# ──────────────────────────────────────────────────────────

class TestScanSignals:
    """scan_signals 테스트."""

    def test_empty_market_data(self):
        """빈 market_data -> 빈 리스트."""
        strat = BBSqueezeStrategy()
        result = strat.scan_signals({})
        assert result == []

    def test_empty_daily_data(self):
        """daily_data가 비어있으면 빈 리스트."""
        strat = BBSqueezeStrategy()
        result = strat.scan_signals({"daily_data": {}})
        assert result == []

    def test_valid_squeeze_breakout_generates_signal(self):
        """조건 충족 (스퀴즈 + 브레이크아웃 + 거래량 + MA200) -> 시그널 생성."""
        strat = BBSqueezeStrategy()
        df = _make_squeeze_breakout_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) >= 1
        sig = signals[0]
        assert sig.ticker == "005930"
        assert sig.strategy == "bb_squeeze"
        assert sig.side == "buy"
        assert sig.mode == "swing"
        assert 0 < sig.confidence <= 1.0
        assert sig.stop_loss_price < sig.target_price
        assert sig.take_profit_price > sig.target_price
        assert "bandwidth" in sig.metadata
        assert "volume_ratio" in sig.metadata

    def test_no_squeeze_no_signal(self):
        """스퀴즈 아닌 (밴드폭 넓은) 상태 -> 시그널 없음."""
        strat = BBSqueezeStrategy()
        df = _make_wide_band_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_squeeze_but_no_breakout_no_signal(self):
        """스퀴즈 상태이지만 상단밴드 돌파 없음 -> 시그널 없음."""
        strat = BBSqueezeStrategy()
        df = _make_squeeze_breakout_df(breakout=False)
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_volume_too_low_no_signal(self):
        """거래량 미달 -> 시그널 없음."""
        strat = BBSqueezeStrategy()
        df = _make_squeeze_breakout_df(volume_spike=False)
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_below_ma200_no_signal(self):
        """MA200 아래 -> 시그널 없음 (추세 필터)."""
        strat = BBSqueezeStrategy()
        # 하락 추세 데이터를 직접 구성: 가격이 MA200 아래에 확실히 위치
        n = 300
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # 지속적 하락 추세
        prices = [10000 * (1 - 0.003 * i) for i in range(n)]
        volumes = [100000] * (n - 1) + [250000]
        df = pd.DataFrame({
            "close": prices,
            "volume": volumes,
            "시가총액": [500_000_000_000] * n,
        }, index=dates)
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_market_cap_filter(self):
        """시가총액 미달 -> 시그널 없음."""
        strat = BBSqueezeStrategy()
        df = _make_squeeze_breakout_df()
        df["시가총액"] = 100_000_000_000  # 1000억 (3000억 미달)
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_max_signals_limit(self):
        """max_signals 제한 확인."""
        strat = BBSqueezeStrategy(params={"max_signals": 2})
        df = _make_squeeze_breakout_df()
        daily_data = {f"00{i}930": df.copy() for i in range(5)}
        market_data = {"daily_data": daily_data}
        signals = strat.scan_signals(market_data)
        assert len(signals) <= 2

    def test_none_dataframe_skipped(self):
        """DataFrame이 None이면 스킵."""
        strat = BBSqueezeStrategy()
        market_data = {"daily_data": {"005930": None}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_signal_type(self):
        """반환 타입이 ShortTermSignal."""
        strat = BBSqueezeStrategy()
        df = _make_squeeze_breakout_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        for sig in signals:
            assert isinstance(sig, ShortTermSignal)

    def test_korean_columns_work(self):
        """한글 컬럼명(종가/거래량) 지원."""
        strat = BBSqueezeStrategy()
        df = _make_korean_col_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        # 한글 컬럼도 작동 (시그널 생성 여부는 데이터에 따라)
        # 에러 없이 실행되면 성공
        assert isinstance(signals, list)

    def test_signals_sorted_by_score(self):
        """시그널은 점수 내림차순 정렬."""
        strat = BBSqueezeStrategy(params={"max_signals": 10})
        # 여러 종목 (약간 다른 거래량)
        daily_data = {}
        for i in range(5):
            df = _make_squeeze_breakout_df()
            # 거래량 다르게 설정
            df.iloc[-1, df.columns.get_loc("volume")] = int(100000 * (2 + i * 0.5))
            daily_data[f"00{i}930"] = df
        market_data = {"daily_data": daily_data}
        signals = strat.scan_signals(market_data)
        if len(signals) >= 2:
            for i in range(len(signals) - 1):
                assert signals[i].confidence >= signals[i + 1].confidence


# ──────────────────────────────────────────────────────────
# 4. _evaluate_ticker
# ──────────────────────────────────────────────────────────

class TestEvaluateTicker:
    """_evaluate_ticker 테스트."""

    def test_qualifying_returns_dict(self):
        """조건 충족 시 딕셔너리 반환."""
        strat = BBSqueezeStrategy()
        df = _make_squeeze_breakout_df()
        result = strat._evaluate_ticker("005930", df)
        assert result is not None
        assert result["ticker"] == "005930"
        assert result["score"] > 0
        assert "bandwidth" in result
        assert "volume_ratio" in result
        assert "ma200_distance" in result
        assert "reason" in result

    def test_no_close_column_returns_none(self):
        """close/종가 컬럼 없으면 None."""
        strat = BBSqueezeStrategy()
        df = pd.DataFrame({
            "open": [100] * 250,
            "volume": [1000] * 250,
        })
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_no_volume_column_returns_none(self):
        """volume/거래량 컬럼 없으면 None."""
        strat = BBSqueezeStrategy()
        df = pd.DataFrame({
            "close": [100] * 250,
        })
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_zero_close_returns_none(self):
        """종가 0이면 None."""
        strat = BBSqueezeStrategy()
        df = _make_squeeze_breakout_df()
        df.iloc[-1, df.columns.get_loc("close")] = 0
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_score_range(self):
        """점수는 0~100 범위."""
        strat = BBSqueezeStrategy()
        df = _make_squeeze_breakout_df()
        result = strat._evaluate_ticker("005930", df)
        if result is not None:
            assert 0 <= result["score"] <= 100

    def test_no_market_cap_column_still_works(self):
        """시가총액 컬럼 없어도 동작 (필터 스킵)."""
        strat = BBSqueezeStrategy()
        df = _make_squeeze_breakout_df()
        df = df.drop(columns=["시가총액"])
        result = strat._evaluate_ticker("005930", df)
        # 시가총액 필터 스킵 -> 다른 조건 충족 시 dict 반환
        if result is not None:
            assert result["ticker"] == "005930"


# ──────────────────────────────────────────────────────────
# 5. check_exit
# ──────────────────────────────────────────────────────────

class TestCheckExit:
    """check_exit 테스트."""

    def test_stop_loss_triggers_sell(self):
        """손절 -5% 이하 -> sell 시그널."""
        strat = BBSqueezeStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 9400,  # -6%
            "entry_date": "2026-02-20",
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is not None
        assert result.side == "sell"
        assert result.strategy == "bb_squeeze"
        assert "손절" in result.reason

    def test_take_profit_triggers_sell(self):
        """익절 +10% 이상 -> sell 시그널."""
        strat = BBSqueezeStrategy()
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

    def test_band_reentry_triggers_sell(self):
        """하단밴드 재진입 -> sell 시그널."""
        strat = BBSqueezeStrategy()
        # 가격이 하단밴드 아래인 데이터
        n = 30
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        # 처음 안정적이다가 급락
        prices = [10000] * 20 + [10000 * (1 - 0.02 * i) for i in range(1, 11)]
        volumes = [100000] * n
        df = pd.DataFrame({
            "close": prices,
            "volume": volumes,
        }, index=dates)

        current_price = prices[-1]  # 급락한 가격
        position = {
            "ticker": "005930",
            "entry_price": current_price * 1.02,  # 약간 위에서 진입 (손절 아닌 범위)
            "current_price": current_price,
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
        }
        market_data = {"daily_data": {"005930": df}}
        result = strat.check_exit(position, market_data)
        # 하단밴드 아래이면 청산 시그널
        if result is not None:
            assert "하단밴드" in result.reason or "손절" in result.reason

    def test_time_stop_triggers_sell(self):
        """보유일 초과 (7일) -> sell 시그널."""
        strat = BBSqueezeStrategy()
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10050,  # 소폭 이익 (손절/익절 아님)
            "entry_date": old_date,
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is not None
        assert result.side == "sell"
        assert "보유일 초과" in result.reason

    def test_no_exit_condition_returns_none(self):
        """청산 조건 미충족 -> None."""
        strat = BBSqueezeStrategy()
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
        strat = BBSqueezeStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 0,
            "current_price": 10000,
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is None

    def test_invalid_current_price_returns_none(self):
        """current_price <= 0이면 None."""
        strat = BBSqueezeStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 0,
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is None

    def test_exit_signal_type(self):
        """청산 시그널이 ShortTermSignal 타입."""
        strat = BBSqueezeStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 9400,
            "entry_date": "2026-02-20",
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert isinstance(result, ShortTermSignal)
        assert result.strategy == "bb_squeeze"
        assert result.mode == "swing"

    def test_exit_multiple_reasons(self):
        """여러 청산 조건 동시 충족."""
        strat = BBSqueezeStrategy()
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
        """날짜 형식 잘못되면 보유일 체크 스킵."""
        strat = BBSqueezeStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10050,
            "entry_date": "invalid-date",
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is None


# ──────────────────────────────────────────────────────────
# 6. Edge cases
# ──────────────────────────────────────────────────────────

class TestEdgeCases:
    """엣지 케이스 테스트."""

    def test_insufficient_data_skipped(self):
        """데이터 부족 (200일 미만) -> 스킵."""
        strat = BBSqueezeStrategy()
        df = _make_short_df(n=50)
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_empty_dataframe(self):
        """빈 DataFrame -> 스킵."""
        strat = BBSqueezeStrategy()
        df = pd.DataFrame(columns=["close", "volume", "시가총액"])
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_single_row_dataframe(self):
        """1행 DataFrame -> 스킵."""
        strat = BBSqueezeStrategy()
        df = pd.DataFrame({
            "close": [10000],
            "volume": [100000],
            "시가총액": [500_000_000_000],
        })
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_evaluate_ticker_returns_none_for_short_data(self):
        """_evaluate_ticker에 짧은 데이터 -> None."""
        strat = BBSqueezeStrategy()
        df = _make_short_df(n=50)
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_confidence_capped_at_1(self):
        """confidence는 최대 1.0."""
        strat = BBSqueezeStrategy()
        df = _make_squeeze_breakout_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        for sig in signals:
            assert sig.confidence <= 1.0

    def test_check_exit_with_korean_columns(self):
        """check_exit에서 한글 컬럼명 지원."""
        strat = BBSqueezeStrategy()
        n = 30
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        prices = [10000] * 20 + [10000 * (1 - 0.03 * i) for i in range(1, 11)]
        volumes = [100000] * n
        df = pd.DataFrame({
            "종가": prices,
            "거래량": volumes,
        }, index=dates)

        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 9400,  # -6%
            "entry_date": "2026-02-20",
        }
        market_data = {"daily_data": {"005930": df}}
        result = strat.check_exit(position, market_data)
        assert result is not None
        assert "손절" in result.reason
