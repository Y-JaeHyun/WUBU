"""HighBreakoutStrategy 테스트 — 52주 신고가 돌파 모멘텀 전략."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategy.short_term_base import ShortTermSignal, ShortTermStrategy
from src.strategy.high_breakout import HighBreakoutStrategy


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _make_breakout_df(
    n: int = 260,
    base_close: float = 10000,
    base_volume: float = 100000,
    breakout_pct: float = 0.03,
    volume_spike: float = 3.0,
    market_cap: float = 500_000_000_000,
    high_at: int = 130,
):
    """52주 신고가 돌파 조건을 만족하는 테스트 DataFrame 생성.

    Args:
        n: 전체 기간 (일)
        base_close: 기본 종가
        base_volume: 기본 거래량
        breakout_pct: 신고가 대비 돌파 비율
        volume_spike: 마지막 날 거래량 배수 (평균 대비)
        market_cap: 시가총액
        high_at: 과거 52주 최고가 발생 위치 (끝에서 n번째)

    Returns:
        pd.DataFrame with close, volume, 시가총액 columns
    """
    dates = pd.date_range("2025-01-01", periods=n, freq="B")

    # 기본 가격: base_close를 중심으로 작은 변동
    np.random.seed(42)
    prices = [base_close * (1 + 0.001 * np.random.randn()) for _ in range(n)]

    # high_at 위치에 과거 최고가 설정 (끝에서 high_at번째)
    prev_high = base_close * 1.05  # 기본가 대비 5% 높은 과거 최고가
    prices[-(high_at + 1)] = prev_high

    # 마지막 날: 과거 최고가를 breakout_pct만큼 돌파
    prices[-1] = prev_high * (1 + breakout_pct)

    # 거래량: 평소 base_volume, 마지막 날 spike
    volumes = [base_volume] * n
    volumes[-1] = base_volume * volume_spike

    df = pd.DataFrame({
        "close": prices,
        "volume": volumes,
        "시가총액": [market_cap] * n,
    }, index=dates)
    return df


def _make_no_breakout_df(n: int = 260, base_close: float = 10000, base_volume: float = 100000):
    """52주 신고가를 돌파하지 않는 DataFrame."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    np.random.seed(42)
    prices = [base_close * (1 + 0.001 * np.random.randn()) for _ in range(n)]

    # 과거에 최고가가 있고, 오늘은 그 아래
    prices[-130] = base_close * 1.10  # 과거 최고가: +10%
    prices[-1] = base_close * 1.05    # 오늘: +5% (최고가 미달)

    volumes = [base_volume] * n
    volumes[-1] = base_volume * 3

    df = pd.DataFrame({
        "close": prices,
        "volume": volumes,
        "시가총액": [500_000_000_000] * n,
    }, index=dates)
    return df


def _make_low_volume_df(n: int = 260, base_close: float = 10000, base_volume: float = 100000):
    """52주 신고가 돌파했지만 거래량이 부족한 DataFrame."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    np.random.seed(42)
    prices = [base_close * (1 + 0.001 * np.random.randn()) for _ in range(n)]
    prices[-130] = base_close * 1.05
    prices[-1] = base_close * 1.08  # 돌파

    volumes = [base_volume] * n
    volumes[-1] = base_volume * 1.2  # 평균 대비 1.2배 (2배 미만)

    df = pd.DataFrame({
        "close": prices,
        "volume": volumes,
        "시가총액": [500_000_000_000] * n,
    }, index=dates)
    return df


def _make_korean_col_df(n: int = 260, base_close: float = 10000, base_volume: float = 100000):
    """한글 컬럼명 DataFrame (pykrx 호환)."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    np.random.seed(42)
    prices = [base_close * (1 + 0.001 * np.random.randn()) for _ in range(n)]
    prices[-130] = base_close * 1.05
    prices[-1] = base_close * 1.08

    volumes = [base_volume] * n
    volumes[-1] = base_volume * 3

    df = pd.DataFrame({
        "종가": prices,
        "거래량": volumes,
        "시가총액": [500_000_000_000] * n,
    }, index=dates)
    return df


# ──────────────────────────────────────────────────────────
# 1. 기본 초기화
# ──────────────────────────────────────────────────────────

class TestInitialization:
    """초기화 관련 테스트."""

    def test_name(self):
        """전략 이름이 high_breakout."""
        strat = HighBreakoutStrategy()
        assert strat.name == "high_breakout"

    def test_mode(self):
        """모드가 swing."""
        strat = HighBreakoutStrategy()
        assert strat.mode == "swing"

    def test_default_params(self):
        """기본 파라미터로 초기화."""
        strat = HighBreakoutStrategy()
        p = strat.params
        assert p["lookback_days"] == 252
        assert p["volume_multiplier"] == 2.0
        assert p["volume_avg_days"] == 20
        assert p["take_profit_pct"] == 0.15
        assert p["stop_loss_pct"] == -0.05
        assert p["trailing_stop_pct"] == 0.10
        assert p["max_holding_days"] == 10
        assert p["max_signals"] == 3
        assert p["min_market_cap"] == 300_000_000_000

    def test_custom_params(self):
        """커스텀 파라미터 오버라이드."""
        strat = HighBreakoutStrategy(params={
            "lookback_days": 200,
            "volume_multiplier": 3.0,
            "max_signals": 5,
        })
        p = strat.params
        assert p["lookback_days"] == 200
        assert p["volume_multiplier"] == 3.0
        assert p["max_signals"] == 5
        # 나머지는 기본값 유지
        assert p["take_profit_pct"] == 0.15
        assert p["stop_loss_pct"] == -0.05

    def test_params_returns_copy(self):
        """params 프로퍼티가 복사본을 반환하는지 확인."""
        strat = HighBreakoutStrategy()
        p1 = strat.params
        p1["lookback_days"] = 999
        assert strat.params["lookback_days"] == 252  # 원본 변경 없음

    def test_inherits_short_term_strategy(self):
        """ShortTermStrategy를 상속하는지 확인."""
        strat = HighBreakoutStrategy()
        assert isinstance(strat, ShortTermStrategy)


# ──────────────────────────────────────────────────────────
# 2. scan_signals - 52주 신고가 돌파
# ──────────────────────────────────────────────────────────

class TestScanSignals:
    """scan_signals 테스트."""

    def test_valid_breakout_generates_signal(self):
        """52주 신고가 돌파 + 거래량 확인 -> 시그널 생성."""
        strat = HighBreakoutStrategy()
        df = _make_breakout_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) >= 1
        sig = signals[0]
        assert sig.ticker == "005930"
        assert sig.strategy == "high_breakout"
        assert sig.side == "buy"
        assert sig.mode == "swing"
        assert 0 < sig.confidence <= 1.0

    def test_no_breakout_no_signal(self):
        """종가가 52주 최고가 미달 -> 시그널 없음."""
        strat = HighBreakoutStrategy()
        df = _make_no_breakout_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_volume_too_low_no_signal(self):
        """거래량 부족 -> 시그널 없음."""
        strat = HighBreakoutStrategy()
        df = _make_low_volume_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_market_cap_too_low_no_signal(self):
        """시가총액 미달 -> 시그널 없음."""
        strat = HighBreakoutStrategy()
        df = _make_breakout_df(market_cap=100_000_000_000)  # 1000억 (3000억 미만)
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_multiple_tickers_max_signals_limit(self):
        """여러 종목이 조건 충족해도 max_signals 제한."""
        strat = HighBreakoutStrategy(params={"max_signals": 2})
        daily_data = {}
        for i in range(5):
            df = _make_breakout_df(breakout_pct=0.03 + 0.01 * i, volume_spike=3.0 + i)
            daily_data[f"00{i}930"] = df
        market_data = {"daily_data": daily_data}
        signals = strat.scan_signals(market_data)
        assert len(signals) <= 2

    def test_signal_attributes(self):
        """시그널 속성이 올바른지 확인."""
        strat = HighBreakoutStrategy()
        df = _make_breakout_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) >= 1
        sig = signals[0]

        # strategy, mode
        assert sig.strategy == "high_breakout"
        assert sig.mode == "swing"

        # metadata keys
        assert "breakout_strength" in sig.metadata
        assert "volume_ratio" in sig.metadata
        assert "prev_52w_high" in sig.metadata
        assert "market_cap" in sig.metadata

        # price levels
        assert sig.stop_loss_price < sig.target_price
        assert sig.take_profit_price > sig.target_price

        # signal type
        assert isinstance(sig, ShortTermSignal)

    def test_empty_market_data(self):
        """빈 market_data -> 빈 리스트."""
        strat = HighBreakoutStrategy()
        result = strat.scan_signals({})
        assert result == []

    def test_empty_daily_data(self):
        """daily_data가 비어있으면 빈 리스트."""
        strat = HighBreakoutStrategy()
        result = strat.scan_signals({"daily_data": {}})
        assert result == []

    def test_none_dataframe_skipped(self):
        """DataFrame이 None이면 스킵."""
        strat = HighBreakoutStrategy()
        market_data = {"daily_data": {"005930": None}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_signals_sorted_by_score(self):
        """시그널은 점수 내림차순 정렬."""
        strat = HighBreakoutStrategy(params={"max_signals": 10})
        daily_data = {}
        for i in range(5):
            # 각 종목의 돌파 강도를 다르게 설정
            df = _make_breakout_df(breakout_pct=0.01 + 0.02 * i, volume_spike=2.5 + i * 0.5)
            daily_data[f"00{i}930"] = df
        market_data = {"daily_data": daily_data}
        signals = strat.scan_signals(market_data)

        if len(signals) >= 2:
            for i in range(len(signals) - 1):
                assert signals[i].confidence >= signals[i + 1].confidence

    def test_korean_columns_work(self):
        """한글 컬럼명(종가, 거래량)으로도 동작."""
        strat = HighBreakoutStrategy()
        df = _make_korean_col_df()
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        # 한글 컬럼 돌파 시 시그널 생성 가능
        for sig in signals:
            assert sig.strategy == "high_breakout"


# ──────────────────────────────────────────────────────────
# 3. _evaluate_ticker
# ──────────────────────────────────────────────────────────

class TestEvaluateTicker:
    """_evaluate_ticker 테스트."""

    def test_qualifying_returns_dict(self):
        """조건 충족 시 딕셔너리 반환."""
        strat = HighBreakoutStrategy()
        df = _make_breakout_df()
        result = strat._evaluate_ticker("005930", df)
        assert result is not None
        assert result["ticker"] == "005930"
        assert result["score"] > 0
        assert "breakout_strength" in result
        assert "volume_ratio" in result
        assert "prev_52w_high" in result
        assert "days_since_high" in result
        assert "reason" in result

    def test_no_breakout_returns_none(self):
        """종가 <= 52주 최고가 -> None."""
        strat = HighBreakoutStrategy()
        df = _make_no_breakout_df()
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_low_volume_returns_none(self):
        """거래량 부족 -> None."""
        strat = HighBreakoutStrategy()
        df = _make_low_volume_df()
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_low_market_cap_returns_none(self):
        """시가총액 미달 -> None."""
        strat = HighBreakoutStrategy()
        df = _make_breakout_df(market_cap=100_000_000_000)
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_zero_close_returns_none(self):
        """종가 0이면 None."""
        strat = HighBreakoutStrategy()
        df = _make_breakout_df()
        df.iloc[-1, df.columns.get_loc("close")] = 0
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_zero_volume_returns_none(self):
        """거래량 0이면 None."""
        strat = HighBreakoutStrategy()
        df = _make_breakout_df()
        df.iloc[-1, df.columns.get_loc("volume")] = 0
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_score_range(self):
        """점수는 0~100 범위."""
        strat = HighBreakoutStrategy()
        df = _make_breakout_df()
        result = strat._evaluate_ticker("005930", df)
        assert result is not None
        assert 0 <= result["score"] <= 100

    def test_higher_breakout_higher_score(self):
        """돌파 강도가 높은 종목이 더 높은 점수."""
        strat = HighBreakoutStrategy()
        df1 = _make_breakout_df(breakout_pct=0.01)
        df2 = _make_breakout_df(breakout_pct=0.08)

        r1 = strat._evaluate_ticker("A", df1)
        r2 = strat._evaluate_ticker("B", df2)

        assert r1 is not None and r2 is not None
        assert r2["score"] > r1["score"]

    def test_higher_volume_higher_score(self):
        """거래량이 더 높은 종목이 더 높은 점수 (다른 조건 동일)."""
        strat = HighBreakoutStrategy()
        df1 = _make_breakout_df(volume_spike=2.5)
        df2 = _make_breakout_df(volume_spike=5.0)

        r1 = strat._evaluate_ticker("A", df1)
        r2 = strat._evaluate_ticker("B", df2)

        assert r1 is not None and r2 is not None
        assert r2["score"] > r1["score"]

    def test_missing_close_column_returns_none(self):
        """close/종가 컬럼이 없으면 None."""
        strat = HighBreakoutStrategy()
        dates = pd.date_range("2025-01-01", periods=260, freq="B")
        df = pd.DataFrame({
            "open": [10000] * 260,
            "volume": [100000] * 260,
        }, index=dates)
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_missing_volume_column_returns_none(self):
        """volume/거래량 컬럼이 없으면 None."""
        strat = HighBreakoutStrategy()
        dates = pd.date_range("2025-01-01", periods=260, freq="B")
        df = pd.DataFrame({
            "close": [10000] * 260,
        }, index=dates)
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_market_cap_inf_uses_default_score(self):
        """시가총액 정보가 없으면(inf) 중간 점수 적용."""
        strat = HighBreakoutStrategy()
        df = _make_breakout_df()
        # 시가총액 컬럼 제거
        df = df.drop(columns=["시가총액"])
        result = strat._evaluate_ticker("005930", df)
        # 시가총액 inf -> min_market_cap 미만이 아니므로 통과
        assert result is not None

    def test_confidence_capped_at_1(self):
        """confidence는 최대 1.0."""
        strat = HighBreakoutStrategy()
        df = _make_breakout_df(breakout_pct=0.20, volume_spike=6.0)
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        for sig in signals:
            assert sig.confidence <= 1.0


# ──────────────────────────────────────────────────────────
# 4. check_exit
# ──────────────────────────────────────────────────────────

class TestCheckExit:
    """check_exit 테스트."""

    def test_stop_loss_triggers_sell(self):
        """손절 -5% 이하 -> sell 시그널."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 9400,  # -6%
            "entry_date": "2026-02-20",
            "metadata": {"peak_price": 10000},
        }
        result = strat.check_exit(position, {"daily_data": {}, "date": "2026-02-21"})
        assert result is not None
        assert result.side == "sell"
        assert "손절" in result.reason

    def test_take_profit_triggers_sell(self):
        """익절 +15% 이상 -> sell 시그널."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 11600,  # +16%
            "entry_date": "2026-02-20",
            "metadata": {"peak_price": 11600},
        }
        result = strat.check_exit(position, {"daily_data": {}, "date": "2026-02-21"})
        assert result is not None
        assert result.side == "sell"
        assert "익절" in result.reason

    def test_trailing_stop_triggers_sell(self):
        """트레일링 스탑: 고점 대비 10% 하락 -> sell 시그널."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10500,  # 고점 12000 대비 12.5% 하락
            "entry_date": "2026-02-20",
            "metadata": {"peak_price": 12000},
        }
        result = strat.check_exit(position, {"daily_data": {}, "date": "2026-02-21"})
        assert result is not None
        assert result.side == "sell"
        assert "트레일링" in result.reason

    def test_time_stop_triggers_sell(self):
        """보유일 초과(10일) -> sell 시그널."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10050,  # 소폭 이익
            "entry_date": "2026-02-01",
            "metadata": {"peak_price": 10050},
        }
        result = strat.check_exit(position, {"daily_data": {}, "date": "2026-02-15"})
        assert result is not None
        assert result.side == "sell"
        assert "보유일 초과" in result.reason

    def test_no_exit_condition_returns_none(self):
        """청산 조건 없으면 None."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10200,  # +2% (손절/익절 아님)
            "entry_date": "2026-02-20",
            "metadata": {"peak_price": 10200},  # 트레일링 아님
        }
        result = strat.check_exit(position, {"daily_data": {}, "date": "2026-02-21"})
        assert result is None

    def test_invalid_entry_price_returns_none(self):
        """entry_price <= 0이면 None."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 0,
            "current_price": 10000,
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is None

    def test_invalid_current_price_returns_none(self):
        """current_price <= 0이면 None."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 0,
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is None

    def test_exit_signal_type(self):
        """청산 시그널이 ShortTermSignal 타입."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 9400,
            "entry_date": "2026-02-20",
            "metadata": {"peak_price": 10000},
        }
        result = strat.check_exit(position, {"daily_data": {}, "date": "2026-02-21"})
        assert isinstance(result, ShortTermSignal)
        assert result.strategy == "high_breakout"
        assert result.mode == "swing"

    def test_exit_multiple_reasons(self):
        """여러 청산 조건이 동시에 충족."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 9400,  # -6% (손절)
            "entry_date": "2026-01-01",  # 보유일 초과
            "metadata": {"peak_price": 12000},  # 트레일링 (고점 대비 21.7% 하락)
        }
        result = strat.check_exit(position, {"daily_data": {}, "date": "2026-02-20"})
        assert result is not None
        assert "손절" in result.reason
        assert "보유일 초과" in result.reason
        assert "트레일링" in result.reason

    def test_exit_date_format_yyyymmdd(self):
        """YYYYMMDD 날짜 형식도 지원."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10050,
            "entry_date": "20260201",  # YYYYMMDD
            "metadata": {"peak_price": 10050},
        }
        result = strat.check_exit(position, {"daily_data": {}, "date": "20260215"})
        assert result is not None
        assert "보유일 초과" in result.reason

    def test_exit_invalid_date_format_ignored(self):
        """날짜 형식이 잘못되면 보유일 체크 스킵."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10050,
            "entry_date": "invalid-date",
            "metadata": {"peak_price": 10050},
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is None  # 다른 조건 미충족

    def test_trailing_stop_no_metadata(self):
        """metadata에 peak_price가 없으면 entry_price를 사용."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 8900,  # entry_price 대비 -11% (트레일링 + 손절)
            "entry_date": "2026-02-20",
            "metadata": {},  # peak_price 없음
        }
        result = strat.check_exit(position, {"daily_data": {}, "date": "2026-02-21"})
        assert result is not None
        # 손절 (-11%) 또는 트레일링 (entry_price=peak 대비 11% 하락)
        assert "손절" in result.reason or "트레일링" in result.reason

    def test_trailing_stop_not_triggered_when_near_peak(self):
        """현재가가 고점 근처면 트레일링 미발동."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10800,  # 고점 11000 대비 -1.8% (10% 미만)
            "entry_date": "2026-02-20",
            "metadata": {"peak_price": 11000},
        }
        result = strat.check_exit(position, {"daily_data": {}, "date": "2026-02-21"})
        assert result is None


# ──────────────────────────────────────────────────────────
# 5. Edge Cases
# ──────────────────────────────────────────────────────────

class TestEdgeCases:
    """엣지 케이스 테스트."""

    def test_insufficient_data_skipped(self):
        """데이터 252일 미만이면 스킵."""
        strat = HighBreakoutStrategy()
        dates = pd.date_range("2025-01-01", periods=100, freq="B")
        df = pd.DataFrame({
            "close": [10000] * 100,
            "volume": [100000] * 100,
            "시가총액": [500_000_000_000] * 100,
        }, index=dates)
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0

    def test_exactly_253_days(self):
        """정확히 253일(lookback+1) 데이터에서 동작."""
        strat = HighBreakoutStrategy()
        n = 253
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        np.random.seed(42)
        prices = [10000 * (1 + 0.001 * np.random.randn()) for _ in range(n)]
        # 중간에 최고가 설정, 마지막 날 돌파
        prices[50] = 10500
        prices[-1] = 10800

        volumes = [100000] * n
        volumes[-1] = 300000  # 거래량 3배

        df = pd.DataFrame({
            "close": prices,
            "volume": volumes,
            "시가총액": [500_000_000_000] * n,
        }, index=dates)
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        # 조건 충족하면 시그널 생성 (동작 자체가 에러 없이 수행)
        assert isinstance(signals, list)

    def test_all_same_prices_no_breakout(self):
        """모든 가격이 동일하면 돌파 없음 (close == prev_high)."""
        strat = HighBreakoutStrategy()
        n = 260
        dates = pd.date_range("2025-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "close": [10000] * n,
            "volume": [100000] * (n - 1) + [300000],
            "시가총액": [500_000_000_000] * n,
        }, index=dates)
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        assert len(signals) == 0  # close == prev_52w_high, not >

    def test_zero_price_history(self):
        """과거 최고가가 0이면 필터링."""
        strat = HighBreakoutStrategy()
        n = 260
        dates = pd.date_range("2025-01-01", periods=n, freq="B")
        prices = [0] * (n - 1) + [100]
        df = pd.DataFrame({
            "close": prices,
            "volume": [100000] * n,
            "시가총액": [500_000_000_000] * n,
        }, index=dates)
        # close(100) > prev_52w_high(0) but we check prev_52w_high > 0
        result = strat._evaluate_ticker("005930", df)
        assert result is None

    def test_evaluate_ticker_exception_handling(self):
        """_evaluate_ticker에서 예외 발생 시 scan_signals가 건너뜀."""
        strat = HighBreakoutStrategy()
        # 올바른 종목과 문제 있는 종목 혼합
        good_df = _make_breakout_df()
        # 빈 DataFrame (예외 유발 가능)
        bad_df = pd.DataFrame()

        market_data = {"daily_data": {
            "005930": good_df,
            "000000": bad_df,
        }}
        # 에러 없이 실행되어야 함
        signals = strat.scan_signals(market_data)
        assert isinstance(signals, list)

    def test_custom_lookback_shorter(self):
        """lookback_days를 짧게 설정해도 동작."""
        strat = HighBreakoutStrategy(params={"lookback_days": 20})
        n = 30
        dates = pd.date_range("2025-01-01", periods=n, freq="B")
        np.random.seed(42)
        prices = [10000] * n
        prices[5] = 10500  # 과거 최고가
        prices[-1] = 10800  # 돌파

        volumes = [100000] * n
        volumes[-1] = 300000

        df = pd.DataFrame({
            "close": prices,
            "volume": volumes,
            "시가총액": [500_000_000_000] * n,
        }, index=dates)
        market_data = {"daily_data": {"005930": df}}
        signals = strat.scan_signals(market_data)
        # 21일 이상 데이터가 있으므로 동작
        assert isinstance(signals, list)

    def test_market_cap_column_english(self):
        """market_cap (영문) 컬럼도 인식."""
        strat = HighBreakoutStrategy()
        df = _make_breakout_df()
        df = df.rename(columns={"시가총액": "market_cap"})
        result = strat._evaluate_ticker("005930", df)
        assert result is not None

    def test_check_exit_uses_market_data_date(self):
        """check_exit가 market_data의 date를 기준일로 사용."""
        strat = HighBreakoutStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10050,
            "entry_date": "2026-02-01",
            "metadata": {"peak_price": 10050},
        }
        # date가 entry_date로부터 5일 후 -> 보유일 5일 (10일 미만)
        result = strat.check_exit(position, {"daily_data": {}, "date": "2026-02-06"})
        assert result is None  # 5일 < 10일 -> 미발동

        # date가 entry_date로부터 11일 후 -> 보유일 11일 (10일 이상)
        result = strat.check_exit(position, {"daily_data": {}, "date": "2026-02-12"})
        assert result is not None
        assert "보유일 초과" in result.reason

    def test_check_exit_no_date_uses_now(self):
        """market_data에 date가 없으면 현재 시간 사용."""
        strat = HighBreakoutStrategy()
        old_date = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
        position = {
            "ticker": "005930",
            "entry_price": 10000,
            "current_price": 10050,
            "entry_date": old_date,
            "metadata": {"peak_price": 10050},
        }
        result = strat.check_exit(position, {"daily_data": {}})
        assert result is not None
        assert "보유일 초과" in result.reason
