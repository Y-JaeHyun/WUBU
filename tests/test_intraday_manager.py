"""장중 분봉 데이터 관리 모듈(src/data/intraday_manager.py) 테스트.

IntradayDataManager의 틱 집계, 분봉 조회, 기술적 지표 계산,
데이터 헬스 체크, 일일 초기화를 검증한다.
"""

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from src.data.intraday_manager import IntradayBar, IntradayDataManager

KST = timezone(timedelta(hours=9))


# ===================================================================
# 헬퍼 함수
# ===================================================================


def make_tick(
    ticker: str = "005930",
    price: int = 70000,
    volume: int = 100,
    timestamp: str = "2026-02-22T09:01:23",
) -> dict:
    """테스트용 틱 데이터를 생성한다."""
    return {
        "ticker": ticker,
        "price": price,
        "volume": volume,
        "timestamp": timestamp,
    }


def feed_n_ticks(
    manager: IntradayDataManager,
    n: int,
    ticker: str = "005930",
    start_minute: int = 0,
    base_price: int = 70000,
    interval_minutes: int = 1,
) -> None:
    """n개의 분봉을 확정시키기 위한 틱을 주입한다.

    각 분봉마다 2개의 틱(시작 + 마감)을 보내고,
    다음 분봉의 첫 틱으로 이전 분봉을 확정시킨다.
    마지막에 추가 틱 하나를 보내 마지막 분봉도 확정한다.
    """
    for i in range(n):
        minute = start_minute + i * interval_minutes
        hour = 9 + minute // 60
        m = minute % 60
        ts = f"2026-02-22T{hour:02d}:{m:02d}:00"
        price = base_price + i * 100

        # 해당 분의 첫 틱
        manager.on_tick(make_tick(ticker, price, 50, ts))

        # 해당 분의 두번째 틱 (high/low/close 변동)
        ts2 = f"2026-02-22T{hour:02d}:{m:02d}:30"
        manager.on_tick(make_tick(ticker, price + 50, 30, ts2))

    # 다음 분의 첫 틱으로 마지막 바를 확정
    final_minute = start_minute + n * interval_minutes
    final_hour = 9 + final_minute // 60
    final_m = final_minute % 60
    ts_final = f"2026-02-22T{final_hour:02d}:{final_m:02d}:00"
    manager.on_tick(make_tick(ticker, base_price, 10, ts_final))


# ===================================================================
# IntradayBar 데이터 클래스
# ===================================================================


class TestIntradayBar:
    """IntradayBar 데이터 클래스 기본 테스트."""

    def test_fields(self):
        """모든 필드가 정상적으로 설정된다."""
        ts = datetime(2026, 2, 22, 9, 0, 0)
        bar = IntradayBar(
            timestamp=ts, open=70000, high=71000,
            low=69000, close=70500, volume=1000,
        )
        assert bar.timestamp == ts
        assert bar.open == 70000
        assert bar.high == 71000
        assert bar.low == 69000
        assert bar.close == 70500
        assert bar.volume == 1000


# ===================================================================
# on_tick 테스트
# ===================================================================


class TestOnTick:
    """on_tick() 틱 집계 테스트."""

    def test_on_tick_creates_first_bar(self):
        """첫 틱 수신 시 current_bar가 생성된다."""
        mgr = IntradayDataManager(intervals=[1])
        mgr.on_tick(make_tick(timestamp="2026-02-22T09:00:10"))

        # 아직 확정된 바는 없지만 current_bar가 존재
        assert "005930" in mgr._current_bar
        assert 1 in mgr._current_bar["005930"]
        current = mgr._current_bar["005930"][1]
        assert current["open"] == 70000
        assert current["close"] == 70000

    def test_on_tick_updates_current_bar(self):
        """같은 분 내의 추가 틱이 high/low/close/volume을 갱신한다."""
        mgr = IntradayDataManager(intervals=[1])

        # 첫 틱: open = 70000
        mgr.on_tick(make_tick(price=70000, volume=100,
                              timestamp="2026-02-22T09:01:00"))

        # 두번째 틱: 가격 상승 -> high 갱신
        mgr.on_tick(make_tick(price=71000, volume=200,
                              timestamp="2026-02-22T09:01:15"))

        # 세번째 틱: 가격 하락 -> low 갱신
        mgr.on_tick(make_tick(price=69000, volume=50,
                              timestamp="2026-02-22T09:01:45"))

        current = mgr._current_bar["005930"][1]
        assert current["open"] == 70000
        assert current["high"] == 71000
        assert current["low"] == 69000
        assert current["close"] == 69000
        assert current["volume"] == 350  # 100 + 200 + 50

    def test_on_tick_new_minute_finalizes_bar(self):
        """분 경계를 넘어가면 이전 바가 확정된다."""
        mgr = IntradayDataManager(intervals=[1])

        # 09:01 분봉
        mgr.on_tick(make_tick(price=70000, volume=100,
                              timestamp="2026-02-22T09:01:00"))
        mgr.on_tick(make_tick(price=70500, volume=200,
                              timestamp="2026-02-22T09:01:30"))

        # 09:02 분봉 시작 -> 09:01 바 확정
        mgr.on_tick(make_tick(price=71000, volume=50,
                              timestamp="2026-02-22T09:02:00"))

        bars = mgr.get_bars("005930", interval=1)
        assert len(bars) == 1
        bar = bars[0]
        assert bar.open == 70000
        assert bar.high == 70500
        assert bar.low == 70000
        assert bar.close == 70500
        assert bar.volume == 300  # 100 + 200
        assert bar.timestamp.minute == 1

    def test_on_tick_multi_interval(self):
        """1분봉과 5분봉이 동시에 집계된다."""
        mgr = IntradayDataManager(intervals=[1, 5])

        # 09:00 ~ 09:04 동안 5개 분의 틱
        for m in range(5):
            ts = f"2026-02-22T09:{m:02d}:15"
            mgr.on_tick(make_tick(price=70000 + m * 100, volume=100, timestamp=ts))

        # 09:05 틱으로 09:04 1분봉 확정 + 09:00 5분봉 확정
        mgr.on_tick(make_tick(price=70500, volume=50,
                              timestamp="2026-02-22T09:05:00"))

        bars_1m = mgr.get_bars("005930", interval=1)
        bars_5m = mgr.get_bars("005930", interval=5)

        # 1분봉: 09:00, 09:01, 09:02, 09:03, 09:04 = 5개
        assert len(bars_1m) == 5

        # 5분봉: 09:00~09:04 = 1개
        assert len(bars_5m) == 1
        bar_5m = bars_5m[0]
        # 5분봉의 open = 첫 틱(09:00) 가격
        assert bar_5m.open == 70000
        # 5분봉의 close = 마지막 틱(09:04) 가격
        assert bar_5m.close == 70400
        assert bar_5m.high == 70400
        assert bar_5m.low == 70000

    def test_on_tick_ignores_empty_ticker(self):
        """ticker가 비어있으면 무시한다."""
        mgr = IntradayDataManager(intervals=[1])
        mgr.on_tick({"ticker": "", "price": 100, "volume": 10,
                      "timestamp": "2026-02-22T09:00:00"})
        assert len(mgr._bars) == 0

    def test_on_tick_ignores_zero_price(self):
        """price가 0이면 무시한다."""
        mgr = IntradayDataManager(intervals=[1])
        mgr.on_tick(make_tick(price=0))
        assert len(mgr._bars) == 0


# ===================================================================
# get_bars / get_bars_df 테스트
# ===================================================================


class TestGetBars:
    """get_bars() / get_bars_df() 조회 테스트."""

    def test_get_bars_returns_correct_count(self):
        """n 파라미터로 반환 개수를 제한한다."""
        mgr = IntradayDataManager(intervals=[1])
        feed_n_ticks(mgr, 10)

        bars = mgr.get_bars("005930", interval=1, n=5)
        assert len(bars) == 5

    def test_get_bars_returns_all_when_n_exceeds(self):
        """n이 전체 개수보다 크면 전체를 반환한다."""
        mgr = IntradayDataManager(intervals=[1])
        feed_n_ticks(mgr, 3)

        bars = mgr.get_bars("005930", interval=1, n=100)
        assert len(bars) == 3

    def test_get_bars_empty_ticker(self):
        """존재하지 않는 종목은 빈 리스트를 반환한다."""
        mgr = IntradayDataManager(intervals=[1])
        bars = mgr.get_bars("999999", interval=1)
        assert bars == []

    def test_get_bars_df_columns(self):
        """DataFrame의 컬럼이 올바른지 확인한다."""
        mgr = IntradayDataManager(intervals=[1])
        feed_n_ticks(mgr, 3)

        df = mgr.get_bars_df("005930", interval=1)
        expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_cols
        assert len(df) == 3

    def test_get_bars_df_empty(self):
        """데이터가 없으면 빈 DataFrame을 반환한다."""
        mgr = IntradayDataManager(intervals=[1])
        df = mgr.get_bars_df("005930", interval=1)
        assert len(df) == 0
        assert "close" in df.columns


# ===================================================================
# 기술적 지표 테스트
# ===================================================================


class TestTechnicalIndicators:
    """기술적 지표 계산 테스트."""

    def test_get_rsi_with_enough_data(self):
        """15개 이상의 바로 RSI를 계산한다."""
        mgr = IntradayDataManager(intervals=[1])
        feed_n_ticks(mgr, 20, interval_minutes=1)

        rsi = mgr.get_rsi("005930", period=14, interval=1)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_get_rsi_insufficient_data_returns_none(self):
        """바 수가 부족하면 None을 반환한다."""
        mgr = IntradayDataManager(intervals=[1])
        feed_n_ticks(mgr, 5)

        rsi = mgr.get_rsi("005930", period=14, interval=1)
        assert rsi is None

    def test_get_rsi_nonexistent_ticker(self):
        """존재하지 않는 종목은 None을 반환한다."""
        mgr = IntradayDataManager(intervals=[1])
        assert mgr.get_rsi("999999", interval=1) is None

    def test_get_macd(self):
        """MACD 계산 결과가 올바른 구조를 갖는다."""
        mgr = IntradayDataManager(intervals=[1])
        feed_n_ticks(mgr, 50, interval_minutes=1)

        result = mgr.get_macd("005930", interval=1)
        assert result is not None
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result
        assert isinstance(result["macd"], float)
        assert isinstance(result["signal"], float)
        assert isinstance(result["histogram"], float)

    def test_get_macd_insufficient_data(self):
        """바 수가 부족하면 None을 반환한다."""
        mgr = IntradayDataManager(intervals=[1])
        feed_n_ticks(mgr, 10)

        assert mgr.get_macd("005930", interval=1) is None

    def test_get_bollinger_bands(self):
        """볼린저 밴드 계산 결과가 올바른 구조를 갖는다."""
        mgr = IntradayDataManager(intervals=[1])
        feed_n_ticks(mgr, 30, interval_minutes=1)

        result = mgr.get_bollinger_bands("005930", period=20, interval=1)
        assert result is not None
        assert "upper" in result
        assert "middle" in result
        assert "lower" in result
        # upper > middle > lower
        assert result["upper"] >= result["middle"]
        assert result["middle"] >= result["lower"]

    def test_get_bollinger_bands_insufficient_data(self):
        """바 수가 부족하면 None을 반환한다."""
        mgr = IntradayDataManager(intervals=[1])
        feed_n_ticks(mgr, 5)

        assert mgr.get_bollinger_bands("005930", period=20, interval=1) is None

    def test_get_vwap_cumulative(self):
        """VWAP이 누적 가중 평균으로 계산된다."""
        mgr = IntradayDataManager(intervals=[1])

        # price=70000, volume=100 -> sum_pv=7_000_000, sum_vol=100
        mgr.on_tick(make_tick(price=70000, volume=100,
                              timestamp="2026-02-22T09:00:01"))

        # price=72000, volume=200 -> sum_pv=7_000_000+14_400_000=21_400_000
        #                            sum_vol=300
        mgr.on_tick(make_tick(price=72000, volume=200,
                              timestamp="2026-02-22T09:00:30"))

        vwap = mgr.get_vwap("005930")
        assert vwap is not None
        # (70000*100 + 72000*200) / (100 + 200) = 21_400_000 / 300 = 71333.33...
        expected = round((70000 * 100 + 72000 * 200) / 300, 2)
        assert abs(vwap - expected) < 0.01

    def test_get_vwap_no_data(self):
        """데이터가 없으면 None을 반환한다."""
        mgr = IntradayDataManager(intervals=[1])
        assert mgr.get_vwap("005930") is None


# ===================================================================
# 데이터 헬스 테스트
# ===================================================================


class TestDataHealth:
    """데이터 헬스 체크 테스트."""

    def test_is_data_stale_fresh(self):
        """최근 틱이 있으면 stale이 아니다."""
        mgr = IntradayDataManager(intervals=[1])
        now_str = datetime.now(KST).isoformat()
        mgr.on_tick(make_tick(timestamp=now_str))

        assert mgr.is_data_stale("005930", max_delay_seconds=60) is False

    def test_is_data_stale_expired(self):
        """오래된 틱이면 stale이다."""
        mgr = IntradayDataManager(intervals=[1])
        old_time = datetime.now(KST) - timedelta(minutes=5)
        mgr.on_tick(make_tick(timestamp=old_time.isoformat()))

        assert mgr.is_data_stale("005930", max_delay_seconds=60) is True

    def test_is_data_stale_no_data(self):
        """데이터가 없으면 stale이다."""
        mgr = IntradayDataManager(intervals=[1])
        assert mgr.is_data_stale("005930") is True

    def test_get_last_tick_time(self):
        """마지막 틱 시각을 반환한다."""
        mgr = IntradayDataManager(intervals=[1])
        ts_str = "2026-02-22T09:05:23"
        mgr.on_tick(make_tick(timestamp=ts_str))

        last = mgr.get_last_tick_time("005930")
        assert last is not None
        assert last.hour == 9
        assert last.minute == 5

    def test_get_last_tick_time_none(self):
        """데이터가 없으면 None을 반환한다."""
        mgr = IntradayDataManager(intervals=[1])
        assert mgr.get_last_tick_time("005930") is None


# ===================================================================
# reset_day 테스트
# ===================================================================


class TestResetDay:
    """reset_day() 일일 초기화 테스트."""

    def test_reset_day_clears_data(self):
        """reset_day() 호출 후 모든 데이터가 초기화된다."""
        mgr = IntradayDataManager(intervals=[1])
        feed_n_ticks(mgr, 5)

        # 데이터 존재 확인
        assert len(mgr.get_bars("005930", interval=1)) > 0
        assert mgr.get_last_tick_time("005930") is not None
        assert mgr.get_vwap("005930") is not None

        mgr.reset_day()

        assert mgr.get_bars("005930", interval=1) == []
        assert mgr.get_last_tick_time("005930") is None
        assert mgr.get_vwap("005930") is None
        assert len(mgr._current_bar) == 0


# ===================================================================
# MAX_BARS_PER_TICKER 테스트
# ===================================================================


class TestMaxBarsLimit:
    """MAX_BARS_PER_TICKER 제한 테스트."""

    def test_max_bars_limit(self):
        """확정 바가 MAX_BARS_PER_TICKER를 초과하면 오래된 바가 제거된다."""
        mgr = IntradayDataManager(intervals=[1])
        original_max = IntradayDataManager.MAX_BARS_PER_TICKER

        # 테스트를 위해 제한을 작게 설정
        IntradayDataManager.MAX_BARS_PER_TICKER = 10
        try:
            feed_n_ticks(mgr, 15, interval_minutes=1)

            bars = mgr.get_bars("005930", interval=1, n=500)
            assert len(bars) <= 10
        finally:
            IntradayDataManager.MAX_BARS_PER_TICKER = original_max


# ===================================================================
# 초기화 / 엣지 케이스
# ===================================================================


class TestEdgeCases:
    """엣지 케이스 테스트."""

    def test_unsupported_interval_fallback(self):
        """지원하지 않는 interval만 주면 기본값 [1]을 사용한다."""
        mgr = IntradayDataManager(intervals=[3, 7, 99])
        assert mgr._intervals == [1]

    def test_default_intervals(self):
        """intervals를 지정하지 않으면 [1, 5]이 기본값이다."""
        mgr = IntradayDataManager()
        assert mgr._intervals == [1, 5]

    def test_invalid_timestamp_uses_now(self):
        """타임스탬프 파싱 실패 시 현재 시각을 사용한다."""
        mgr = IntradayDataManager(intervals=[1])
        mgr.on_tick({
            "ticker": "005930",
            "price": 70000,
            "volume": 100,
            "timestamp": "invalid-timestamp",
        })
        # 에러 없이 current_bar가 생성되었는지 확인
        assert "005930" in mgr._current_bar

    def test_multiple_tickers(self):
        """여러 종목을 동시에 관리할 수 있다."""
        mgr = IntradayDataManager(intervals=[1])

        # 종목 A: 09:00
        mgr.on_tick(make_tick("005930", 70000, 100, "2026-02-22T09:00:00"))
        # 종목 B: 09:00
        mgr.on_tick(make_tick("000660", 80000, 200, "2026-02-22T09:00:00"))

        # 종목 A: 09:01 -> 09:00 바 확정
        mgr.on_tick(make_tick("005930", 70100, 50, "2026-02-22T09:01:00"))
        # 종목 B: 09:01 -> 09:00 바 확정
        mgr.on_tick(make_tick("000660", 80200, 80, "2026-02-22T09:01:00"))

        bars_a = mgr.get_bars("005930", interval=1)
        bars_b = mgr.get_bars("000660", interval=1)

        assert len(bars_a) == 1
        assert len(bars_b) == 1
        assert bars_a[0].open == 70000
        assert bars_b[0].open == 80000

    def test_15min_interval(self):
        """15분봉이 올바르게 집계된다."""
        mgr = IntradayDataManager(intervals=[15])

        # 09:00 ~ 09:14 동안 틱 전송
        mgr.on_tick(make_tick(price=70000, volume=100,
                              timestamp="2026-02-22T09:00:00"))
        mgr.on_tick(make_tick(price=71000, volume=100,
                              timestamp="2026-02-22T09:07:00"))
        mgr.on_tick(make_tick(price=69000, volume=100,
                              timestamp="2026-02-22T09:14:59"))

        # 09:15 틱으로 바 확정
        mgr.on_tick(make_tick(price=70500, volume=50,
                              timestamp="2026-02-22T09:15:00"))

        bars = mgr.get_bars("005930", interval=15)
        assert len(bars) == 1
        assert bars[0].open == 70000
        assert bars[0].high == 71000
        assert bars[0].low == 69000
        assert bars[0].close == 69000
        assert bars[0].volume == 300
