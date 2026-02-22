"""ORB 데이트레이딩 전략 테스트."""

import pytest
from datetime import datetime, time, timedelta
from unittest.mock import patch
import pandas as pd
import numpy as np

from src.strategy.orb_daytrading import ORBDaytradingStrategy
from src.strategy.short_term_base import ShortTermSignal


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def _make_intraday_df(
    or_high=10150,
    or_low=10000,
    breakout_price=10200,
    breakout_volume=2000,
    avg_volume=1000,
    opening_bars=30,
    post_bars=30,
):
    """Create intraday DataFrame with opening range and breakout.

    Default range = (10150-10000)/10000 = 1.5% (within 0.5%~3%).

    Args:
        or_high: 시가범위 내 최고가.
        or_low: 시가범위 내 최저가.
        breakout_price: 시가범위 이후 종가(돌파가).
        breakout_volume: 돌파 봉 거래량.
        avg_volume: 시가범위 평균 거래량.
        opening_bars: 시가범위 봉 수.
        post_bars: 시가범위 이후 봉 수.
    """
    rows = []
    base = datetime(2026, 2, 22, 9, 0)
    # Opening range bars (09:00~09:29)
    for i in range(opening_bars):
        ts = base + timedelta(minutes=i)
        rows.append({
            "timestamp": ts.isoformat(),
            "open": 10050,
            "high": or_high,
            "low": or_low,
            "close": 10080,
            "volume": avg_volume,
        })
    # Post-opening range bars (09:30~)
    for i in range(post_bars):
        ts = base + timedelta(minutes=opening_bars + i)
        rows.append({
            "timestamp": ts.isoformat(),
            "open": breakout_price,
            "high": breakout_price + 50,
            "low": breakout_price - 20,
            "close": breakout_price,
            "volume": breakout_volume,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────
# 1. 기본 속성 테스트
# ──────────────────────────────────────────────────────────


class TestORBBasic:
    """기본 속성 테스트."""

    def test_name_and_mode(self):
        """name과 mode 확인."""
        s = ORBDaytradingStrategy()
        assert s.name == "orb_daytrading"
        assert s.mode == "daytrading"

    def test_default_params(self):
        """기본 파라미터가 올바르게 설정되는지 확인."""
        s = ORBDaytradingStrategy()
        p = s.params
        assert p["opening_range_minutes"] == 30
        assert p["volume_confirm_ratio"] == 1.5
        assert p["profit_target_ratio"] == 1.5
        assert p["stop_loss_buffer_pct"] == 0.002
        assert p["min_range_pct"] == 0.005
        assert p["max_range_pct"] == 0.03
        assert p["max_signals"] == 2
        assert p["no_entry_after"] == "14:30"
        assert p["force_close_time"] == "15:20"
        assert p["min_market_cap"] == 500_000_000_000

    def test_custom_params(self):
        """사용자 지정 파라미터 오버라이드."""
        s = ORBDaytradingStrategy(params={"max_signals": 5, "min_range_pct": 0.01})
        p = s.params
        assert p["max_signals"] == 5
        assert p["min_range_pct"] == 0.01
        # 나머지는 기본값 유지
        assert p["opening_range_minutes"] == 30

    def test_params_property_returns_copy(self):
        """params 프로퍼티가 복사본을 반환하는지 확인."""
        s = ORBDaytradingStrategy()
        p1 = s.params
        p1["max_signals"] = 99
        p2 = s.params
        assert p2["max_signals"] == 2  # 원본 불변


# ──────────────────────────────────────────────────────────
# 2. 시가범위 계산 테스트
# ──────────────────────────────────────────────────────────


class TestOpeningRange:
    """시가범위 계산 테스트."""

    def test_opening_range_with_timestamp_column(self):
        """timestamp 컬럼이 있는 DataFrame에서 시가범위 계산."""
        s = ORBDaytradingStrategy()
        df = _make_intraday_df(or_high=10150, or_low=10000, avg_volume=1000)
        or_high, or_low, avg_vol = s._get_opening_range(df, 30)
        assert or_high == 10150
        assert or_low == 10000
        assert avg_vol == 1000.0

    def test_opening_range_with_row_based_fallback(self):
        """timestamp 컬럼 없이 행 수 기반 fallback."""
        s = ORBDaytradingStrategy()
        df = _make_intraday_df(or_high=10200, or_low=9900, avg_volume=500)
        # timestamp 컬럼 제거
        df = df.drop(columns=["timestamp"])
        or_high, or_low, avg_vol = s._get_opening_range(df, 30)
        assert or_high == 10200
        assert or_low == 9900
        assert avg_vol == 500.0

    def test_opening_range_empty_df(self):
        """빈 DataFrame에서 None 반환."""
        s = ORBDaytradingStrategy()
        df = pd.DataFrame()
        or_high, or_low, avg_vol = s._get_opening_range(df, 30)
        assert or_high is None
        assert or_low is None
        assert avg_vol == 0.0

    def test_opening_range_missing_columns(self):
        """필수 컬럼(high, low)이 없으면 None 반환."""
        s = ORBDaytradingStrategy()
        # timestamp는 있으나 high/low 없음
        df = pd.DataFrame({
            "timestamp": [(datetime(2026, 2, 22, 9, 0) + timedelta(minutes=i)).isoformat()
                          for i in range(30)],
            "close": [10000] * 30,
        })
        or_high, or_low, avg_vol = s._get_opening_range(df, 30)
        assert or_high is None
        assert or_low is None


# ──────────────────────────────────────────────────────────
# 3. 시그널 스캔 테스트
# ──────────────────────────────────────────────────────────


class TestScanSignals:
    """시그널 스캔 테스트."""

    def test_scan_no_data(self):
        """intraday_data가 비어 있으면 빈 리스트 반환."""
        s = ORBDaytradingStrategy()
        result = s.scan_signals({"intraday_data": {}, "current_time": datetime(2026, 2, 22, 10, 0)})
        assert result == []

    def test_scan_before_opening_range_end(self):
        """시가범위 확정 전(09:20)에는 빈 리스트 반환."""
        s = ORBDaytradingStrategy()
        df = _make_intraday_df()
        result = s.scan_signals({
            "intraday_data": {"005930": df},
            "current_time": datetime(2026, 2, 22, 9, 20),
        })
        assert result == []

    def test_scan_after_no_entry_time(self):
        """신규 진입 마감 시각(14:30) 이후에는 빈 리스트 반환."""
        s = ORBDaytradingStrategy()
        df = _make_intraday_df()
        result = s.scan_signals({
            "intraday_data": {"005930": df},
            "current_time": datetime(2026, 2, 22, 14, 35),
        })
        assert result == []

    def test_scan_upward_breakout(self):
        """상향 돌파 시 buy 시그널 생성."""
        s = ORBDaytradingStrategy()
        # range = (10150-10000)/10000 = 1.5%, breakout_price=10200 > or_high=10150
        df = _make_intraday_df(or_high=10150, or_low=10000, breakout_price=10200,
                               breakout_volume=2000, avg_volume=1000)
        result = s.scan_signals({
            "intraday_data": {"005930": df},
            "current_time": datetime(2026, 2, 22, 10, 0),
        })
        assert len(result) == 1
        assert result[0].side == "buy"
        assert result[0].ticker == "005930"

    def test_scan_downward_breakout(self):
        """하향 돌파 시 sell 시그널 생성."""
        s = ORBDaytradingStrategy()
        # range = (10150-10000)/10000 = 1.5%, breakout=9950 < or_low=10000
        df = _make_intraday_df(or_high=10150, or_low=10000, breakout_price=9950,
                               breakout_volume=2000, avg_volume=1000)
        result = s.scan_signals({
            "intraday_data": {"005930": df},
            "current_time": datetime(2026, 2, 22, 10, 0),
        })
        assert len(result) == 1
        assert result[0].side == "sell"

    def test_scan_no_breakout(self):
        """시가범위 내 가격이면 시그널 없음."""
        s = ORBDaytradingStrategy()
        # breakout_price 10080 is within [10000, 10150]
        df = _make_intraday_df(or_high=10150, or_low=10000, breakout_price=10080,
                               breakout_volume=2000, avg_volume=1000)
        result = s.scan_signals({
            "intraday_data": {"005930": df},
            "current_time": datetime(2026, 2, 22, 10, 0),
        })
        assert result == []

    def test_scan_volume_too_low(self):
        """돌파했지만 거래량이 1.5배 미만이면 시그널 없음."""
        s = ORBDaytradingStrategy()
        # breakout_volume=1200 < avg_volume*1.5=1500
        df = _make_intraday_df(or_high=10150, or_low=10000, breakout_price=10200,
                               breakout_volume=1200, avg_volume=1000)
        result = s.scan_signals({
            "intraday_data": {"005930": df},
            "current_time": datetime(2026, 2, 22, 10, 0),
        })
        assert result == []

    def test_scan_range_too_narrow(self):
        """범위가 0.5% 미만이면 시그널 없음."""
        s = ORBDaytradingStrategy()
        # range = (10040-10000)/10000 = 0.4% < 0.5%
        df = _make_intraday_df(or_high=10040, or_low=10000, breakout_price=10050,
                               breakout_volume=2000, avg_volume=1000)
        result = s.scan_signals({
            "intraday_data": {"005930": df},
            "current_time": datetime(2026, 2, 22, 10, 0),
        })
        assert result == []

    def test_scan_range_too_wide(self):
        """범위가 3% 초과이면 시그널 없음."""
        s = ORBDaytradingStrategy()
        # range = (10400-10000)/10000 = 4% > 3%
        df = _make_intraday_df(or_high=10400, or_low=10000, breakout_price=10500,
                               breakout_volume=2000, avg_volume=1000)
        result = s.scan_signals({
            "intraday_data": {"005930": df},
            "current_time": datetime(2026, 2, 22, 10, 0),
        })
        assert result == []

    def test_scan_market_cap_filter(self):
        """시가총액이 5000억 미만이면 필터링."""
        s = ORBDaytradingStrategy()
        # Valid range: 1.5%, breakout above or_high
        df = _make_intraday_df(or_high=10150, or_low=10000, breakout_price=10200,
                               breakout_volume=2000, avg_volume=1000)
        daily_df = pd.DataFrame([{"market_cap": 100_000_000_000}])  # 1000억 < 5000억
        result = s.scan_signals({
            "intraday_data": {"005930": df},
            "daily_data": {"005930": daily_df},
            "current_time": datetime(2026, 2, 22, 10, 0),
        })
        assert result == []

    def test_scan_max_signals_limit(self):
        """max_signals=2 설정 시 상위 2개만 반환."""
        s = ORBDaytradingStrategy(params={"max_signals": 2})
        intraday = {}
        for i, ticker in enumerate(["A", "B", "C", "D", "E"]):
            # 모두 돌파, 거래량 확인, 범위 유효 (1.5%)
            bv = 2000 + i * 500  # 서로 다른 거래량(=다른 점수)
            df = _make_intraday_df(or_high=10150, or_low=10000, breakout_price=10200,
                                   breakout_volume=bv, avg_volume=1000)
            intraday[ticker] = df
        result = s.scan_signals({
            "intraday_data": intraday,
            "current_time": datetime(2026, 2, 22, 10, 0),
        })
        assert len(result) <= 2

    def test_scan_signal_attributes(self):
        """시그널의 strategy, mode, metadata 검증."""
        s = ORBDaytradingStrategy()
        df = _make_intraday_df(or_high=10150, or_low=10000, breakout_price=10200,
                               breakout_volume=2000, avg_volume=1000)
        result = s.scan_signals({
            "intraday_data": {"005930": df},
            "current_time": datetime(2026, 2, 22, 10, 0),
        })
        assert len(result) == 1
        sig = result[0]
        assert sig.strategy == "orb_daytrading"
        assert sig.mode == "daytrading"
        assert isinstance(sig.metadata, dict)
        assert "opening_range_high" in sig.metadata
        assert "opening_range_low" in sig.metadata
        assert "range_pct" in sig.metadata
        assert "volume_ratio" in sig.metadata
        assert sig.metadata["breakout_type"] == "buy"

    def test_scan_signal_expires_at_force_close(self):
        """시그널의 expires_at이 당일 15:20으로 설정되는지 확인."""
        s = ORBDaytradingStrategy()
        current = datetime(2026, 2, 22, 10, 0)
        df = _make_intraday_df(or_high=10150, or_low=10000, breakout_price=10200,
                               breakout_volume=2000, avg_volume=1000)
        result = s.scan_signals({
            "intraday_data": {"005930": df},
            "current_time": current,
        })
        assert len(result) == 1
        sig = result[0]
        expires = datetime.fromisoformat(sig.expires_at)
        assert expires.hour == 15
        assert expires.minute == 20
        assert expires.date() == current.date()


# ──────────────────────────────────────────────────────────
# 4. 청산 체크 테스트
# ──────────────────────────────────────────────────────────


class TestCheckExit:
    """청산 체크 테스트."""

    def test_exit_stop_loss_buy(self):
        """매수 포지션, 현재가가 손절가 이하이면 sell 시그널."""
        s = ORBDaytradingStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10200,
            "current_price": 9950,
            "side": "buy",
            "stop_loss_price": 9980,
            "metadata": {"opening_range_high": 10150, "opening_range_low": 10000},
        }
        result = s.check_exit(position, {"current_time": datetime(2026, 2, 22, 11, 0)})
        assert result is not None
        assert result.side == "sell"
        assert result.confidence == 1.0
        assert "손절" in result.reason

    def test_exit_stop_loss_sell(self):
        """매도 포지션, 현재가가 손절가 이상이면 buy 시그널."""
        s = ORBDaytradingStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 9950,
            "current_price": 10200,
            "side": "sell",
            "stop_loss_price": 10170,  # or_high * 1.002 = ~10170
            "metadata": {"opening_range_high": 10150, "opening_range_low": 10000},
        }
        result = s.check_exit(position, {"current_time": datetime(2026, 2, 22, 11, 0)})
        assert result is not None
        assert result.side == "buy"
        assert "손절" in result.reason

    def test_exit_take_profit_buy(self):
        """매수 포지션, 현재가가 익절가 이상이면 sell 시그널."""
        s = ORBDaytradingStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10200,
            "current_price": 10500,
            "side": "buy",
            "stop_loss_price": 9980,
            "take_profit_price": 10425,  # current > take_profit
            "metadata": {"opening_range_high": 10150, "opening_range_low": 10000},
        }
        result = s.check_exit(position, {"current_time": datetime(2026, 2, 22, 11, 0)})
        assert result is not None
        assert result.side == "sell"
        assert "익절" in result.reason

    def test_exit_take_profit_sell(self):
        """매도 포지션, 현재가가 익절가 이하이면 buy 시그널."""
        s = ORBDaytradingStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 9950,
            "current_price": 9700,
            "side": "sell",
            "stop_loss_price": 10170,
            "take_profit_price": 9725,  # current < take_profit
            "metadata": {"opening_range_high": 10150, "opening_range_low": 10000},
        }
        result = s.check_exit(position, {"current_time": datetime(2026, 2, 22, 11, 0)})
        assert result is not None
        assert result.side == "buy"
        assert "익절" in result.reason

    def test_exit_force_close(self):
        """현재 시각이 15:20 이후이면 강제 청산 시그널."""
        s = ORBDaytradingStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10200,
            "current_price": 10250,
            "side": "buy",
            "metadata": {"opening_range_high": 10150, "opening_range_low": 10000},
        }
        result = s.check_exit(position, {"current_time": datetime(2026, 2, 22, 15, 25)})
        assert result is not None
        assert result.side == "sell"
        assert "강제청산" in result.reason
        assert result.confidence == 1.0

    def test_exit_no_trigger(self):
        """트리거 조건 없으면 None 반환."""
        s = ORBDaytradingStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 10200,
            "current_price": 10220,
            "side": "buy",
            "stop_loss_price": 9980,
            "take_profit_price": 10500,
            "metadata": {"opening_range_high": 10150, "opening_range_low": 10000},
        }
        result = s.check_exit(position, {"current_time": datetime(2026, 2, 22, 11, 0)})
        assert result is None

    def test_exit_invalid_prices(self):
        """entry_price=0이면 None 반환."""
        s = ORBDaytradingStrategy()
        position = {
            "ticker": "005930",
            "entry_price": 0,
            "current_price": 10600,
            "side": "buy",
            "metadata": {},
        }
        result = s.check_exit(position, {"current_time": datetime(2026, 2, 22, 11, 0)})
        assert result is None

    def test_exit_opening_range_based_stop(self):
        """stop_loss_price가 없고 시가범위 하단 이탈 시 손절."""
        s = ORBDaytradingStrategy()
        # or_low * (1 - 0.002) = 10000 * 0.998 = 9980
        position = {
            "ticker": "005930",
            "entry_price": 10200,
            "current_price": 9970,  # < 9980
            "side": "buy",
            "stop_loss_price": 0,  # 없음
            "metadata": {"opening_range_high": 10150, "opening_range_low": 10000},
        }
        result = s.check_exit(position, {"current_time": datetime(2026, 2, 22, 11, 0)})
        assert result is not None
        assert "시가범위 하단 이탈" in result.reason


# ──────────────────────────────────────────────────────────
# 5. 점수 계산 테스트
# ──────────────────────────────────────────────────────────


class TestBreakoutScore:
    """점수 계산 테스트."""

    def test_perfect_breakout_score(self):
        """이른 시간, 높은 거래량, 적절한 범위, 강한 돌파 -> 높은 점수."""
        s = ORBDaytradingStrategy()
        score = s._calculate_breakout_score(
            side="buy",
            current_price=10250,  # or_high 10150 대비 100 돌파 (range 150의 67%)
            or_high=10150,
            or_low=10000,
            volume_ratio=5.0,       # 매우 높은 거래량
            range_pct=0.015,        # 1.5% 범위 (최적)
            current_time=datetime(2026, 2, 22, 9, 35),  # 이른 시간
        )
        assert score > 0.6  # 높은 점수 기대

    def test_minimal_breakout_score(self):
        """늦은 시간, 낮은 거래량, 엣지 범위 -> 낮은 점수."""
        s = ORBDaytradingStrategy()
        score = s._calculate_breakout_score(
            side="buy",
            current_price=10155,  # 아주 약한 돌파 (5/150 ~ 3%)
            or_high=10150,
            or_low=10000,
            volume_ratio=1.6,       # 최소 기준 약간 초과
            range_pct=0.005,        # 최소 범위
            current_time=datetime(2026, 2, 22, 14, 25),  # 마감 직전
        )
        assert score < 0.3  # 낮은 점수 기대

    def test_score_bounded_0_1(self):
        """점수는 항상 0.0~1.0 범위."""
        s = ORBDaytradingStrategy()
        # 극단적 파라미터
        for vol in [0.0, 1.5, 10.0, 100.0]:
            for rp in [0.005, 0.015, 0.03]:
                for price in [10151, 10200, 10500, 11000]:
                    score = s._calculate_breakout_score(
                        side="buy",
                        current_price=price,
                        or_high=10150,
                        or_low=10000,
                        volume_ratio=vol,
                        range_pct=rp,
                        current_time=datetime(2026, 2, 22, 10, 0),
                    )
                    assert 0.0 <= score <= 1.0, f"Score {score} out of range for vol={vol}, rp={rp}, price={price}"


# ──────────────────────────────────────────────────────────
# 6. 헬퍼 메서드 테스트
# ──────────────────────────────────────────────────────────


class TestHelpers:
    """헬퍼 메서드 테스트."""

    def test_parse_time(self):
        """HH:MM 파싱 확인."""
        assert ORBDaytradingStrategy._parse_time("14:30") == time(14, 30)

    def test_parse_time_midnight(self):
        """자정 시각 파싱."""
        assert ORBDaytradingStrategy._parse_time("00:00") == time(0, 0)

    def test_get_opening_range_end_default(self):
        """기본 시가범위 종료 시각 = 09:30."""
        s = ORBDaytradingStrategy()
        end = s._get_opening_range_end()
        assert end == time(9, 30)

    def test_get_opening_range_end_custom(self):
        """커스텀 opening_range_minutes에 따른 종료 시각."""
        s = ORBDaytradingStrategy(params={"opening_range_minutes": 15})
        end = s._get_opening_range_end()
        assert end == time(9, 15)
