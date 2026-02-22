"""장중 분봉 데이터 관리 모듈.

WebSocket 틱 데이터를 분봉(1분, 5분, 15분)으로 집계하고
기술적 지표(RSI, MACD, 볼린저밴드, VWAP)를 계산한다.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from src.utils.logger import get_logger

logger = get_logger(__name__)

KST = timezone(timedelta(hours=9))


@dataclass
class IntradayBar:
    """분봉 데이터 하나를 나타내는 데이터 클래스.

    Attributes:
        timestamp: 해당 분봉의 시작 시각 (예: 09:01:00).
        open: 시가.
        high: 고가.
        low: 저가.
        close: 종가.
        volume: 거래량.
    """

    timestamp: datetime
    open: int
    high: int
    low: int
    close: int
    volume: int


class IntradayDataManager:
    """장중 분봉 데이터 관리기.

    WebSocket 틱 데이터를 수신하여 여러 간격의 분봉으로 집계하고,
    기술적 지표를 계산한다.

    Args:
        intervals: 집계할 분봉 간격 리스트 (기본 [1, 5]).
    """

    SUPPORTED_INTERVALS = [1, 5, 15]
    MAX_BARS_PER_TICKER = 500

    def __init__(self, intervals: Optional[list[int]] = None) -> None:
        """IntradayDataManager를 초기화한다.

        Args:
            intervals: 분봉 간격 리스트. None이면 [1, 5].
                       지원 범위: 1, 5, 15분.
        """
        requested = intervals or [1, 5]
        self._intervals = [
            iv for iv in requested if iv in self.SUPPORTED_INTERVALS
        ]
        if not self._intervals:
            self._intervals = [1]
            logger.warning(
                "유효한 interval이 없어 기본값 [1]을 사용합니다. "
                "지원: %s",
                self.SUPPORTED_INTERVALS,
            )

        # {ticker: {interval: [IntradayBar, ...]}}
        self._bars: dict[str, dict[int, list[IntradayBar]]] = {}

        # 현재 구축 중인 부분 바
        # {ticker: {interval: {"timestamp": dt, "open": int, "high": int,
        #                       "low": int, "close": int, "volume": int}}}
        self._current_bar: dict[str, dict[int, dict]] = {}

        # 데이터 헬스 체크용 마지막 틱 시각
        self._last_tick_time: dict[str, datetime] = {}

        # VWAP 누적 데이터
        # {ticker: {"sum_pv": float, "sum_vol": int}}
        self._vwap_data: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 틱 처리
    # ------------------------------------------------------------------

    def on_tick(self, tick: dict) -> None:
        """틱 데이터 수신 시 호출. 분봉으로 집계한다.

        tick format (KIS WebSocket 호환):
            {"ticker": "005930", "price": 70000, "volume": 100,
             "timestamp": "2026-02-22T09:01:23", ...}

        분 경계(예: 09:01 -> 09:02)를 넘어가면 현재 바를 확정하고
        새 바를 시작한다.

        Args:
            tick: 틱 데이터 딕셔너리.
        """
        ticker = tick.get("ticker", "")
        if not ticker:
            return

        price = tick.get("price", 0)
        volume = tick.get("volume", 0)
        ts_str = tick.get("timestamp", "")

        if price <= 0:
            return

        # 타임스탬프 파싱
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            ts = datetime.now(KST)

        # 데이터 헬스 갱신
        self._last_tick_time[ticker] = ts

        # VWAP 누적
        if ticker not in self._vwap_data:
            self._vwap_data[ticker] = {"sum_pv": 0.0, "sum_vol": 0}
        self._vwap_data[ticker]["sum_pv"] += price * volume
        self._vwap_data[ticker]["sum_vol"] += volume

        # 종목별 저장소 초기화
        if ticker not in self._bars:
            self._bars[ticker] = {iv: [] for iv in self._intervals}
            self._current_bar[ticker] = {}

        # 각 interval에 대해 분봉 집계
        for interval in self._intervals:
            self._process_tick_for_interval(
                ticker, interval, price, volume, ts
            )

    def _process_tick_for_interval(
        self,
        ticker: str,
        interval: int,
        price: int,
        volume: int,
        ts: datetime,
    ) -> None:
        """특정 interval에 대해 틱을 분봉으로 집계한다.

        Args:
            ticker: 종목 코드.
            interval: 분봉 간격 (분).
            price: 체결 가격.
            volume: 체결 수량.
            ts: 체결 시각.
        """
        bar_start = self._get_bar_start(ts, interval)

        current = self._current_bar[ticker].get(interval)

        if current is None:
            # 첫 틱: 새 바 시작
            self._current_bar[ticker][interval] = {
                "timestamp": bar_start,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
            return

        if current["timestamp"] == bar_start:
            # 같은 바 내의 틱: 업데이트
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price
            current["volume"] += volume
        else:
            # 새 분봉 시작: 이전 바 확정
            finalized = IntradayBar(
                timestamp=current["timestamp"],
                open=current["open"],
                high=current["high"],
                low=current["low"],
                close=current["close"],
                volume=current["volume"],
            )
            self._bars[ticker][interval].append(finalized)

            # MAX_BARS_PER_TICKER 제한
            if len(self._bars[ticker][interval]) > self.MAX_BARS_PER_TICKER:
                self._bars[ticker][interval] = self._bars[ticker][interval][
                    -self.MAX_BARS_PER_TICKER:
                ]

            # 새 바 시작
            self._current_bar[ticker][interval] = {
                "timestamp": bar_start,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }

    @staticmethod
    def _get_bar_start(ts: datetime, interval: int) -> datetime:
        """해당 틱이 속하는 분봉의 시작 시각을 계산한다.

        Args:
            ts: 틱 타임스탬프.
            interval: 분봉 간격.

        Returns:
            분봉 시작 시각 (초, 마이크로초 = 0).
        """
        minute_slot = (ts.minute // interval) * interval
        return ts.replace(minute=minute_slot, second=0, microsecond=0)

    # ------------------------------------------------------------------
    # 바 조회
    # ------------------------------------------------------------------

    def get_bars(
        self, ticker: str, interval: int = 1, n: int = 100
    ) -> list[IntradayBar]:
        """확정된 분봉 리스트를 반환한다.

        Args:
            ticker: 종목 코드.
            interval: 분봉 간격 (기본 1분).
            n: 반환할 최대 개수 (기본 100).

        Returns:
            IntradayBar 리스트 (시간순, 최신이 뒤).
        """
        bars = self._bars.get(ticker, {}).get(interval, [])
        return bars[-n:]

    def get_bars_df(
        self, ticker: str, interval: int = 1, n: int = 100
    ) -> pd.DataFrame:
        """확정된 분봉을 DataFrame으로 반환한다.

        Args:
            ticker: 종목 코드.
            interval: 분봉 간격 (기본 1분).
            n: 반환할 최대 개수 (기본 100).

        Returns:
            DataFrame (columns: timestamp, open, high, low, close, volume).
            데이터가 없으면 빈 DataFrame.
        """
        bars = self.get_bars(ticker, interval, n)
        if not bars:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        records = [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 기술적 지표
    # ------------------------------------------------------------------

    def get_rsi(
        self,
        ticker: str,
        period: int = 14,
        interval: int = 5,
    ) -> Optional[float]:
        """RSI(Relative Strength Index)를 계산한다.

        period + 1개 이상의 확정 바가 필요하다.

        Args:
            ticker: 종목 코드.
            period: RSI 기간 (기본 14).
            interval: 분봉 간격 (기본 5분).

        Returns:
            RSI 값 (0~100). 데이터 부족 시 None.
        """
        bars = self.get_bars(ticker, interval, n=period + 50)
        if len(bars) < period + 1:
            return None

        close_series = pd.Series([b.close for b in bars], dtype=float)
        rsi_indicator = RSIIndicator(close=close_series, window=period)
        rsi_values = rsi_indicator.rsi()

        last_rsi = rsi_values.iloc[-1]
        if pd.isna(last_rsi):
            return None
        return round(float(last_rsi), 2)

    def get_macd(
        self,
        ticker: str,
        interval: int = 5,
    ) -> Optional[dict]:
        """MACD를 계산한다.

        EMA(12) - EMA(26), signal = EMA(9, of MACD).
        최소 26 + 9 = 35개 이상의 확정 바가 필요하다.

        Args:
            ticker: 종목 코드.
            interval: 분봉 간격 (기본 5분).

        Returns:
            {"macd": float, "signal": float, "histogram": float}.
            데이터 부족 시 None.
        """
        min_bars = 35
        bars = self.get_bars(ticker, interval, n=min_bars + 50)
        if len(bars) < min_bars:
            return None

        close_series = pd.Series([b.close for b in bars], dtype=float)
        macd_indicator = MACD(
            close=close_series,
            window_slow=26,
            window_fast=12,
            window_sign=9,
        )

        macd_val = macd_indicator.macd().iloc[-1]
        signal_val = macd_indicator.macd_signal().iloc[-1]
        hist_val = macd_indicator.macd_diff().iloc[-1]

        if pd.isna(macd_val) or pd.isna(signal_val):
            return None

        return {
            "macd": round(float(macd_val), 2),
            "signal": round(float(signal_val), 2),
            "histogram": round(float(hist_val), 2),
        }

    def get_bollinger_bands(
        self,
        ticker: str,
        period: int = 20,
        interval: int = 5,
    ) -> Optional[dict]:
        """볼린저 밴드를 계산한다.

        SMA(period) +/- 2 * stddev(period).

        Args:
            ticker: 종목 코드.
            period: 이동평균 기간 (기본 20).
            interval: 분봉 간격 (기본 5분).

        Returns:
            {"upper": float, "middle": float, "lower": float}.
            데이터 부족 시 None.
        """
        bars = self.get_bars(ticker, interval, n=period + 50)
        if len(bars) < period:
            return None

        close_series = pd.Series([b.close for b in bars], dtype=float)
        bb = BollingerBands(
            close=close_series,
            window=period,
            window_dev=2,
        )

        upper = bb.bollinger_hband().iloc[-1]
        middle = bb.bollinger_mavg().iloc[-1]
        lower = bb.bollinger_lband().iloc[-1]

        if pd.isna(upper) or pd.isna(middle) or pd.isna(lower):
            return None

        return {
            "upper": round(float(upper), 2),
            "middle": round(float(middle), 2),
            "lower": round(float(lower), 2),
        }

    def get_vwap(self, ticker: str) -> Optional[float]:
        """VWAP(Volume Weighted Average Price)을 계산한다.

        당일 누적 sum(price * volume) / sum(volume).

        Args:
            ticker: 종목 코드.

        Returns:
            VWAP 값 (float). 데이터가 없으면 None.
        """
        vwap = self._vwap_data.get(ticker)
        if vwap is None or vwap["sum_vol"] == 0:
            return None
        return round(vwap["sum_pv"] / vwap["sum_vol"], 2)

    # ------------------------------------------------------------------
    # 데이터 헬스
    # ------------------------------------------------------------------

    def get_last_tick_time(self, ticker: str) -> Optional[datetime]:
        """해당 종목의 마지막 틱 수신 시각을 반환한다.

        Args:
            ticker: 종목 코드.

        Returns:
            마지막 틱 시각. 데이터가 없으면 None.
        """
        return self._last_tick_time.get(ticker)

    def is_data_stale(
        self, ticker: str, max_delay_seconds: int = 60
    ) -> bool:
        """데이터가 오래되었는지(stale) 판단한다.

        마지막 틱 이후 max_delay_seconds 이상 경과하면 stale.

        Args:
            ticker: 종목 코드.
            max_delay_seconds: 허용 지연 시간 (초, 기본 60).

        Returns:
            데이터가 오래되었으면 True. 데이터가 없어도 True.
        """
        last = self._last_tick_time.get(ticker)
        if last is None:
            return True

        now = datetime.now(last.tzinfo)
        elapsed = (now - last).total_seconds()
        return elapsed > max_delay_seconds

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------

    def reset_day(self) -> None:
        """장 마감 후 호출. 당일 데이터를 모두 초기화한다."""
        self._bars.clear()
        self._current_bar.clear()
        self._last_tick_time.clear()
        self._vwap_data.clear()
        logger.info("IntradayDataManager: 일일 데이터 초기화 완료.")
