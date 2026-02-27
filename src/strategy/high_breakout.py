"""52주 신고가 돌파 모멘텀 전략.

보유기간: 3~10일
진입: 종가 > 252일 최고가 + 거래량 > 20일 평균 x 2
청산: 익절 +15%, 손절 -5%, 트레일링 10%, 시간 10일
"""

import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.short_term_base import ShortTermSignal, ShortTermStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HighBreakoutStrategy(ShortTermStrategy):
    """52주 신고가 돌파 모멘텀 전략.

    보유기간: 3~10일
    진입: 종가 > 252일 최고가 + 거래량 > 20일 평균 x 2
    청산: 익절 +15%, 손절 -5%, 트레일링 10%, 시간 10일
    """

    name = "high_breakout"
    mode = "swing"

    DEFAULT_PARAMS = {
        "lookback_days": 252,               # 52주
        "volume_multiplier": 2.0,           # 거래량 확인 배수
        "volume_avg_days": 20,              # 평균 거래량 기간
        "take_profit_pct": 0.15,            # +15% 익절
        "stop_loss_pct": -0.05,             # -5% 손절
        "trailing_stop_pct": 0.10,          # 고점 대비 10% 하락
        "max_holding_days": 10,             # 최대 보유일
        "max_signals": 3,
        "min_market_cap": 300_000_000_000,  # 3000억
        "confirm_close": False,             # 종가확인형 돌파 (True: 종가 확인 후 다음날 진입)
        # ATR 동적 손절/익절
        "use_atr_stops": True,
        "atr_period": 14,
        "atr_stop_mult": 2.0,              # 손절: 진입가 - ATR * mult
        "atr_profit_mult": 3.0,            # 익절: 진입가 + ATR * mult
    }

    def __init__(self, params: dict = None):
        """HighBreakoutStrategy 초기화.

        Args:
            params: 파라미터 오버라이드 딕셔너리.
        """
        self._params = dict(self.DEFAULT_PARAMS)
        if params:
            self._params.update(params)

    @property
    def params(self) -> dict:
        return dict(self._params)

    def scan_signals(self, market_data: dict) -> list[ShortTermSignal]:
        """시그널 스캔.

        market_data에 필요한 키:
        - "daily_data": {ticker: DataFrame} -- 일봉 데이터 (columns: close, volume, 시가총액/market_cap 등)
        - "date": str -- 기준일

        Returns:
            상위 max_signals개의 ShortTermSignal 리스트
        """
        daily_data = market_data.get("daily_data", {})
        if not daily_data:
            logger.warning("시장 데이터 없음 — 스캔 스킵")
            return []

        candidates = []

        for ticker, df in daily_data.items():
            if df is None or len(df) < self._params["lookback_days"] + 1:
                continue

            try:
                result = self._evaluate_ticker(ticker, df)
                if result is not None:
                    candidates.append(result)
            except Exception as e:
                logger.debug("종목 %s 평가 실패: %s", ticker, e)

        # 점수 정렬 후 상위 N개
        candidates.sort(key=lambda x: x["score"], reverse=True)
        max_signals = self._params["max_signals"]
        top = candidates[:max_signals]

        signals = []
        for c in top:
            close = c.get("close", 0)
            sig = ShortTermSignal(
                id="",  # ShortTermTrader가 자동 생성
                ticker=c["ticker"],
                strategy=self.name,
                side="buy",
                mode=self.mode,
                confidence=min(c["score"] / 100.0, 1.0),
                target_price=close,
                stop_loss_price=close * (1 + self._params["stop_loss_pct"]),
                take_profit_price=close * (1 + self._params["take_profit_pct"]),
                reason=c.get("reason", ""),
                metadata={
                    "breakout_strength": c.get("breakout_strength", 0),
                    "volume_ratio": c.get("volume_ratio", 0),
                    "prev_52w_high": c.get("prev_52w_high", 0),
                    "market_cap": c.get("market_cap", 0),
                    "confirm_close": self._params.get("confirm_close", False),
                },
            )
            signals.append(sig)

        logger.info("HighBreakout 스캔: %d 후보 중 %d 시그널", len(candidates), len(signals))
        return signals

    def _evaluate_ticker(self, ticker: str, df: pd.DataFrame) -> Optional[dict]:
        """개별 종목을 52주 신고가 돌파 조건으로 평가한다.

        Returns:
            평가 결과 딕셔너리 또는 None (조건 미달)
        """
        lookback = self._params["lookback_days"]
        vol_avg_days = self._params["volume_avg_days"]

        # 컬럼명 결정 (영문 / 한글)
        close_col = "close" if "close" in df.columns else "종가"
        volume_col = "volume" if "volume" in df.columns else "거래량"

        if close_col not in df.columns or volume_col not in df.columns:
            return None

        # 최신 행 기준
        latest = df.iloc[-1]
        close = float(latest[close_col])
        volume = float(latest[volume_col])

        if close <= 0 or volume <= 0:
            return None

        # 1. 52주 최고가 (오늘 제외)
        hist_close = df[close_col].astype(float).iloc[-(lookback + 1):-1]
        prev_52w_high = float(hist_close.max())

        if prev_52w_high <= 0:
            return None

        # 신고가 돌파 확인: 오늘 종가 > 과거 252일 최고가
        if close <= prev_52w_high:
            return None

        breakout_strength = (close - prev_52w_high) / prev_52w_high

        # 2. 거래량 확인: 오늘 거래량 > 20일 평균 거래량 x multiplier
        recent_volumes = df[volume_col].astype(float).iloc[-vol_avg_days:]
        avg_volume = float(recent_volumes.mean())

        if avg_volume <= 0:
            return None

        volume_ratio = volume / avg_volume
        if volume_ratio < self._params["volume_multiplier"]:
            return None

        # 3. 시가총액 체크 (선택)
        market_cap = float(
            latest.get("시가총액", latest.get("market_cap", float("inf")))
        )
        if market_cap < self._params["min_market_cap"]:
            return None

        # 4. 마지막 52주 최고가가 얼마나 오래전인지 (recency)
        hist_close_values = hist_close.values
        last_high_idx = int(np.argmax(hist_close_values))
        days_since_high = len(hist_close_values) - 1 - last_high_idx

        # 점수 계산 (100점 만점)
        score = 0.0

        # (a) Breakout strength (25pts): 돌파 강도, max 10%까지 선형
        bs_score = min(breakout_strength / 0.10, 1.0) * 25
        score += bs_score

        # (b) Volume ratio (35pts): 거래량 배수, max 5x까지 선형
        vr_score = min(volume_ratio / 5.0, 1.0) * 35
        score += vr_score

        # (c) Recency of last high (20pts): 오래전일수록 높은 점수 (더 큰 돌파)
        # 0일 전 → 0점, 252일 전 → 20점
        recency_score = min(days_since_high / lookback, 1.0) * 20
        score += recency_score

        # (d) Market cap (20pts): 로그 스케일, 3000억~10조 구간
        min_cap = self._params["min_market_cap"]  # 3000억
        max_cap = min_cap * 33.3  # ~10조
        if market_cap != float("inf") and market_cap > 0:
            log_min = math.log10(min_cap)
            log_max = math.log10(max_cap)
            log_cap = math.log10(min(market_cap, max_cap))
            cap_ratio = (log_cap - log_min) / (log_max - log_min) if log_max > log_min else 0
            cap_score = min(max(cap_ratio, 0), 1.0) * 20
        else:
            cap_score = 10.0  # 시가총액 정보 없으면 중간 점수
        score += cap_score

        reason = (
            f"52W돌파={breakout_strength:.2%}, "
            f"거래량={volume_ratio:.1f}배, "
            f"최고가이후={days_since_high}일"
        )

        return {
            "ticker": ticker,
            "score": score,
            "close": close,
            "breakout_strength": breakout_strength,
            "volume_ratio": volume_ratio,
            "prev_52w_high": prev_52w_high,
            "days_since_high": days_since_high,
            "market_cap": market_cap,
            "reason": reason,
        }

    def check_exit(self, position: dict, market_data: dict) -> Optional[ShortTermSignal]:
        """보유 포지션 청산 체크.

        Args:
            position: {"ticker", "entry_price", "current_price", "entry_date", "metadata": {"peak_price": float}}
            market_data: {"daily_data": {ticker: DataFrame}, "date": str}

        Exit conditions (checked in order):
        1. Stop loss: pnl <= -5%
        2. Take profit: pnl >= +15%
        3. Trailing stop: current_price < peak_price * (1 - 0.10)
        4. Time stop: days_held >= 10

        Returns:
            청산 시그널 또는 None
        """
        ticker = position.get("ticker", "")
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", 0)

        if entry_price <= 0 or current_price <= 0:
            return None

        pnl_pct = (current_price - entry_price) / entry_price
        reasons = []

        # 1. ATR 또는 고정 % 손절/익절
        use_atr = self._params.get("use_atr_stops", False)
        atr_applied = False
        if use_atr:
            atr_reasons, atr_applied = self._check_atr_exit(position, market_data)
            reasons.extend(atr_reasons)
        if not atr_applied:
            # 고정 % fallback
            if pnl_pct <= self._params["stop_loss_pct"]:
                reasons.append(f"손절: {pnl_pct:.2%}")
            if pnl_pct >= self._params["take_profit_pct"]:
                reasons.append(f"익절: {pnl_pct:.2%}")

        # 2. 트레일링 스탑
        metadata = position.get("metadata", {})
        peak_price = metadata.get("peak_price", entry_price)
        if peak_price > 0 and current_price < peak_price * (1 - self._params["trailing_stop_pct"]):
            drop_from_peak = (current_price - peak_price) / peak_price
            reasons.append(f"트레일링: 고점대비 {drop_from_peak:.2%}")

        # 3. 시간 손절
        entry_date = position.get("entry_date", "")
        if entry_date:
            try:
                # "YYYY-MM-DD" 또는 "YYYYMMDD" 형식 지원
                date_str = entry_date.replace("-", "")
                entry_dt = datetime.strptime(date_str, "%Y%m%d")
                # 기준일: market_data의 date 또는 현재
                ref_date_str = market_data.get("date", "")
                if ref_date_str:
                    ref_str = ref_date_str.replace("-", "")
                    ref_dt = datetime.strptime(ref_str, "%Y%m%d")
                else:
                    ref_dt = datetime.now()
                days_held = (ref_dt - entry_dt).days
                if days_held >= self._params["max_holding_days"]:
                    reasons.append(f"보유일 초과: {days_held}일")
            except ValueError:
                pass

        if not reasons:
            return None

        return ShortTermSignal(
            id="",
            ticker=ticker,
            strategy=self.name,
            side="sell",
            mode=self.mode,
            confidence=0.9,
            target_price=current_price,
            reason=", ".join(reasons),
        )
