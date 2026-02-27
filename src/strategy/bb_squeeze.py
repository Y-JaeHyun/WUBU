"""볼린저밴드 스퀴즈 브레이크아웃 전략.

보유기간: 3~7일
진입: 밴드폭 6개월 최저 → 종가 > 상단밴드 + 거래량 1.5배 + MA200 위
청산: 익절 +10%, 손절 -5%, 하단밴드 재진입, 시간 7일
"""

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.strategy.short_term_base import ShortTermSignal, ShortTermStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BBSqueezeStrategy(ShortTermStrategy):
    """볼린저밴드 스퀴즈 브레이크아웃 전략.

    보유기간: 3~7일
    진입: 밴드폭 6개월 최저 → 종가 > 상단밴드 + 거래량 1.5배 + MA200 위
    청산: 익절 +10%, 손절 -5%, 하단밴드 재진입, 시간 7일
    """

    name = "bb_squeeze"
    mode = "swing"

    DEFAULT_PARAMS = {
        "bb_period": 20,
        "bb_std": 2.0,
        "bandwidth_lookback": 126,       # 6개월 최저 밴드폭 탐색
        "bandwidth_percentile": 5,       # 하위 5% 밴드폭 = 스퀴즈
        "volume_multiplier": 1.5,        # 거래량 1.5배
        "volume_avg_days": 20,
        "ma_trend_period": 200,          # MA200 추세 필터
        "trend_filter_period": 0,        # 추가 추세필터 MA 기간 (0=비활성, 권장=100)
        "take_profit_pct": 0.10,         # +10%
        "stop_loss_pct": -0.05,          # -5%
        "max_holding_days": 7,
        "max_signals": 3,
        "min_market_cap": 300_000_000_000,
        # ATR 동적 손절/익절
        "use_atr_stops": True,
        "atr_period": 14,
        "atr_stop_mult": 2.0,           # 손절: 진입가 - ATR * mult
        "atr_profit_mult": 3.0,         # 익절: 진입가 + ATR * mult
    }

    def __init__(self, params: dict = None):
        """BBSqueezeStrategy 초기화.

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
        - "daily_data": {ticker: DataFrame} -- 일봉 데이터 (columns: close, volume, 시가총액 등)
        - "date": str -- 기준일

        Returns:
            상위 max_signals개의 ShortTermSignal 리스트
        """
        daily_data = market_data.get("daily_data", {})
        if not daily_data:
            logger.warning("시장 데이터 없음 -- 스캔 스킵")
            return []

        min_rows = max(
            self._params["ma_trend_period"],
            self._params["bb_period"] + self._params["bandwidth_lookback"],
        ) + 5

        candidates = []

        for ticker, df in daily_data.items():
            if df is None or len(df) < min_rows:
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
                    "bandwidth": c.get("bandwidth", 0),
                    "bandwidth_percentile": c.get("bandwidth_pct", 0),
                    "volume_ratio": c.get("volume_ratio", 0),
                    "ma200_distance": c.get("ma200_distance", 0),
                    "upper_band": c.get("upper_band", 0),
                    "lower_band": c.get("lower_band", 0),
                },
            )
            signals.append(sig)

        logger.info("BBSqueeze 스캔: %d 후보 중 %d 시그널", len(candidates), len(signals))
        return signals

    def _evaluate_ticker(self, ticker: str, df: pd.DataFrame) -> Optional[dict]:
        """개별 종목을 평가한다.

        Returns:
            평가 결과 딕셔너리 또는 None (조건 미달)
        """
        close_col = "close" if "close" in df.columns else "종가"
        volume_col = "volume" if "volume" in df.columns else "거래량"

        if close_col not in df.columns or volume_col not in df.columns:
            return None

        closes = df[close_col].astype(float)
        volumes = df[volume_col].astype(float)

        latest_close = float(closes.iloc[-1])
        if latest_close <= 0:
            return None

        # 1. Bollinger Bandwidth 계산
        bb_period = self._params["bb_period"]
        bb_std = self._params["bb_std"]
        bandwidth = self._calculate_bandwidth(closes, bb_period, bb_std)

        if bandwidth.iloc[-1] is None or pd.isna(bandwidth.iloc[-1]):
            return None

        current_bw = float(bandwidth.iloc[-1])

        # 2. 스퀴즈 체크: 최근 bandwidth_lookback일 중 하위 bandwidth_percentile%
        lookback = self._params["bandwidth_lookback"]
        bw_window = bandwidth.iloc[-lookback:]
        bw_valid = bw_window.dropna()

        if len(bw_valid) < 20:  # 최소 20일은 필요
            return None

        percentile_threshold = bw_valid.quantile(
            self._params["bandwidth_percentile"] / 100.0
        )
        is_squeeze = current_bw <= percentile_threshold
        if not is_squeeze:
            return None

        # 3. 상단밴드 브레이크아웃 체크
        sma = closes.rolling(bb_period).mean()
        rolling_std = closes.rolling(bb_period).std()
        upper_band = sma + bb_std * rolling_std
        lower_band = sma - bb_std * rolling_std

        current_upper = float(upper_band.iloc[-1])
        current_lower = float(lower_band.iloc[-1])

        if pd.isna(current_upper):
            return None

        if latest_close <= current_upper:
            return None  # 브레이크아웃 아님

        # 4. 거래량 체크
        vol_avg_days = self._params["volume_avg_days"]
        if len(volumes) < vol_avg_days + 1:
            return None

        avg_volume = float(volumes.iloc[-vol_avg_days - 1:-1].mean())
        if avg_volume <= 0:
            return None

        current_volume = float(volumes.iloc[-1])
        volume_ratio = current_volume / avg_volume
        if volume_ratio < self._params["volume_multiplier"]:
            return None

        # 5. MA200 추세 필터
        ma_period = self._params["ma_trend_period"]
        if len(closes) < ma_period:
            return None

        ma200 = float(closes.rolling(ma_period).mean().iloc[-1])
        if pd.isna(ma200) or ma200 <= 0:
            return None

        if latest_close <= ma200:
            return None  # 하락 추세

        # 5b. 추가 추세필터 (trend_filter_period > 0이면 활성)
        trend_filter_period = self._params.get("trend_filter_period", 0)
        if trend_filter_period > 0 and len(closes) >= trend_filter_period:
            ma_trend = float(closes.rolling(trend_filter_period).mean().iloc[-1])
            if not pd.isna(ma_trend) and latest_close <= ma_trend:
                return None  # 추가 추세필터 미달

        # 6. 시가총액 체크
        market_cap_col = "시가총액" if "시가총액" in df.columns else "market_cap"
        if market_cap_col in df.columns:
            market_cap = float(df[market_cap_col].iloc[-1])
            if market_cap < self._params["min_market_cap"]:
                return None
        # market_cap 컬럼이 없으면 필터 스킵 (데이터 불완전 허용)

        # --- 점수 계산 (100점 만점) ---
        score = 0.0

        # 스퀴즈 심각도 (30점): 밴드폭이 낮을수록 높은 점수
        bw_min = float(bw_valid.min())
        bw_max = float(bw_valid.max())
        if bw_max > bw_min:
            squeeze_severity = 1.0 - (current_bw - bw_min) / (bw_max - bw_min)
        else:
            squeeze_severity = 1.0
        score += squeeze_severity * 30

        # 거래량 비율 (30점): 1.5x -> 0점, 3x+ -> 30점
        vol_score = min((volume_ratio - self._params["volume_multiplier"]) / 1.5, 1.0)
        score += max(0, vol_score) * 30

        # 추세 강도 (20점): MA200 위 거리
        ma200_distance = (latest_close - ma200) / ma200
        trend_score = min(ma200_distance / 0.20, 1.0)  # 20% 위면 만점
        score += max(0, trend_score) * 20

        # 시가총액 (20점): log 스케일
        if market_cap_col in df.columns:
            mcap = float(df[market_cap_col].iloc[-1])
            if mcap > 0:
                # 3000억(11.47) ~ 100조(14.0) 범위
                log_mcap = math.log10(mcap)
                mcap_score = min(max((log_mcap - 11.47) / 2.53, 0), 1.0)
                score += mcap_score * 20
            else:
                score += 10  # 기본 중간 점수
        else:
            score += 10  # market_cap 정보 없으면 중간 점수

        reason = (
            f"BB스퀴즈: BW={current_bw:.4f}(하위{self._params['bandwidth_percentile']}%), "
            f"거래량={volume_ratio:.1f}배, MA200+{ma200_distance:.1%}"
        )

        return {
            "ticker": ticker,
            "score": score,
            "close": latest_close,
            "bandwidth": current_bw,
            "bandwidth_pct": float(
                (bw_valid <= current_bw).mean() * 100
            ),
            "volume_ratio": volume_ratio,
            "ma200_distance": ma200_distance,
            "upper_band": current_upper,
            "lower_band": current_lower,
            "reason": reason,
        }

    def check_exit(self, position: dict, market_data: dict) -> Optional[ShortTermSignal]:
        """보유 포지션 청산 체크.

        Args:
            position: {"ticker", "entry_price", "current_price", "entry_date", ...}
            market_data: {"daily_data": {ticker: DataFrame}}

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

        # 2. 하단밴드 재진입 (평균회귀 실패)
        daily_data = market_data.get("daily_data", {})
        df = daily_data.get(ticker)
        if df is not None:
            close_col = "close" if "close" in df.columns else "종가"
            if close_col in df.columns and len(df) >= self._params["bb_period"]:
                closes = df[close_col].astype(float)
                sma = closes.rolling(self._params["bb_period"]).mean()
                rolling_std = closes.rolling(self._params["bb_period"]).std()
                lower_band = sma - self._params["bb_std"] * rolling_std

                current_lower = lower_band.iloc[-1]
                if not pd.isna(current_lower) and current_price < float(current_lower):
                    reasons.append(f"하단밴드 재진입: 현재가={current_price:.0f}, 하단={float(current_lower):.0f}")

        # 3. 시간 손절
        entry_date = position.get("entry_date", "")
        if entry_date:
            try:
                entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
                days_held = (datetime.now() - entry_dt).days
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

    @staticmethod
    def _calculate_bandwidth(closes: pd.Series, period: int, std: float) -> pd.Series:
        """Calculate Bollinger Bandwidth series.

        Bandwidth = (Upper - Lower) / Middle
        """
        sma = closes.rolling(period).mean()
        rolling_std = closes.rolling(period).std()
        upper = sma + std * rolling_std
        lower = sma - std * rolling_std
        bandwidth = (upper - lower) / sma
        return bandwidth
