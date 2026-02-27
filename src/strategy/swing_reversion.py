"""거래량 급등 + RSI 평균회귀 스윙 트레이딩 전략.

보유기간: 2~5일
진입: 거래량 2배 + RSI < 30 + 볼린저밴드 하한선
청산: 익절 +10%, 손절 -5%, 시간 5일, RSI > 70
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.strategy.short_term_base import ShortTermSignal, ShortTermStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SwingReversionStrategy(ShortTermStrategy):
    """거래량 급등 + RSI 평균회귀 스윙 전략."""

    name = "swing_reversion"
    mode = "swing"

    # 기본 파라미터
    DEFAULT_PARAMS = {
        "volume_multiplier": 2.0,   # 평균 대비 거래량 배수
        "volume_avg_days": 20,      # 거래량 비교 기간 (이동평균)
        "rsi_period": 14,           # RSI 기간
        "rsi_oversold": 30,         # 과매도 기준
        "rsi_overbought": 70,       # 과매수 기준 (청산)
        "bollinger_period": 20,     # 볼린저 기간
        "bollinger_std": 2.0,       # 볼린저 표준편차
        "min_market_cap": 300_000_000_000,  # 최소 시가총액 3000억
        "max_signals": 3,           # 최대 시그널 수
        "stop_loss_pct": -0.05,     # 손절 -5%
        "take_profit_pct": 0.10,    # 익절 +10%
        "max_holding_days": 5,      # 최대 보유일
        "use_obv_filter": False,    # OBV 다이버전스 필터 (기본 OFF)
        "obv_divergence_days": 5,   # 다이버전스 확인 기간
        "regime_filter": False,     # 추세하락 레짐 필터 (MA50<MA200 시 비활성)
        # ATR 동적 손절/익절
        "use_atr_stops": True,
        "atr_period": 14,
        "atr_stop_mult": 2.0,      # 손절: 진입가 - ATR * mult
        "atr_profit_mult": 3.0,    # 익절: 진입가 + ATR * mult
    }

    def __init__(self, params: dict = None):
        """SwingReversionStrategy 초기화.

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
            logger.warning("시장 데이터 없음 — 스캔 스킵")
            return []

        candidates = []

        for ticker, df in daily_data.items():
            if df is None or len(df) < max(self._params["rsi_period"], self._params["bollinger_period"]) + 5:
                continue

            try:
                score = self._evaluate_ticker(ticker, df)
                if score is not None:
                    candidates.append(score)
            except Exception as e:
                logger.debug("종목 %s 평가 실패: %s", ticker, e)

        # 점수 정렬 후 상위 N개
        candidates.sort(key=lambda x: x["score"], reverse=True)
        max_signals = self._params["max_signals"]
        top = candidates[:max_signals]

        signals = []
        for c in top:
            sig = ShortTermSignal(
                id="",  # ShortTermTrader가 자동 생성
                ticker=c["ticker"],
                strategy=self.name,
                side="buy",
                mode=self.mode,
                confidence=min(c["score"] / 100.0, 1.0),
                target_price=c.get("close", 0),
                stop_loss_price=c.get("close", 0) * (1 + self._params["stop_loss_pct"]),
                take_profit_price=c.get("close", 0) * (1 + self._params["take_profit_pct"]),
                reason=c.get("reason", ""),
                metadata={
                    "rsi": c.get("rsi", 0),
                    "volume_ratio": c.get("volume_ratio", 0),
                    "bb_position": c.get("bb_position", 0),
                },
            )
            signals.append(sig)

        logger.info("SwingReversion 스캔: %d 후보 중 %d 시그널", len(candidates), len(signals))
        return signals

    def _evaluate_ticker(self, ticker: str, df: pd.DataFrame) -> Optional[dict]:
        """개별 종목을 평가한다.

        Returns:
            평가 결과 딕셔너리 또는 None (조건 미달)
        """
        # 최신 데이터
        latest = df.iloc[-1]

        close_col = "close" if "close" in df.columns else "종가"
        volume_col = "volume" if "volume" in df.columns else "거래량"

        close = float(latest.get(close_col, 0))
        volume = float(latest.get(volume_col, 0))

        if close <= 0 or volume <= 0:
            return None

        # 0. 추세하락 레짐 필터 (MA50 < MA200이면 비활성)
        if self._params.get("regime_filter", False):
            closes_all = df[close_col].astype(float)
            if len(closes_all) >= 200:
                ma50 = float(closes_all.rolling(50).mean().iloc[-1])
                ma200 = float(closes_all.rolling(200).mean().iloc[-1])
                if not pd.isna(ma50) and not pd.isna(ma200) and ma50 < ma200:
                    return None  # 데드크로스 구간: 매수 비활성

        # 1. 거래량 급등 체크 (20일 이동평균 대비)
        vol_avg_days = self._params.get("volume_avg_days", 20)
        vol_series = df[volume_col].astype(float)
        if len(vol_series) < vol_avg_days + 1:
            return None
        # 오늘 제외한 최근 N일 평균
        avg_volume = float(vol_series.iloc[-(vol_avg_days + 1):-1].mean())
        if avg_volume <= 0:
            return None
        volume_ratio = volume / avg_volume
        if volume_ratio < self._params["volume_multiplier"]:
            return None

        # 2. RSI 계산
        rsi = self._calculate_rsi(df, self._params["rsi_period"])
        if rsi is None or rsi >= self._params["rsi_oversold"]:
            return None

        # 3. 볼린저밴드 하한선 체크
        bb_lower, bb_middle, bb_upper = self._calculate_bollinger(
            df, self._params["bollinger_period"], self._params["bollinger_std"]
        )
        if bb_lower is None:
            return None

        # 볼린저 위치: 0=하한, 0.5=중간, 1=상한
        bb_range = bb_upper - bb_lower if bb_upper != bb_lower else 1
        bb_position = (close - bb_lower) / bb_range

        # 하한선 근처 (20% 이하)
        if bb_position > 0.2:
            return None

        # 4. OBV 다이버전스 필터 (옵션)
        if self._params.get("use_obv_filter", False):
            if not self._check_obv_divergence(df, self._params.get("obv_divergence_days", 5)):
                return None

        # 5. 시가총액 체크 (선택)
        market_cap = float(latest.get("시가총액", latest.get("market_cap", float("inf"))))
        if market_cap < self._params["min_market_cap"]:
            return None

        # 점수 계산 (100점 만점)
        score = 0.0
        # RSI 점수: 30->0점, 0->50점
        score += max(0, (self._params["rsi_oversold"] - rsi) / self._params["rsi_oversold"]) * 50
        # 거래량 점수: 2배->0점, 5배->30점
        score += min((volume_ratio - self._params["volume_multiplier"]) / 3.0, 1.0) * 30
        # 볼린저 점수: 0.2->0점, 0->20점
        score += max(0, (0.2 - bb_position) / 0.2) * 20

        reason = (
            f"RSI={rsi:.1f}, 거래량={volume_ratio:.1f}배, "
            f"BB위치={bb_position:.2f}"
        )

        return {
            "ticker": ticker,
            "score": score,
            "close": close,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "bb_position": bb_position,
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

        # 2. RSI > 70 (과매수)
        daily_data = market_data.get("daily_data", {})
        df = daily_data.get(ticker)
        if df is not None and len(df) >= self._params["rsi_period"] + 1:
            rsi = self._calculate_rsi(df, self._params["rsi_period"])
            if rsi is not None and rsi >= self._params["rsi_overbought"]:
                reasons.append(f"RSI 과매수: {rsi:.1f}")

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
    def _check_obv_divergence(df: pd.DataFrame, lookback: int = 5) -> bool:
        """OBV 강세 다이버전스를 확인한다.

        가격은 하락하지만 OBV는 상승하는 경우 = 강세 다이버전스.
        매집 신호로 해석되어 평균회귀 시그널의 신뢰도를 높인다.

        Args:
            df: 가격/거래량 DataFrame.
            lookback: 다이버전스 비교 기간.

        Returns:
            강세 다이버전스가 발견되면 True.
        """
        close_col = "close" if "close" in df.columns else "종가"
        volume_col = "volume" if "volume" in df.columns else "거래량"

        if close_col not in df.columns or volume_col not in df.columns:
            return False
        if len(df) < lookback + 1:
            return False

        closes = df[close_col].astype(float)
        volumes = df[volume_col].astype(float)

        # OBV 계산
        price_diff = closes.diff()
        sign = price_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (volumes * sign).cumsum()

        # 강세 다이버전스: 가격 하락 + OBV 상승
        price_down = float(closes.iloc[-1]) < float(closes.iloc[-lookback - 1])
        obv_up = float(obv.iloc[-1]) > float(obv.iloc[-lookback - 1])

        return bool(price_down and obv_up)

    @staticmethod
    def _calculate_rsi(df: pd.DataFrame, period: int) -> Optional[float]:
        """RSI를 계산한다."""
        close_col = "close" if "close" in df.columns else "종가"
        if close_col not in df.columns:
            return None

        closes = df[close_col].astype(float)
        if len(closes) < period + 1:
            return None

        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))

        avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
        avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    @staticmethod
    def _calculate_bollinger(
        df: pd.DataFrame, period: int, num_std: float
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """볼린저밴드를 계산한다."""
        close_col = "close" if "close" in df.columns else "종가"
        if close_col not in df.columns:
            return None, None, None

        closes = df[close_col].astype(float)
        if len(closes) < period:
            return None, None, None

        sma = closes.rolling(window=period).mean().iloc[-1]
        std = closes.rolling(window=period).std().iloc[-1]

        if pd.isna(sma) or pd.isna(std):
            return None, None, None

        upper = sma + num_std * std
        lower = sma - num_std * std

        return float(lower), float(sma), float(upper)
