"""단기 트레이딩 전략 추상 클래스."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ShortTermSignal:
    """단기 매매 시그널."""
    id: str                        # "sig_20260222_001" 형태
    ticker: str                    # 종목코드 6자리
    strategy: str                  # 전략 이름 (e.g., "swing_reversion")
    side: str                      # "buy" or "sell"
    mode: str                      # "swing" or "daytrading"
    confidence: float              # 0.0 ~ 1.0
    target_price: float = 0.0     # 목표가 (0이면 시장가)
    stop_loss_price: float = 0.0  # 손절가
    take_profit_price: float = 0.0 # 익절가
    reason: str = ""               # 시그널 사유
    state: str = "pending"         # pending -> confirmed -> executing -> done/expired/rejected
    created_at: str = ""           # ISO format
    expires_at: str = ""           # ISO format, 만료 시각
    confirmed_at: Optional[str] = None
    executed_at: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """딕셔너리로 변환."""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "strategy": self.strategy,
            "side": self.side,
            "mode": self.mode,
            "confidence": self.confidence,
            "target_price": self.target_price,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "reason": self.reason,
            "state": self.state,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "confirmed_at": self.confirmed_at,
            "executed_at": self.executed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ShortTermSignal':
        """딕셔너리에서 생성."""
        return cls(
            id=data.get("id", ""),
            ticker=data.get("ticker", ""),
            strategy=data.get("strategy", ""),
            side=data.get("side", "buy"),
            mode=data.get("mode", "swing"),
            confidence=data.get("confidence", 0.0),
            target_price=data.get("target_price", 0.0),
            stop_loss_price=data.get("stop_loss_price", 0.0),
            take_profit_price=data.get("take_profit_price", 0.0),
            reason=data.get("reason", ""),
            state=data.get("state", "pending"),
            created_at=data.get("created_at", ""),
            expires_at=data.get("expires_at", ""),
            confirmed_at=data.get("confirmed_at"),
            executed_at=data.get("executed_at"),
            metadata=data.get("metadata", {}),
        )


class ShortTermStrategy(ABC):
    """단기 트레이딩 전략 추상 클래스.

    모든 단기 전략은 이 클래스를 상속받아 구현한다.
    """

    name: str = "base"
    mode: str = "swing"  # "swing" or "daytrading"

    @abstractmethod
    def scan_signals(self, market_data: dict) -> list[ShortTermSignal]:
        """시그널을 스캔한다. 서브클래스에서 구현."""
        ...

    @abstractmethod
    def check_exit(self, position: dict, market_data: dict) -> Optional[ShortTermSignal]:
        """보유 포지션의 청산 여부를 확인한다. 서브클래스에서 구현."""
        ...

    # ── ATR 유틸리티 ──────────────────────────────────────

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """ATR(Average True Range) 계산.

        ATR = SMA of True Range over *period* days.
        True Range = max(high-low, |high-prev_close|, |low-prev_close|).
        high/low 컬럼이 없으면 close 로 대체한다.

        Returns:
            최신 ATR 값, 계산 불가 시 None.
        """
        close_col = "close" if "close" in df.columns else "종가"
        high_col = "high" if "high" in df.columns else "고가"
        low_col = "low" if "low" in df.columns else "저가"

        if close_col not in df.columns:
            return None

        closes = df[close_col].astype(float)
        if len(closes) < period + 1:
            return None

        # high/low가 있으면 사용, 없으면 close로 대체
        if high_col in df.columns and low_col in df.columns:
            highs = df[high_col].astype(float)
            lows = df[low_col].astype(float)
        else:
            highs = closes
            lows = closes

        prev_closes = closes.shift(1)

        tr1 = highs - lows
        tr2 = (highs - prev_closes).abs()
        tr3 = (lows - prev_closes).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]

        if pd.isna(atr) or atr <= 0:
            return None

        return float(atr)

    def _check_atr_exit(
        self, position: dict, market_data: dict
    ) -> tuple[list[str], bool]:
        """ATR 기반 동적 손절/익절 체크.

        Returns:
            (reasons, atr_available):
              - reasons: 청산 사유 문자열 리스트
              - atr_available: ATR 계산 성공 여부 (False면 고정% fallback 필요)
        """
        params = getattr(self, "_params", {})
        ticker = position.get("ticker", "")
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", 0)

        if entry_price <= 0 or current_price <= 0:
            return [], False

        daily_data = market_data.get("daily_data", {})
        df = daily_data.get(ticker)
        if df is None:
            return [], False

        atr_period = params.get("atr_period", 14)
        atr = self._calculate_atr(df, atr_period)
        if atr is None:
            return [], False

        atr_stop_mult = params.get("atr_stop_mult", 2.0)
        atr_profit_mult = params.get("atr_profit_mult", 3.0)

        atr_stop_price = entry_price - atr * atr_stop_mult
        atr_profit_price = entry_price + atr * atr_profit_mult

        pnl_pct = (current_price - entry_price) / entry_price
        reasons: list[str] = []

        if current_price <= atr_stop_price:
            reasons.append(f"ATR손절: {pnl_pct:.2%}(ATR={atr:.0f})")

        if current_price >= atr_profit_price:
            reasons.append(f"ATR익절: {pnl_pct:.2%}(ATR={atr:.0f})")

        return reasons, True
