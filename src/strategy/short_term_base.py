"""단기 트레이딩 전략 추상 클래스."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

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
