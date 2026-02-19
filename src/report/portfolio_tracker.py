"""포트폴리오 성과 추적 모듈.

잔고 이력을 저장하고, MDD(최대 낙폭)를 계산한다.
JSON 파일 기반으로 재시작 후에도 이력을 유지한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioTracker:
    """포트폴리오 성과 추적기.

    매시간 호출되는 모니터링에서 잔고를 기록하고,
    고점 대비 최대 낙폭(MDD)을 실시간 계산한다.

    Attributes:
        history_path: 이력 저장 파일 경로.
        peak: 기록된 최고 평가금액.
        current_mdd: 현재 MDD (음수 비율).
    """

    def __init__(
        self,
        history_path: Optional[str] = None,
        max_records: int = 500,
    ) -> None:
        """PortfolioTracker를 초기화한다.

        Args:
            history_path: 이력 JSON 파일 경로. None이면 기본 경로 사용.
            max_records: 보관할 최대 레코드 수.
        """
        if history_path is None:
            self._path = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "portfolio_history.json"
            )
        else:
            self._path = Path(history_path)

        self._max_records = max_records
        self._history: list[dict] = []
        self.peak: float = 0.0
        self.current_mdd: float = 0.0

        self._load()
        logger.info(
            "PortfolioTracker 초기화 (이력 %d건, peak=%s, mdd=%.2f%%)",
            len(self._history),
            f"{self.peak:,.0f}" if self.peak else "N/A",
            self.current_mdd * 100,
        )

    def _load(self) -> None:
        """저장된 이력 파일을 로드한다."""
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._history = data.get("history", [])
                self.peak = data.get("peak", 0.0)
                self.current_mdd = data.get("current_mdd", 0.0)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("포트폴리오 이력 로드 실패: %s", e)
            self._history = []

    def _save(self) -> None:
        """이력을 파일에 저장한다."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "peak": self.peak,
                "current_mdd": self.current_mdd,
                "history": self._history[-self._max_records :],
            }
            self._path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("포트폴리오 이력 저장 실패: %s", e)

    def update(self, total_eval: int) -> float:
        """현재 평가금액을 기록하고 MDD를 갱신한다.

        Args:
            total_eval: 현재 총 평가금액 (원).

        Returns:
            갱신된 현재 MDD (음수 비율, 예: -0.15 = -15%).
        """
        if total_eval <= 0:
            return self.current_mdd

        now = datetime.now().isoformat(timespec="seconds")

        # 고점 갱신
        if total_eval > self.peak:
            self.peak = float(total_eval)

        # MDD 계산: (현재 - 고점) / 고점
        if self.peak > 0:
            self.current_mdd = (total_eval - self.peak) / self.peak
        else:
            self.current_mdd = 0.0

        # 이력 추가
        self._history.append(
            {
                "ts": now,
                "eval": total_eval,
                "peak": self.peak,
                "mdd": round(self.current_mdd, 6),
            }
        )

        # 오래된 레코드 정리
        if len(self._history) > self._max_records:
            self._history = self._history[-self._max_records :]

        self._save()
        return self.current_mdd

    def get_peak(self) -> float:
        """기록된 최고 평가금액을 반환한다."""
        return self.peak

    def get_mdd(self) -> float:
        """현재 MDD를 반환한다."""
        return self.current_mdd

    def get_history_count(self) -> int:
        """저장된 이력 건수를 반환한다."""
        return len(self._history)
