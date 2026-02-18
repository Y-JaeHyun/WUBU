"""리포트/차트/스캐너/성과지표 패키지.

백테스트 결과 시각화, 일일 리포트, 시장 스캐너, 성과 지표 계산 기능을 제공한다.
"""

from src.report.backtest_report import BacktestReport  # noqa: F401
from src.report.daily_report import DailyReport  # noqa: F401
from src.report.scanner import MarketScanner  # noqa: F401
from src.report import charts  # noqa: F401
from src.report.metrics import (  # noqa: F401
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    win_rate,
    turnover,
    tracking_error,
    information_ratio,
)

__all__ = [
    "BacktestReport",
    "DailyReport",
    "MarketScanner",
    "charts",
    "cagr",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "win_rate",
    "turnover",
    "tracking_error",
    "information_ratio",
]
