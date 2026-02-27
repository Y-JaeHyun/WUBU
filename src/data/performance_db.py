"""포트폴리오 성과 DB 모듈.

SQLite 기반으로 일별 NAV, 포지션, 거래, 전략별 성과를 저장하고 조회한다.
장기 성과 추적과 롤링 지표 계산을 지원한다.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceDB:
    """SQLite 기반 포트폴리오 성과 저장소.

    Args:
        db_path: SQLite DB 파일 경로.
    """

    def __init__(self, db_path: str = "data/performance.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """DB 연결을 반환한다."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = self._dict_factory
        return conn

    @staticmethod
    def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict:
        """sqlite3 row를 dict로 변환한다."""
        return {
            col[0]: row[idx]
            for idx, col in enumerate(cursor.description)
        }

    def _init_db(self) -> None:
        """테이블을 초기화한다."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS daily_nav (
                    date TEXT PRIMARY KEY,
                    nav REAL NOT NULL,
                    cash REAL DEFAULT 0,
                    positions_value REAL DEFAULT 0,
                    benchmark_return REAL,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    name TEXT,
                    qty INTEGER DEFAULT 0,
                    avg_price REAL DEFAULT 0,
                    market_value REAL DEFAULT 0,
                    weight REAL DEFAULT 0,
                    UNIQUE(date, ticker)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    name TEXT,
                    side TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    price REAL NOT NULL,
                    amount REAL DEFAULT 0,
                    fee REAL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    return_1d REAL,
                    return_7d REAL,
                    return_30d REAL,
                    sharpe REAL,
                    mdd REAL,
                    UNIQUE(date, strategy_name)
                );

                CREATE INDEX IF NOT EXISTS idx_positions_date
                    ON positions(date);
                CREATE INDEX IF NOT EXISTS idx_trades_date
                    ON trades(date);
                CREATE INDEX IF NOT EXISTS idx_strategy_perf_date
                    ON strategy_performance(date);
            """)
            conn.commit()
            logger.info("성과 DB 초기화 완료: %s", self.db_path)
        except sqlite3.Error as e:
            logger.error("성과 DB 초기화 실패: %s", e)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # NAV 기록/조회
    # ------------------------------------------------------------------

    def record_daily_nav(
        self,
        date: str,
        nav: float,
        cash: float = 0,
        positions_value: float = 0,
        benchmark_return: Optional[float] = None,
    ) -> bool:
        """일별 NAV를 기록한다.

        Args:
            date: 날짜 ('YYYY-MM-DD').
            nav: 순자산가치(총 평가).
            cash: 보유 현금.
            positions_value: 포지션 평가금액.
            benchmark_return: 벤치마크 수익률.

        Returns:
            성공 시 True.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO daily_nav
                   (date, nav, cash, positions_value, benchmark_return)
                   VALUES (?, ?, ?, ?, ?)""",
                (date, nav, cash, positions_value, benchmark_return),
            )
            conn.commit()
            logger.info("NAV 기록: %s, %.0f원", date, nav)
            return True
        except sqlite3.Error as e:
            logger.error("NAV 기록 실패: %s", e)
            return False
        finally:
            conn.close()

    def get_nav_history(
        self,
        days: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """NAV 히스토리를 조회한다.

        Args:
            days: 최근 N일간. None이면 전체.

        Returns:
            날짜 오름차순 정렬된 NAV 리스트.
        """
        conn = self._get_conn()
        try:
            if days is not None:
                cutoff = (
                    datetime.now() - timedelta(days=days)
                ).strftime("%Y-%m-%d")
                rows = conn.execute(
                    "SELECT * FROM daily_nav WHERE date >= ? ORDER BY date",
                    (cutoff,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM daily_nav ORDER BY date"
                ).fetchall()

            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error("NAV 히스토리 조회 실패: %s", e)
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 포지션 기록/조회
    # ------------------------------------------------------------------

    def record_positions(
        self,
        date: str,
        positions: list[dict[str, Any]],
    ) -> bool:
        """일별 포지션을 기록한다.

        Args:
            date: 날짜 ('YYYY-MM-DD').
            positions: 포지션 리스트.
                [{ticker, name, qty, avg_price, market_value, weight}, ...]

        Returns:
            성공 시 True.
        """
        conn = self._get_conn()
        try:
            for pos in positions:
                conn.execute(
                    """INSERT OR REPLACE INTO positions
                       (date, ticker, name, qty, avg_price, market_value, weight)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        date,
                        pos.get("ticker", ""),
                        pos.get("name", ""),
                        pos.get("qty", 0),
                        pos.get("avg_price", 0),
                        pos.get("market_value", 0),
                        pos.get("weight", 0),
                    ),
                )
            conn.commit()
            logger.info("포지션 기록: %s, %d종목", date, len(positions))
            return True
        except sqlite3.Error as e:
            logger.error("포지션 기록 실패: %s", e)
            return False
        finally:
            conn.close()

    def get_positions(self, date: str) -> list[dict[str, Any]]:
        """특정 날짜의 포지션을 조회한다.

        Args:
            date: 날짜 ('YYYY-MM-DD').

        Returns:
            포지션 리스트.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM positions WHERE date = ? ORDER BY weight DESC",
                (date,),
            ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error("포지션 조회 실패: %s", e)
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 거래 기록/조회
    # ------------------------------------------------------------------

    def record_trade(
        self,
        date: str,
        ticker: str,
        side: str,
        qty: int,
        price: float,
        fee: float = 0,
        name: str = "",
    ) -> bool:
        """거래를 기록한다.

        Args:
            date: 거래일 ('YYYY-MM-DD').
            ticker: 종목코드.
            side: 매매 구분 ('buy' 또는 'sell').
            qty: 수량.
            price: 가격.
            fee: 수수료.
            name: 종목명.

        Returns:
            성공 시 True.
        """
        amount = qty * price
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO trades
                   (date, ticker, name, side, qty, price, amount, fee)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (date, ticker, name, side, qty, price, amount, fee),
            )
            conn.commit()
            logger.info(
                "거래 기록: %s %s %s %d주 @%.0f원",
                date, side.upper(), ticker, qty, price,
            )
            return True
        except sqlite3.Error as e:
            logger.error("거래 기록 실패: %s", e)
            return False
        finally:
            conn.close()

    def get_trades(
        self,
        days: Optional[int] = None,
        ticker: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """거래 내역을 조회한다.

        Args:
            days: 최근 N일간. None이면 전체.
            ticker: 특정 종목만. None이면 전체.

        Returns:
            거래 리스트 (날짜 내림차순).
        """
        conn = self._get_conn()
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params: list[Any] = []

            if days is not None:
                cutoff = (
                    datetime.now() - timedelta(days=days)
                ).strftime("%Y-%m-%d")
                query += " AND date >= ?"
                params.append(cutoff)

            if ticker is not None:
                query += " AND ticker = ?"
                params.append(ticker)

            query += " ORDER BY date DESC, id DESC"

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error("거래 조회 실패: %s", e)
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 전략 성과 기록/조회
    # ------------------------------------------------------------------

    def record_strategy_performance(
        self,
        date: str,
        strategy_name: str,
        return_1d: Optional[float] = None,
        return_7d: Optional[float] = None,
        return_30d: Optional[float] = None,
        sharpe: Optional[float] = None,
        mdd: Optional[float] = None,
    ) -> bool:
        """전략별 성과를 기록한다.

        Args:
            date: 날짜.
            strategy_name: 전략 이름.
            return_1d: 1일 수익률.
            return_7d: 7일 수익률.
            return_30d: 30일 수익률.
            sharpe: Sharpe Ratio.
            mdd: Maximum Drawdown.

        Returns:
            성공 시 True.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO strategy_performance
                   (date, strategy_name, return_1d, return_7d, return_30d, sharpe, mdd)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (date, strategy_name, return_1d, return_7d, return_30d, sharpe, mdd),
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error("전략 성과 기록 실패: %s", e)
            return False
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 성과 분석
    # ------------------------------------------------------------------

    def get_performance_summary(
        self,
        period: str = "30d",
    ) -> dict[str, Any]:
        """기간별 성과 요약을 반환한다.

        Args:
            period: 기간 ('7d', '30d', '90d', '1y', 'all').

        Returns:
            성과 요약 딕셔너리.
        """
        days_map = {
            "7d": 7,
            "30d": 30,
            "90d": 90,
            "1y": 365,
            "all": None,
        }
        days = days_map.get(period, 30)
        nav_history = self.get_nav_history(days=days)

        if not nav_history:
            return {"period": period, "error": "데이터 없음"}

        navs = [row["nav"] for row in nav_history]
        dates = [row["date"] for row in nav_history]

        first_nav = navs[0]
        last_nav = navs[-1]
        total_return = (last_nav / first_nav - 1) if first_nav > 0 else 0

        # 일별 수익률
        daily_returns: list[float] = []
        for i in range(1, len(navs)):
            if navs[i - 1] > 0:
                daily_returns.append(navs[i] / navs[i - 1] - 1)

        # MDD 계산
        peak = navs[0]
        max_dd = 0.0
        for nav in navs:
            if nav > peak:
                peak = nav
            dd = (nav / peak - 1) if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd

        # Sharpe Ratio (연율화, 무위험 0 가정)
        sharpe = 0.0
        if daily_returns:
            import statistics

            mean_r = statistics.mean(daily_returns)
            std_r = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0
            if std_r > 0:
                sharpe = round((mean_r / std_r) * (252 ** 0.5), 4)

        return {
            "period": period,
            "start_date": dates[0],
            "end_date": dates[-1],
            "start_nav": round(first_nav, 0),
            "end_nav": round(last_nav, 0),
            "total_return": round(total_return, 4),
            "mdd": round(max_dd, 4),
            "sharpe": sharpe,
            "data_points": len(navs),
        }

    def calculate_rolling_metrics(
        self,
        window: int = 30,
    ) -> list[dict[str, Any]]:
        """롤링 성과 지표를 계산한다.

        Args:
            window: 롤링 윈도우 (일수).

        Returns:
            롤링 지표 리스트.
        """
        nav_history = self.get_nav_history()

        if len(nav_history) < window + 1:
            return []

        navs = [row["nav"] for row in nav_history]
        dates = [row["date"] for row in nav_history]

        results: list[dict[str, Any]] = []

        for i in range(window, len(navs)):
            window_navs = navs[i - window : i + 1]
            window_date = dates[i]

            # 수익률
            ret = (
                (window_navs[-1] / window_navs[0] - 1)
                if window_navs[0] > 0
                else 0
            )

            # 일별 수익률
            daily_returns: list[float] = []
            for j in range(1, len(window_navs)):
                if window_navs[j - 1] > 0:
                    daily_returns.append(window_navs[j] / window_navs[j - 1] - 1)

            # MDD
            peak = window_navs[0]
            max_dd = 0.0
            for nav in window_navs:
                if nav > peak:
                    peak = nav
                dd = (nav / peak - 1) if peak > 0 else 0
                if dd < max_dd:
                    max_dd = dd

            # Sharpe
            sharpe = 0.0
            if daily_returns:
                import statistics

                mean_r = statistics.mean(daily_returns)
                std_r = (
                    statistics.stdev(daily_returns)
                    if len(daily_returns) > 1
                    else 0
                )
                if std_r > 0:
                    sharpe = round((mean_r / std_r) * (252 ** 0.5), 4)

            results.append({
                "date": window_date,
                "return": round(ret, 4),
                "mdd": round(max_dd, 4),
                "sharpe": sharpe,
            })

        return results

    def format_summary_report(self, period: str = "30d") -> str:
        """텔레그램 발송용 성과 요약을 생성한다.

        Args:
            period: 기간.

        Returns:
            포매팅된 성과 요약 문자열.
        """
        summary = self.get_performance_summary(period)

        if "error" in summary:
            return f"[성과 요약] {summary['error']}"

        total_ret = summary.get("total_return", 0)
        sign = "+" if total_ret >= 0 else ""

        lines = [
            f"[성과 요약] {summary.get('period', '?')}",
            "=" * 35,
            f"  기간: {summary.get('start_date')} ~ {summary.get('end_date')}",
            f"  시작 NAV: {summary.get('start_nav', 0):,.0f}원",
            f"  현재 NAV: {summary.get('end_nav', 0):,.0f}원",
            f"  수익률: {sign}{total_ret:.2%}",
            f"  MDD: {summary.get('mdd', 0):.2%}",
            f"  Sharpe: {summary.get('sharpe', 0):.2f}",
            f"  데이터: {summary.get('data_points', 0)}일",
            "=" * 35,
        ]
        return "\n".join(lines)
