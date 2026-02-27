"""성과 DB(src/data/performance_db.py) 테스트.

PerformanceDB 클래스의 NAV 기록/조회, 포지션 기록/조회,
거래 기록/조회, 성과 분석, 롤링 지표 계산 기능을 검증한다.
tmp_path를 사용하여 테스트마다 격리된 DB를 사용한다.
"""

import os

import pytest

from src.data.performance_db import PerformanceDB


# ===================================================================
# 공통 픽스처
# ===================================================================


@pytest.fixture
def db(tmp_path):
    """임시 디렉토리에 생성된 PerformanceDB."""
    db_path = str(tmp_path / "test_performance.db")
    return PerformanceDB(db_path=db_path)


@pytest.fixture
def db_with_nav(db):
    """NAV 데이터가 미리 입력된 DB."""
    for i in range(30):
        date = f"2026-02-{i + 1:02d}" if i < 28 else f"2026-03-{i - 27:02d}"
        nav = 1_000_000 + i * 10_000
        db.record_daily_nav(date, nav, cash=100_000, positions_value=nav - 100_000)
    return db


# ===================================================================
# 초기화 테스트
# ===================================================================


class TestInit:
    """PerformanceDB 초기화 테스트."""

    def test_db_file_created(self, tmp_path):
        """DB 파일이 생성된다."""
        db_path = str(tmp_path / "test.db")
        PerformanceDB(db_path=db_path)
        assert os.path.exists(db_path)

    def test_parent_dir_created(self, tmp_path):
        """부모 디렉토리가 자동으로 생성된다."""
        db_path = str(tmp_path / "subdir" / "test.db")
        PerformanceDB(db_path=db_path)
        assert os.path.exists(db_path)

    def test_tables_created(self, db):
        """필요한 테이블이 생성된다."""
        conn = db._get_conn()
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t["name"] for t in tables}
            assert "daily_nav" in table_names
            assert "positions" in table_names
            assert "trades" in table_names
            assert "strategy_performance" in table_names
        finally:
            conn.close()


# ===================================================================
# NAV 기록/조회 테스트
# ===================================================================


class TestDailyNav:
    """record_daily_nav() / get_nav_history() 테스트."""

    def test_record_and_retrieve(self, db):
        """NAV를 기록하고 조회할 수 있다."""
        db.record_daily_nav("2026-02-26", 1_000_000, cash=200_000, positions_value=800_000)
        history = db.get_nav_history()

        assert len(history) == 1
        assert history[0]["nav"] == 1_000_000
        assert history[0]["cash"] == 200_000

    def test_upsert_same_date(self, db):
        """같은 날짜에 다시 기록하면 업데이트된다."""
        db.record_daily_nav("2026-02-26", 1_000_000)
        db.record_daily_nav("2026-02-26", 1_100_000)

        history = db.get_nav_history()
        assert len(history) == 1
        assert history[0]["nav"] == 1_100_000

    def test_multiple_dates(self, db):
        """여러 날짜를 기록하고 조회한다."""
        db.record_daily_nav("2026-02-24", 1_000_000)
        db.record_daily_nav("2026-02-25", 1_010_000)
        db.record_daily_nav("2026-02-26", 1_020_000)

        history = db.get_nav_history()
        assert len(history) == 3

    def test_history_sorted_by_date(self, db):
        """결과가 날짜 오름차순으로 정렬된다."""
        db.record_daily_nav("2026-02-26", 1_020_000)
        db.record_daily_nav("2026-02-24", 1_000_000)
        db.record_daily_nav("2026-02-25", 1_010_000)

        history = db.get_nav_history()
        dates = [h["date"] for h in history]
        assert dates == sorted(dates)

    def test_days_filter(self, db_with_nav):
        """days 파라미터로 최근 N일만 조회한다."""
        all_history = db_with_nav.get_nav_history()
        history_7d = db_with_nav.get_nav_history(days=7)
        # days 필터가 적용되어 전체보다 적거나 같은 결과를 반환
        assert len(history_7d) <= len(all_history)
        # 실제로 필터링이 되는지 확인
        history_1d = db_with_nav.get_nav_history(days=1)
        assert len(history_1d) < len(all_history)

    def test_empty_history(self, db):
        """데이터가 없으면 빈 리스트를 반환한다."""
        assert db.get_nav_history() == []

    def test_benchmark_return_recorded(self, db):
        """벤치마크 수익률이 기록된다."""
        db.record_daily_nav("2026-02-26", 1_000_000, benchmark_return=0.01)
        history = db.get_nav_history()
        assert history[0]["benchmark_return"] == 0.01


# ===================================================================
# 포지션 기록/조회 테스트
# ===================================================================


class TestPositions:
    """record_positions() / get_positions() 테스트."""

    def test_record_and_retrieve(self, db):
        """포지션을 기록하고 조회한다."""
        positions = [
            {"ticker": "005930", "name": "삼성전자", "qty": 10, "avg_price": 70000,
             "market_value": 700000, "weight": 0.7},
            {"ticker": "000660", "name": "SK하이닉스", "qty": 5, "avg_price": 120000,
             "market_value": 600000, "weight": 0.3},
        ]
        db.record_positions("2026-02-26", positions)
        result = db.get_positions("2026-02-26")

        assert len(result) == 2

    def test_sorted_by_weight(self, db):
        """포지션이 비중 내림차순으로 정렬된다."""
        positions = [
            {"ticker": "A", "weight": 0.3},
            {"ticker": "B", "weight": 0.7},
        ]
        db.record_positions("2026-02-26", positions)
        result = db.get_positions("2026-02-26")

        assert result[0]["ticker"] == "B"
        assert result[1]["ticker"] == "A"

    def test_upsert_position(self, db):
        """같은 날짜/종목에 다시 기록하면 업데이트된다."""
        db.record_positions("2026-02-26", [{"ticker": "005930", "qty": 10, "weight": 0.5}])
        db.record_positions("2026-02-26", [{"ticker": "005930", "qty": 20, "weight": 0.8}])

        result = db.get_positions("2026-02-26")
        assert len(result) == 1
        assert result[0]["qty"] == 20

    def test_empty_positions(self, db):
        """데이터가 없으면 빈 리스트를 반환한다."""
        assert db.get_positions("2026-02-26") == []


# ===================================================================
# 거래 기록/조회 테스트
# ===================================================================


class TestTrades:
    """record_trade() / get_trades() 테스트."""

    def test_record_and_retrieve(self, db):
        """거래를 기록하고 조회한다."""
        db.record_trade("2026-02-26", "005930", "buy", 10, 70000, fee=700, name="삼성전자")

        trades = db.get_trades()
        assert len(trades) == 1
        assert trades[0]["ticker"] == "005930"
        assert trades[0]["side"] == "buy"
        assert trades[0]["qty"] == 10
        assert trades[0]["amount"] == 700000
        assert trades[0]["fee"] == 700

    def test_multiple_trades(self, db):
        """여러 거래를 기록한다."""
        db.record_trade("2026-02-26", "005930", "buy", 10, 70000)
        db.record_trade("2026-02-26", "000660", "sell", 5, 120000)

        trades = db.get_trades()
        assert len(trades) == 2

    def test_filter_by_days(self, db):
        """days 파라미터로 최근 N일 거래만 조회한다."""
        db.record_trade("2026-01-01", "005930", "buy", 10, 70000)
        db.record_trade("2026-02-26", "000660", "sell", 5, 120000)

        trades = db.get_trades(days=7)
        # 최근 7일 이내 거래만 반환
        assert all(t["date"] >= "2026-02-19" for t in trades)

    def test_filter_by_ticker(self, db):
        """ticker 파라미터로 특정 종목 거래만 조회한다."""
        db.record_trade("2026-02-26", "005930", "buy", 10, 70000)
        db.record_trade("2026-02-26", "000660", "sell", 5, 120000)

        trades = db.get_trades(ticker="005930")
        assert len(trades) == 1
        assert trades[0]["ticker"] == "005930"

    def test_trades_sorted_desc(self, db):
        """거래가 날짜 내림차순으로 정렬된다."""
        db.record_trade("2026-02-24", "005930", "buy", 10, 70000)
        db.record_trade("2026-02-26", "000660", "sell", 5, 120000)

        trades = db.get_trades()
        assert trades[0]["date"] >= trades[1]["date"]


# ===================================================================
# 전략 성과 기록 테스트
# ===================================================================


class TestStrategyPerformance:
    """record_strategy_performance() 테스트."""

    def test_record_strategy(self, db):
        """전략 성과를 기록한다."""
        result = db.record_strategy_performance(
            "2026-02-26", "MultiFactor",
            return_1d=0.01, return_7d=0.03, return_30d=0.05,
            sharpe=1.5, mdd=-0.08,
        )
        assert result is True

    def test_upsert_strategy(self, db):
        """같은 날짜/전략에 다시 기록하면 업데이트된다."""
        db.record_strategy_performance("2026-02-26", "MultiFactor", return_1d=0.01)
        db.record_strategy_performance("2026-02-26", "MultiFactor", return_1d=0.02)

        conn = db._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM strategy_performance WHERE date=? AND strategy_name=?",
                ("2026-02-26", "MultiFactor"),
            ).fetchall()
            assert len(rows) == 1
            assert rows[0]["return_1d"] == 0.02
        finally:
            conn.close()


# ===================================================================
# get_performance_summary() 테스트
# ===================================================================


class TestPerformanceSummary:
    """get_performance_summary() 성과 요약 테스트."""

    def test_empty_data_returns_error(self, db):
        """데이터가 없으면 에러를 반환한다."""
        summary = db.get_performance_summary()
        assert "error" in summary

    def test_positive_return(self, db):
        """양의 수익률을 계산한다."""
        db.record_daily_nav("2026-02-01", 1_000_000)
        db.record_daily_nav("2026-02-28", 1_100_000)

        summary = db.get_performance_summary(period="all")
        assert summary["total_return"] == pytest.approx(0.1, abs=0.001)

    def test_negative_return(self, db):
        """음의 수익률을 계산한다."""
        db.record_daily_nav("2026-02-01", 1_000_000)
        db.record_daily_nav("2026-02-28", 900_000)

        summary = db.get_performance_summary(period="all")
        assert summary["total_return"] == pytest.approx(-0.1, abs=0.001)

    def test_mdd_calculation(self, db):
        """MDD를 올바르게 계산한다."""
        db.record_daily_nav("2026-02-01", 1_000_000)
        db.record_daily_nav("2026-02-02", 1_100_000)  # 고점
        db.record_daily_nav("2026-02-03", 990_000)     # 저점
        db.record_daily_nav("2026-02-04", 1_050_000)

        summary = db.get_performance_summary(period="all")
        # MDD = (990000 / 1100000 - 1) = -0.1
        assert summary["mdd"] == pytest.approx(-0.1, abs=0.001)

    def test_summary_includes_sharpe(self, db_with_nav):
        """Sharpe Ratio가 포함된다."""
        summary = db_with_nav.get_performance_summary(period="all")
        assert "sharpe" in summary

    def test_period_filter(self, db_with_nav):
        """period 파라미터로 기간을 제한한다."""
        summary_7d = db_with_nav.get_performance_summary("7d")
        summary_all = db_with_nav.get_performance_summary("all")
        assert summary_7d["data_points"] <= summary_all["data_points"]


# ===================================================================
# calculate_rolling_metrics() 테스트
# ===================================================================


class TestRollingMetrics:
    """calculate_rolling_metrics() 롤링 지표 테스트."""

    def test_insufficient_data_returns_empty(self, db):
        """데이터가 부족하면 빈 리스트를 반환한다."""
        db.record_daily_nav("2026-02-26", 1_000_000)
        result = db.calculate_rolling_metrics(window=30)
        assert result == []

    def test_rolling_metrics_calculated(self, db_with_nav):
        """롤링 지표가 계산된다."""
        result = db_with_nav.calculate_rolling_metrics(window=5)
        assert len(result) > 0
        assert "date" in result[0]
        assert "return" in result[0]
        assert "mdd" in result[0]
        assert "sharpe" in result[0]

    def test_rolling_window_size(self, db_with_nav):
        """윈도우 크기에 따라 결과 수가 달라진다."""
        result_5 = db_with_nav.calculate_rolling_metrics(window=5)
        result_10 = db_with_nav.calculate_rolling_metrics(window=10)
        assert len(result_5) >= len(result_10)


# ===================================================================
# format_summary_report() 테스트
# ===================================================================


class TestFormatSummaryReport:
    """format_summary_report() 포맷 테스트."""

    def test_no_data_message(self, db):
        """데이터가 없으면 에러 메시지를 반환한다."""
        result = db.format_summary_report()
        assert "데이터 없음" in result

    def test_with_data(self, db_with_nav):
        """데이터가 있으면 포매팅된 보고서를 반환한다."""
        result = db_with_nav.format_summary_report(period="all")

        assert "[성과 요약]" in result
        assert "NAV" in result
        assert "수익률" in result
        assert "MDD" in result
        assert "Sharpe" in result
