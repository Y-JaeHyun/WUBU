"""리포트/차트/스캐너 모듈 테스트.

차트 생성, 백테스트 리포트, 마켓 스캐너, 일일 리포트 등을 검증한다.
외부 API(pykrx 등)는 mock 처리한다.
"""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from src.report.charts import plot_cumulative_returns, plot_drawdown
from src.report.backtest_report import BacktestReport
from src.report.scanner import MarketScanner


# ===================================================================
# 헬퍼: 테스트용 데이터 생성
# ===================================================================


def _make_portfolio_history(periods: int = 100):
    """테스트용 포트폴리오 히스토리를 생성한다."""
    dates = pd.bdate_range("2024-01-02", periods=periods)
    np.random.seed(42)
    values = 100_000_000 * np.exp(
        np.cumsum(np.random.randn(periods) * 0.005 + 0.0003)
    )
    return pd.DataFrame(
        {
            "portfolio_value": values,
            "cash": values * 0.05,
            "num_holdings": [20] * periods,
        },
        index=dates,
    )


def _make_backtest_results():
    """테스트용 백테스트 결과 dict를 생성한다."""
    return {
        "strategy_name": "TestStrategy",
        "start_date": "20240102",
        "end_date": "20241231",
        "initial_capital": 100_000_000,
        "final_value": 115_000_000,
        "total_return": 15.0,
        "cagr": 15.0,
        "sharpe_ratio": 1.2,
        "mdd": -8.5,
        "win_rate": 60.0,
        "total_trades": 240,
        "rebalance_count": 12,
    }


def _make_mock_backtest():
    """BacktestReport가 기대하는 mock Backtest 객체를 생성한다."""
    mock_bt = MagicMock()
    mock_bt.get_results.return_value = _make_backtest_results()
    mock_bt.get_portfolio_history.return_value = _make_portfolio_history()
    mock_bt.get_trades.return_value = pd.DataFrame(columns=[
        "date", "ticker", "action", "quantity", "price", "cost",
    ])
    return mock_bt


# ===================================================================
# 차트 생성 테스트
# ===================================================================


class TestCharts:
    """차트 함수 검증."""

    def test_plot_cumulative_returns_creates_figure(self):
        """plot_cumulative_returns가 matplotlib Figure를 생성한다."""
        import matplotlib

        matplotlib.use("Agg")  # 비-GUI 백엔드

        history = _make_portfolio_history()

        fig = plot_cumulative_returns(history)

        # matplotlib Figure 객체여야 함
        import matplotlib.figure

        assert isinstance(fig, matplotlib.figure.Figure), (
            "plot_cumulative_returns가 matplotlib Figure를 반환해야 합니다."
        )

    def test_plot_drawdown_creates_figure(self):
        """plot_drawdown가 matplotlib Figure를 생성한다."""
        import matplotlib

        matplotlib.use("Agg")

        history = _make_portfolio_history()

        fig = plot_drawdown(history)

        import matplotlib.figure

        assert isinstance(fig, matplotlib.figure.Figure), (
            "plot_drawdown가 matplotlib Figure를 반환해야 합니다."
        )

    def test_plot_monthly_heatmap(self):
        """월별 수익률 히트맵을 생성한다."""
        import matplotlib

        matplotlib.use("Agg")

        # 충분한 기간의 데이터 (2년)
        history = _make_portfolio_history(periods=500)
        # 일별 수익률 시리즈로 변환
        daily_returns = history["portfolio_value"].pct_change().dropna()

        # 모듈에 plot_monthly_returns_heatmap이 있으면 테스트
        try:
            from src.report.charts import plot_monthly_returns_heatmap

            fig = plot_monthly_returns_heatmap(daily_returns)

            import matplotlib.figure

            assert isinstance(fig, matplotlib.figure.Figure), (
                "plot_monthly_returns_heatmap이 matplotlib Figure를 반환해야 합니다."
            )
        except ImportError:
            pytest.skip("plot_monthly_returns_heatmap이 아직 구현되지 않았습니다.")


# ===================================================================
# BacktestReport 테스트
# ===================================================================


class TestBacktestReport:
    """BacktestReport 검증."""

    def test_backtest_report_text(self):
        """텍스트 리포트가 올바르게 생성된다."""
        mock_bt = _make_mock_backtest()

        report = BacktestReport(mock_bt)
        text = report.generate_text_report()

        assert isinstance(text, str), "텍스트 리포트는 문자열이어야 합니다."
        assert len(text) > 0, "텍스트 리포트가 비어 있으면 안 됩니다."

        # 핵심 정보가 포함되어야 함
        assert "TestStrategy" in text or "전략" in text, (
            "리포트에 전략 이름이 포함되어야 합니다."
        )

    def test_backtest_report_contains_metrics(self):
        """리포트에 주요 성과 지표가 포함된다."""
        mock_bt = _make_mock_backtest()

        report = BacktestReport(mock_bt)
        text = report.generate_text_report()

        # 수익률, CAGR, MDD 등의 정보가 리포트에 존재해야 함
        assert "15" in text or "수익" in text, (
            "리포트에 수익률 정보가 포함되어야 합니다."
        )

    def test_backtest_report_html(self, tmp_path):
        """HTML 리포트가 올바르게 생성된다."""
        mock_bt = _make_mock_backtest()

        report = BacktestReport(mock_bt)
        output = str(tmp_path / "report.html")
        html = report.generate_html_report(output)

        assert isinstance(html, str), "HTML 리포트는 문자열이어야 합니다."
        assert "<html" in html, "HTML 태그가 포함되어야 합니다."
        assert "TestStrategy" in html, "전략 이름이 HTML에 포함되어야 합니다."
        assert (tmp_path / "report.html").exists(), "HTML 파일이 저장되어야 합니다."


# ===================================================================
# MarketScanner 테스트
# ===================================================================


class TestMarketScanner:
    """MarketScanner 검증."""

    @patch("src.report.scanner.get_all_fundamentals")
    def test_scanner_value_top(self, mock_fund):
        """밸류 상위 종목 스캔이 올바르게 동작한다."""
        np.random.seed(42)
        n = 20
        mock_fund.return_value = pd.DataFrame(
            {
                "ticker": [f"{i:06d}" for i in range(1, n + 1)],
                "name": [f"종목{i}" for i in range(1, n + 1)],
                "market": ["KOSPI"] * n,
                "pbr": np.random.uniform(0.3, 3.0, n).round(2),
                "per": np.random.uniform(3, 30, n).round(2),
                "close": np.random.randint(5000, 200000, n),
                "market_cap": np.random.randint(
                    100_000_000_000, 5_000_000_000_000, n
                ),
                "volume": np.random.randint(100_000, 5_000_000, n),
            }
        )

        scanner = MarketScanner()
        result = scanner.scan_value_top(date="20240101", n=5)

        assert isinstance(result, pd.DataFrame), (
            "스캔 결과가 DataFrame이어야 합니다."
        )

        # DataFrame인 경우 길이 확인
        assert len(result) <= 5, "상위 5개 이하만 반환해야 합니다."
        assert not result.empty, "스캔 결과가 비어 있으면 안 됩니다."

    @patch("src.report.scanner.get_all_fundamentals")
    def test_scanner_format_result(self, mock_fund):
        """스캔 결과 포매팅이 올바르게 동작한다."""
        np.random.seed(42)
        n = 10
        mock_fund.return_value = pd.DataFrame(
            {
                "ticker": [f"{i:06d}" for i in range(1, n + 1)],
                "name": [f"종목{i}" for i in range(1, n + 1)],
                "market": ["KOSPI"] * n,
                "pbr": np.random.uniform(0.3, 3.0, n).round(2),
                "per": np.random.uniform(3, 30, n).round(2),
                "close": np.random.randint(5000, 200000, n),
                "market_cap": np.random.randint(
                    100_000_000_000, 5_000_000_000_000, n
                ),
                "volume": np.random.randint(100_000, 5_000_000, n),
            }
        )

        scanner = MarketScanner()
        result = scanner.scan_value_top(date="20240101", n=3)
        formatted = scanner.format_scan_result(result, "밸류 상위")

        assert isinstance(formatted, str), "포매팅 결과가 문자열이어야 합니다."
        assert len(formatted) > 0, "포매팅 결과가 비어 있으면 안 됩니다."


# ===================================================================
# DailyReport 테스트
# ===================================================================


class TestDailyReport:
    """DailyReport 생성 검증."""

    @patch("src.report.scanner.get_all_fundamentals")
    def test_daily_report_generate(self, mock_fund):
        """일일 리포트가 에러 없이 생성된다."""
        np.random.seed(42)
        n = 10
        mock_fund.return_value = pd.DataFrame(
            {
                "ticker": [f"{i:06d}" for i in range(1, n + 1)],
                "name": [f"종목{i}" for i in range(1, n + 1)],
                "market": ["KOSPI"] * n,
                "pbr": np.random.uniform(0.3, 3.0, n).round(2),
                "per": np.random.uniform(3, 30, n).round(2),
                "close": np.random.randint(5000, 200000, n),
                "market_cap": np.random.randint(
                    100_000_000_000, 5_000_000_000_000, n
                ),
                "volume": np.random.randint(100_000, 5_000_000, n),
            }
        )

        try:
            from src.report.daily_report import DailyReport

            report = DailyReport()

            portfolio_state = {
                "holdings": {"005930": 100, "000660": 50},
                "cash": 5_000_000,
                "total_value": 105_000_000,
                "daily_return": 0.5,
                "cumulative_return": 5.0,
                "ticker_prices": {"005930": 70000, "000660": 130000},
                "ticker_names": {"005930": "삼성전자", "000660": "SK하이닉스"},
            }
            market_data = {
                "kospi": 2650.0,
                "kosdaq": 870.0,
                "kospi_change": 0.5,
                "kosdaq_change": -0.3,
            }

            text = report.generate(portfolio_state, market_data)

            assert isinstance(text, str), "일일 리포트는 문자열이어야 합니다."
            assert len(text) > 0, "일일 리포트가 비어 있으면 안 됩니다."
        except ImportError:
            pytest.skip("DailyReport 모듈이 아직 구현되지 않았습니다.")
