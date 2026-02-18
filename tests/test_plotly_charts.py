"""Plotly 차트 모듈(src/report/plotly_charts.py) 테스트.

plot_interactive_equity, plot_candlestick, plot_correlation_heatmap,
plot_risk_contribution, plot_factor_exposure, figure_to_html
함수의 Figure 생성 및 HTML 변환을 검증한다.

src/report/plotly_charts.py가 아직 구현되지 않았으면 테스트를 스킵한다.
"""

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _try_import_plotly_charts():
    """plotly_charts 모듈을 임포트한다."""
    try:
        import src.report.plotly_charts as pc
        return pc
    except ImportError:
        return None


def _try_import_go():
    """plotly.graph_objects를 임포트한다."""
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        return None


def _make_equity_curve(n=100, seed=42):
    """포트폴리오 가치 시계열을 생성한다."""
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    values = 100_000_000 * np.exp(np.cumsum(np.random.randn(n) * 0.005 + 0.002))
    return pd.Series(values, index=dates, name="portfolio_value")


def _make_ohlcv(n=60, seed=42):
    """OHLCV DataFrame을 생성한다."""
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    base = 70000
    close = base + np.cumsum(np.random.randn(n) * 500)
    close = np.maximum(close, 1000)
    return pd.DataFrame({
        "open": close - np.random.randint(0, 500, n).astype(float),
        "high": close + np.random.randint(0, 1000, n).astype(float),
        "low": close - np.random.randint(0, 1000, n).astype(float),
        "close": close,
        "volume": np.random.randint(100000, 5000000, n),
    }, index=dates)


def _make_returns_df(n_assets=5, n_days=100, seed=42):
    """다자산 수익률 DataFrame을 생성한다 (상관 히트맵용)."""
    np.random.seed(seed)
    tickers = [f"{i:06d}" for i in range(1, n_assets + 1)]
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    returns = np.random.randn(n_days, n_assets) * 0.02
    return pd.DataFrame(returns, index=dates, columns=tickers)


# ===================================================================
# plotly_charts 검증
# ===================================================================

class TestPlotlyCharts:
    """Plotly 차트 모듈 검증."""

    def test_interactive_equity(self):
        """인터랙티브 수익률 곡선 Figure가 생성된다."""
        pc = _try_import_plotly_charts()
        go = _try_import_go()
        if pc is None:
            pytest.skip("plotly_charts 모듈이 아직 구현되지 않았습니다.")
        if go is None:
            pytest.skip("plotly가 설치되어 있지 않습니다.")

        if not hasattr(pc, "plot_interactive_equity"):
            pytest.skip("plot_interactive_equity 함수를 찾을 수 없습니다.")

        equity = _make_equity_curve()
        fig = pc.plot_interactive_equity(equity)

        assert isinstance(fig, go.Figure), "반환값이 go.Figure여야 합니다."
        assert len(fig.data) > 0, "Figure에 데이터가 있어야 합니다."

    def test_candlestick(self):
        """캔들스틱 차트 Figure가 생성된다."""
        pc = _try_import_plotly_charts()
        go = _try_import_go()
        if pc is None:
            pytest.skip("plotly_charts 모듈이 아직 구현되지 않았습니다.")
        if go is None:
            pytest.skip("plotly가 설치되어 있지 않습니다.")

        if not hasattr(pc, "plot_candlestick"):
            pytest.skip("plot_candlestick 함수를 찾을 수 없습니다.")

        ohlcv = _make_ohlcv()
        fig = pc.plot_candlestick(ohlcv)

        assert isinstance(fig, go.Figure), "반환값이 go.Figure여야 합니다."
        assert len(fig.data) > 0, "Figure에 데이터가 있어야 합니다."

    def test_correlation_heatmap(self):
        """상관계수 히트맵 Figure가 생성된다."""
        pc = _try_import_plotly_charts()
        go = _try_import_go()
        if pc is None:
            pytest.skip("plotly_charts 모듈이 아직 구현되지 않았습니다.")
        if go is None:
            pytest.skip("plotly가 설치되어 있지 않습니다.")

        if not hasattr(pc, "plot_correlation_heatmap"):
            pytest.skip("plot_correlation_heatmap 함수를 찾을 수 없습니다.")

        # plot_correlation_heatmap은 수익률 DataFrame을 받아 내부에서 .corr() 호출
        returns_df = _make_returns_df()
        fig = pc.plot_correlation_heatmap(returns_df)

        assert isinstance(fig, go.Figure), "반환값이 go.Figure여야 합니다."
        assert len(fig.data) > 0, "Figure에 데이터가 있어야 합니다."

    def test_risk_contribution_chart(self):
        """리스크 기여도 차트 Figure가 생성된다."""
        pc = _try_import_plotly_charts()
        go = _try_import_go()
        if pc is None:
            pytest.skip("plotly_charts 모듈이 아직 구현되지 않았습니다.")
        if go is None:
            pytest.skip("plotly가 설치되어 있지 않습니다.")

        if not hasattr(pc, "plot_risk_contribution"):
            pytest.skip("plot_risk_contribution 함수를 찾을 수 없습니다.")

        np.random.seed(42)
        tickers = [f"{i:06d}" for i in range(1, 6)]
        rc_data = pd.DataFrame({
            "ticker": tickers,
            "contribution": np.random.rand(5),
        })
        # 정규화하여 합이 1이 되도록
        rc_data["contribution"] = rc_data["contribution"] / rc_data["contribution"].sum()

        fig = pc.plot_risk_contribution(rc_data)

        assert isinstance(fig, go.Figure), "반환값이 go.Figure여야 합니다."
        assert len(fig.data) > 0, "Figure에 데이터가 있어야 합니다."

    def test_factor_exposure_chart(self):
        """팩터 노출도 차트 Figure가 생성된다."""
        pc = _try_import_plotly_charts()
        go = _try_import_go()
        if pc is None:
            pytest.skip("plotly_charts 모듈이 아직 구현되지 않았습니다.")
        if go is None:
            pytest.skip("plotly가 설치되어 있지 않습니다.")

        if not hasattr(pc, "plot_factor_exposure"):
            pytest.skip("plot_factor_exposure 함수를 찾을 수 없습니다.")

        # plot_factor_exposure는 dict[str, float]을 받는다
        exposure = {
            "value": 0.5,
            "momentum": -0.2,
            "quality": 0.8,
            "size": -0.3,
        }

        fig = pc.plot_factor_exposure(exposure)

        assert isinstance(fig, go.Figure), "반환값이 go.Figure여야 합니다."
        assert len(fig.data) > 0, "Figure에 데이터가 있어야 합니다."

    def test_figure_to_html(self):
        """Figure를 HTML 문자열로 변환할 수 있다."""
        pc = _try_import_plotly_charts()
        go = _try_import_go()
        if pc is None:
            pytest.skip("plotly_charts 모듈이 아직 구현되지 않았습니다.")
        if go is None:
            pytest.skip("plotly가 설치되어 있지 않습니다.")

        if not hasattr(pc, "figure_to_html"):
            pytest.skip("figure_to_html 함수를 찾을 수 없습니다.")

        # 간단한 Figure 생성
        equity = _make_equity_curve()

        if hasattr(pc, "plot_interactive_equity"):
            fig = pc.plot_interactive_equity(equity)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=[1, 2, 3]))

        html = pc.figure_to_html(fig)
        assert isinstance(html, str), "HTML이 문자열이어야 합니다."
        assert len(html) > 0, "HTML이 비어 있으면 안 됩니다."
        assert "<" in html, "HTML 태그가 포함되어야 합니다."
