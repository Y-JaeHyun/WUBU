"""Plotly 인터랙티브 차트 모듈.

matplotlib 기반 charts.py의 Plotly 대응 버전으로,
인터랙티브 차트를 생성하여 HTML 리포트에 인라인으로 삽입한다.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 공통 레이아웃 설정
# ---------------------------------------------------------------------------

_DARK_THEME = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(family="sans-serif", size=12, color="#c9d1d9"),
    title_font=dict(size=16, color="#ffffff"),
    legend=dict(
        bgcolor="rgba(22,27,34,0.8)",
        bordercolor="#30363d",
        borderwidth=1,
    ),
    xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
)

_GREEN = "#3fb950"
_RED = "#f85149"
_BLUE = "#58a6ff"
_GRAY = "#8b949e"
_YELLOW = "#d29922"


def _apply_theme(fig: go.Figure) -> go.Figure:
    """다크 테마를 Figure에 적용한다.

    Args:
        fig: Plotly Figure 객체.

    Returns:
        테마가 적용된 Figure 객체.
    """
    fig.update_layout(**_DARK_THEME)
    return fig


# ---------------------------------------------------------------------------
# 1. 인터랙티브 에쿼티 커브 + 드로다운
# ---------------------------------------------------------------------------

def plot_interactive_equity(
    portfolio_values: pd.Series,
    benchmark_values: Optional[pd.Series] = None,
    title: str = "포트폴리오 수익률",
) -> go.Figure:
    """인터랙티브 에쿼티 커브와 드로다운 서브차트를 생성한다.

    상단: 누적 수익률 곡선 (포트폴리오 + 벤치마크)
    하단: 드로다운 영역 차트

    Args:
        portfolio_values: 포트폴리오 가치 시계열 (DatetimeIndex).
        benchmark_values: 벤치마크 가치 시계열 (선택).
        title: 차트 제목.

    Returns:
        Plotly Figure 객체.
    """
    if portfolio_values.empty:
        logger.warning("포트폴리오 데이터가 비어 있어 빈 차트를 반환합니다.")
        fig = go.Figure()
        fig.add_annotation(text="데이터 없음", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return _apply_theme(fig)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=["누적 수익률 (%)", "드로다운 (%)"],
    )

    # 누적 수익률 계산
    initial = portfolio_values.iloc[0]
    if initial == 0:
        logger.warning("초기 포트폴리오 가치가 0입니다.")
        initial = 1

    cum_return = (portfolio_values / initial - 1) * 100

    fig.add_trace(
        go.Scatter(
            x=cum_return.index,
            y=cum_return.values,
            mode="lines",
            name="포트폴리오",
            line=dict(color=_BLUE, width=2),
            hovertemplate="날짜: %{x}<br>수익률: %{y:.2f}%<extra></extra>",
        ),
        row=1, col=1,
    )

    # 벤치마크 오버레이
    if benchmark_values is not None and not benchmark_values.empty:
        bm_initial = benchmark_values.iloc[0]
        if bm_initial != 0:
            bm_return = (benchmark_values / bm_initial - 1) * 100
            fig.add_trace(
                go.Scatter(
                    x=bm_return.index,
                    y=bm_return.values,
                    mode="lines",
                    name="벤치마크",
                    line=dict(color=_GRAY, width=1.5, dash="dash"),
                    hovertemplate="날짜: %{x}<br>수익률: %{y:.2f}%<extra></extra>",
                ),
                row=1, col=1,
            )

    # 드로다운 계산
    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax * 100

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="드로다운",
            fill="tozeroy",
            line=dict(color=_RED, width=1),
            fillcolor="rgba(248,81,73,0.2)",
            hovertemplate="날짜: %{x}<br>드로다운: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    # MDD 표시
    mdd_idx = drawdown.idxmin()
    mdd_val = drawdown.min()
    fig.add_annotation(
        x=mdd_idx, y=mdd_val,
        text=f"MDD: {mdd_val:.1f}%",
        showarrow=True, arrowhead=2,
        arrowcolor=_RED, font=dict(color=_RED, size=11),
        row=2, col=1,
    )

    fig.update_layout(
        title=title,
        height=600,
        showlegend=True,
        hovermode="x unified",
    )

    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# 2. 캔들스틱 + 거래량
# ---------------------------------------------------------------------------

def plot_candlestick(
    ohlcv: pd.DataFrame,
    ticker: str = "",
    title: str = "",
) -> go.Figure:
    """캔들스틱 차트와 거래량 서브차트를 생성한다.

    Args:
        ohlcv: OHLCV DataFrame. 컬럼: open, high, low, close, volume.
            DatetimeIndex 필요.
        ticker: 종목 코드 (차트 제목용).
        title: 차트 제목. 비어 있으면 ticker 기반 자동 생성.

    Returns:
        Plotly Figure 객체.
    """
    if ohlcv.empty:
        logger.warning("OHLCV 데이터가 비어 있어 빈 차트를 반환합니다.")
        fig = go.Figure()
        fig.add_annotation(text="데이터 없음", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return _apply_theme(fig)

    # 컬럼명 표준화 (대소문자 무시)
    col_map = {c.lower(): c for c in ohlcv.columns}
    open_col = col_map.get("open", "open")
    high_col = col_map.get("high", "high")
    low_col = col_map.get("low", "low")
    close_col = col_map.get("close", "close")
    volume_col = col_map.get("volume", "volume")

    has_volume = volume_col in ohlcv.columns

    if has_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.75, 0.25],
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # 캔들스틱
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv[open_col],
            high=ohlcv[high_col],
            low=ohlcv[low_col],
            close=ohlcv[close_col],
            name=ticker or "가격",
            increasing_line_color=_RED,    # 한국 주식: 상승=빨강
            decreasing_line_color=_BLUE,   # 하락=파랑
            increasing_fillcolor=_RED,
            decreasing_fillcolor=_BLUE,
        ),
        row=1, col=1,
    )

    # 거래량
    if has_volume:
        # 상승/하락 색상 구분
        colors = [
            _RED if c >= o else _BLUE
            for c, o in zip(ohlcv[close_col], ohlcv[open_col])
        ]
        fig.add_trace(
            go.Bar(
                x=ohlcv.index,
                y=ohlcv[volume_col],
                name="거래량",
                marker_color=colors,
                opacity=0.6,
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="거래량", row=2, col=1)

    chart_title = title or (f"{ticker} 캔들스틱" if ticker else "캔들스틱")
    fig.update_layout(
        title=chart_title,
        height=600 if has_volume else 450,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="가격", row=1, col=1)

    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# 3. 상관 히트맵
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "종목 간 상관관계",
) -> go.Figure:
    """수익률 기반 상관 히트맵을 생성한다.

    Args:
        returns: 일별 수익률 DataFrame (컬럼: 종목명/코드).
        title: 차트 제목.

    Returns:
        Plotly Figure 객체.
    """
    if returns.empty or returns.shape[1] < 2:
        logger.warning("상관관계 계산에 필요한 데이터가 부족합니다.")
        fig = go.Figure()
        fig.add_annotation(text="데이터 부족", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return _apply_theme(fig)

    corr = returns.corr()
    labels = corr.columns.tolist()

    # 주석 텍스트 (상관계수 값)
    annotations_text = [[f"{corr.iloc[i, j]:.2f}" for j in range(len(labels))]
                        for i in range(len(labels))]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            text=annotations_text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale=[
                [0.0, _RED],
                [0.5, "#1a1e24"],
                [1.0, _GREEN],
            ],
            zmin=-1,
            zmax=1,
            colorbar=dict(title="상관계수"),
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        height=max(400, len(labels) * 40 + 150),
        width=max(500, len(labels) * 40 + 200),
    )

    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# 4. 리스크 기여도
# ---------------------------------------------------------------------------

def plot_risk_contribution(
    risk_decomp: pd.DataFrame,
    title: str = "리스크 기여도",
) -> go.Figure:
    """리스크 기여도를 파이 차트와 수평 바 차트로 표시한다.

    Args:
        risk_decomp: 리스크 분해 DataFrame.
            컬럼: 'ticker' (또는 인덱스), 'contribution' (0~1).
        title: 차트 제목.

    Returns:
        Plotly Figure 객체.
    """
    if risk_decomp.empty:
        logger.warning("리스크 기여도 데이터가 비어 있습니다.")
        fig = go.Figure()
        fig.add_annotation(text="데이터 없음", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return _apply_theme(fig)

    # 데이터 준비
    if "ticker" in risk_decomp.columns:
        labels = risk_decomp["ticker"].tolist()
    else:
        labels = risk_decomp.index.tolist()

    if "contribution" in risk_decomp.columns:
        values = risk_decomp["contribution"].tolist()
    else:
        # 첫 번째 숫자 컬럼 사용
        num_cols = risk_decomp.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            logger.warning("리스크 기여도 숫자 컬럼이 없습니다.")
            fig = go.Figure()
            return _apply_theme(fig)
        values = risk_decomp[num_cols[0]].tolist()

    pct_values = [v * 100 for v in values]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=["구성 비율", "기여도 (%)"],
    )

    # 파이 차트
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=pct_values,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            marker=dict(line=dict(color="#0d1117", width=2)),
        ),
        row=1, col=1,
    )

    # 바 차트 (내림차순 정렬)
    sorted_pairs = sorted(zip(labels, pct_values), key=lambda x: x[1], reverse=True)
    sorted_labels = [p[0] for p in sorted_pairs]
    sorted_values = [p[1] for p in sorted_pairs]

    fig.add_trace(
        go.Bar(
            x=sorted_values,
            y=sorted_labels,
            orientation="h",
            marker_color=_BLUE,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=title,
        height=max(400, len(labels) * 30 + 200),
        showlegend=False,
    )

    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# 5. 팩터 노출도 (레이더 차트)
# ---------------------------------------------------------------------------

def plot_factor_exposure(
    exposures: dict[str, float],
    title: str = "팩터 노출도",
) -> go.Figure:
    """팩터 노출도를 레이더(polar) 차트로 표시한다.

    Args:
        exposures: {팩터명: 노출도} 딕셔너리.
            예: {"Size": 0.3, "Value": -0.2, "Momentum": 0.5, ...}
        title: 차트 제목.

    Returns:
        Plotly Figure 객체.
    """
    if not exposures:
        logger.warning("팩터 노출도 데이터가 비어 있습니다.")
        fig = go.Figure()
        fig.add_annotation(text="데이터 없음", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return _apply_theme(fig)

    factors = list(exposures.keys())
    values = list(exposures.values())

    # 레이더 차트는 시작점과 끝점이 연결되어야 함
    factors_closed = factors + [factors[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=factors_closed,
            fill="toself",
            name="노출도",
            line=dict(color=_BLUE, width=2),
            fillcolor="rgba(88,166,255,0.2)",
            hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
        )
    )

    # 0 기준선
    fig.add_trace(
        go.Scatterpolar(
            r=[0] * len(factors_closed),
            theta=factors_closed,
            mode="lines",
            name="기준선",
            line=dict(color=_GRAY, width=1, dash="dot"),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        height=500,
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(
                visible=True,
                gridcolor="#30363d",
                linecolor="#30363d",
            ),
            angularaxis=dict(
                gridcolor="#30363d",
                linecolor="#30363d",
            ),
        ),
        showlegend=False,
    )

    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def figure_to_html(fig: go.Figure) -> str:
    """Figure를 HTML div 문자열로 변환한다.

    Plotly CDN을 사용하여 JavaScript를 로드하므로
    별도의 plotly.js 파일이 필요하지 않다.

    Args:
        fig: Plotly Figure 객체.

    Returns:
        HTML div 문자열 (인라인 삽입용).
    """
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def save_figure_html(fig: go.Figure, output_path: str) -> None:
    """Figure를 완전한 HTML 파일로 저장한다.

    Args:
        fig: Plotly Figure 객체.
        output_path: HTML 파일 저장 경로.
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    logger.info("Plotly 차트 저장 완료: %s", output_path)


def save_figure_image(
    fig: go.Figure,
    output_path: str,
    width: int = 1200,
    height: int = 600,
) -> bool:
    """Figure를 정적 이미지(PNG)로 저장한다.

    kaleido 패키지가 필요하다. 미설치 시 경고 로그를 남기고 False를 반환한다.

    Args:
        fig: Plotly Figure 객체.
        output_path: 이미지 파일 저장 경로.
        width: 이미지 너비 (px).
        height: 이미지 높이 (px).

    Returns:
        저장 성공 시 True, 실패 시 False.
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        fig.write_image(output_path, width=width, height=height)
        logger.info("Plotly 이미지 저장 완료: %s", output_path)
        return True
    except ValueError as e:
        logger.warning(
            "Plotly 이미지 저장 실패 (kaleido 미설치 가능): %s", e
        )
        return False
    except Exception as e:
        logger.error("Plotly 이미지 저장 오류: %s", e)
        return False
