"""차트 생성 유틸리티 모듈.

matplotlib 기반으로 백테스트 결과 및 포트폴리오 성과를 시각화한다.
서버 환경(Agg 백엔드)에서 동작하도록 설계되었다.
"""

from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 한글 폰트 설정 (NanumBarunGothic — pykrx 설치 시 포함됨)
# ---------------------------------------------------------------------------
_FONT_SET = False


def _setup_korean_font() -> None:
    """한글 폰트(NanumBarunGothic)를 matplotlib에 등록한다."""
    global _FONT_SET
    if _FONT_SET:
        return

    try:
        import matplotlib.font_manager as fm
        import os

        # NanumBarunGothic 경로 탐색
        font_candidates = [
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
            "/usr/share/fonts/nanum/NanumBarunGothic.ttf",
        ]

        # site-packages 내 pykrx 번들 폰트 탐색
        try:
            import pykrx
            pykrx_dir = os.path.dirname(pykrx.__file__)
            font_candidates.append(os.path.join(pykrx_dir, "NanumBarunGothic.ttf"))
        except ImportError:
            pass

        # 시스템 폰트 매니저에서도 탐색
        for f in fm.findSystemFonts():
            if "NanumBarunGothic" in f:
                font_candidates.insert(0, f)

        for fpath in font_candidates:
            if os.path.isfile(fpath):
                fm.fontManager.addfont(fpath)
                prop = fm.FontProperties(fname=fpath)
                plt.rcParams["font.family"] = prop.get_name()
                plt.rcParams["axes.unicode_minus"] = False
                logger.info(f"한글 폰트 설정 완료: {fpath}")
                _FONT_SET = True
                return

        # 후보가 없으면 rcParams만 시도
        plt.rcParams["font.family"] = "NanumBarunGothic"
        plt.rcParams["axes.unicode_minus"] = False
        _FONT_SET = True
        logger.warning("NanumBarunGothic 파일을 직접 찾지 못해 이름만 설정합니다.")
    except Exception as e:
        logger.warning(f"한글 폰트 설정 실패 (영문으로 표시됩니다): {e}")
        _FONT_SET = True  # 반복 시도 방지


def _save_or_return(fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
    """save_path가 주어지면 파일로 저장하고, 아니면 Figure를 반환한다."""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"차트 저장 완료: {save_path}")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 1. 누적 수익률 차트
# ---------------------------------------------------------------------------
def plot_cumulative_returns(
    history_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """누적 수익률 곡선을 그린다.

    history_df가 단일 컬럼('portfolio_value')이면 단일 전략,
    여러 컬럼이면 복수 전략 비교 차트를 생성한다.
    benchmark_df가 주어지면 벤치마크를 오버레이한다.

    Args:
        history_df: DatetimeIndex, 컬럼은 'portfolio_value' 또는 전략별 가치 컬럼들
        benchmark_df: 벤치마크 시계열 (DatetimeIndex, 'close' 또는 단일 숫자 컬럼)
        title: 차트 제목
        save_path: 저장 경로 (None이면 Figure 반환)

    Returns:
        matplotlib Figure 객체
    """
    _setup_korean_font()
    fig, ax = plt.subplots(figsize=(12, 6))

    # history_df 처리: 'portfolio_value' 컬럼이 있으면 그것만, 아니면 모든 숫자 컬럼
    if "portfolio_value" in history_df.columns:
        cols = ["portfolio_value"]
    else:
        cols = history_df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols:
        series = history_df[col]
        initial = series.iloc[0]
        if initial == 0:
            continue
        cum_return = (series / initial - 1) * 100
        label = col.replace("portfolio_value", "Portfolio")
        ax.plot(cum_return.index, cum_return.values, linewidth=1.5, label=label)

    # 벤치마크 오버레이
    if benchmark_df is not None and not benchmark_df.empty:
        if "close" in benchmark_df.columns:
            bm_series = benchmark_df["close"]
        else:
            bm_series = benchmark_df.iloc[:, 0]
        bm_initial = bm_series.iloc[0]
        if bm_initial != 0:
            bm_return = (bm_series / bm_initial - 1) * 100
            ax.plot(
                bm_return.index,
                bm_return.values,
                linewidth=1.5,
                linestyle="--",
                color="gray",
                alpha=0.7,
                label="Benchmark",
            )

    ax.set_title(title or "누적 수익률", fontsize=14, fontweight="bold")
    ax.set_xlabel("날짜")
    ax.set_ylabel("수익률 (%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()

    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 2. 드로다운 차트
# ---------------------------------------------------------------------------
def plot_drawdown(
    history_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """드로다운 시계열 차트를 그린다.

    Args:
        history_df: DatetimeIndex, 'portfolio_value' 컬럼 필요
        save_path: 저장 경로 (None이면 Figure 반환)

    Returns:
        matplotlib Figure 객체
    """
    _setup_korean_font()
    fig, ax = plt.subplots(figsize=(12, 4))

    if "portfolio_value" in history_df.columns:
        values = history_df["portfolio_value"]
    else:
        values = history_df.iloc[:, 0]

    cummax = values.cummax()
    drawdown = (values - cummax) / cummax * 100

    ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    ax.plot(drawdown.index, drawdown.values, color="red", linewidth=0.8)

    # MDD 표시
    mdd_idx = drawdown.idxmin()
    mdd_val = drawdown.min()
    ax.annotate(
        f"MDD: {mdd_val:.1f}%",
        xy=(mdd_idx, mdd_val),
        xytext=(mdd_idx, mdd_val - 2),
        fontsize=10,
        color="darkred",
        fontweight="bold",
        ha="center",
    )

    ax.set_title("드로다운 (Drawdown)", fontsize=14, fontweight="bold")
    ax.set_xlabel("날짜")
    ax.set_ylabel("드로다운 (%)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 3. 월별 수익률 히트맵
# ---------------------------------------------------------------------------
def plot_monthly_returns_heatmap(
    returns_series: pd.Series,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """연도 x 월 히트맵을 그린다.

    빨간색은 손실, 녹색은 이익을 나타낸다.

    Args:
        returns_series: 일별 수익률 Series (DatetimeIndex)
        save_path: 저장 경로 (None이면 Figure 반환)

    Returns:
        matplotlib Figure 객체
    """
    _setup_korean_font()

    # 월별 수익률 계산 (일별 수익률 → 월별 누적)
    monthly = returns_series.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

    # 연도 x 월 피벗
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot_table(index="year", columns="month", values="return", aggfunc="first")
    pivot.columns = [f"{m}월" for m in pivot.columns]

    # 차트 크기 결정
    n_years = len(pivot)
    fig_height = max(3, n_years * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # 커스텀 컬러맵: 빨간(손실) ↔ 흰(0) ↔ 녹색(이익)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ["#d32f2f", "#ffcdd2", "white", "#c8e6c9", "#2e7d32"]
    cmap = LinearSegmentedColormap.from_list("rg", colors_list, N=256)

    # 데이터 범위 대칭화
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 1)
    vmin = -vmax

    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # 축 레이블
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # 셀 내 숫자 표시
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                text_color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    ax.set_title("월별 수익률 (%)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="수익률 (%)", shrink=0.8)
    fig.tight_layout()

    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 4. 연도별 수익률 바 차트
# ---------------------------------------------------------------------------
def plot_annual_returns_bar(
    returns_series: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """연도별 수익률 막대 차트를 그린다.

    Args:
        returns_series: 일별 수익률 Series (DatetimeIndex)
        benchmark_returns: 벤치마크 일별 수익률 Series (선택)
        save_path: 저장 경로 (None이면 Figure 반환)

    Returns:
        matplotlib Figure 객체
    """
    _setup_korean_font()

    # 연도별 수익률 계산
    annual = returns_series.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    years = annual.index.year.astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    x = np.arange(len(years))

    # 색상: 양수=파란, 음수=빨간
    colors = ["#1976d2" if v >= 0 else "#d32f2f" for v in annual.values]
    bars = ax.bar(x, annual.values, bar_width, color=colors, label="전략", edgecolor="white")

    # 벤치마크 비교
    if benchmark_returns is not None:
        bm_annual = benchmark_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
        # 동일 연도만 매칭
        bm_years = bm_annual.index.year.astype(str)
        common_mask = years.isin(bm_years)

        bm_values = []
        for y in years:
            match = bm_annual[bm_annual.index.year == int(y)]
            bm_values.append(match.iloc[0] if len(match) > 0 else 0)
        bm_values = np.array(bm_values)

        ax.bar(x + bar_width, bm_values, bar_width, color="gray", alpha=0.6,
               label="벤치마크", edgecolor="white")

    # 바 위에 수치 표시
    for bar, val in zip(bars, annual.values):
        y_pos = bar.get_height()
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:.1f}%",
                ha="center", va=va, fontsize=9, fontweight="bold")

    ax.set_xticks(x + (bar_width / 2 if benchmark_returns is not None else 0))
    ax.set_xticklabels(years, fontsize=10)
    ax.set_title("연도별 수익률", fontsize=14, fontweight="bold")
    ax.set_ylabel("수익률 (%)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 5. 포트폴리오 구성 차트
# ---------------------------------------------------------------------------
def plot_portfolio_composition(
    weights_dict: dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """포트폴리오 구성 비중을 수평 바 차트로 그린다.

    종목 수가 적으면(10개 이하) 파이 차트, 많으면 수평 바 차트를 사용한다.

    Args:
        weights_dict: {종목명 또는 종목코드: 비중(0~1)} 딕셔너리
        save_path: 저장 경로 (None이면 Figure 반환)

    Returns:
        matplotlib Figure 객체
    """
    _setup_korean_font()

    if not weights_dict:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return _save_or_return(fig, save_path)

    # 비중 기준 정렬 (내림차순)
    sorted_items = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] * 100 for item in sorted_items]  # 퍼센트 변환

    if len(labels) <= 10:
        # 파이 차트
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            pctdistance=0.85,
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title("포트폴리오 구성", fontsize=14, fontweight="bold")
    else:
        # 수평 바 차트
        fig_height = max(6, len(labels) * 0.35 + 2)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        y_pos = np.arange(len(labels))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
        bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()  # 상위 종목이 위에 오도록
        ax.set_xlabel("비중 (%)")
        ax.set_title("포트폴리오 구성", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # 바 옆에 수치 표시
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", ha="left", va="center", fontsize=8)

    fig.tight_layout()
    return _save_or_return(fig, save_path)
