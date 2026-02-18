"""성과 지표 모듈.

백테스트 결과의 다양한 성과 지표를 계산하는 함수를 제공한다.
CAGR, MDD, 샤프비율, 소르티노비율, 칼마비율, 승률, 회전율,
추적오차, 정보비율 등을 지원한다.
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 연간 거래일 수
TRADING_DAYS_PER_YEAR = 252


def cagr(returns_series: pd.Series) -> float:
    """연평균 복합 성장률(CAGR)을 계산한다.

    Args:
        returns_series: 일별 수익률 시리즈 (DatetimeIndex)

    Returns:
        CAGR (소수, 예: 0.12 = 12%)
    """
    if returns_series.empty:
        return 0.0

    # 누적 수익률 계산
    cumulative = (1 + returns_series).cumprod()
    total_return = cumulative.iloc[-1]

    # 기간 계산 (년)
    n_days = (returns_series.index[-1] - returns_series.index[0]).days
    if n_days <= 0:
        return 0.0

    n_years = n_days / 365.25

    if total_return <= 0:
        return -1.0

    return float(total_return ** (1 / n_years) - 1)


def max_drawdown(returns_series: pd.Series) -> float:
    """최대 낙폭(MDD)을 계산한다.

    Args:
        returns_series: 일별 수익률 시리즈 (DatetimeIndex)

    Returns:
        최대 낙폭 (음수, 예: -0.25 = -25%)
    """
    if returns_series.empty:
        return 0.0

    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()

    return float(mdd)


def sharpe_ratio(
    returns_series: pd.Series,
    risk_free: float = 0.025,
) -> float:
    """샤프비율을 계산한다.

    연율화된 초과수익률을 연율화된 변동성으로 나눈 값이다.

    Args:
        returns_series: 일별 수익률 시리즈 (DatetimeIndex)
        risk_free: 연간 무위험이자율 (기본 2.5%)

    Returns:
        샤프비율 (float)
    """
    if returns_series.empty or len(returns_series) < 2:
        return 0.0

    risk_free_daily = risk_free / TRADING_DAYS_PER_YEAR
    excess_returns = returns_series - risk_free_daily

    mean_excess = excess_returns.mean()
    std_excess = returns_series.std()

    if std_excess == 0 or np.isnan(std_excess):
        return 0.0

    return float(mean_excess / std_excess * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(
    returns_series: pd.Series,
    risk_free: float = 0.025,
) -> float:
    """소르티노비율을 계산한다.

    샤프비율과 유사하지만 하방 변동성만을 사용한다.

    Args:
        returns_series: 일별 수익률 시리즈 (DatetimeIndex)
        risk_free: 연간 무위험이자율 (기본 2.5%)

    Returns:
        소르티노비율 (float)
    """
    if returns_series.empty or len(returns_series) < 2:
        return 0.0

    risk_free_daily = risk_free / TRADING_DAYS_PER_YEAR
    excess_returns = returns_series - risk_free_daily

    mean_excess = excess_returns.mean()

    # 하방 변동성: 무위험이자율 이하의 수익률만 대상
    downside_returns = returns_series[returns_series < risk_free_daily]

    if downside_returns.empty:
        # 하방 변동성이 없으면 무한대 (실용적으로 큰 값 반환)
        return float("inf") if mean_excess > 0 else 0.0

    downside_std = np.sqrt(np.mean((downside_returns - risk_free_daily) ** 2))

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return float(mean_excess / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def calmar_ratio(returns_series: pd.Series) -> float:
    """칼마비율을 계산한다.

    CAGR을 최대 낙폭(절대값)으로 나눈 값이다.

    Args:
        returns_series: 일별 수익률 시리즈 (DatetimeIndex)

    Returns:
        칼마비율 (float)
    """
    if returns_series.empty:
        return 0.0

    annual_return = cagr(returns_series)
    mdd = max_drawdown(returns_series)

    if mdd == 0:
        return float("inf") if annual_return > 0 else 0.0

    return float(annual_return / abs(mdd))


def win_rate(
    returns_series: pd.Series,
    freq: str = "monthly",
) -> float:
    """승률을 계산한다.

    지정된 주기(월별/주별)에서 양의 수익률을 기록한 비율이다.

    Args:
        returns_series: 일별 수익률 시리즈 (DatetimeIndex)
        freq: 집계 주기 - "monthly" 또는 "weekly" (기본 "monthly")

    Returns:
        승률 (0 ~ 1, 예: 0.6 = 60%)
    """
    if returns_series.empty:
        return 0.0

    # 누적 수익률로 변환하여 주기별 수익률 계산
    cumulative = (1 + returns_series).cumprod()

    if freq == "monthly":
        period_returns = cumulative.resample("ME").last().pct_change().dropna()
    elif freq == "weekly":
        period_returns = cumulative.resample("W").last().pct_change().dropna()
    else:
        raise ValueError(f"지원하지 않는 주기: {freq}. 'monthly' 또는 'weekly' 중 선택하세요.")

    if period_returns.empty:
        return 0.0

    wins = (period_returns > 0).sum()
    total = len(period_returns)

    return float(wins / total)


def turnover(portfolio_weights_list: list[dict[str, float]]) -> float:
    """포트폴리오 회전율을 계산한다.

    연속된 리밸런싱 시점 간 비중 변화의 평균이다.
    회전율 = mean(sum(|w_new - w_old|) / 2) for each rebalance

    Args:
        portfolio_weights_list: 리밸런싱 시점별 {ticker: weight} 리스트

    Returns:
        평균 회전율 (0 ~ 1)
    """
    if len(portfolio_weights_list) < 2:
        return 0.0

    turnover_values: list[float] = []

    for i in range(1, len(portfolio_weights_list)):
        prev_weights = portfolio_weights_list[i - 1]
        curr_weights = portfolio_weights_list[i]

        # 모든 종목 코드 합집합
        all_tickers = set(prev_weights.keys()) | set(curr_weights.keys())

        total_change = sum(
            abs(curr_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
            for t in all_tickers
        )

        # 편도 회전율 (매수 또는 매도 한쪽만)
        turnover_values.append(total_change / 2)

    return float(np.mean(turnover_values))


def tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """추적오차를 계산한다.

    포트폴리오 수익률과 벤치마크 수익률 차이의 표준편차를 연율화한 값이다.

    Args:
        returns: 포트폴리오 일별 수익률 시리즈
        benchmark_returns: 벤치마크 일별 수익률 시리즈

    Returns:
        추적오차 (연율화, float)
    """
    if returns.empty or benchmark_returns.empty:
        return 0.0

    # 공통 날짜만 사용
    common_idx = returns.index.intersection(benchmark_returns.index)
    if common_idx.empty or len(common_idx) < 2:
        return 0.0

    active_returns = returns.loc[common_idx] - benchmark_returns.loc[common_idx]

    return float(active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """정보비율을 계산한다.

    초과수익률의 평균을 추적오차로 나눈 값이다.

    Args:
        returns: 포트폴리오 일별 수익률 시리즈
        benchmark_returns: 벤치마크 일별 수익률 시리즈

    Returns:
        정보비율 (float)
    """
    if returns.empty or benchmark_returns.empty:
        return 0.0

    # 공통 날짜만 사용
    common_idx = returns.index.intersection(benchmark_returns.index)
    if common_idx.empty or len(common_idx) < 2:
        return 0.0

    active_returns = returns.loc[common_idx] - benchmark_returns.loc[common_idx]
    te = active_returns.std()

    if te == 0 or np.isnan(te):
        return 0.0

    # 연율화된 정보비율
    return float(active_returns.mean() / te * np.sqrt(TRADING_DAYS_PER_YEAR))
