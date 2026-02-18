"""고급 리스크 지표 모듈.

VaR, CVaR, 분산화 비율, 팩터 노출 등 포트폴리오 리스크 분석에
필요한 지표를 계산한다.

기존 metrics.py가 수익률 기반 성과 지표(CAGR, MDD, 샤프 등)를 제공하는 반면,
이 모듈은 리스크 관리에 초점을 맞춘 지표를 보완적으로 제공한다.

사용 예시:
    var_95 = value_at_risk(returns, confidence=0.95)
    cvar_95 = conditional_var(returns, confidence=0.95)
    dr = diversification_ratio(weights, covariance)
    factor_exp = factor_exposure(portfolio_weights, factor_scores)
    rc = risk_contribution(weights, covariance)
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """VaR(Value at Risk)를 계산한다.

    지정 신뢰수준에서 최대 예상 손실을 산출한다.
    양수로 반환한다 (예: 0.03 = 3% 최대 손실).

    Args:
        returns: 일별 수익률 시리즈
        confidence: 신뢰수준 (기본 0.95 = 95%)
        method: 계산 방법 - "historical" (역사적) 또는 "parametric" (정규분포 가정)

    Returns:
        VaR 값 (양수, float).
        빈 데이터이면 0.0 반환.
    """
    if returns.empty or len(returns) < 2:
        logger.warning("빈 수익률 데이터: VaR=0.0 반환")
        return 0.0

    returns_clean = returns.dropna()
    if returns_clean.empty:
        return 0.0

    method = method.lower()

    if method == "historical":
        # 역사적 VaR: 하위 (1-confidence) 분위수
        var = float(-np.percentile(returns_clean, (1 - confidence) * 100))

    elif method == "parametric":
        # 정규분포 가정 VaR
        from scipy.stats import norm
        mean_ret = returns_clean.mean()
        std_ret = returns_clean.std()

        if std_ret == 0 or np.isnan(std_ret):
            return 0.0

        z_score = norm.ppf(1 - confidence)
        var = float(-(mean_ret + z_score * std_ret))

    else:
        raise ValueError(
            f"지원하지 않는 VaR 방법: {method}. "
            "'historical' 또는 'parametric' 중 선택하세요."
        )

    logger.info(
        f"VaR 계산: confidence={confidence:.0%}, method={method}, "
        f"VaR={var:.4f}"
    )

    return max(var, 0.0)


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """CVaR(Conditional Value at Risk, Expected Shortfall)를 계산한다.

    VaR를 초과하는 손실의 기대값이다. VaR보다 보수적인 리스크 측정치로,
    꼬리 리스크(tail risk)를 더 잘 반영한다.

    양수로 반환한다 (예: 0.05 = 5% 기대 손실).

    Args:
        returns: 일별 수익률 시리즈
        confidence: 신뢰수준 (기본 0.95 = 95%)

    Returns:
        CVaR 값 (양수, float).
        빈 데이터이면 0.0 반환.
    """
    if returns.empty or len(returns) < 2:
        logger.warning("빈 수익률 데이터: CVaR=0.0 반환")
        return 0.0

    returns_clean = returns.dropna()
    if returns_clean.empty:
        return 0.0

    # VaR 임계값
    var_threshold = np.percentile(returns_clean, (1 - confidence) * 100)

    # VaR 이하의 수익률 평균 (꼬리 기대 손실)
    tail_returns = returns_clean[returns_clean <= var_threshold]

    if tail_returns.empty:
        # VaR 이하 데이터가 없으면 VaR 자체를 반환
        cvar = float(-var_threshold)
    else:
        cvar = float(-tail_returns.mean())

    logger.info(
        f"CVaR 계산: confidence={confidence:.0%}, CVaR={cvar:.4f}"
    )

    return max(cvar, 0.0)


def diversification_ratio(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """분산화 비율(Diversification Ratio)을 계산한다.

    DR = (w^T * sigma) / sqrt(w^T * Sigma * w)

    개별 자산 변동성의 가중합을 포트폴리오 변동성으로 나눈 비율이다.
    DR > 1이면 분산화 효과가 있으며, 높을수록 분산 투자가 잘 되어 있다.

    Args:
        weights: 비중 벡터 (n_assets,)
        covariance: NxN 공분산 행렬 (numpy array)

    Returns:
        분산화 비율 (float, >= 1.0).
        유효하지 않은 입력이면 1.0 반환.
    """
    if len(weights) == 0 or covariance.size == 0:
        logger.warning("빈 입력 데이터: DR=1.0 반환")
        return 1.0

    weights = np.asarray(weights, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    # 개별 변동성
    individual_vols = np.sqrt(np.diag(covariance))

    # 가중 변동성 합
    weighted_vol_sum = float(weights @ individual_vols)

    # 포트폴리오 변동성
    port_var = float(weights @ covariance @ weights)
    if port_var <= 0:
        logger.warning("포트폴리오 분산 <= 0: DR=1.0 반환")
        return 1.0

    port_vol = np.sqrt(port_var)

    if port_vol == 0:
        return 1.0

    dr = weighted_vol_sum / port_vol

    logger.info(
        f"분산화 비율: DR={dr:.4f}, "
        f"가중변동성합={weighted_vol_sum:.4f}, "
        f"포트폴리오변동성={port_vol:.4f}"
    )

    return float(max(dr, 1.0))


def factor_exposure(
    portfolio_weights: dict[str, float],
    factor_scores: dict[str, pd.Series],
) -> dict[str, float]:
    """포트폴리오의 팩터별 노출도를 계산한다.

    각 팩터에 대해 포트폴리오 보유 종목의 가중 평균 팩터 스코어를 산출한다.
    양의 노출도는 해당 팩터에 대한 롱(매수) 포지션을 의미한다.

    Args:
        portfolio_weights: {종목코드: 비중} 딕셔너리
        factor_scores: {팩터명: Series(index=ticker, values=score)} 딕셔너리
            예: {"value": value_scores, "momentum": mom_scores}

    Returns:
        {팩터명: 노출도} 딕셔너리.
        빈 입력이면 빈 딕셔너리 반환.
    """
    if not portfolio_weights or not factor_scores:
        logger.warning("빈 입력 데이터: 빈 팩터 노출도 반환")
        return {}

    exposures = {}
    portfolio_tickers = set(portfolio_weights.keys())

    for factor_name, scores in factor_scores.items():
        if scores.empty:
            exposures[factor_name] = 0.0
            continue

        # 포트폴리오 종목과 팩터 스코어의 교집합
        common_tickers = portfolio_tickers.intersection(set(scores.index))

        if not common_tickers:
            exposures[factor_name] = 0.0
            continue

        # 가중 평균 팩터 스코어
        weighted_score = sum(
            portfolio_weights[t] * float(scores.loc[t])
            for t in common_tickers
            if not np.isnan(scores.loc[t])
        )

        # 비중 합으로 정규화 (포트폴리오 일부만 커버할 수 있으므로)
        weight_sum = sum(
            portfolio_weights[t]
            for t in common_tickers
            if not np.isnan(scores.loc[t])
        )

        if weight_sum > 0:
            exposures[factor_name] = float(weighted_score / weight_sum)
        else:
            exposures[factor_name] = 0.0

    logger.info(
        f"팩터 노출도 계산: {len(exposures)}개 팩터, "
        f"{len(portfolio_tickers)}개 종목"
    )

    return exposures


def risk_contribution(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """각 자산의 리스크 기여도를 계산한다.

    RC_i = w_i * (Sigma @ w)_i / sigma_p

    리스크 기여도의 합은 포트폴리오 전체 리스크(변동성)와 동일하다.

    Args:
        weights: 비중 벡터 (n_assets,)
        covariance: NxN 공분산 행렬 (numpy array)

    Returns:
        리스크 기여도 벡터 (n_assets,).
        유효하지 않은 입력이면 빈 배열 반환.
    """
    if len(weights) == 0 or covariance.size == 0:
        logger.warning("빈 입력 데이터: 빈 리스크 기여도 반환")
        return np.array([])

    weights = np.asarray(weights, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    # 포트폴리오 분산
    port_var = float(weights @ covariance @ weights)
    if port_var <= 0:
        logger.warning("포트폴리오 분산 <= 0: 0 벡터 반환")
        return np.zeros(len(weights))

    port_vol = np.sqrt(port_var)

    # 한계 리스크: (Sigma @ w)
    marginal_risk = covariance @ weights

    # 리스크 기여도: w_i * marginal_risk_i / sigma_p
    rc = weights * marginal_risk / port_vol

    logger.info(
        f"리스크 기여도 계산: {len(rc)}개 자산, "
        f"총 리스크={port_vol:.6f}, "
        f"RC 합={rc.sum():.6f}"
    )

    return rc


def risk_contribution_pct(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """각 자산의 리스크 기여 비율을 계산한다.

    전체 리스크 대비 각 자산의 리스크 기여 비율(%)을 반환한다.
    비율의 합은 1.0이다.

    Args:
        weights: 비중 벡터 (n_assets,)
        covariance: NxN 공분산 행렬 (numpy array)

    Returns:
        리스크 기여 비율 벡터 (n_assets,, 합=1.0).
        유효하지 않은 입력이면 빈 배열 반환.
    """
    rc = risk_contribution(weights, covariance)

    if len(rc) == 0:
        return np.array([])

    rc_sum = rc.sum()
    if rc_sum == 0:
        return np.zeros(len(rc))

    return rc / rc_sum
