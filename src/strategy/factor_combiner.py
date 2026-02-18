"""팩터 결합 모듈.

밸류, 모멘텀 등 다양한 팩터의 스코어를 결합하여
통합 점수를 생성하는 유틸리티를 제공한다.
Z-Score 결합과 순위(Rank) 결합 두 가지 방법을 지원한다.
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def combine_zscore(
    value_scores: pd.Series,
    momentum_scores: pd.Series,
    value_weight: float = 0.5,
    momentum_weight: float = 0.5,
) -> pd.Series:
    """Z-Score 기반 팩터 결합.

    각 팩터 스코어를 Z-Score로 표준화한 후 가중 합산한다.
    이상치는 Z-Score +/-3으로 클리핑한다.

    Args:
        value_scores: 밸류 팩터 스코어 (index=ticker)
        momentum_scores: 모멘텀 팩터 스코어 (index=ticker)
        value_weight: 밸류 가중치 (기본 0.5)
        momentum_weight: 모멘텀 가중치 (기본 0.5)

    Returns:
        결합된 스코어 (pd.Series, index=ticker)
    """
    # 공통 종목만 대상
    common_tickers = value_scores.index.intersection(momentum_scores.index)
    if common_tickers.empty:
        logger.warning("공통 종목 없음: 빈 시리즈 반환")
        return pd.Series(dtype=float)

    val = value_scores.loc[common_tickers].copy()
    mom = momentum_scores.loc[common_tickers].copy()

    # NaN 제거
    valid_mask = val.notna() & mom.notna()
    val = val[valid_mask]
    mom = mom[valid_mask]

    if val.empty:
        logger.warning("유효한 공통 종목 없음: 빈 시리즈 반환")
        return pd.Series(dtype=float)

    # Z-Score 표준화
    val_z = _zscore_with_clip(val)
    mom_z = _zscore_with_clip(mom)

    # 가중 합산
    combined = val_z * value_weight + mom_z * momentum_weight

    logger.info(
        f"Z-Score 결합 완료: {len(combined)}개 종목, "
        f"value_weight={value_weight}, momentum_weight={momentum_weight}"
    )

    return combined


def combine_rank(
    value_scores: pd.Series,
    momentum_scores: pd.Series,
    value_weight: float = 0.5,
    momentum_weight: float = 0.5,
) -> pd.Series:
    """순위(Rank) 기반 팩터 결합.

    각 팩터 스코어를 백분위 순위(percentile rank)로 변환한 후 가중 합산한다.

    Args:
        value_scores: 밸류 팩터 스코어 (index=ticker)
        momentum_scores: 모멘텀 팩터 스코어 (index=ticker)
        value_weight: 밸류 가중치 (기본 0.5)
        momentum_weight: 모멘텀 가중치 (기본 0.5)

    Returns:
        결합된 스코어 (pd.Series, index=ticker)
    """
    # 공통 종목만 대상
    common_tickers = value_scores.index.intersection(momentum_scores.index)
    if common_tickers.empty:
        logger.warning("공통 종목 없음: 빈 시리즈 반환")
        return pd.Series(dtype=float)

    val = value_scores.loc[common_tickers].copy()
    mom = momentum_scores.loc[common_tickers].copy()

    # NaN 제거
    valid_mask = val.notna() & mom.notna()
    val = val[valid_mask]
    mom = mom[valid_mask]

    if val.empty:
        logger.warning("유효한 공통 종목 없음: 빈 시리즈 반환")
        return pd.Series(dtype=float)

    # 백분위 순위 (0~1 범위, 높을수록 좋음)
    val_pctrank = val.rank(pct=True)
    mom_pctrank = mom.rank(pct=True)

    # 가중 합산
    combined = val_pctrank * value_weight + mom_pctrank * momentum_weight

    logger.info(
        f"Rank 결합 완료: {len(combined)}개 종목, "
        f"value_weight={value_weight}, momentum_weight={momentum_weight}"
    )

    return combined


def _zscore_with_clip(series: pd.Series, clip_limit: float = 3.0) -> pd.Series:
    """Z-Score 표준화 후 이상치 클리핑.

    Args:
        series: 원본 시리즈
        clip_limit: 클리핑 한계값 (기본 +-3.0)

    Returns:
        Z-Score 변환 및 클리핑된 시리즈
    """
    mean = series.mean()
    std = series.std()

    if std == 0 or np.isnan(std):
        logger.warning("표준편차가 0이므로 Z-Score를 0으로 설정")
        return pd.Series(0.0, index=series.index)

    z = (series - mean) / std
    z = z.clip(lower=-clip_limit, upper=clip_limit)

    return z
