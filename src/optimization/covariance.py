"""공분산 행렬 추정 모듈.

수익률 데이터로부터 공분산 행렬을 추정하는 다양한 방법을 제공한다.
표본 공분산, Ledoit-Wolf 축소 추정, 지수가중 이동평균(EWM) 공분산을 지원한다.

사용 예시:
    estimator = CovarianceEstimator(method="ledoit_wolf", lookback_days=252)
    cov_matrix = estimator.estimate(returns_df)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 지원하는 공분산 추정 방법
VALID_METHODS = ("sample", "ledoit_wolf", "ewm")


class CovarianceEstimator:
    """공분산 행렬 추정기.

    수익률 DataFrame을 입력받아 공분산 행렬을 추정한다.
    세 가지 방법을 지원한다:
    - sample: 단순 표본 공분산 행렬
    - ledoit_wolf: Ledoit-Wolf 축소 추정 (추천)
    - ewm: 지수가중 이동평균 공분산

    Args:
        method: 추정 방법 - "sample", "ledoit_wolf" (추천), "ewm"
        lookback_days: 룩백 기간 (기본 252 = 약 1년)
        halflife: EWM 반감기 (기본 63 = 약 3개월). ewm 방법에서만 사용.
    """

    def __init__(
        self,
        method: str = "ledoit_wolf",
        lookback_days: int = 252,
        halflife: Optional[int] = None,
    ):
        method = method.lower()
        if method not in VALID_METHODS:
            raise ValueError(
                f"지원하지 않는 공분산 추정 방법: {method}. "
                f"{VALID_METHODS} 중 선택하세요."
            )

        self.method = method
        self.lookback_days = lookback_days
        self.halflife = halflife or 63

        logger.info(
            f"CovarianceEstimator 초기화: method={method}, "
            f"lookback_days={lookback_days}, halflife={self.halflife}"
        )

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """수익률 DataFrame으로부터 공분산 행렬을 추정한다.

        Args:
            returns: 일별 수익률 DataFrame (index=날짜, columns=종목코드)

        Returns:
            NxN 공분산 행렬 DataFrame (index=종목코드, columns=종목코드).
            빈 입력이면 빈 DataFrame 반환.
        """
        if returns.empty:
            logger.warning("빈 수익률 데이터: 빈 공분산 행렬 반환")
            return pd.DataFrame()

        # 룩백 기간 적용
        if len(returns) > self.lookback_days:
            returns = returns.iloc[-self.lookback_days:]

        # NaN이 많은 종목 제거 (50% 이상 NaN이면 제외)
        valid_ratio = returns.notna().mean()
        valid_columns = valid_ratio[valid_ratio >= 0.5].index.tolist()

        if not valid_columns:
            logger.warning("유효한 종목이 없음: 빈 공분산 행렬 반환")
            return pd.DataFrame()

        returns = returns[valid_columns].copy()

        # NaN을 0으로 채움 (남은 NaN)
        returns = returns.fillna(0.0)

        if len(returns) < 2:
            logger.warning("수익률 데이터 부족 (2일 미만): 빈 공분산 행렬 반환")
            return pd.DataFrame()

        if len(valid_columns) < 2:
            logger.warning("유효한 종목 수 부족 (2개 미만): 단일 종목 분산 반환")
            var_value = returns[valid_columns[0]].var()
            return pd.DataFrame(
                [[var_value]],
                index=valid_columns,
                columns=valid_columns,
            )

        # 추정 방법 선택
        if self.method == "sample":
            cov_matrix = self._sample(returns)
        elif self.method == "ledoit_wolf":
            cov_matrix = self._ledoit_wolf(returns)
        elif self.method == "ewm":
            cov_matrix = self._exponential_weighted(returns)
        else:
            cov_matrix = self._sample(returns)

        # 대칭성 보장
        cov_matrix = (cov_matrix + cov_matrix.T) / 2

        # 대각 원소가 0 이하이면 작은 양수로 보정
        min_var = 1e-10
        for col in cov_matrix.columns:
            if cov_matrix.loc[col, col] <= 0:
                cov_matrix.loc[col, col] = min_var

        logger.info(
            f"공분산 행렬 추정 완료: method={self.method}, "
            f"size={cov_matrix.shape[0]}x{cov_matrix.shape[1]}, "
            f"lookback={len(returns)}일"
        )

        return cov_matrix

    def _sample(self, returns: pd.DataFrame) -> pd.DataFrame:
        """표본 공분산 행렬을 계산한다.

        가장 단순한 방법. 종목 수가 관측 수보다 많으면 불안정할 수 있다.

        Args:
            returns: 일별 수익률 DataFrame

        Returns:
            표본 공분산 행렬 DataFrame
        """
        cov = returns.cov()
        logger.info(f"표본 공분산 행렬 계산: {cov.shape[0]}x{cov.shape[1]}")
        return cov

    def _ledoit_wolf(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Ledoit-Wolf 축소 추정 공분산 행렬을 계산한다.

        표본 공분산과 구조화된 추정치의 가중 평균으로 안정적인 추정을 제공한다.
        sklearn이 없으면 표본 공분산으로 fallback한다.

        Args:
            returns: 일별 수익률 DataFrame

        Returns:
            Ledoit-Wolf 공분산 행렬 DataFrame
        """
        try:
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf()
            lw.fit(returns.values)

            cov = pd.DataFrame(
                lw.covariance_,
                index=returns.columns,
                columns=returns.columns,
            )

            logger.info(
                f"Ledoit-Wolf 공분산 추정 완료: "
                f"shrinkage={lw.shrinkage_:.4f}"
            )
            return cov

        except ImportError:
            logger.warning(
                "sklearn이 설치되어 있지 않음: 표본 공분산으로 대체"
            )
            return self._sample(returns)

    def _exponential_weighted(self, returns: pd.DataFrame) -> pd.DataFrame:
        """지수가중 이동평균(EWM) 공분산 행렬을 계산한다.

        최근 데이터에 더 높은 가중치를 부여하여 시간 변동성을 반영한다.

        Args:
            returns: 일별 수익률 DataFrame

        Returns:
            EWM 공분산 행렬 DataFrame
        """
        n_assets = len(returns.columns)
        tickers = returns.columns.tolist()

        # EWM 가중치 생성
        n_obs = len(returns)
        decay = np.log(2) / self.halflife
        weights = np.exp(-decay * np.arange(n_obs)[::-1])
        weights = weights / weights.sum()

        # 가중 평균
        returns_values = returns.values
        weighted_mean = np.average(returns_values, axis=0, weights=weights)

        # 가중 공분산 행렬
        demeaned = returns_values - weighted_mean
        cov_matrix = np.zeros((n_assets, n_assets))

        for t in range(n_obs):
            cov_matrix += weights[t] * np.outer(demeaned[t], demeaned[t])

        # 비편향 보정 계수
        w_sum_sq = np.sum(weights ** 2)
        correction = 1.0 / (1.0 - w_sum_sq) if (1.0 - w_sum_sq) > 0 else 1.0
        cov_matrix *= correction

        cov = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)

        logger.info(
            f"EWM 공분산 추정 완료: halflife={self.halflife}, "
            f"유효 관측수={n_obs}"
        )
        return cov

    def get_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """상관계수 행렬을 계산한다.

        공분산 행렬에서 상관계수 행렬을 도출한다.

        Args:
            returns: 일별 수익률 DataFrame

        Returns:
            NxN 상관계수 행렬 DataFrame
        """
        cov_matrix = self.estimate(returns)

        if cov_matrix.empty:
            return pd.DataFrame()

        # 표준편차 벡터
        std = np.sqrt(np.diag(cov_matrix.values))
        std[std == 0] = 1e-10  # 0 나눗셈 방지

        # 상관계수 행렬
        corr_matrix = cov_matrix.values / np.outer(std, std)

        # -1 ~ 1 클리핑
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        # 대각선 = 1
        np.fill_diagonal(corr_matrix, 1.0)

        return pd.DataFrame(
            corr_matrix,
            index=cov_matrix.index,
            columns=cov_matrix.columns,
        )
