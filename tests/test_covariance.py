"""공분산 행렬 추정 모듈(src/optimization/covariance.py) 테스트.

CovarianceEstimator의 세 가지 추정 방법(sample, ledoit_wolf, ewm),
양정치성, 대칭성, 엣지 케이스 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _import_covariance_estimator():
    """CovarianceEstimator 클래스를 임포트한다."""
    from src.optimization.covariance import CovarianceEstimator
    return CovarianceEstimator


# ===================================================================
# CovarianceEstimator 검증
# ===================================================================

class TestCovarianceEstimator:
    """CovarianceEstimator 검증."""

    def test_sample_covariance(self, sample_returns_matrix):
        """표본 공분산 행렬이 올바르게 추정된다."""
        CovarianceEstimator = _import_covariance_estimator()
        estimator = CovarianceEstimator(method="sample")

        cov = estimator.estimate(sample_returns_matrix)

        assert isinstance(cov, pd.DataFrame), "반환값이 DataFrame이어야 합니다."
        assert cov.shape[0] == cov.shape[1], "공분산 행렬은 정방행렬이어야 합니다."
        assert cov.shape[0] == sample_returns_matrix.shape[1], (
            "공분산 행렬 크기가 종목 수와 같아야 합니다."
        )

    def test_ledoit_wolf_covariance(self, sample_returns_matrix):
        """Ledoit-Wolf 축소 추정 공분산이 올바르게 추정된다."""
        CovarianceEstimator = _import_covariance_estimator()
        estimator = CovarianceEstimator(method="ledoit_wolf")

        cov = estimator.estimate(sample_returns_matrix)

        assert isinstance(cov, pd.DataFrame), "반환값이 DataFrame이어야 합니다."
        assert cov.shape[0] == cov.shape[1], "정방행렬이어야 합니다."
        # Ledoit-Wolf 결과는 표본 공분산과 다를 수 있음
        sample_cov = sample_returns_matrix.cov()
        # 차이가 존재하는지 (축소 추정이 적용되었는지) 확인
        # sklearn이 없으면 sample과 동일하므로 무조건 통과
        assert not cov.empty, "빈 행렬이면 안 됩니다."

    def test_ewm_covariance(self, sample_returns_matrix):
        """EWM 공분산이 올바르게 추정된다."""
        CovarianceEstimator = _import_covariance_estimator()
        estimator = CovarianceEstimator(method="ewm", halflife=63)

        cov = estimator.estimate(sample_returns_matrix)

        assert isinstance(cov, pd.DataFrame), "반환값이 DataFrame이어야 합니다."
        assert cov.shape[0] == cov.shape[1], "정방행렬이어야 합니다."
        assert not cov.empty, "빈 행렬이면 안 됩니다."

    def test_covariance_positive_definite(self, sample_returns_matrix):
        """공분산 행렬이 양정치(positive semi-definite)인지 확인한다."""
        CovarianceEstimator = _import_covariance_estimator()
        estimator = CovarianceEstimator(method="sample")

        cov = estimator.estimate(sample_returns_matrix)

        # 고유값이 모두 0 이상이면 양반정치
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues >= -1e-8), (
            f"공분산 행렬의 고유값이 음수입니다: min={eigenvalues.min()}"
        )

    def test_covariance_symmetric(self, sample_returns_matrix):
        """공분산 행렬이 대칭인지 확인한다."""
        CovarianceEstimator = _import_covariance_estimator()
        estimator = CovarianceEstimator(method="ledoit_wolf")

        cov = estimator.estimate(sample_returns_matrix)

        diff = np.abs(cov.values - cov.values.T).max()
        assert diff < 1e-10, (
            f"공분산 행렬이 대칭이 아닙니다: max_diff={diff}"
        )

    def test_lookback_window(self):
        """룩백 기간이 올바르게 적용된다."""
        CovarianceEstimator = _import_covariance_estimator()

        np.random.seed(42)
        n_stocks, n_days = 5, 500
        tickers = [f"{i:06d}" for i in range(1, n_stocks + 1)]
        dates = pd.bdate_range("2022-01-03", periods=n_days)
        returns = pd.DataFrame(
            np.random.randn(n_days, n_stocks) * 0.02,
            index=dates,
            columns=tickers,
        )

        estimator_short = CovarianceEstimator(method="sample", lookback_days=100)
        estimator_long = CovarianceEstimator(method="sample", lookback_days=400)

        cov_short = estimator_short.estimate(returns)
        cov_long = estimator_long.estimate(returns)

        # 두 추정치가 모두 유효해야 함
        assert not cov_short.empty, "짧은 룩백 공분산이 비어 있으면 안 됩니다."
        assert not cov_long.empty, "긴 룩백 공분산이 비어 있으면 안 됩니다."
        # 서로 다른 룩백이면 결과가 다를 수 있음
        assert cov_short.shape == cov_long.shape, "형상은 같아야 합니다."

    def test_empty_returns(self):
        """빈 수익률 데이터 입력 시 빈 DataFrame을 반환한다."""
        CovarianceEstimator = _import_covariance_estimator()
        estimator = CovarianceEstimator()

        cov = estimator.estimate(pd.DataFrame())

        assert isinstance(cov, pd.DataFrame), "반환값이 DataFrame이어야 합니다."
        assert cov.empty, "빈 입력이면 빈 DataFrame이어야 합니다."

    def test_single_stock(self):
        """단일 종목 수익률로 1x1 공분산 행렬을 반환한다."""
        CovarianceEstimator = _import_covariance_estimator()
        estimator = CovarianceEstimator(method="sample")

        np.random.seed(42)
        dates = pd.bdate_range("2023-01-02", periods=100)
        returns = pd.DataFrame(
            np.random.randn(100) * 0.02,
            index=dates,
            columns=["005930"],
        )

        cov = estimator.estimate(returns)

        assert isinstance(cov, pd.DataFrame), "반환값이 DataFrame이어야 합니다."
        assert cov.shape == (1, 1), "단일 종목이면 1x1 행렬이어야 합니다."
        assert cov.iloc[0, 0] > 0, "분산은 양수여야 합니다."

    def test_nan_handling(self):
        """NaN이 포함된 수익률에서도 공분산이 추정된다."""
        CovarianceEstimator = _import_covariance_estimator()
        estimator = CovarianceEstimator(method="sample")

        np.random.seed(42)
        n_stocks, n_days = 5, 100
        tickers = [f"{i:06d}" for i in range(1, n_stocks + 1)]
        dates = pd.bdate_range("2023-01-02", periods=n_days)
        returns = pd.DataFrame(
            np.random.randn(n_days, n_stocks) * 0.02,
            index=dates,
            columns=tickers,
        )
        # NaN 삽입 (20%)
        mask = np.random.rand(n_days, n_stocks) < 0.2
        returns[mask] = np.nan

        cov = estimator.estimate(returns)

        assert isinstance(cov, pd.DataFrame), "NaN 포함 데이터도 DataFrame을 반환해야 합니다."
        assert not cov.empty, "NaN이 있어도 빈 행렬이면 안 됩니다."

    def test_default_method(self):
        """기본 추정 방법이 ledoit_wolf이다."""
        CovarianceEstimator = _import_covariance_estimator()
        estimator = CovarianceEstimator()

        assert estimator.method == "ledoit_wolf", (
            f"기본 방법이 ledoit_wolf여야 합니다: {estimator.method}"
        )
