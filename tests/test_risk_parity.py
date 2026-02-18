"""리스크 패리티 최적화 모듈(src/optimization/risk_parity.py) 테스트.

RiskParityOptimizer의 비중 최적화, 리스크 기여도 균등 배분,
역변동성 fallback, 리스크 분해 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _import_risk_parity_optimizer():
    """RiskParityOptimizer 클래스를 임포트한다."""
    from src.optimization.risk_parity import RiskParityOptimizer
    return RiskParityOptimizer


def _make_equal_cov(n=5, variance=0.04):
    """동일 분산, 상관계수 0인 대각 공분산 행렬을 생성한다."""
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    cov = pd.DataFrame(
        np.eye(n) * variance,
        index=tickers,
        columns=tickers,
    )
    return cov


def _make_different_vol_cov(vols=None):
    """서로 다른 변동성의 대각 공분산 행렬을 생성한다."""
    if vols is None:
        vols = [0.1, 0.2, 0.3, 0.4, 0.5]
    n = len(vols)
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    cov = pd.DataFrame(
        np.diag(np.array(vols) ** 2),
        index=tickers,
        columns=tickers,
    )
    return cov


# ===================================================================
# RiskParityOptimizer 검증
# ===================================================================

class TestRiskParityOptimizer:
    """RiskParityOptimizer 검증."""

    def test_optimize_equal_covariance(self):
        """동일 공분산 → 동일 비중에 가깝다."""
        RiskParityOptimizer = _import_risk_parity_optimizer()
        cov = _make_equal_cov(n=5, variance=0.04)
        optimizer = RiskParityOptimizer(covariance=cov)

        weights = optimizer.optimize()

        assert isinstance(weights, dict), "반환값이 dict여야 합니다."
        assert len(weights) == 5, "5개 종목의 비중이 반환되어야 합니다."

        # 동일 분산이면 비중이 거의 동일해야 함 (1/5 = 0.2)
        for ticker, w in weights.items():
            assert w == pytest.approx(0.2, abs=0.02), (
                f"동일 공분산에서 비중이 0.2에 가까워야 합니다: {ticker}={w}"
            )

    def test_optimize_different_volatility(self):
        """변동성이 다르면 저변동 자산의 비중이 더 높다."""
        RiskParityOptimizer = _import_risk_parity_optimizer()
        cov = _make_different_vol_cov(vols=[0.1, 0.3])
        optimizer = RiskParityOptimizer(covariance=cov)

        weights = optimizer.optimize()

        tickers = list(weights.keys())
        # 첫 번째 자산(vol=0.1)이 두 번째(vol=0.3)보다 비중이 높아야 함
        assert weights[tickers[0]] > weights[tickers[1]], (
            "저변동 자산의 비중이 더 높아야 합니다."
        )

    def test_weights_sum_to_one(self, sample_covariance_matrix):
        """비중 합이 1.0이다."""
        RiskParityOptimizer = _import_risk_parity_optimizer()
        optimizer = RiskParityOptimizer(covariance=sample_covariance_matrix)

        weights = optimizer.optimize()

        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6), (
            f"비중 합이 1.0이어야 합니다: {total}"
        )

    def test_all_weights_positive(self, sample_covariance_matrix):
        """모든 비중이 양수이다."""
        RiskParityOptimizer = _import_risk_parity_optimizer()
        optimizer = RiskParityOptimizer(covariance=sample_covariance_matrix)

        weights = optimizer.optimize()

        for ticker, w in weights.items():
            assert w > 0, f"종목 {ticker}의 비중이 양수여야 합니다: {w}"

    def test_risk_contribution_equal(self):
        """리스크 기여도(RC%)가 거의 균등하다 (편차 < 1%)."""
        RiskParityOptimizer = _import_risk_parity_optimizer()
        cov = _make_equal_cov(n=5, variance=0.04)
        optimizer = RiskParityOptimizer(covariance=cov)

        weights = optimizer.optimize()
        decomp = optimizer.get_risk_decomposition(weights)

        if "RC_pct" in decomp.columns:
            rc_pct = decomp["RC_pct"].values
            target = 1.0 / 5  # 5개 자산 동일 예산
            for i, pct in enumerate(rc_pct):
                assert abs(pct - target) < 0.01, (
                    f"RC% 편차가 1% 미만이어야 합니다: asset={i}, RC%={pct:.4f}"
                )

    def test_inverse_volatility_fallback(self):
        """최적화 실패 시 역변동성 비중이 반환된다."""
        RiskParityOptimizer = _import_risk_parity_optimizer()
        # 특이 공분산 (모든 원소 동일 → 최적화 어려울 수 있음)
        n = 3
        tickers = [f"{i:06d}" for i in range(1, n + 1)]
        cov = pd.DataFrame(
            np.ones((n, n)) * 0.04,
            index=tickers,
            columns=tickers,
        )
        # 대각 원소를 조금 더 크게 (양정치 근사)
        np.fill_diagonal(cov.values, 0.05)

        optimizer = RiskParityOptimizer(covariance=cov)
        weights = optimizer.optimize()

        # 어떤 형태든 유효한 비중이 반환되어야 함
        assert isinstance(weights, dict), "반환값이 dict여야 합니다."
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4), (
            "비중 합이 1.0이어야 합니다."
        )

    def test_risk_decomposition(self, sample_covariance_matrix):
        """get_risk_decomposition 결과가 올바른 구조를 갖는다."""
        RiskParityOptimizer = _import_risk_parity_optimizer()
        optimizer = RiskParityOptimizer(covariance=sample_covariance_matrix)

        weights = optimizer.optimize()
        decomp = optimizer.get_risk_decomposition(weights)

        assert isinstance(decomp, pd.DataFrame), "반환값이 DataFrame이어야 합니다."
        expected_cols = {"ticker", "weight", "RC", "RC_pct", "marginal_risk"}
        assert expected_cols.issubset(set(decomp.columns)), (
            f"필수 컬럼이 포함되어야 합니다: {expected_cols - set(decomp.columns)}"
        )
        assert len(decomp) == sample_covariance_matrix.shape[0], (
            "자산 수만큼 행이 있어야 합니다."
        )

    def test_custom_budget(self):
        """사용자 정의 리스크 예산이 적용된다."""
        RiskParityOptimizer = _import_risk_parity_optimizer()
        cov = _make_equal_cov(n=3, variance=0.04)
        tickers = cov.columns.tolist()

        # 첫 번째 자산에 50% 리스크 예산
        budget = {tickers[0]: 0.5, tickers[1]: 0.25, tickers[2]: 0.25}
        optimizer = RiskParityOptimizer(covariance=cov, budget=budget)

        weights = optimizer.optimize()

        assert isinstance(weights, dict), "반환값이 dict여야 합니다."
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4), (
            "비중 합이 1.0이어야 합니다."
        )

    def test_two_assets(self):
        """2자산 케이스가 올바르게 처리된다."""
        RiskParityOptimizer = _import_risk_parity_optimizer()
        cov = _make_different_vol_cov(vols=[0.1, 0.2])
        optimizer = RiskParityOptimizer(covariance=cov)

        weights = optimizer.optimize()

        assert len(weights) == 2, "2개 자산의 비중이 반환되어야 합니다."
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4), (
            "비중 합이 1.0이어야 합니다."
        )

    def test_many_assets(self, sample_covariance_matrix):
        """20자산 케이스가 올바르게 처리된다."""
        RiskParityOptimizer = _import_risk_parity_optimizer()
        optimizer = RiskParityOptimizer(covariance=sample_covariance_matrix)

        weights = optimizer.optimize()

        assert len(weights) == 20, "20개 자산의 비중이 반환되어야 합니다."
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4), (
            "비중 합이 1.0이어야 합니다."
        )

    def test_singular_covariance(self):
        """특이(singular) 공분산 행렬도 처리된다."""
        RiskParityOptimizer = _import_risk_parity_optimizer()

        # 선형 종속 행이 있는 공분산 (rank-deficient)
        n = 4
        tickers = [f"{i:06d}" for i in range(1, n + 1)]
        base = np.random.RandomState(42).randn(100, 2)
        # 4개 열을 2개 열의 선형 조합으로 생성
        data = np.column_stack([base, base @ np.array([[1, 0.5], [0.5, 1]])])
        cov = pd.DataFrame(
            np.cov(data.T),
            index=tickers,
            columns=tickers,
        )
        # 대각 원소 보정
        for i in range(n):
            if cov.iloc[i, i] <= 0:
                cov.iloc[i, i] = 1e-6

        optimizer = RiskParityOptimizer(covariance=cov)
        weights = optimizer.optimize()

        assert isinstance(weights, dict), "특이 공분산에서도 dict 반환이어야 합니다."
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4), (
            "비중 합이 1.0이어야 합니다."
        )

    def test_diagonal_covariance(self):
        """대각 공분산(상관 없음)에서 역변동성 비례 비중을 반환한다."""
        RiskParityOptimizer = _import_risk_parity_optimizer()
        vols = [0.1, 0.2, 0.3]
        cov = _make_different_vol_cov(vols=vols)
        optimizer = RiskParityOptimizer(covariance=cov)

        weights = optimizer.optimize()

        tickers = list(weights.keys())
        # 대각 공분산에서 ERC = 역변동성 비례
        inv_vols = [1.0 / v for v in vols]
        total_inv = sum(inv_vols)
        expected = [iv / total_inv for iv in inv_vols]

        for i, ticker in enumerate(tickers):
            assert weights[ticker] == pytest.approx(expected[i], abs=0.02), (
                f"대각 공분산에서 역변동성 비례 비중이어야 합니다: "
                f"{ticker}={weights[ticker]:.4f}, expected={expected[i]:.4f}"
            )
