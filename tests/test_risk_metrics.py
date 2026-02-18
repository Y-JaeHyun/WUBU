"""리스크 지표 모듈(src/report/risk_metrics.py) 테스트.

VaR, CVaR, 분산비율, 팩터 노출도, 리스크 기여도 등을 검증한다.

src/report/risk_metrics.py가 아직 구현되지 않았으면 테스트를 스킵한다.
"""

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _try_import_risk_metrics():
    """risk_metrics 모듈 함수들을 임포트한다."""
    try:
        import src.report.risk_metrics as rm
        return rm
    except ImportError:
        return None


def _make_returns(n=252, seed=42, mean=0.0005, std=0.02):
    """일별 수익률 Series를 생성한다."""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    returns = np.random.randn(n) * std + mean
    return pd.Series(returns, index=dates, name="daily_return")


def _make_portfolio_weights(n_assets=5):
    """포트폴리오 비중 dict를 생성한다."""
    tickers = [f"{i:06d}" for i in range(1, n_assets + 1)]
    weight = 1.0 / n_assets
    return {t: weight for t in tickers}


def _make_returns_matrix(n_assets=5, n_days=252, seed=42):
    """다자산 수익률 DataFrame을 생성한다."""
    np.random.seed(seed)
    tickers = [f"{i:06d}" for i in range(1, n_assets + 1)]
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    returns = np.random.randn(n_days, n_assets) * 0.02
    return pd.DataFrame(returns, index=dates, columns=tickers)


# ===================================================================
# 리스크 지표 검증
# ===================================================================

class TestRiskMetrics:
    """리스크 지표 모듈 검증."""

    def test_var_historical(self):
        """Historical VaR이 올바르게 계산된다."""
        rm = _try_import_risk_metrics()
        if rm is None:
            pytest.skip("risk_metrics 모듈이 아직 구현되지 않았습니다.")

        if not hasattr(rm, "value_at_risk"):
            pytest.skip("value_at_risk 함수를 찾을 수 없습니다.")

        returns = _make_returns()
        var = rm.value_at_risk(returns, method="historical", confidence=0.95)

        assert isinstance(var, (float, np.floating)), "VaR은 숫자여야 합니다."

    def test_var_parametric(self):
        """Parametric VaR이 올바르게 계산된다."""
        rm = _try_import_risk_metrics()
        if rm is None:
            pytest.skip("risk_metrics 모듈이 아직 구현되지 않았습니다.")

        if not hasattr(rm, "value_at_risk"):
            pytest.skip("value_at_risk 함수를 찾을 수 없습니다.")

        returns = _make_returns()
        var = rm.value_at_risk(returns, method="parametric", confidence=0.95)

        assert isinstance(var, (float, np.floating)), "VaR은 숫자여야 합니다."

    def test_var_positive(self):
        """VaR은 양수(손실 크기)이다."""
        rm = _try_import_risk_metrics()
        if rm is None:
            pytest.skip("risk_metrics 모듈이 아직 구현되지 않았습니다.")

        if not hasattr(rm, "value_at_risk"):
            pytest.skip("value_at_risk 함수를 찾을 수 없습니다.")

        returns = _make_returns()
        var = rm.value_at_risk(returns, method="historical", confidence=0.95)

        assert var > 0, f"VaR은 양수(손실 크기)여야 합니다: {var}"

    def test_cvar_greater_than_var(self):
        """CVaR은 VaR 이상이다 (더 큰 손실 기대값)."""
        rm = _try_import_risk_metrics()
        if rm is None:
            pytest.skip("risk_metrics 모듈이 아직 구현되지 않았습니다.")

        if not hasattr(rm, "value_at_risk") or not hasattr(rm, "conditional_var"):
            pytest.skip("value_at_risk/conditional_var 함수를 찾을 수 없습니다.")

        returns = _make_returns()
        var = rm.value_at_risk(returns, method="historical", confidence=0.95)
        cvar = rm.conditional_var(returns, confidence=0.95)

        assert cvar >= var, (
            f"CVaR({cvar:.4f})은 VaR({var:.4f}) 이상이어야 합니다."
        )

    def test_cvar_basic(self):
        """CVaR이 올바르게 계산된다."""
        rm = _try_import_risk_metrics()
        if rm is None:
            pytest.skip("risk_metrics 모듈이 아직 구현되지 않았습니다.")

        if not hasattr(rm, "conditional_var"):
            pytest.skip("conditional_var 함수를 찾을 수 없습니다.")

        returns = _make_returns()
        cvar = rm.conditional_var(returns, confidence=0.95)

        assert isinstance(cvar, (float, np.floating)), "CVaR은 숫자여야 합니다."
        assert cvar > 0, f"CVaR은 양수여야 합니다: {cvar}"

    def test_diversification_ratio_single(self):
        """단일 자산 분산비율은 1.0이다."""
        rm = _try_import_risk_metrics()
        if rm is None:
            pytest.skip("risk_metrics 모듈이 아직 구현되지 않았습니다.")

        if not hasattr(rm, "diversification_ratio"):
            pytest.skip("diversification_ratio 함수가 없습니다.")

        # 단일 자산
        weights = np.array([1.0])
        cov = np.array([[0.04]])  # 분산 = 0.04

        dr = rm.diversification_ratio(weights, cov)

        assert dr == pytest.approx(1.0, abs=0.01), (
            f"단일 자산 분산비율은 1.0이어야 합니다: {dr}"
        )

    def test_diversification_ratio_diversified(self):
        """분산 포트폴리오의 DR은 1.0보다 크다."""
        rm = _try_import_risk_metrics()
        if rm is None:
            pytest.skip("risk_metrics 모듈이 아직 구현되지 않았습니다.")

        if not hasattr(rm, "diversification_ratio"):
            pytest.skip("diversification_ratio 함수가 없습니다.")

        # 상관계수가 낮은 2자산
        weights = np.array([0.5, 0.5])
        cov = np.array([
            [0.04, 0.005],
            [0.005, 0.04],
        ])

        dr = rm.diversification_ratio(weights, cov)

        assert dr > 1.0, (
            f"분산 포트폴리오의 DR은 1.0보다 커야 합니다: {dr}"
        )

    def test_factor_exposure(self):
        """팩터 노출도가 올바르게 계산된다."""
        rm = _try_import_risk_metrics()
        if rm is None:
            pytest.skip("risk_metrics 모듈이 아직 구현되지 않았습니다.")

        if not hasattr(rm, "factor_exposure"):
            pytest.skip("factor_exposure 함수가 없습니다.")

        np.random.seed(42)
        n = 10
        tickers = [f"{i:06d}" for i in range(1, n + 1)]

        # portfolio_weights: dict[str, float]
        portfolio_weights = {t: 1.0 / n for t in tickers}

        # factor_scores: dict[str, pd.Series]
        factor_scores = {
            "value": pd.Series(np.random.randn(n), index=tickers),
            "momentum": pd.Series(np.random.randn(n), index=tickers),
            "quality": pd.Series(np.random.randn(n), index=tickers),
        }

        exposure = rm.factor_exposure(portfolio_weights, factor_scores)

        assert isinstance(exposure, dict), (
            "팩터 노출도가 dict여야 합니다."
        )
        assert len(exposure) == 3, (
            f"팩터 수가 3개여야 합니다: {len(exposure)}"
        )
        for factor_name in ["value", "momentum", "quality"]:
            assert factor_name in exposure, (
                f"'{factor_name}' 팩터가 결과에 있어야 합니다."
            )

    def test_risk_contribution(self):
        """리스크 기여도 합이 포트폴리오 변동성과 같다."""
        rm = _try_import_risk_metrics()
        if rm is None:
            pytest.skip("risk_metrics 모듈이 아직 구현되지 않았습니다.")

        if not hasattr(rm, "risk_contribution"):
            pytest.skip("risk_contribution 함수가 없습니다.")

        np.random.seed(42)
        weights = np.array([0.3, 0.3, 0.4])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.01],
            [0.005, 0.01, 0.0625],
        ])

        rc = rm.risk_contribution(weights, cov)

        # RC 합 = sigma_p
        sigma_p = np.sqrt(weights @ cov @ weights)
        rc_sum = np.sum(rc)
        assert rc_sum == pytest.approx(sigma_p, abs=1e-6), (
            f"RC 합({rc_sum:.6f})이 sigma_p({sigma_p:.6f})와 같아야 합니다."
        )

    def test_empty_returns(self):
        """빈 수익률 데이터도 에러 없이 처리된다."""
        rm = _try_import_risk_metrics()
        if rm is None:
            pytest.skip("risk_metrics 모듈이 아직 구현되지 않았습니다.")

        empty_returns = pd.Series(dtype=float)

        if hasattr(rm, "value_at_risk"):
            try:
                var = rm.value_at_risk(empty_returns, confidence=0.95)
                # 빈 데이터에서 0.0 반환 허용
                assert var == 0.0, (
                    f"빈 데이터에서 VaR은 0.0이어야 합니다: {var}"
                )
            except (ValueError, ZeroDivisionError):
                # 빈 데이터에서 에러도 정상
                pass

        if hasattr(rm, "conditional_var"):
            try:
                cvar = rm.conditional_var(empty_returns, confidence=0.95)
                # 빈 데이터에서 0.0 반환 허용
                assert cvar == 0.0, (
                    f"빈 데이터에서 CVaR은 0.0이어야 합니다: {cvar}"
                )
            except (ValueError, ZeroDivisionError):
                pass
