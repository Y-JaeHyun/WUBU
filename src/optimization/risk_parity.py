"""ERC 리스크 패리티 최적화 모듈.

Equal Risk Contribution(ERC) 방식으로 각 자산의 리스크 기여도가
동일(또는 지정된 비율)하도록 포트폴리오 비중을 최적화한다.

사용 예시:
    optimizer = RiskParityOptimizer(covariance=cov_df, budget=None)
    weights = optimizer.optimize()
    risk_decomp = optimizer.get_risk_decomposition(weights)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskParityOptimizer:
    """Equal Risk Contribution 최적화기.

    각 자산의 리스크 기여도(Risk Contribution)가 목표 비율과 동일하도록
    비중을 최적화한다. 기본적으로 모든 자산이 동일한 리스크를 기여한다.

    최적화 실패 시 역변동성(Inverse Volatility) 비중으로 fallback한다.

    Args:
        covariance: NxN 공분산 행렬 DataFrame (index=종목코드, columns=종목코드)
        budget: 자산별 리스크 예산 딕셔너리 (None이면 동일 비중 예산).
                예: {'005930': 0.3, '000660': 0.3, '035720': 0.4}
                비율의 합은 1.0이어야 한다.
    """

    def __init__(
        self,
        covariance: pd.DataFrame,
        budget: dict[str, float] | None = None,
    ):
        if covariance.empty:
            raise ValueError("공분산 행렬이 비어 있습니다.")

        self.covariance = covariance
        self.tickers = covariance.columns.tolist()
        self.n_assets = len(self.tickers)
        self.cov_matrix = covariance.values.astype(float)

        # 리스크 예산 설정
        if budget is not None:
            # budget 키와 covariance 인덱스 매핑
            self.budget = np.array(
                [budget.get(t, 1.0 / self.n_assets) for t in self.tickers]
            )
            # 정규화
            budget_sum = self.budget.sum()
            if budget_sum > 0:
                self.budget = self.budget / budget_sum
            else:
                self.budget = np.ones(self.n_assets) / self.n_assets
        else:
            self.budget = np.ones(self.n_assets) / self.n_assets

        logger.info(
            f"RiskParityOptimizer 초기화: {self.n_assets}개 자산, "
            f"budget={'동일' if budget is None else '사용자 지정'}"
        )

    def _portfolio_risk(self, w: np.ndarray) -> float:
        """포트폴리오 전체 리스크(변동성)를 계산한다.

        sigma_p = sqrt(w^T * Sigma * w)

        Args:
            w: 비중 벡터 (n_assets,)

        Returns:
            포트폴리오 변동성 (float)
        """
        port_var = w @ self.cov_matrix @ w
        if port_var <= 0:
            return 1e-10
        return np.sqrt(port_var)

    def _risk_contribution(self, w: np.ndarray) -> np.ndarray:
        """각 자산의 리스크 기여도(RC)를 계산한다.

        RC_i = w_i * (Sigma @ w)_i / sigma_p

        Args:
            w: 비중 벡터 (n_assets,)

        Returns:
            리스크 기여도 벡터 (n_assets,)
        """
        sigma_p = self._portfolio_risk(w)
        marginal_risk = self.cov_matrix @ w  # (Sigma @ w)

        # RC_i = w_i * marginal_risk_i / sigma_p
        rc = w * marginal_risk / sigma_p

        return rc

    def _objective(self, w: np.ndarray) -> float:
        """ERC 목적 함수: 리스크 기여도와 목표 비율의 차이를 최소화한다.

        sum((RC_i / sigma_p - budget_i)^2)

        Args:
            w: 비중 벡터 (n_assets,)

        Returns:
            목적 함수 값 (float)
        """
        sigma_p = self._portfolio_risk(w)
        rc = self._risk_contribution(w)

        # 리스크 기여 비율 (전체 리스크 대비)
        rc_pct = rc / sigma_p if sigma_p > 0 else np.zeros(self.n_assets)

        # 목표 리스크 기여 비율과의 차이
        diff = rc_pct - self.budget
        return float(np.sum(diff ** 2))

    def _objective_log(self, w: np.ndarray) -> float:
        """로그 장벽 함수 기반 ERC 목적 함수.

        Spinu(2013) 방식: sum((w_i*(Sigma@w)_i - w_j*(Sigma@w)_j)^2)
        제곱합이 작을수록 리스크가 균등 배분된다.

        Args:
            w: 비중 벡터 (n_assets,)

        Returns:
            목적 함수 값 (float)
        """
        marginal_risk = self.cov_matrix @ w
        risk_contrib = w * marginal_risk

        # 목표 리스크 기여도
        target_rc = self.budget * np.sum(risk_contrib)

        # 차이의 제곱합
        diff = risk_contrib - target_rc
        return float(np.sum(diff ** 2))

    def optimize(self) -> dict[str, float]:
        """SLSQP로 ERC 비중을 최적화한다.

        제약 조건:
        - 비중 합 = 1
        - 개별 비중 >= 0.001 (공매도 불가)
        - 개별 비중 <= 1.0

        최적화 실패 시 역변동성 비중으로 fallback한다.

        Returns:
            {종목코드: 비중} 딕셔너리
        """
        if self.n_assets == 1:
            logger.info("단일 자산: 비중 1.0 할당")
            return {self.tickers[0]: 1.0}

        # 초기 비중: 역변동성 비중
        initial_weights = self._inverse_volatility_array()

        # 제약 조건
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        # 개별 비중 범위
        bounds = [(0.001, 1.0) for _ in range(self.n_assets)]

        try:
            result = minimize(
                self._objective_log,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )

            if result.success:
                weights = result.x
                # 음수 보정 및 재정규화
                weights = np.maximum(weights, 0.0)
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights = weights / weight_sum
                else:
                    logger.warning("최적화 결과 비중 합 0: 역변동성 fallback")
                    return self._inverse_volatility()

                result_dict = {
                    self.tickers[i]: float(weights[i])
                    for i in range(self.n_assets)
                }

                logger.info(
                    f"ERC 최적화 성공: {self.n_assets}개 자산, "
                    f"objective={result.fun:.2e}"
                )
                return result_dict

            else:
                logger.warning(
                    f"ERC 최적화 실패 ({result.message}): 역변동성 fallback"
                )
                return self._inverse_volatility()

        except Exception as e:
            logger.warning(f"ERC 최적화 예외 ({e}): 역변동성 fallback")
            return self._inverse_volatility()

    def _inverse_volatility_array(self) -> np.ndarray:
        """역변동성 비중 벡터를 반환한다.

        Returns:
            비중 벡터 (n_assets,)
        """
        vols = np.sqrt(np.diag(self.cov_matrix))
        vols[vols <= 0] = 1e-10  # 0 나눗셈 방지

        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()

        return weights

    def _inverse_volatility(self) -> dict[str, float]:
        """역변동성 비중 딕셔너리를 반환한다 (fallback용).

        각 자산의 변동성(표준편차) 역수에 비례하여 비중을 할당한다.
        변동성이 낮은 자산에 더 많은 비중을 부여한다.

        Returns:
            {종목코드: 비중} 딕셔너리
        """
        weights = self._inverse_volatility_array()

        result = {
            self.tickers[i]: float(weights[i])
            for i in range(self.n_assets)
        }

        logger.info(f"역변동성 비중 산출: {self.n_assets}개 자산")
        return result

    def get_risk_decomposition(self, weights: dict[str, float]) -> pd.DataFrame:
        """포트폴리오 리스크 분해를 수행한다.

        각 자산의 비중, 리스크 기여도(RC), 리스크 기여 비율(RC%),
        한계 리스크(Marginal Risk)를 산출한다.

        Args:
            weights: {종목코드: 비중} 딕셔너리

        Returns:
            DataFrame with columns: [ticker, weight, RC, RC_pct, marginal_risk]
            빈 weights이면 빈 DataFrame 반환.
        """
        if not weights:
            logger.warning("빈 비중 딕셔너리: 빈 리스크 분해 반환")
            return pd.DataFrame(
                columns=["ticker", "weight", "RC", "RC_pct", "marginal_risk"]
            )

        # 비중 벡터 구성 (공분산 행렬 순서에 맞춤)
        w = np.array([weights.get(t, 0.0) for t in self.tickers])

        # 포트폴리오 리스크
        sigma_p = self._portfolio_risk(w)

        # 한계 리스크: (Sigma @ w) / sigma_p
        marginal_risk = self.cov_matrix @ w
        if sigma_p > 0:
            marginal_risk_normalized = marginal_risk / sigma_p
        else:
            marginal_risk_normalized = np.zeros(self.n_assets)

        # 리스크 기여도
        rc = self._risk_contribution(w)

        # 리스크 기여 비율
        rc_sum = rc.sum()
        rc_pct = rc / rc_sum if rc_sum > 0 else np.zeros(self.n_assets)

        result = pd.DataFrame({
            "ticker": self.tickers,
            "weight": w,
            "RC": rc,
            "RC_pct": rc_pct,
            "marginal_risk": marginal_risk_normalized,
        })

        logger.info(
            f"리스크 분해 완료: sigma_p={sigma_p:.6f}, "
            f"RC 범위=[{rc.min():.6f}, {rc.max():.6f}]"
        )

        return result
