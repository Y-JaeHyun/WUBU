"""변동성 타겟팅 오버레이 모듈.

포트폴리오의 실현 변동성을 목표 수준으로 조절한다.
한국 시장 제약(레버리지 불가)으로 exposure는 최대 1.0으로 캡핑한다.
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VolTargetingOverlay:
    """변동성 타겟팅 오버레이.

    target_vol / realized_vol 비율로 전체 포지션을 스케일링한다.
    레버리지 불가이므로 exposure_ratio <= max_exposure (기본 1.0).

    Args:
        target_vol: 목표 연율화 변동성 (기본 0.15 = 15%)
        lookback_days: 실현 변동성 계산 기간 (기본 20 거래일)
        min_vol: 최소 실현 변동성 (0 나눗셈 방지, 기본 0.05)
        max_exposure: 최대 노출 비율 (기본 1.0, 레버리지 불가)
        use_downside_only: True이면 하방 변동성만 사용 (기본 True)
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        lookback_days: int = 20,
        min_vol: float = 0.05,
        max_exposure: float = 1.0,
        use_downside_only: bool = True,
    ):
        self.target_vol = target_vol
        self.lookback_days = lookback_days
        self.min_vol = min_vol
        self.max_exposure = max_exposure
        self.use_downside_only = use_downside_only

        logger.info(
            f"VolTargetingOverlay 초기화: target_vol={target_vol:.0%}, "
            f"lookback={lookback_days}d, "
            f"downside_only={use_downside_only}"
        )

    def compute_realized_vol(self, portfolio_values: pd.Series) -> float:
        """포트폴리오 가치 시계열에서 실현 변동성을 계산한다.

        Args:
            portfolio_values: 최근 N일 포트폴리오 가치 시계열

        Returns:
            연율화 실현 변동성
        """
        if len(portfolio_values) < self.lookback_days + 1:
            return self.target_vol

        daily_returns = portfolio_values.pct_change().dropna()
        if len(daily_returns) < self.lookback_days:
            return self.target_vol

        recent = daily_returns.iloc[-self.lookback_days :]

        if self.use_downside_only:
            downside = recent[recent < 0]
            if len(downside) < 3:
                return self.target_vol
            vol = downside.std() * np.sqrt(252)
        else:
            vol = recent.std() * np.sqrt(252)

        return max(vol, self.min_vol)

    def get_exposure(self, portfolio_values: pd.Series) -> float:
        """현재 포트폴리오 변동성 기반 노출 비율을 계산한다.

        exposure = min(target_vol / realized_vol, max_exposure)

        Args:
            portfolio_values: 최근 포트폴리오 가치 시계열

        Returns:
            exposure ratio (0.0 ~ max_exposure)
        """
        realized = self.compute_realized_vol(portfolio_values)
        exposure = self.target_vol / realized
        result = min(exposure, self.max_exposure)

        logger.info(
            f"변동성 타겟팅: realized_vol={realized:.2%}, "
            f"target={self.target_vol:.2%}, exposure={result:.2%}"
        )
        return result

    def apply(self, weights: dict, portfolio_values: pd.Series) -> dict:
        """변동성 타겟팅을 적용하여 비중을 조절한다.

        Args:
            weights: {ticker: weight}
            portfolio_values: 최근 포트폴리오 가치 시계열

        Returns:
            조절된 {ticker: weight}
        """
        exposure = self.get_exposure(portfolio_values)

        if exposure >= 1.0:
            return weights
        if exposure <= 0.0:
            return {}

        return {t: w * exposure for t, w in weights.items()}
