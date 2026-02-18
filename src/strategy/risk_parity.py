"""리스크 패리티 전략 모듈.

기존 팩터 전략(예: ThreeFactorStrategy)으로 종목을 선정한 후,
ERC(Equal Risk Contribution) 방식으로 비중을 최적화하는 전략이다.

동일 비중 대신 리스크 기여도가 균등하도록 비중을 배분하므로
분산 투자 효과가 향상된다.

사용 예시:
    from src.strategy.three_factor import ThreeFactorStrategy
    selector = ThreeFactorStrategy(num_stocks=20)
    strategy = RiskParityStrategy(stock_selector=selector)
    signals = strategy.generate_signals(date, data)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.optimization.covariance import CovarianceEstimator
from src.optimization.risk_parity import RiskParityOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskParityStrategy(Strategy):
    """리스크 패리티 전략.

    기존 팩터 전략으로 종목을 선정하고, 선정된 종목의 수익률 공분산을
    기반으로 ERC 최적화를 수행하여 비중을 결정한다.

    Args:
        stock_selector: Strategy 객체 (종목 선정용, 예: ThreeFactorStrategy)
        cov_method: 공분산 추정 방법 - "sample", "ledoit_wolf", "ewm" (기본 "ledoit_wolf")
        lookback_days: 공분산 추정 룩백 기간 (기본 252)
        max_weight: 단일 종목 최대 비중 (기본 0.15)
        min_weight: 단일 종목 최소 비중 (기본 0.01)
        risk_budget: 자산별 리스크 예산 딕셔너리 (None이면 동일 예산)
    """

    def __init__(
        self,
        stock_selector: Strategy,
        cov_method: str = "ledoit_wolf",
        lookback_days: int = 252,
        max_weight: float = 0.15,
        min_weight: float = 0.01,
        risk_budget: Optional[dict[str, float]] = None,
    ):
        self.stock_selector = stock_selector
        self.cov_method = cov_method
        self.lookback_days = lookback_days
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.risk_budget = risk_budget

        # 종목 선정용 전략 이름 추출
        if hasattr(stock_selector, "name"):
            self._selector_name = stock_selector.name
        else:
            self._selector_name = type(stock_selector).__name__

        # 공분산 추정기 초기화
        self._cov_estimator = CovarianceEstimator(
            method=cov_method,
            lookback_days=lookback_days,
        )

        logger.info(
            f"RiskParityStrategy 초기화: selector={self._selector_name}, "
            f"cov_method={cov_method}, lookback={lookback_days}, "
            f"max_weight={max_weight}, min_weight={min_weight}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"RiskParity({self._selector_name})"

    def _extract_returns(
        self,
        prices: dict[str, pd.DataFrame],
        tickers: list[str],
    ) -> pd.DataFrame:
        """종목별 가격 데이터에서 일별 수익률 DataFrame을 생성한다.

        Args:
            prices: {종목코드: OHLCV DataFrame} 딕셔너리
            tickers: 대상 종목 코드 리스트

        Returns:
            일별 수익률 DataFrame (index=날짜, columns=종목코드)
        """
        returns_dict = {}

        for ticker in tickers:
            if ticker not in prices:
                continue

            price_df = prices[ticker]

            if price_df.empty or "close" not in price_df.columns:
                continue

            close = price_df["close"].dropna()
            if len(close) < 2:
                continue

            ret = close.pct_change().dropna()
            returns_dict[ticker] = ret

        if not returns_dict:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_dict)

        # NaN 행 제거 (모든 종목에 데이터가 있는 날짜만)
        returns_df = returns_df.dropna(how="all")

        return returns_df

    def _clip_and_normalize(self, weights: dict[str, float]) -> dict[str, float]:
        """비중을 min_weight/max_weight로 클리핑하고 정규화한다.

        Args:
            weights: {종목코드: 비중} 딕셔너리

        Returns:
            클리핑 및 정규화된 {종목코드: 비중} 딕셔너리
        """
        if not weights:
            return {}

        # 클리핑
        clipped = {}
        for ticker, w in weights.items():
            clipped[ticker] = max(self.min_weight, min(self.max_weight, w))

        # 정규화 (합이 1.0이 되도록)
        total = sum(clipped.values())
        if total <= 0:
            return {}

        normalized = {t: w / total for t, w in clipped.items()}

        return normalized

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        1. stock_selector로 종목 후보를 선정한다.
        2. 후보 종목의 수익률로 공분산 행렬을 추정한다.
        3. ERC 최적화로 비중을 결정한다.
        4. max_weight/min_weight 클리핑 후 정규화한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {'fundamentals': DataFrame, 'prices': dict[ticker, DataFrame],
                   ...} 형태

        Returns:
            종목코드: 비중 딕셔너리
        """
        # 1. 종목 선정 (기존 팩터 전략 사용)
        raw_signals = self.stock_selector.generate_signals(date, data)

        if not raw_signals:
            logger.warning(f"종목 선정 결과 없음 ({date})")
            return {}

        candidate_tickers = list(raw_signals.keys())
        logger.info(
            f"리스크 패리티 종목 후보: {len(candidate_tickers)}개 ({date})"
        )

        # 2. 수익률 추출
        prices = data.get("prices", {})
        if not prices:
            logger.warning(
                f"가격 데이터 없음 ({date}): 동일 비중 fallback"
            )
            return raw_signals

        returns_df = self._extract_returns(prices, candidate_tickers)

        if returns_df.empty or len(returns_df.columns) < 2:
            logger.warning(
                f"수익률 데이터 부족 ({date}): 동일 비중 fallback"
            )
            return raw_signals

        # 수익률 데이터가 있는 종목만 필터
        valid_tickers = returns_df.columns.tolist()

        if len(valid_tickers) < 2:
            logger.warning(
                f"유효 종목 수 부족 ({date}): 동일 비중 fallback"
            )
            weight = 1.0 / len(candidate_tickers)
            return {t: weight for t in candidate_tickers}

        # 3. 공분산 추정
        cov_matrix = self._cov_estimator.estimate(returns_df)

        if cov_matrix.empty:
            logger.warning(
                f"공분산 추정 실패 ({date}): 동일 비중 fallback"
            )
            return raw_signals

        # 4. ERC 최적화
        try:
            optimizer = RiskParityOptimizer(
                covariance=cov_matrix,
                budget=self.risk_budget,
            )
            optimized_weights = optimizer.optimize()
        except Exception as e:
            logger.warning(
                f"ERC 최적화 실패 ({date}): {e}. 동일 비중 fallback"
            )
            return raw_signals

        # 5. 클리핑 및 정규화
        final_weights = self._clip_and_normalize(optimized_weights)

        if not final_weights:
            logger.warning(
                f"비중 클리핑 후 빈 결과 ({date}): 동일 비중 fallback"
            )
            return raw_signals

        logger.info(
            f"리스크 패리티 시그널 생성 ({date}): {len(final_weights)}개 종목, "
            f"비중 범위=[{min(final_weights.values()):.4f}, "
            f"{max(final_weights.values()):.4f}]"
        )

        return final_weights
