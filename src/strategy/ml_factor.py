"""ML 팩터 전략 모듈.

ML 파이프라인의 예측 결과를 기반으로 종목을 선별하고
포트폴리오를 구성하는 전략이다.

학습된 MLPipeline이 예측한 기대 수익률 스코어 상위 N개 종목을
동일 비중 또는 리스크 패리티 비중으로 포트폴리오를 구성한다.

사용 예시:
    pipeline = MLPipeline(model_type="ridge")
    pipeline.train(feature_history, target_history)
    strategy = MLFactorStrategy(ml_pipeline=pipeline, num_stocks=20)
    signals = strategy.generate_signals(date, data)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.ml.features import build_factor_features
from src.ml.pipeline import MLPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLFactorStrategy(Strategy):
    """ML 팩터 모델 전략.

    ML 파이프라인의 예측 스코어가 높은 상위 종목을 선별하여
    포트폴리오를 구성한다.

    Args:
        ml_pipeline: MLPipeline 객체 (학습 완료 상태)
        num_stocks: 포트폴리오 종목 수 (기본 20)
        use_risk_parity: 리스크 패리티 비중 사용 여부 (기본 False)
        min_market_cap: 최소 시가총액 (기본 1000억원)
        max_weight: 단일 종목 최대 비중 (기본 0.15, 리스크 패리티 시 사용)
    """

    def __init__(
        self,
        ml_pipeline: MLPipeline,
        num_stocks: int = 20,
        use_risk_parity: bool = False,
        min_market_cap: int = 100_000_000_000,
        max_weight: float = 0.15,
    ):
        self.ml_pipeline = ml_pipeline
        self.num_stocks = num_stocks
        self.use_risk_parity = use_risk_parity
        self.min_market_cap = min_market_cap
        self.max_weight = max_weight

        logger.info(
            f"MLFactorStrategy 초기화: model_type={ml_pipeline.model_type}, "
            f"num_stocks={num_stocks}, use_risk_parity={use_risk_parity}, "
            f"min_market_cap={min_market_cap:,}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        rp_str = "+RP" if self.use_risk_parity else ""
        return f"MLFactor({self.ml_pipeline.model_type}, top{self.num_stocks}{rp_str})"

    def _filter_by_market_cap(self, fundamentals: pd.DataFrame) -> set[str]:
        """시가총액 기준으로 유니버스를 필터링한다.

        Args:
            fundamentals: 전종목 기본 지표 DataFrame

        Returns:
            시가총액 조건을 충족하는 종목 코드 집합
        """
        if fundamentals.empty or "ticker" not in fundamentals.columns:
            return set()

        df = fundamentals.copy()

        if "market_cap" in df.columns:
            df = df[df["market_cap"] >= self.min_market_cap]

        return set(df["ticker"].tolist())

    def _apply_risk_parity(
        self,
        tickers: list[str],
        prices: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """리스크 패리티 비중을 산출한다.

        Args:
            tickers: 대상 종목 코드 리스트
            prices: {종목코드: OHLCV DataFrame} 딕셔너리

        Returns:
            {종목코드: 비중} 딕셔너리
        """
        from src.optimization.covariance import CovarianceEstimator
        from src.optimization.risk_parity import RiskParityOptimizer

        # 수익률 추출
        returns_dict = {}
        for ticker in tickers:
            if ticker not in prices:
                continue
            price_df = prices[ticker]
            if price_df.empty or "close" not in price_df.columns:
                continue
            close = price_df["close"].dropna()
            if len(close) < 60:
                continue
            returns_dict[ticker] = close.pct_change().dropna()

        if len(returns_dict) < 2:
            # 리스크 패리티 불가: 동일 비중
            weight = 1.0 / len(tickers)
            return {t: weight for t in tickers}

        returns_df = pd.DataFrame(returns_dict).dropna(how="all")

        # 공분산 추정
        estimator = CovarianceEstimator(method="ledoit_wolf")
        cov_matrix = estimator.estimate(returns_df)

        if cov_matrix.empty:
            weight = 1.0 / len(tickers)
            return {t: weight for t in tickers}

        # ERC 최적화
        try:
            optimizer = RiskParityOptimizer(covariance=cov_matrix)
            optimized = optimizer.optimize()

            # max_weight 클리핑
            clipped = {}
            for t, w in optimized.items():
                clipped[t] = min(self.max_weight, w)

            total = sum(clipped.values())
            if total > 0:
                return {t: w / total for t, w in clipped.items()}

        except Exception as e:
            logger.warning(f"리스크 패리티 비중 산출 실패: {e}")

        # Fallback: 동일 비중
        weight = 1.0 / len(tickers)
        return {t: weight for t in tickers}

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        1. 펀더멘탈 + 가격 데이터로 피처를 생성한다.
        2. ML 모델로 기대 수익률을 예측한다.
        3. 시가총액 필터링 후 상위 N개 종목을 선정한다.
        4. 동일 비중 또는 리스크 패리티 비중을 할당한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {'fundamentals': DataFrame, 'prices': dict[ticker, DataFrame],
                   ...} 형태

        Returns:
            종목코드: 비중 딕셔너리
        """
        fundamentals = data.get("fundamentals", pd.DataFrame())
        prices = data.get("prices", {})

        if fundamentals.empty:
            logger.warning(f"펀더멘탈 데이터 없음 ({date}), 빈 시그널 반환")
            return {}

        # 1. 피처 생성
        features = build_factor_features(fundamentals, prices)

        if features.empty:
            logger.warning(f"피처 생성 실패 ({date})")
            return {}

        # 2. ML 예측
        predictions = self.ml_pipeline.predict(features)

        if predictions.empty:
            logger.warning(f"ML 예측 실패 ({date})")
            return {}

        # 3. 시가총액 필터링
        eligible_tickers = self._filter_by_market_cap(fundamentals)

        if eligible_tickers:
            predictions = predictions[predictions.index.isin(eligible_tickers)]

        if predictions.empty:
            logger.warning(f"시가총액 필터 후 종목 없음 ({date})")
            return {}

        # 4. 상위 N개 선정 (예측 스코어 내림차순)
        top_predictions = predictions.sort_values(ascending=False).head(
            self.num_stocks
        )

        if top_predictions.empty:
            return {}

        selected_tickers = top_predictions.index.tolist()

        # 5. 비중 할당
        if self.use_risk_parity and prices:
            signals = self._apply_risk_parity(selected_tickers, prices)
        else:
            # 동일 비중
            weight = 1.0 / len(selected_tickers)
            signals = {ticker: weight for ticker in selected_tickers}

        logger.info(
            f"ML 팩터 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"예측 스코어 범위=[{top_predictions.min():.4f}, "
            f"{top_predictions.max():.4f}]"
        )

        return signals
