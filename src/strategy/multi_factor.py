"""멀티팩터 전략 모듈.

밸류, 모멘텀 등 복수의 팩터를 결합하여 포트폴리오를 구성하는 전략을 제공한다.
FactorCombiner를 통해 Z-Score 또는 순위 기반으로 팩터를 통합하고,
선택적으로 MarketTimingOverlay를 적용할 수 있다.
"""

from typing import Optional

import pandas as pd

from src.backtest.engine import Strategy
from src.strategy.conglomerate import detect_conglomerate
from src.strategy.value import ValueStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.factor_combiner import combine_zscore, combine_rank
from src.strategy.market_timing import MarketTimingOverlay
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiFactorStrategy(Strategy):
    """멀티팩터 전략.

    밸류와 모멘텀 팩터를 결합하여 종합 점수 기반으로 종목을 선별한다.
    팩터 결합 방식(Z-Score, 순위)과 마켓 타이밍 오버레이 적용 여부를 설정할 수 있다.

    Args:
        factors: 사용할 팩터 리스트 (기본 ["value", "momentum"])
        weights: 팩터별 가중치 리스트 (기본 [0.5, 0.5])
        combine_method: 결합 방식 - "zscore" 또는 "rank" (기본 "zscore")
        num_stocks: 포트폴리오 종목 수 (기본 20)
        apply_market_timing: 마켓 타이밍 오버레이 적용 여부 (기본 False)
        market_timing_params: MarketTimingOverlay 초기화 파라미터 (기본 {})
        value_params: ValueStrategy 초기화 파라미터 (기본 {})
        momentum_params: MomentumStrategy 초기화 파라미터 (기본 {})
        turnover_penalty: 회전율 페널티 (기본 0.0, 비활성).
            설정 시 이전 리밸런싱 포트폴리오에 없던 새 종목의 스코어에서
            페널티만큼 차감하여 불필요한 교체를 억제.
        max_group_weight: 동일 업종 합산 비중 상한 (기본 0.25 = 25%, 0이면 제한 없음).
            fundamentals에 'sector' 또는 'industry' 컬럼이 있을 때 활성화.
            업종 집중도를 제한하여 특정 섹터 쏠림을 방지한다.
        max_stocks_per_conglomerate: 동일 기업집단(계열사) 최대 종목 수
            (기본 2, 0이면 제한 없음). 종목명 기반 계열사 탐지.
    """

    def __init__(
        self,
        factors: Optional[list[str]] = None,
        weights: Optional[list[float]] = None,
        combine_method: str = "zscore",
        num_stocks: int = 20,
        apply_market_timing: bool = False,
        market_timing_params: Optional[dict] = None,
        value_params: Optional[dict] = None,
        momentum_params: Optional[dict] = None,
        turnover_penalty: float = 0.0,
        max_group_weight: float = 0.25,
        max_stocks_per_conglomerate: int = 2,
    ):
        self.factors = factors or ["value", "momentum"]
        self.weights = weights or [0.5, 0.5]
        self.combine_method = combine_method.lower()
        self.num_stocks = num_stocks
        self.apply_market_timing = apply_market_timing
        self.turnover_penalty = turnover_penalty
        self.max_group_weight = max_group_weight
        self.max_stocks_per_conglomerate = max_stocks_per_conglomerate
        # 이전 포트폴리오 종목 저장 (회전율 페널티용)
        self._prev_holdings: set[str] = set()

        if len(self.factors) != len(self.weights):
            raise ValueError(
                f"팩터 수({len(self.factors)})와 가중치 수({len(self.weights)})가 일치하지 않습니다."
            )

        if self.combine_method not in ("zscore", "rank"):
            raise ValueError(
                f"지원하지 않는 결합 방식: {combine_method}. 'zscore' 또는 'rank' 중 선택하세요."
            )

        # 팩터별 전략 객체 생성
        self._value_strategy: Optional[ValueStrategy] = None
        self._momentum_strategy: Optional[MomentumStrategy] = None

        vp = value_params or {}
        mp = momentum_params or {}

        if "value" in self.factors:
            self._value_strategy = ValueStrategy(**vp)
        if "momentum" in self.factors:
            self._momentum_strategy = MomentumStrategy(**mp)

        # 마켓 타이밍 오버레이
        self._market_timing: Optional[MarketTimingOverlay] = None
        if apply_market_timing:
            mtp = market_timing_params or {}
            self._market_timing = MarketTimingOverlay(**mtp)

        logger.info(
            f"MultiFactorStrategy 초기화: factors={self.factors}, "
            f"weights={self.weights}, combine_method={self.combine_method}, "
            f"num_stocks={num_stocks}, market_timing={apply_market_timing}, "
            f"turnover_penalty={turnover_penalty}, "
            f"max_group_weight={max_group_weight}, "
            f"max_stocks_per_conglomerate={max_stocks_per_conglomerate}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        factor_str = "+".join(self.factors)
        mt_str = "+MT" if self.apply_market_timing else ""
        return f"MultiFactor({factor_str}, {self.combine_method}, top{self.num_stocks}{mt_str})"

    def _get_value_scores(self, fundamentals: pd.DataFrame) -> pd.Series:
        """밸류 팩터 스코어를 추출한다.

        밸류 스코어는 PBR의 역수(1/PBR)로 계산한다.
        PBR이 낮을수록 저평가이므로, 역수를 취해 높은 값이 좋은 것으로 변환한다.

        Args:
            fundamentals: 전 종목 기본 지표 DataFrame

        Returns:
            pd.Series (index=ticker, values=value_score)
        """
        if fundamentals.empty or "ticker" not in fundamentals.columns:
            return pd.Series(dtype=float)

        df = fundamentals.copy()

        # 기본 필터: PBR > 0
        if "pbr" in df.columns:
            df = df[df["pbr"] > 0]
        else:
            return pd.Series(dtype=float)

        if df.empty:
            return pd.Series(dtype=float)

        # 밸류 스코어 = 1 / PBR (높을수록 저평가)
        scores = pd.Series(
            (1.0 / df["pbr"].values),
            index=df["ticker"].values,
            name="value_score",
        )

        # PER도 있으면 복합 스코어
        if "per" in df.columns:
            per_valid = df[df["per"] > 0]
            if not per_valid.empty:
                per_scores = pd.Series(
                    (1.0 / per_valid["per"].values),
                    index=per_valid["ticker"].values,
                )
                # PBR 역수와 PER 역수의 평균 (공통 종목)
                common = scores.index.intersection(per_scores.index)
                if not common.empty:
                    scores.loc[common] = (scores.loc[common] + per_scores.loc[common]) / 2

        return scores

    def _get_momentum_scores(self, data: dict) -> pd.Series:
        """모멘텀 팩터 스코어를 추출한다.

        Args:
            data: {'fundamentals': DataFrame, 'prices': dict} 형태

        Returns:
            pd.Series (index=ticker, values=momentum_score)
        """
        if self._momentum_strategy is None:
            return pd.Series(dtype=float)

        return self._momentum_strategy.get_scores(data)

    def _apply_concentration_filter(
        self,
        ranked_scores: pd.Series,
        fundamentals: pd.DataFrame,
        num_stocks: int,
    ) -> pd.Series:
        """업종 비중 + 계열사 집중도를 동시 제한하며 상위 종목을 선정한다.

        스코어 순서대로 순회하면서 두 조건을 동시 체크:
        1. 해당 업종 합산 비중이 max_group_weight를 초과하면 스킵
        2. 해당 계열사 종목 수가 max_stocks_per_conglomerate를 초과하면 스킵

        Args:
            ranked_scores: 스코어 내림차순 정렬된 Series (index=ticker).
            fundamentals: 종목 기본 지표 DataFrame.
            num_stocks: 선정할 종목 수.

        Returns:
            선정된 종목의 스코어 Series.
        """
        no_sector_limit = self.max_group_weight <= 0
        no_conglomerate_limit = self.max_stocks_per_conglomerate <= 0

        if no_sector_limit and no_conglomerate_limit:
            return ranked_scores.head(num_stocks)

        # --- 업종 매핑 ---
        sector_map: dict[str, str] = {}
        if not no_sector_limit and "ticker" in fundamentals.columns:
            for col in ("sector", "industry"):
                if col in fundamentals.columns:
                    sector_map = dict(
                        zip(fundamentals["ticker"], fundamentals[col])
                    )
                    break

        # --- 종목명 매핑 (계열사 탐지용) ---
        name_map: dict[str, str] = {}
        if not no_conglomerate_limit and "ticker" in fundamentals.columns:
            if "name" in fundamentals.columns:
                name_map = dict(
                    zip(fundamentals["ticker"], fundamentals["name"])
                )

        stock_weight = 1.0 / num_stocks
        selected: list[str] = []
        sector_weights: dict[str, float] = {}
        conglomerate_counts: dict[str, int] = {}
        sector_skipped = 0
        conglomerate_skipped = 0

        for ticker in ranked_scores.index:
            if len(selected) >= num_stocks:
                break

            # 업종 비중 체크
            if sector_map and not no_sector_limit:
                sector = sector_map.get(ticker, "기타")
                if sector_weights.get(sector, 0.0) + stock_weight > self.max_group_weight:
                    sector_skipped += 1
                    continue
            else:
                sector = "기타"

            # 계열사 체크
            conglomerate = None
            if name_map and not no_conglomerate_limit:
                name = name_map.get(ticker, "")
                conglomerate = detect_conglomerate(name)
                if conglomerate is not None:
                    if conglomerate_counts.get(conglomerate, 0) >= self.max_stocks_per_conglomerate:
                        conglomerate_skipped += 1
                        continue

            # 두 조건 모두 통과 → 선정
            selected.append(ticker)
            sector_weights[sector] = sector_weights.get(sector, 0.0) + stock_weight
            if conglomerate is not None:
                conglomerate_counts[conglomerate] = conglomerate_counts.get(conglomerate, 0) + 1

        if sector_skipped > 0:
            logger.info(
                f"업종 비중 필터: {sector_skipped}개 스킵 "
                f"(max_weight={self.max_group_weight:.0%})"
            )
        if conglomerate_skipped > 0:
            logger.info(
                f"계열사 필터: {conglomerate_skipped}개 스킵 "
                f"(max_per_conglomerate={self.max_stocks_per_conglomerate})"
            )

        return ranked_scores[selected]

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        복수 팩터의 스코어를 결합하여 상위 종목을 선별하고 동일 비중을 부여한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {'fundamentals': DataFrame, 'prices': dict[ticker, DataFrame],
                   'index_prices': pd.Series (optional)} 형태

        Returns:
            종목코드: 비중 딕셔너리
        """
        fundamentals = data.get("fundamentals", pd.DataFrame())

        if fundamentals.empty:
            logger.warning(f"펀더멘탈 데이터 없음 ({date}), 빈 시그널 반환")
            return {}

        # 팩터별 스코어 계산
        factor_scores: dict[str, pd.Series] = {}
        factor_weights: dict[str, float] = {}

        for factor, weight in zip(self.factors, self.weights):
            if factor == "value":
                scores = self._get_value_scores(fundamentals)
                if not scores.empty:
                    factor_scores["value"] = scores
                    factor_weights["value"] = weight
            elif factor == "momentum":
                scores = self._get_momentum_scores(data)
                if not scores.empty:
                    factor_scores["momentum"] = scores
                    factor_weights["momentum"] = weight

        if len(factor_scores) < 2:
            # 팩터가 하나만 있으면 해당 팩터만으로 종목 선정
            if factor_scores:
                factor_name, scores = next(iter(factor_scores.items()))
                logger.info(f"단일 팩터({factor_name})로 종목 선정")
                combined = scores
            else:
                logger.warning(f"유효한 팩터 스코어 없음 ({date})")
                return {}
        else:
            # 팩터 결합
            value_scores = factor_scores.get("value", pd.Series(dtype=float))
            momentum_scores = factor_scores.get("momentum", pd.Series(dtype=float))
            vw = factor_weights.get("value", 0.5)
            mw = factor_weights.get("momentum", 0.5)

            if self.combine_method == "zscore":
                combined = combine_zscore(value_scores, momentum_scores, vw, mw)
            elif self.combine_method == "rank":
                combined = combine_rank(value_scores, momentum_scores, vw, mw)
            else:
                combined = pd.Series(dtype=float)

            if combined.empty:
                logger.warning(f"팩터 결합 결과 없음 ({date})")
                return {}

        # 회전율 페널티 적용: 이전 포트폴리오에 없던 새 종목 스코어 차감
        if self.turnover_penalty > 0 and self._prev_holdings:
            new_tickers = set(combined.index) - self._prev_holdings
            if new_tickers:
                combined = combined.copy()
                combined.loc[list(new_tickers)] -= self.turnover_penalty
                logger.info(
                    f"회전율 페널티 적용: {len(new_tickers)}개 신규 종목에 "
                    f"{self.turnover_penalty} 차감"
                )

        ranked = combined.sort_values(ascending=False)
        top = self._apply_concentration_filter(ranked, fundamentals, self.num_stocks)

        # 동일 비중 할당
        weight = 1.0 / len(top)
        signals = {ticker: weight for ticker in top.index}

        # 마켓 타이밍 오버레이 적용
        if self.apply_market_timing and self._market_timing is not None:
            index_prices = data.get("index_prices", pd.Series(dtype=float))
            if not index_prices.empty:
                if self._market_timing.switch_mode == "binary":
                    signal = self._market_timing.get_signal(index_prices)
                    signals = self._market_timing.apply_overlay(signals, signal)
                elif self._market_timing.switch_mode == "gradual":
                    signals = self._market_timing.apply_overlay_gradual(
                        signals, index_prices
                    )

        # 이전 포트폴리오 종목 갱신 (회전율 페널티용)
        self._prev_holdings = set(signals.keys())

        logger.info(
            f"멀티팩터 시그널 생성 ({date}): {len(signals)}개 종목"
        )

        return signals
