"""N팩터 전략 모듈.

밸류(Value), 모멘텀(Momentum), 퀄리티(Quality), 저변동성(LowVol) 팩터를
결합하여 종합 점수 기반으로 종목을 선별하는 전략을 제공한다.

N-factor combiner를 사용하여 Z-Score 또는 순위 기반으로 팩터를 통합하고,
선택적으로 MarketTimingOverlay, 섹터 중립화, 회전율 감소를 적용할 수 있다.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.strategy.conglomerate import detect_conglomerate
from src.strategy.factor_combiner import combine_n_factors_zscore, combine_n_factors_rank
from src.strategy.momentum import MomentumStrategy
from src.strategy.quality import QualityStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ThreeFactorStrategy(Strategy):
    """N팩터 전략 (밸류 + 모멘텀 + 퀄리티 + 선택적 저변동성).

    밸류, 모멘텀, 퀄리티 (+ 저변동성) 팩터를 결합하여 종합 순위가 높은
    종목을 선별하고 동일 비중 포트폴리오를 구성한다.

    Args:
        num_stocks: 포트폴리오 종목 수 (기본 20)
        value_weight: 밸류 팩터 가중치 (기본 0.33)
        momentum_weight: 모멘텀 팩터 가중치 (기본 0.33)
        quality_weight: 퀄리티 팩터 가중치 (기본 0.34)
        low_vol_weight: 저변동성 팩터 가중치 (기본 0.0 = 비활성)
        combination_method: 팩터 결합 방식 - "zscore" 또는 "rank" (기본 "zscore")
        min_market_cap: 최소 시가총액 (기본 1000억원)
        market_timing: MarketTimingOverlay 객체 (기본 None)
        momentum_params: MomentumStrategy 초기화 파라미터 (기본 {})
        quality_params: QualityStrategy 초기화 파라미터 (기본 {})
        max_group_weight: 동일 업종 합산 비중 상한 (기본 0.25, 0이면 제한 없음)
        max_stocks_per_conglomerate: 동일 기업집단 최대 종목 수 (기본 2, 0이면 제한 없음)
        low_vol_params: LowVolatilityStrategy 초기화 파라미터 (기본 {})
        sector_neutral: 섹터 중립화 활성화 여부 (기본 False)
        max_sector_pct: 단일 섹터 최대 비중 (기본 0.25)
        turnover_buffer: 회전율 감소 버퍼 크기 (기본 0 = 비활성)
        holding_bonus: 기존 보유 종목 스코어 가산 (기본 0.0)
        regime_model: 레짐 메타 모델 객체 (기본 None)
    """

    def __init__(
        self,
        num_stocks: int = 20,
        value_weight: float = 0.33,
        momentum_weight: float = 0.33,
        quality_weight: float = 0.34,
        low_vol_weight: float = 0.0,
        combination_method: str = "zscore",
        min_market_cap: int = 100_000_000_000,
        market_timing: Optional[object] = None,
        momentum_params: Optional[dict] = None,
        quality_params: Optional[dict] = None,
        max_group_weight: float = 0.25,
        max_stocks_per_conglomerate: int = 2,
        low_vol_params: Optional[dict] = None,
        sector_neutral: bool = False,
        max_sector_pct: float = 0.25,
        turnover_buffer: int = 0,
        holding_bonus: float = 0.0,
        regime_model: Optional[object] = None,
    ):
        self.num_stocks = num_stocks
        self.value_weight = value_weight
        self.momentum_weight = momentum_weight
        self.quality_weight = quality_weight
        self.low_vol_weight = low_vol_weight
        self.combination_method = combination_method.lower()
        self.min_market_cap = min_market_cap
        self.market_timing = market_timing
        self.max_group_weight = max_group_weight
        self.max_stocks_per_conglomerate = max_stocks_per_conglomerate
        self.sector_neutral = sector_neutral
        self.max_sector_pct = max_sector_pct
        self.turnover_buffer = turnover_buffer
        self.holding_bonus = holding_bonus
        self.regime_model = regime_model

        # Turnover 감소용 보유 종목 추적
        self._current_holdings: set[str] = set()

        if self.combination_method not in ("zscore", "rank"):
            raise ValueError(
                f"지원하지 않는 결합 방식: {combination_method}. "
                "'zscore' 또는 'rank' 중 선택하세요."
            )

        # 하위 전략 객체 생성
        mp = momentum_params or {}
        qp = quality_params or {}
        self._momentum_strategy = MomentumStrategy(**mp)
        self._quality_strategy = QualityStrategy(**qp)

        # 저변동성 전략 (가중치 > 0일 때만 생성)
        self._low_vol_strategy = None
        if low_vol_weight > 0:
            from src.strategy.low_volatility import LowVolatilityStrategy
            lvp = low_vol_params or {}
            self._low_vol_strategy = LowVolatilityStrategy(**lvp)

        logger.info(
            f"ThreeFactorStrategy 초기화: num_stocks={num_stocks}, "
            f"weights=(V={value_weight}, M={momentum_weight}, "
            f"Q={quality_weight}, LV={low_vol_weight}), "
            f"method={self.combination_method}, "
            f"market_timing={'있음' if market_timing else '없음'}, "
            f"max_group_weight={max_group_weight}, "
            f"max_stocks_per_conglomerate={max_stocks_per_conglomerate}, "
            f"sector_neutral={sector_neutral}, "
            f"turnover_buffer={turnover_buffer}"
        )

    def update_holdings(self, holdings: set[str]) -> None:
        """현재 보유 종목 집합을 업데이트한다.

        백테스트 엔진에서 리밸런싱 전에 호출하여
        현재 보유 종목 정보를 전달한다 (Turnover 감소용).

        Args:
            holdings: 현재 보유 종목 코드 집합
        """
        self._current_holdings = set(holdings)

    @property
    def name(self) -> str:
        """전략 이름."""
        mt_str = "+MT" if self.market_timing is not None else ""
        parts = [
            f"V{self.value_weight:.0%}",
            f"M{self.momentum_weight:.0%}",
            f"Q{self.quality_weight:.0%}",
        ]
        if self.low_vol_weight > 0:
            parts.append(f"LV{self.low_vol_weight:.0%}")
        factor_str = "+".join(parts)
        extras = []
        if self.sector_neutral:
            extras.append("SN")
        if self.turnover_buffer > 0:
            extras.append(f"TB{self.turnover_buffer}")
        if self.regime_model is not None:
            extras.append("RM")
        extra_str = "+" + "+".join(extras) if extras else ""
        return (
            f"MultiFactor({factor_str}, {self.combination_method}, "
            f"top{self.num_stocks}{mt_str}{extra_str})"
        )

    def _get_value_scores(self, fundamentals: pd.DataFrame) -> pd.Series:
        """밸류 팩터 스코어를 계산한다.

        PBR 역수를 밸류 스코어로 사용한다 (PBR이 낮을수록 저평가).

        Args:
            fundamentals: 전 종목 기본 지표 DataFrame

        Returns:
            pd.Series (index=ticker, values=value_score)
        """
        if fundamentals.empty or "ticker" not in fundamentals.columns:
            return pd.Series(dtype=float)

        df = fundamentals.copy()

        # PBR > 0 필터
        if "pbr" not in df.columns:
            logger.warning("PBR 컬럼 없음: 빈 밸류 스코어 반환")
            return pd.Series(dtype=float)

        df = df[df["pbr"] > 0].copy()

        if df.empty:
            return pd.Series(dtype=float)

        # 밸류 스코어 = 1 / PBR (높을수록 저평가)
        scores = pd.Series(
            (1.0 / df["pbr"].values),
            index=df["ticker"].values,
            name="value_score",
        )

        # PER도 있으면 복합 밸류 스코어
        if "per" in df.columns:
            per_valid = df[df["per"] > 0]
            if not per_valid.empty:
                per_scores = pd.Series(
                    (1.0 / per_valid["per"].values),
                    index=per_valid["ticker"].values,
                )
                common = scores.index.intersection(per_scores.index)
                if not common.empty:
                    scores.loc[common] = (scores.loc[common] + per_scores.loc[common]) / 2

        logger.info(f"밸류 스코어 계산 완료: {len(scores)}개 종목")
        return scores

    def _get_momentum_scores(self, data: dict) -> pd.Series:
        """모멘텀 팩터 스코어를 계산한다."""
        return self._momentum_strategy.get_scores(data)

    def _get_quality_scores(self, data: dict) -> pd.Series:
        """퀄리티 팩터 스코어를 계산한다."""
        return self._quality_strategy.get_scores(data)

    def _filter_by_market_cap(self, fundamentals: pd.DataFrame) -> set[str]:
        """시가총액 기준으로 유니버스를 필터링한다."""
        if fundamentals.empty or "ticker" not in fundamentals.columns:
            return set()

        df = fundamentals.copy()

        if "market_cap" in df.columns:
            df = df[df["market_cap"] >= self.min_market_cap]

        return set(df["ticker"].tolist())

    def _apply_concentration_filter(
        self,
        ranked_scores: pd.Series,
        fundamentals: pd.DataFrame,
        num_stocks: int,
    ) -> pd.Series:
        """업종 비중 + 계열사 집중도를 동시 제한하며 상위 종목을 선정한다."""
        no_sector_limit = self.max_group_weight <= 0
        no_conglomerate_limit = self.max_stocks_per_conglomerate <= 0

        if no_sector_limit and no_conglomerate_limit:
            return ranked_scores.head(num_stocks)

        sector_map: dict[str, str] = {}
        if not no_sector_limit and "ticker" in fundamentals.columns:
            for col in ("sector", "industry"):
                if col in fundamentals.columns:
                    sector_map = dict(
                        zip(fundamentals["ticker"], fundamentals[col])
                    )
                    break

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

            if sector_map and not no_sector_limit:
                sector = sector_map.get(ticker, "기타")
                if sector_weights.get(sector, 0.0) + stock_weight > self.max_group_weight:
                    sector_skipped += 1
                    continue
            else:
                sector = "기타"

            conglomerate = None
            if name_map and not no_conglomerate_limit:
                name = name_map.get(ticker, "")
                conglomerate = detect_conglomerate(name)
                if conglomerate is not None:
                    if conglomerate_counts.get(conglomerate, 0) >= self.max_stocks_per_conglomerate:
                        conglomerate_skipped += 1
                        continue

            selected.append(ticker)
            sector_weights[sector] = sector_weights.get(sector, 0.0) + stock_weight
            if conglomerate is not None:
                conglomerate_counts[conglomerate] = conglomerate_counts.get(conglomerate, 0) + 1

        if sector_skipped > 0:
            logger.info(
                f"3팩터 업종 비중 필터: {sector_skipped}개 스킵 "
                f"(max_weight={self.max_group_weight:.0%})"
            )
        if conglomerate_skipped > 0:
            logger.info(
                f"3팩터 계열사 필터: {conglomerate_skipped}개 스킵 "
                f"(max_per_conglomerate={self.max_stocks_per_conglomerate})"
            )

        return ranked_scores[selected]

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다."""
        fundamentals = data.get("fundamentals", pd.DataFrame())

        if fundamentals.empty:
            logger.warning(f"펀더멘탈 데이터 없음 ({date}), 빈 시그널 반환")
            return {}

        eligible_tickers = self._filter_by_market_cap(fundamentals)
        if not eligible_tickers:
            logger.warning(f"시가총액 필터 통과 종목 없음 ({date})")
            return {}

        value_scores = self._get_value_scores(fundamentals)
        momentum_scores = self._get_momentum_scores(data)
        quality_scores = self._get_quality_scores(data)

        factors: dict[str, pd.Series] = {}
        weights: dict[str, float] = {}

        if not value_scores.empty:
            value_scores = value_scores[value_scores.index.isin(eligible_tickers)]
            if not value_scores.empty:
                factors["value"] = value_scores
                weights["value"] = self.value_weight

        if not momentum_scores.empty:
            momentum_scores = momentum_scores[momentum_scores.index.isin(eligible_tickers)]
            if not momentum_scores.empty:
                factors["momentum"] = momentum_scores
                weights["momentum"] = self.momentum_weight

        if not quality_scores.empty:
            quality_scores = quality_scores[quality_scores.index.isin(eligible_tickers)]
            if not quality_scores.empty:
                factors["quality"] = quality_scores
                weights["quality"] = self.quality_weight

        if self._low_vol_strategy is not None and self.low_vol_weight > 0:
            low_vol_scores = self._low_vol_strategy.get_scores(data)
            if not low_vol_scores.empty:
                low_vol_scores = low_vol_scores[
                    low_vol_scores.index.isin(eligible_tickers)
                ]
                if not low_vol_scores.empty:
                    factors["low_vol"] = low_vol_scores
                    weights["low_vol"] = self.low_vol_weight

        if not factors:
            logger.warning(f"유효한 팩터 스코어 없음 ({date})")
            return {}

        # 레짐 메타 모델로 가중치 동적 조절
        if self.regime_model is not None:
            try:
                index_prices = data.get("index_prices", pd.Series(dtype=float))
                regime_weights = self.regime_model.predict(
                    market_prices=index_prices if not index_prices.empty else None
                )
                if regime_weights:
                    for factor_name in weights:
                        if factor_name in regime_weights:
                            weights[factor_name] = regime_weights[factor_name]
                    logger.info(f"레짐 가중치 적용 ({date}): {weights}")
            except Exception as e:
                logger.warning(f"레짐 모델 실패 ({date}): {e}. 정적 가중치 사용.")

        if len(factors) == 1:
            factor_name, scores = next(iter(factors.items()))
            logger.info(f"단일 팩터({factor_name})로 종목 선정 ({date})")
            combined = scores
        else:
            if self.combination_method == "zscore":
                combined = combine_n_factors_zscore(factors, weights)
            elif self.combination_method == "rank":
                combined = combine_n_factors_rank(factors, weights)
            else:
                combined = pd.Series(dtype=float)

        if combined.empty:
            logger.warning(f"팩터 결합 결과 없음 ({date})")
            return {}

        # 종목 선정 전략 분기
        if self.sector_neutral:
            try:
                from src.data.sector_collector import get_sector_for_tickers
                from src.strategy.sector_neutral import (
                    sector_neutral_rank,
                    select_sector_neutral,
                )

                sector_map = get_sector_for_tickers(list(combined.index), date)
                if sector_map:
                    combined = sector_neutral_rank(combined, sector_map)
                    selected_tickers = select_sector_neutral(
                        combined, sector_map, self.num_stocks, self.max_sector_pct
                    )
                    top = combined.loc[
                        [t for t in selected_tickers if t in combined.index]
                    ]
                else:
                    top = combined.sort_values(ascending=False).head(self.num_stocks)
            except Exception as e:
                logger.warning(f"섹터 중립화 실패 ({date}): {e}. 기본 선정 사용.")
                top = combined.sort_values(ascending=False).head(self.num_stocks)
        elif self.turnover_buffer > 0 and self._current_holdings:
            if self.holding_bonus > 0:
                for ticker in self._current_holdings:
                    if ticker in combined.index:
                        combined.loc[ticker] += self.holding_bonus

            ranked = combined.sort_values(ascending=False)
            exit_threshold = self.num_stocks + self.turnover_buffer
            selected: set[str] = set()

            for ticker in self._current_holdings:
                if ticker in ranked.index:
                    rank_position = list(ranked.index).index(ticker)
                    if rank_position < exit_threshold:
                        selected.add(ticker)

            for ticker in ranked.index:
                if len(selected) >= self.num_stocks:
                    break
                if ticker not in selected:
                    selected.add(ticker)

            top = combined.loc[list(selected)]
        else:
            # 기본: 상위 N개 선정 (업종/계열사 집중도 제한 적용)
            ranked = combined.sort_values(ascending=False)
            top = self._apply_concentration_filter(ranked, fundamentals, self.num_stocks)

        if top.empty:
            return {}

        weight = 1.0 / len(top)
        signals = {ticker: weight for ticker in top.index}

        if signals and self.market_timing is not None:
            index_prices = data.get("index_prices", pd.Series(dtype=float))
            if not index_prices.empty:
                try:
                    if hasattr(self.market_timing, "switch_mode"):
                        if self.market_timing.switch_mode == "gradual":
                            signals = self.market_timing.apply_overlay_gradual(
                                signals, index_prices
                            )
                        else:
                            signal = self.market_timing.get_signal(index_prices)
                            signals = self.market_timing.apply_overlay(signals, signal)
                    else:
                        signal = self.market_timing.get_signal(index_prices)
                        signals = self.market_timing.apply_overlay(signals, signal)
                except Exception as e:
                    logger.warning(f"마켓 타이밍 오버레이 적용 실패 ({date}): {e}")

        logger.info(
            f"3팩터 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"사용 팩터: {list(factors.keys())}"
        )

        return signals
