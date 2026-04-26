"""Value-Up 공시 스코어 전략 모듈.

KRX Value-Up 프로그램에 참여하고 구체적인 경영 개선 계획을 공시한 회사들에 투자하는 전략이다.

자발적 공시 여부와 공시 수준(ROE/배당 목표, 실행 일정 포함 여부)이 경영진 질 신호로 기능한다.
Value-Up 정책의 시작(2025년 이후)으로 정책 리스크가 최소화된 상태에서
정부 개혁 정책에 부합하는 종목들이 구조적 재평가를 받는 추세이다.

데이터 소스:
- KRX Value-Up Disclosure Portal (공식 포탈)
- DART 기업공시 (경영설명 자료, 투자 보고서 등)
- 회사 IR 정보
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValueUpDisclosureScore(Strategy):
    """Value-Up 공시 스코어 전략.

    KRX Value-Up 프로그램 참여 여부와 공시 수준을 평가하여
    경영진 질이 높은 종목을 선별한다.

    공시 점수:
    - 0점: 공시 없음
    - 1점: 공시만 함 (구체적 목표 없음)
    - 2점: ROE/배당 목표 공시 (일반)
    - 3점: 구체적 일정과 함께 목표 공시 (강력한 신호)

    Args:
        num_stocks: 포트폴리오 종목 수 (기본 20)
        min_market_cap: 최소 시가총액 (기본 1000억원)
        min_volume: 최소 일 거래대금 (기본 1억원)
        min_disclosure_score: 최소 공시 스코어 (기본 1, 공시가 있는 종목만)
        use_quality_filter: 품질 필터 적용 여부 (기본 True)
    """

    def __init__(
        self,
        num_stocks: int = 20,
        min_market_cap: int = 100_000_000_000,
        min_volume: int = 100_000_000,
        min_disclosure_score: int = 1,
        use_quality_filter: bool = True,
    ):
        self.num_stocks = num_stocks
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.min_disclosure_score = min_disclosure_score
        self.use_quality_filter = use_quality_filter

        logger.info(
            f"ValueUpDisclosureScore 초기화: num_stocks={num_stocks}, "
            f"min_disclosure_score={min_disclosure_score}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"ValueUpDisclosure(top{self.num_stocks})"

    def _filter_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """유니버스 필터링."""
        if df.empty:
            return df

        original_count = len(df)

        # 시가총액 필터
        if "market_cap" in df.columns:
            df = df[df["market_cap"] >= self.min_market_cap]

        # 거래대금 필터
        if "volume" in df.columns and "close" in df.columns:
            trade_value = df["volume"] * df["close"]
            df = df[trade_value >= self.min_volume]

        # 품질 필터 (선택)
        if self.use_quality_filter:
            # ROE가 음수인 종목 제외
            if "roe" in df.columns:
                df = df[df["roe"] >= 0]

        logger.debug(
            f"Value-Up 공시 유니버스 필터링: {original_count} -> {len(df)}개 종목"
        )
        return df

    def _calculate_disclosure_score(
        self,
        fundamentals: pd.DataFrame,
        disclosure_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Value-Up 공시 스코어를 계산한다.

        Args:
            fundamentals: 기본 지표 DataFrame
            disclosure_data: Value-Up 공시 데이터 (optional)
                DataFrame with columns: ['ticker', 'has_disclosure', 'score']
                score: 0 (없음) ~ 3 (구체적 계획)

        Returns:
            pd.Series (index=ticker, values=score)
        """
        if fundamentals.empty or "ticker" not in fundamentals.columns:
            return pd.Series(dtype=float)

        # 공시 데이터가 없으면 기본 스코어 (모든 종목에 동일 가중)
        if (
            disclosure_data is None
            or disclosure_data.empty
            or "score" not in disclosure_data.columns
        ):
            logger.warning("Value-Up 공시 데이터 없음, 기본 스코어 적용")
            # 공시 데이터 부재 시, 기본적으로 모든 종목에 0.5점 부여
            # (즉, 공시가 확인되지 않은 상태)
            base_score = 0.5
            return pd.Series(
                [base_score] * len(fundamentals),
                index=fundamentals["ticker"],
            )

        # 공시 데이터 결합
        disclosure_score = disclosure_data.set_index("ticker")["score"]

        # 펀더멘탈의 ticker를 인덱스로 설정하여 매칭
        result_scores = fundamentals.set_index("ticker")["ticker"].index.map(
            lambda t: disclosure_score.get(t, 0.5)
        )
        result_series = pd.Series(result_scores, index=fundamentals["ticker"])

        return result_series

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {
                'fundamentals': DataFrame,
                'value_up_disclosure': DataFrame (공시 데이터, optional),
            }

        Returns:
            종목코드: 비중 딕셔너리
        """
        fundamentals = data.get("fundamentals", pd.DataFrame())

        if fundamentals.empty:
            logger.warning(f"펀더멘탈 데이터 없음 ({date}), 빈 시그널 반환")
            return {}

        # 유니버스 필터링
        filtered = self._filter_universe(fundamentals.copy())
        if filtered.empty:
            logger.warning(f"필터링 후 종목 없음 ({date})")
            return {}

        # Value-Up 공시 데이터
        disclosure_data = data.get("value_up_disclosure", None)

        # 공시 스코어 계산
        scores = self._calculate_disclosure_score(filtered, disclosure_data)

        if scores.empty:
            logger.warning(f"Value-Up 공시 스코어 계산 실패 ({date})")
            return {}

        # 최소 공시 스코어 필터
        eligible = scores[scores >= self.min_disclosure_score]
        if eligible.empty:
            logger.warning(
                f"최소 공시 스코어({self.min_disclosure_score}) 이상 종목 없음 ({date})"
            )
            return {}

        # 상위 N개 선택
        top = eligible.sort_values(ascending=False).head(self.num_stocks)

        if top.empty:
            return {}

        # 공시 스코어 가중 할당
        # 높은 공시 스코어에 더 높은 가중치 부여
        score_sum = top.sum()
        if score_sum <= 0:
            # 모든 점수가 음수인 경우 동일 비중
            weight = 1.0 / len(top)
            signals = {ticker: weight for ticker in top.index}
        else:
            # 점수 비례 가중
            signals = {ticker: top[ticker] / score_sum for ticker in top.index}

        logger.info(
            f"Value-Up 공시 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"평균 공시점수={scores.loc[list(signals.keys())].mean():.2f}"
        )

        return signals
