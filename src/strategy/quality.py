"""퀄리티 팩터 전략 모듈.

수익성, 재무 건전성, 이익의 질 등 퀄리티 지표를 기반으로
우량 종목을 선별하여 포트폴리오를 구성하는 전략을 제공한다.

퀄리티 지표:
- ROE (자기자본이익률): 높을수록 좋음
- GP/A (매출총이익/총자산): 높을수록 좋음
- 부채비율: 낮을수록 좋음
- 발생액(Accruals): 낮을수록 좋음 (이익의 질)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 기본 퀄리티 팩터 가중치
DEFAULT_QUALITY_WEIGHTS: dict[str, float] = {
    "roe": 0.3,
    "gpa": 0.3,
    "debt": 0.2,
    "accrual": 0.2,
}


class QualityStrategy(Strategy):
    """퀄리티 팩터 전략.

    ROE, GP/A, 부채비율, 발생액 등 퀄리티 지표를 종합하여
    재무적으로 우량한 종목을 선별하는 전략이다.

    Args:
        num_stocks: 포트폴리오 종목 수 (기본 20)
        min_market_cap: 최소 시가총액 (기본 1000억원)
        weights: 퀄리티 팩터별 가중치 딕셔너리
            {"roe": 0.3, "gpa": 0.3, "debt": 0.2, "accrual": 0.2}
        min_volume: 최소 일 거래대금 (기본 1억원)
    """

    def __init__(
        self,
        num_stocks: int = 20,
        min_market_cap: int = 100_000_000_000,
        weights: Optional[dict[str, float]] = None,
        min_volume: int = 100_000_000,
    ):
        self.num_stocks = num_stocks
        self.min_market_cap = min_market_cap
        self.weights = weights or DEFAULT_QUALITY_WEIGHTS.copy()
        self.min_volume = min_volume

        logger.info(
            f"QualityStrategy 초기화: num_stocks={num_stocks}, "
            f"min_market_cap={min_market_cap:,}, "
            f"weights={self.weights}, min_volume={min_volume:,}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"Quality(top{self.num_stocks})"

    def _filter_universe(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """유니버스 필터링을 수행한다.

        조건:
        - 시가총액 >= min_market_cap
        - 일 거래대금 >= min_volume
        - ROE > 0 (연속적자 기업 제외)

        Args:
            fundamentals: 전 종목 기본 지표 DataFrame

        Returns:
            필터링된 DataFrame
        """
        if fundamentals.empty:
            return fundamentals

        df = fundamentals.copy()
        original_count = len(df)

        # 시가총액 필터
        if "market_cap" in df.columns:
            df = df[df["market_cap"] >= self.min_market_cap]

        # 거래대금 필터 (volume * close)
        if "volume" in df.columns and "close" in df.columns:
            trade_value = df["volume"] * df["close"]
            df = df[trade_value >= self.min_volume]

        # ROE > 0 필터 (연속적자 기업 제외)
        if "roe" in df.columns:
            df = df[df["roe"] > 0]

        logger.info(f"퀄리티 유니버스 필터링: {original_count} -> {len(df)}개 종목")
        return df

    def _get_quality_data(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """펀더멘탈 데이터에서 퀄리티 지표를 추출/산출한다.

        DART에서 가져온 퀄리티 데이터가 fundamentals에 포함되어 있으면
        그대로 사용하고, 없으면 pykrx 기본 데이터로 추정한다.

        Args:
            fundamentals: 전 종목 기본 지표 DataFrame

        Returns:
            DataFrame with columns: ['ticker', 'roe', 'gpa', 'debt_ratio', 'accruals']
        """
        df = fundamentals.copy()

        if "ticker" not in df.columns:
            logger.warning("ticker 컬럼이 없습니다.")
            return pd.DataFrame()

        result = pd.DataFrame()
        result["ticker"] = df["ticker"]

        # ROE: 직접 제공되면 사용, 아니면 EPS/BPS로 추정
        if "roe" in df.columns:
            result["roe"] = df["roe"].values
        elif "eps" in df.columns and "bps" in df.columns:
            bps = df["bps"].replace(0, np.nan)
            result["roe"] = (df["eps"] / bps * 100).values
        else:
            result["roe"] = np.nan

        # GP/A: 직접 제공되면 사용, 아니면 ROE 기반 추정
        if "gp_over_assets" in df.columns:
            result["gpa"] = df["gp_over_assets"].values
        elif "gpa" in df.columns:
            result["gpa"] = df["gpa"].values
        else:
            # pykrx fallback: ROE 기반 대략적 추정
            result["gpa"] = result["roe"].abs() / 200

        # 부채비율: 직접 제공되면 사용
        if "debt_ratio" in df.columns:
            result["debt_ratio"] = df["debt_ratio"].values
        else:
            # 부채비율 데이터 없으면 NaN
            result["debt_ratio"] = np.nan

        # 발생액: 직접 제공되면 사용
        if "accruals" in df.columns:
            result["accruals"] = df["accruals"].values
        else:
            result["accruals"] = np.nan

        return result

    def calculate_quality_scores(self, fundamentals: pd.DataFrame) -> pd.Series:
        """종목별 퀄리티 종합 점수를 계산한다.

        각 퀄리티 지표를 백분위 순위로 변환한 후 가중 합산한다.
        - ROE, GP/A: rank(pct=True) - 높을수록 좋음
        - 부채비율, 발생액: 1 - rank(pct=True) - 낮을수록 좋음

        Args:
            fundamentals: 전 종목 기본 지표 DataFrame

        Returns:
            pd.Series (index=ticker, values=quality_score)
            빈 데이터일 경우 빈 Series 반환.
        """
        if fundamentals.empty:
            logger.warning("빈 펀더멘탈 데이터: 빈 스코어 반환")
            return pd.Series(dtype=float)

        quality_data = self._get_quality_data(fundamentals)
        if quality_data.empty or "ticker" not in quality_data.columns:
            logger.warning("퀄리티 데이터 추출 실패: 빈 스코어 반환")
            return pd.Series(dtype=float)

        # ticker를 인덱스로 설정
        quality_data = quality_data.set_index("ticker")

        # 유효 데이터가 있는 팩터만 사용
        scores = pd.Series(0.0, index=quality_data.index)
        total_weight = 0.0

        # ROE 점수 (높을수록 좋음)
        if "roe" in quality_data.columns and quality_data["roe"].notna().sum() > 0:
            roe_valid = quality_data["roe"].dropna()
            if len(roe_valid) > 0:
                roe_pctrank = roe_valid.rank(pct=True)
                w = self.weights.get("roe", 0.3)
                scores.loc[roe_pctrank.index] += roe_pctrank * w
                total_weight += w

        # GP/A 점수 (높을수록 좋음)
        if "gpa" in quality_data.columns and quality_data["gpa"].notna().sum() > 0:
            gpa_valid = quality_data["gpa"].dropna()
            if len(gpa_valid) > 0:
                gpa_pctrank = gpa_valid.rank(pct=True)
                w = self.weights.get("gpa", 0.3)
                scores.loc[gpa_pctrank.index] += gpa_pctrank * w
                total_weight += w

        # 부채비율 점수 (낮을수록 좋음 → 1 - rank)
        if "debt_ratio" in quality_data.columns and quality_data["debt_ratio"].notna().sum() > 0:
            debt_valid = quality_data["debt_ratio"].dropna()
            if len(debt_valid) > 0:
                debt_pctrank = 1 - debt_valid.rank(pct=True)
                w = self.weights.get("debt", 0.2)
                scores.loc[debt_pctrank.index] += debt_pctrank * w
                total_weight += w

        # 발생액 점수 (낮을수록 좋음 → 1 - rank)
        if "accruals" in quality_data.columns and quality_data["accruals"].notna().sum() > 0:
            accrual_valid = quality_data["accruals"].dropna()
            if len(accrual_valid) > 0:
                accrual_pctrank = 1 - accrual_valid.rank(pct=True)
                w = self.weights.get("accrual", 0.2)
                scores.loc[accrual_pctrank.index] += accrual_pctrank * w
                total_weight += w

        if total_weight == 0:
            logger.warning("유효한 퀄리티 팩터 없음: 빈 스코어 반환")
            return pd.Series(dtype=float)

        # 사용된 가중치로 정규화 (누락 팩터 보정)
        scores = scores / total_weight

        # NaN 종목 제거 (0.0은 유효한 스코어)
        scores = scores.dropna()

        logger.info(
            f"퀄리티 스코어 계산 완료: {len(scores)}개 종목, "
            f"사용 가중치 합계={total_weight:.2f}"
        )

        return scores

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        퀄리티 스코어 상위 N개 종목을 동일 비중으로 선택한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {'fundamentals': DataFrame, 'prices': dict[ticker, DataFrame],
                   'quality': DataFrame (optional)} 형태

        Returns:
            종목코드: 비중 딕셔너리
        """
        fundamentals = data.get("fundamentals", pd.DataFrame())

        if fundamentals.empty:
            logger.warning(f"펀더멘탈 데이터 없음 ({date}), 빈 시그널 반환")
            return {}

        # 퀄리티 데이터가 별도 제공되면 merge
        quality_data = data.get("quality", pd.DataFrame())
        if not quality_data.empty and "ticker" in quality_data.columns:
            # 기존 fundamentals에 퀄리티 컬럼 추가
            quality_cols = ["ticker", "roe", "gp_over_assets", "debt_ratio", "accruals"]
            available_quality_cols = [c for c in quality_cols if c in quality_data.columns]
            if len(available_quality_cols) > 1:  # ticker + 최소 1개 퀄리티 컬럼
                fundamentals = fundamentals.merge(
                    quality_data[available_quality_cols],
                    on="ticker",
                    how="left",
                    suffixes=("", "_quality"),
                )

        # 유니버스 필터링
        filtered = self._filter_universe(fundamentals)

        if filtered.empty:
            logger.warning(f"필터링 후 종목 없음 ({date})")
            return {}

        # 퀄리티 스코어 계산
        scores = self.calculate_quality_scores(filtered)

        if scores.empty:
            logger.warning(f"퀄리티 스코어 계산 실패 ({date})")
            return {}

        # 상위 N개 선택 (내림차순)
        top_scores = scores.sort_values(ascending=False).head(self.num_stocks)

        if top_scores.empty:
            return {}

        # 동일 비중 할당
        weight = 1.0 / len(top_scores)
        signals = {ticker: weight for ticker in top_scores.index}

        logger.info(
            f"퀄리티 시그널 생성 ({date}): {len(signals)}개 종목, "
            f"개별 비중={weight:.4f}"
        )

        return signals

    def get_scores(self, data: dict) -> pd.Series:
        """외부에서 퀄리티 스코어에 접근할 수 있도록 제공한다.

        ThreeFactorStrategy 등에서 팩터 결합 시 사용한다.

        Args:
            data: {'fundamentals': DataFrame, ...} 형태

        Returns:
            pd.Series (index=ticker, values=quality_score)
        """
        fundamentals = data.get("fundamentals", pd.DataFrame())
        if fundamentals.empty:
            return pd.Series(dtype=float)

        # 퀄리티 데이터 merge
        quality_data = data.get("quality", pd.DataFrame())
        if not quality_data.empty and "ticker" in quality_data.columns:
            quality_cols = ["ticker", "roe", "gp_over_assets", "debt_ratio", "accruals"]
            available_quality_cols = [c for c in quality_cols if c in quality_data.columns]
            if len(available_quality_cols) > 1:
                fundamentals = fundamentals.merge(
                    quality_data[available_quality_cols],
                    on="ticker",
                    how="left",
                    suffixes=("", "_quality"),
                )

        filtered = self._filter_universe(fundamentals)
        if filtered.empty:
            return pd.Series(dtype=float)

        return self.calculate_quality_scores(filtered)
