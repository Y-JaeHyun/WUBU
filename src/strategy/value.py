"""밸류 팩터 전략 모듈.

저PBR/저PER 종목을 선별하여 동일 비중 포트폴리오를 구성하는 밸류 팩터 전략을 제공한다.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValueStrategy(Strategy):
    """밸류 팩터 전략.

    저PBR 또는 저PER 상위 N개 종목을 동일 비중으로 보유하는 전략이다.

    Args:
        factor: 사용할 밸류 팩터 ('pbr', 'per', 또는 'composite')
        num_stocks: 포트폴리오 종목 수 (기본 20)
        min_market_cap: 최소 시가총액 (기본 1000억원)
        min_volume: 최소 일 거래대금 (기본 1억원)
        exclude_negative: 음수 팩터값(적자 등) 제외 여부 (기본 True)
        industry_neutral: 업종중립 Z-Score 적용 여부 (기본 False).
            활성화 시 업종별로 PBR/PER을 Z-Score 정규화하여 업종간 편향 제거.
            fundamentals에 'sector' 또는 'industry' 컬럼 필요.
        shareholder_yield: 주주환원(배당수익률) 팩터 가산 여부 (기본 False).
            활성화 시 DIV_YIELD 또는 div_yield 컬럼을 밸류 스코어에 가산.
        f_score_filter: 피오트로스키 F-Score 최소 기준 (기본 0, 비활성).
            설정 시 해당 값 이상인 종목만 선택. 간소화 버전(ROA>0, ROA증가, 부채비율감소)
            또는 fundamentals에 'f_score' 컬럼이 있으면 직접 사용.
    """

    def __init__(
        self,
        factor: str = "pbr",
        num_stocks: int = 20,
        min_market_cap: int = 100_000_000_000,
        min_volume: int = 100_000_000,
        exclude_negative: bool = True,
        industry_neutral: bool = False,
        shareholder_yield: bool = False,
        f_score_filter: int = 0,
    ):
        self._factor = factor.lower()
        self.num_stocks = num_stocks
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.exclude_negative = exclude_negative
        self.industry_neutral = industry_neutral
        self.shareholder_yield = shareholder_yield
        self.f_score_filter = f_score_filter

        if self._factor not in ("pbr", "per", "composite"):
            raise ValueError(f"지원하지 않는 팩터: {factor}. 'pbr', 'per', 'composite' 중 선택하세요.")

        logger.info(
            f"ValueStrategy 초기화: factor={self._factor}, "
            f"num_stocks={num_stocks}, min_market_cap={min_market_cap:,}, "
            f"min_volume={min_volume:,}, industry_neutral={industry_neutral}, "
            f"shareholder_yield={shareholder_yield}, f_score_filter={f_score_filter}"
        )

    @property
    def name(self) -> str:
        """전략 이름."""
        return f"Value({self._factor.upper()}, top{self.num_stocks})"

    def _filter_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """유니버스 필터링을 수행한다.

        - 시가총액 하한
        - 거래대금 하한
        - 관리종목 제외 (PER/PBR이 0인 종목 제외)
        - 음수 팩터값 제외 (선택)
        """
        if df.empty:
            return df

        original_count = len(df)

        # 시가총액 필터
        if "market_cap" in df.columns:
            df = df[df["market_cap"] >= self.min_market_cap]

        # 거래대금 필터 (volume * close 근사)
        if "volume" in df.columns and "close" in df.columns:
            trade_value = df["volume"] * df["close"]
            df = df[trade_value >= self.min_volume]

        # PBR/PER이 0이면 관리종목이거나 데이터 없음으로 제외
        if self._factor in ("pbr", "composite") and "pbr" in df.columns:
            df = df[df["pbr"] != 0]

        if self._factor in ("per", "composite") and "per" in df.columns:
            df = df[df["per"] != 0]

        # 음수값 제외 (적자 기업 등)
        if self.exclude_negative:
            if self._factor == "pbr" and "pbr" in df.columns:
                df = df[df["pbr"] > 0]
            elif self._factor == "per" and "per" in df.columns:
                df = df[df["per"] > 0]
            elif self._factor == "composite":
                if "pbr" in df.columns:
                    df = df[df["pbr"] > 0]
                if "per" in df.columns:
                    df = df[df["per"] > 0]

        logger.info(f"유니버스 필터링: {original_count} -> {len(df)}개 종목")
        return df

    def _calculate_f_score(self, df: pd.DataFrame) -> pd.Series:
        """간소화 F-Score를 계산한다.

        fundamentals에 'f_score' 컬럼이 있으면 직접 사용.
        없으면 간소화 버전(ROA>0, ROA증가, 부채비율감소) 3가지로 계산 (0~3점).

        Args:
            df: 펀더멘탈 DataFrame (ticker 포함)

        Returns:
            pd.Series (index=DataFrame index, values=f_score)
        """
        if "f_score" in df.columns:
            return df["f_score"]

        score = pd.Series(0, index=df.index, dtype=int)

        # ROA > 0 (EPS/BPS 또는 ROE로 대용)
        if "roe" in df.columns:
            score += (df["roe"] > 0).astype(int)
        elif "eps" in df.columns:
            score += (df["eps"] > 0).astype(int)

        # ROA 증가 (prev_roe 컬럼이 있는 경우)
        if "prev_roe" in df.columns and "roe" in df.columns:
            score += (df["roe"] > df["prev_roe"]).astype(int)
        elif "prev_eps" in df.columns and "eps" in df.columns:
            score += (df["eps"] > df["prev_eps"]).astype(int)

        # 부채비율 감소 (prev_debt_ratio 컬럼이 있는 경우)
        if "prev_debt_ratio" in df.columns and "debt_ratio" in df.columns:
            score += (df["debt_ratio"] < df["prev_debt_ratio"]).astype(int)

        return score

    def _apply_industry_neutral(self, df: pd.DataFrame, factor_col: str) -> pd.Series:
        """업종별 Z-Score 정규화를 수행한다.

        업종(sector/industry) 내에서 팩터값을 Z-Score로 변환하여
        업종간 편향을 제거한다.

        Args:
            df: 펀더멘탈 DataFrame (sector 또는 industry 컬럼 필요)
            factor_col: Z-Score 정규화할 팩터 컬럼명

        Returns:
            pd.Series: 업종중립 Z-Score (낮을수록 좋은 밸류이므로, 음수가 저평가)
        """
        sector_col = None
        for col in ("sector", "industry"):
            if col in df.columns:
                sector_col = col
                break

        if sector_col is None:
            logger.warning("업종 컬럼(sector/industry) 없음: 전체 Z-Score 사용")
            vals = df[factor_col]
            std = vals.std()
            if std == 0 or np.isnan(std):
                return pd.Series(0.0, index=df.index)
            return (vals - vals.mean()) / std

        def _zscore_group(group):
            std = group.std()
            if std == 0 or np.isnan(std) or len(group) < 2:
                return pd.Series(0.0, index=group.index)
            return (group - group.mean()) / std

        return df.groupby(sector_col)[factor_col].transform(_zscore_group)

    def _rank_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """팩터 기준으로 종목을 랭킹한다 (낮을수록 좋음).

        industry_neutral=True 시 업종별 Z-Score로 정규화.
        shareholder_yield=True 시 배당수익률을 밸류 스코어에 가산.
        """
        if df.empty:
            return df

        df = df.copy()

        # 업종중립 Z-Score 모드
        if self.industry_neutral:
            if self._factor in ("pbr", "composite") and "pbr" in df.columns:
                df["pbr_zscore"] = self._apply_industry_neutral(df, "pbr")
            if self._factor in ("per", "composite") and "per" in df.columns:
                df["per_zscore"] = self._apply_industry_neutral(df, "per")

            if self._factor == "pbr" and "pbr_zscore" in df.columns:
                sort_col = "pbr_zscore"
            elif self._factor == "per" and "per_zscore" in df.columns:
                sort_col = "per_zscore"
            elif self._factor == "composite":
                if "pbr_zscore" in df.columns:
                    df["pbr_rank"] = df["pbr_zscore"].rank(ascending=True)
                else:
                    df["pbr_rank"] = df["pbr"].rank(ascending=True)
                if "per_zscore" in df.columns:
                    df["per_rank"] = df["per_zscore"].rank(ascending=True)
                else:
                    df["per_rank"] = df["per"].rank(ascending=True)
                df["composite_rank"] = df["pbr_rank"] + df["per_rank"]
                sort_col = "composite_rank"
            else:
                sort_col = self._factor

            # 주주환원 팩터 가산 (업종중립 모드)
            if self.shareholder_yield and sort_col != "composite_rank":
                div_col = self._get_div_yield_col(df)
                if div_col is not None:
                    # 배당수익률이 높을수록 좋으므로, Z-Score 값에서 차감 (낮을수록 좋은 체계)
                    div_vals = df[div_col].fillna(0)
                    div_std = div_vals.std()
                    if div_std > 0 and not np.isnan(div_std):
                        div_z = (div_vals - div_vals.mean()) / div_std
                        df[sort_col] = df[sort_col] - div_z

            df = df.sort_values(sort_col, ascending=True)
        else:
            # 기존 로직
            if self._factor == "pbr":
                sort_col = "pbr"
            elif self._factor == "per":
                sort_col = "per"
            elif self._factor == "composite":
                df["pbr_rank"] = df["pbr"].rank(ascending=True)
                df["per_rank"] = df["per"].rank(ascending=True)
                df["composite_rank"] = df["pbr_rank"] + df["per_rank"]
                sort_col = "composite_rank"
            else:
                sort_col = self._factor

            # 주주환원 팩터 가산 (일반 모드)
            if self.shareholder_yield and sort_col != "composite_rank":
                div_col = self._get_div_yield_col(df)
                if div_col is not None:
                    # 배당수익률이 높을수록 좋으므로 랭킹에서 감산 (순위가 낮아짐 = 더 좋은 밸류)
                    div_pctrank = df[div_col].fillna(0).rank(pct=True, ascending=False)
                    factor_rank = df[sort_col].rank(ascending=True, pct=True)
                    # 밸류 순위(70%) + 배당 순위(30%) 결합
                    df["_value_div_rank"] = factor_rank * 0.7 + div_pctrank * 0.3
                    sort_col = "_value_div_rank"
            elif self.shareholder_yield and sort_col == "composite_rank":
                div_col = self._get_div_yield_col(df)
                if div_col is not None:
                    div_rank = df[div_col].fillna(0).rank(ascending=False)
                    df["composite_rank"] = df["composite_rank"] + div_rank * 0.5

            df = df.sort_values(sort_col, ascending=True)

        return df

    def _get_div_yield_col(self, df: pd.DataFrame) -> Optional[str]:
        """배당수익률 컬럼명을 찾는다."""
        for col in ("div_yield", "DIV_YIELD", "dividend_yield"):
            if col in df.columns:
                return col
        return None

    def generate_signals(self, date: str, data: dict) -> dict:
        """날짜별 포트폴리오 비중을 생성한다.

        저PBR/저PER 상위 N개 종목을 동일 비중으로 선택한다.

        Args:
            date: 리밸런싱 날짜 ('YYYYMMDD')
            data: {'fundamentals': DataFrame, 'prices': dict} 형태

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

        # F-Score 필터 적용
        if self.f_score_filter > 0:
            f_scores = self._calculate_f_score(filtered)
            filtered = filtered[f_scores >= self.f_score_filter]
            logger.info(f"F-Score 필터({self.f_score_filter}) 적용 후: {len(filtered)}개 종목")
            if filtered.empty:
                logger.warning(f"F-Score 필터 후 종목 없음 ({date})")
                return {}

        # 팩터 기준 랭킹
        ranked = self._rank_stocks(filtered)

        # 상위 N개 선택
        selected = ranked.head(self.num_stocks)

        # 동일 비중 할당
        weight = 1.0 / len(selected)
        signals = {row["ticker"]: weight for _, row in selected.iterrows()}

        logger.info(
            f"시그널 생성 ({date}): {len(signals)}개 종목, "
            f"개별 비중={weight:.4f}"
        )

        return signals
