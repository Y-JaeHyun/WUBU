"""밸류 팩터 전략 모듈.

저PBR/저PER 종목을 선별하여 동일 비중 포트폴리오를 구성하는 밸류 팩터 전략을 제공한다.
"""

from typing import Optional

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
    """

    def __init__(
        self,
        factor: str = "pbr",
        num_stocks: int = 20,
        min_market_cap: int = 100_000_000_000,
        min_volume: int = 100_000_000,
        exclude_negative: bool = True,
    ):
        self._factor = factor.lower()
        self.num_stocks = num_stocks
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.exclude_negative = exclude_negative

        if self._factor not in ("pbr", "per", "composite"):
            raise ValueError(f"지원하지 않는 팩터: {factor}. 'pbr', 'per', 'composite' 중 선택하세요.")

        logger.info(
            f"ValueStrategy 초기화: factor={self._factor}, "
            f"num_stocks={num_stocks}, min_market_cap={min_market_cap:,}, "
            f"min_volume={min_volume:,}"
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

    def _rank_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """팩터 기준으로 종목을 랭킹한다 (낮을수록 좋음)."""
        if df.empty:
            return df

        if self._factor == "pbr":
            df = df.sort_values("pbr", ascending=True)
        elif self._factor == "per":
            df = df.sort_values("per", ascending=True)
        elif self._factor == "composite":
            # PBR, PER 각각의 랭킹을 구한 후 합산
            df = df.copy()
            df["pbr_rank"] = df["pbr"].rank(ascending=True)
            df["per_rank"] = df["per"].rank(ascending=True)
            df["composite_rank"] = df["pbr_rank"] + df["per_rank"]
            df = df.sort_values("composite_rank", ascending=True)

        return df

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
