"""소형 가치주(Size-Value Interaction) 전략 모듈.

시가총액 하위 종목 중 저PBR 종목을 선별하여 동일 비중 포트폴리오를 구성한다.
한국 시장에서 사이즈 팩터 프리미엄이 가장 큰 점을 활용한 전략이다.

참고: Kim et al., "Enhanced Factor Investing in the Korean Stock Market",
      Pacific-Basin Finance Journal.
"""

import numpy as np
import pandas as pd

from src.backtest.engine import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SizeValueStrategy(Strategy):
    """소형 가치주 집중 전략.

    시가총액 하위 N% 중 PBR이 낮은 종목을 동일 비중으로 보유한다.
    사이즈×밸류 인터랙션 효과를 극대화하는 전략이다.

    Args:
        size_pct: 시가총액 하위 비율 (기본 0.30 = 하위 30%)
        value_pct: PBR 하위 비율 (기본 0.20 = 하위 20%)
        max_stocks: 최대 포트폴리오 종목 수 (기본 20)
        min_volume: 최소 일 거래대금 (기본 5000만원)
        value_factor: 밸류 팩터 ('pbr', 'per', 'composite')
        exclude_negative_earnings: 적자 기업 제외 여부
    """

    def __init__(
        self,
        size_pct: float = 0.30,
        value_pct: float = 0.20,
        max_stocks: int = 20,
        min_volume: int = 50_000_000,
        value_factor: str = "pbr",
        exclude_negative_earnings: bool = True,
    ):
        self.size_pct = size_pct
        self.value_pct = value_pct
        self.max_stocks = max_stocks
        self.min_volume = min_volume
        self.value_factor = value_factor.lower()
        self.exclude_negative_earnings = exclude_negative_earnings

        if self.value_factor not in ("pbr", "per", "composite"):
            raise ValueError(f"지원하지 않는 밸류 팩터: {value_factor}")

        logger.info(
            f"SizeValueStrategy 초기화: size_pct={size_pct}, "
            f"value_pct={value_pct}, max_stocks={max_stocks}, "
            f"min_volume={min_volume:,}, factor={self.value_factor}"
        )

    @property
    def name(self) -> str:
        return f"SizeValue(bottom{int(self.size_pct*100)}%×{self.value_factor.upper()}, top{self.max_stocks})"

    def generate_signals(self, date: str, data: dict) -> dict[str, float]:
        """날짜별 포트폴리오 비중을 생성한다."""
        fundamentals = data.get("fundamentals", pd.DataFrame())

        if fundamentals.empty:
            logger.warning(f"펀더멘탈 데이터 없음 ({date}), 빈 시그널 반환")
            return {}

        df = fundamentals.copy()
        original_count = len(df)

        # 1. 기본 필터: 거래대금, PBR/PER > 0
        df = self._apply_basic_filter(df)
        if df.empty:
            return {}

        # 2. 소형주 필터: 시가총액 하위 N%
        df = self._apply_size_filter(df)
        if df.empty:
            logger.warning(f"소형주 필터 후 종목 없음 ({date})")
            return {}

        # 3. 밸류 필터: PBR/PER 하위 N%
        selected = self._apply_value_filter(df)
        if selected.empty:
            logger.warning(f"밸류 필터 후 종목 없음 ({date})")
            return {}

        # 4. 종목 수 제한
        selected = selected.head(self.max_stocks)

        # 5. 동일 비중
        weight = 1.0 / len(selected)
        signals = {row["ticker"]: weight for _, row in selected.iterrows()}

        logger.info(
            f"SizeValue 시그널 ({date}): {len(signals)}개 종목 "
            f"(전체 {original_count} → 거래대금필터 → "
            f"소형주 {int(self.size_pct*100)}% → "
            f"밸류 {int(self.value_pct*100)}%)"
        )

        return signals

    def _apply_basic_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 유니버스 필터링."""
        # 거래대금 필터
        if "volume" in df.columns and "close" in df.columns:
            trade_value = df["volume"] * df["close"]
            df = df[trade_value >= self.min_volume]

        # PBR/PER > 0 (자본잠식, 적자 제외)
        if self.value_factor in ("pbr", "composite") and "pbr" in df.columns:
            df = df[df["pbr"] > 0]
        if self.value_factor in ("per", "composite") and "per" in df.columns:
            df = df[df["per"] > 0]

        # 적자 기업 제외
        if self.exclude_negative_earnings and "eps" in df.columns:
            df = df[df["eps"] > 0]

        # 시가총액 데이터 필수
        if "market_cap" not in df.columns or df.empty:
            return pd.DataFrame()

        return df

    def _apply_size_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """시가총액 하위 N% 소형주 필터."""
        cutoff = df["market_cap"].quantile(self.size_pct)
        small_caps = df[df["market_cap"] <= cutoff]

        logger.info(
            f"소형주 필터: 시총 <= {cutoff/1e8:,.0f}억원, "
            f"{len(df)} → {len(small_caps)}개"
        )
        return small_caps

    def _apply_value_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """밸류 팩터 하위 N% 선택 (저PBR/저PER)."""
        if self.value_factor == "pbr":
            sort_col = "pbr"
        elif self.value_factor == "per":
            sort_col = "per"
        else:  # composite
            df = df.copy()
            df["pbr_rank"] = df["pbr"].rank(ascending=True)
            df["per_rank"] = df["per"].rank(ascending=True)
            df["composite_rank"] = df["pbr_rank"] + df["per_rank"]
            sort_col = "composite_rank"

        df = df.sort_values(sort_col, ascending=True)
        n_select = max(1, int(len(df) * self.value_pct))
        return df.head(n_select)
