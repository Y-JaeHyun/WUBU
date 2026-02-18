"""시장 스캐너 모듈.

밸류, 모멘텀, 결합 스코어 기준으로 상위 종목을 스캔하는 기능을 제공한다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from src.data.collector import get_all_fundamentals
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketScanner:
    """시장 스캐너.

    다양한 팩터 기준으로 상위 종목을 스캔하고 포매팅된 결과를 제공한다.

    Args:
        market: 스캔 대상 시장 ('KOSPI', 'KOSDAQ', 'ALL')
        min_market_cap: 최소 시가총액 필터 (기본 500억원)
        min_volume: 최소 일 거래량 필터 (기본 10,000주)
    """

    def __init__(
        self,
        market: str = "ALL",
        min_market_cap: int = 50_000_000_000,
        min_volume: int = 10_000,
    ) -> None:
        self.market = market.upper()
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume

    def _fetch_and_filter(self, date: str) -> pd.DataFrame:
        """기본 데이터를 가져와서 기본 필터를 적용한다.

        Args:
            date: 조회일 ('YYYYMMDD' 또는 'YYYY-MM-DD')

        Returns:
            필터링된 DataFrame
        """
        date_str = date.replace("-", "")
        logger.info(f"스캐너 데이터 조회: {date_str} (market={self.market})")

        df = get_all_fundamentals(date_str, market=self.market)

        if df.empty:
            logger.warning(f"스캐너 데이터 없음: {date_str}")
            return df

        original_count = len(df)

        # 시가총액 필터
        if "market_cap" in df.columns:
            df = df[df["market_cap"] >= self.min_market_cap]

        # 거래량 필터
        if "volume" in df.columns:
            df = df[df["volume"] >= self.min_volume]

        # PBR/PER이 0인 종목(관리종목 등) 제외
        if "pbr" in df.columns:
            df = df[df["pbr"] != 0]
        if "per" in df.columns:
            df = df[df["per"] != 0]

        logger.info(f"스캐너 필터링: {original_count} -> {len(df)}개 종목")
        return df

    def scan_value_top(self, date: str, n: int = 20) -> pd.DataFrame:
        """밸류 상위 종목을 스캔한다.

        저PBR + 저PER 결합 랭킹 기준으로 상위 N개를 반환한다.

        Args:
            date: 조회일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
            n: 반환할 종목 수 (기본 20)

        Returns:
            DataFrame with columns: ['ticker', 'name', 'market', 'close',
                                      'market_cap', 'per', 'pbr', 'div_yield',
                                      'value_rank']
        """
        df = self._fetch_and_filter(date)
        if df.empty:
            return pd.DataFrame()

        # 음수 PBR/PER 제외 (적자 기업)
        df = df[(df["pbr"] > 0) & (df["per"] > 0)].copy()

        if df.empty:
            return pd.DataFrame()

        # PBR, PER 랭킹 (낮을수록 좋음)
        df["pbr_rank"] = df["pbr"].rank(ascending=True)
        df["per_rank"] = df["per"].rank(ascending=True)
        df["value_rank"] = (df["pbr_rank"] + df["per_rank"]).rank(ascending=True).astype(int)

        df = df.sort_values("value_rank", ascending=True).head(n)

        # 결과 컬럼 정리
        result_cols = ["ticker", "name", "market", "close", "market_cap", "per", "pbr"]
        if "div_yield" in df.columns:
            result_cols.append("div_yield")
        result_cols.append("value_rank")

        result_cols = [c for c in result_cols if c in df.columns]
        result = df[result_cols].reset_index(drop=True)

        logger.info(f"밸류 스캔 완료: 상위 {len(result)}개 종목")
        return result

    def scan_momentum_top(self, date: str, n: int = 20) -> pd.DataFrame:
        """모멘텀 상위 종목을 스캔한다 (placeholder).

        향후 MomentumStrategy 연동 시 실제 모멘텀 팩터를 사용한다.
        현재는 거래량 기준 상위 종목을 반환한다.

        Args:
            date: 조회일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
            n: 반환할 종목 수 (기본 20)

        Returns:
            DataFrame with columns: ['ticker', 'name', 'market', 'close',
                                      'market_cap', 'volume', 'momentum_rank']
        """
        df = self._fetch_and_filter(date)
        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        # placeholder: 거래량 기준 (향후 모멘텀 팩터로 대체)
        if "volume" in df.columns and "close" in df.columns:
            df["trade_value"] = df["volume"] * df["close"]
            df["momentum_rank"] = df["trade_value"].rank(ascending=False).astype(int)
            df = df.sort_values("momentum_rank", ascending=True).head(n)
        else:
            df = df.head(n)
            df["momentum_rank"] = range(1, len(df) + 1)

        result_cols = ["ticker", "name", "market", "close", "market_cap", "volume", "momentum_rank"]
        result_cols = [c for c in result_cols if c in df.columns]
        result = df[result_cols].reset_index(drop=True)

        logger.info(f"모멘텀 스캔 완료 (placeholder): 상위 {len(result)}개 종목")
        return result

    def scan_combined_top(self, date: str, n: int = 20) -> pd.DataFrame:
        """결합 스코어 상위 종목을 스캔한다 (placeholder).

        밸류 + 모멘텀 결합 랭킹. 향후 모멘텀 팩터 추가 시 완전히 구현된다.
        현재는 밸류 랭킹 + 거래대금 랭킹의 결합으로 대체한다.

        Args:
            date: 조회일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
            n: 반환할 종목 수 (기본 20)

        Returns:
            DataFrame with columns: ['ticker', 'name', 'market', 'close',
                                      'market_cap', 'per', 'pbr', 'volume',
                                      'combined_rank']
        """
        df = self._fetch_and_filter(date)
        if df.empty:
            return pd.DataFrame()

        # 음수 PBR/PER 제외
        df = df[(df["pbr"] > 0) & (df["per"] > 0)].copy()
        if df.empty:
            return pd.DataFrame()

        # 밸류 랭킹
        df["pbr_rank"] = df["pbr"].rank(ascending=True)
        df["per_rank"] = df["per"].rank(ascending=True)
        df["value_score"] = df["pbr_rank"] + df["per_rank"]

        # 모멘텀 대리변수: 거래대금 (placeholder)
        if "volume" in df.columns and "close" in df.columns:
            df["trade_value"] = df["volume"] * df["close"]
            df["momentum_score"] = df["trade_value"].rank(ascending=False)
        else:
            df["momentum_score"] = 0

        # 결합 랭킹 (밸류 70% + 모멘텀 30%)
        df["combined_score"] = df["value_score"].rank() * 0.7 + df["momentum_score"].rank() * 0.3
        df["combined_rank"] = df["combined_score"].rank(ascending=True).astype(int)

        df = df.sort_values("combined_rank", ascending=True).head(n)

        result_cols = [
            "ticker", "name", "market", "close", "market_cap",
            "per", "pbr", "volume", "combined_rank",
        ]
        result_cols = [c for c in result_cols if c in df.columns]
        result = df[result_cols].reset_index(drop=True)

        logger.info(f"결합 스캔 완료 (placeholder): 상위 {len(result)}개 종목")
        return result

    def format_scan_result(self, df: pd.DataFrame, title: str) -> str:
        """스캔 결과를 터미널 출력용 텍스트로 포매팅한다.

        Args:
            df: 스캔 결과 DataFrame
            title: 섹션 제목

        Returns:
            포매팅된 문자열
        """
        if df.empty:
            return f"\n  {title}\n  데이터 없음\n"

        lines: list[str] = []
        lines.append("")
        lines.append(f"  {title}")
        lines.append(f"  {'─' * (len(title) + 10)}")

        # 컬럼 설정
        display_cols = []
        col_formats: dict[str, dict] = {}

        for col in df.columns:
            if col == "ticker":
                display_cols.append(("ticker", "종목코드"))
                col_formats["ticker"] = {"width": 8, "align": "c"}
            elif col == "name":
                display_cols.append(("name", "종목명"))
                col_formats["name"] = {"width": 14, "align": "l"}
            elif col == "market":
                display_cols.append(("market", "시장"))
                col_formats["market"] = {"width": 8, "align": "c"}
            elif col == "close":
                display_cols.append(("close", "현재가"))
                col_formats["close"] = {"width": 12, "align": "r", "fmt": ",.0f"}
            elif col == "market_cap":
                display_cols.append(("market_cap", "시가총액(억)"))
                col_formats["market_cap"] = {"width": 14, "align": "r", "fmt": ",.0f", "div": 100_000_000}
            elif col == "per":
                display_cols.append(("per", "PER"))
                col_formats["per"] = {"width": 8, "align": "r", "fmt": ".1f"}
            elif col == "pbr":
                display_cols.append(("pbr", "PBR"))
                col_formats["pbr"] = {"width": 8, "align": "r", "fmt": ".2f"}
            elif col == "div_yield":
                display_cols.append(("div_yield", "배당률"))
                col_formats["div_yield"] = {"width": 8, "align": "r", "fmt": ".2f"}
            elif col == "volume":
                display_cols.append(("volume", "거래량"))
                col_formats["volume"] = {"width": 12, "align": "r", "fmt": ",.0f"}
            elif col.endswith("_rank"):
                display_cols.append((col, "순위"))
                col_formats[col] = {"width": 6, "align": "c", "fmt": "d"}

        if not display_cols:
            return f"\n  {title}\n  표시할 컬럼 없음\n"

        headers = [h for _, h in display_cols]
        col_keys = [k for k, _ in display_cols]

        # 컬럼 너비 결정
        widths = [max(col_formats.get(k, {}).get("width", 10), len(h) + 2) for k, h in display_cols]

        # 헤더
        header_line = "  "
        sep_line = "  "
        for i, h in enumerate(headers):
            header_line += h.center(widths[i])
            sep_line += "─" * widths[i]

        lines.append(header_line)
        lines.append(sep_line)

        # 데이터 행
        for _, row in df.iterrows():
            row_line = "  "
            for i, key in enumerate(col_keys):
                val = row.get(key, "")
                fmt_info = col_formats.get(key, {})
                fmt_str = fmt_info.get("fmt", "")
                divisor = fmt_info.get("div", 1)
                align = fmt_info.get("align", "r")

                try:
                    if pd.isna(val):
                        cell = "-"
                    elif fmt_str and isinstance(val, (int, float)):
                        cell = format(val / divisor, fmt_str)
                    else:
                        cell = str(val)
                except (ValueError, TypeError):
                    cell = str(val)

                if align == "l":
                    row_line += cell.ljust(widths[i])
                elif align == "c":
                    row_line += cell.center(widths[i])
                else:
                    row_line += cell.rjust(widths[i])

            lines.append(row_line)

        lines.append("")
        return "\n".join(lines)

    def run_full_scan(self, date: str) -> str:
        """전체 스캔(밸류 + 모멘텀 + 결합)을 실행하고 통합 텍스트 결과를 반환한다.

        Args:
            date: 조회일 ('YYYYMMDD' 또는 'YYYY-MM-DD')

        Returns:
            전체 스캔 결과 문자열
        """
        date_str = date.replace("-", "")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("  MARKET SCANNER REPORT")
        lines.append(f"  조회일: {date_str}  |  시장: {self.market}")
        lines.append(f"  생성일시: {now_str}")
        lines.append(f"  최소 시가총액: {self.min_market_cap / 100_000_000:,.0f}억원")
        lines.append("=" * 70)

        # 1. 밸류 상위
        try:
            value_df = self.scan_value_top(date, n=20)
            lines.append(self.format_scan_result(value_df, "밸류 상위 20 (저PBR + 저PER)"))
        except Exception as e:
            lines.append(f"\n  [밸류 스캔 실패] {e}\n")
            logger.error(f"밸류 스캔 실패: {e}")

        # 2. 모멘텀 상위
        try:
            momentum_df = self.scan_momentum_top(date, n=20)
            lines.append(self.format_scan_result(
                momentum_df,
                "모멘텀 상위 20 (거래대금 기준 — placeholder)",
            ))
        except Exception as e:
            lines.append(f"\n  [모멘텀 스캔 실패] {e}\n")
            logger.error(f"모멘텀 스캔 실패: {e}")

        # 3. 결합 상위
        try:
            combined_df = self.scan_combined_top(date, n=20)
            lines.append(self.format_scan_result(
                combined_df,
                "결합 상위 20 (밸류 70% + 모멘텀 30% — placeholder)",
            ))
        except Exception as e:
            lines.append(f"\n  [결합 스캔 실패] {e}\n")
            logger.error(f"결합 스캔 실패: {e}")

        lines.append("=" * 70)
        report = "\n".join(lines)

        logger.info(f"전체 스캔 완료: {date_str}")
        return report
