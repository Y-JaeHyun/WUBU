"""매크로 경제 데이터 수집 모듈.

한국은행 ECOS API, FRED API, FinanceDataReader를 활용하여
기준금리, 미국 국채 금리, 달러 인덱스, VIX 등 매크로 지표를 수집한다.
API 키가 없으면 FinanceDataReader/yfinance로 fallback한다.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUEST_TIMEOUT = 10


class MacroCollector:
    """매크로 경제 데이터 수집기.

    한국은행 ECOS + FRED API를 사용하고,
    미설정 시 FinanceDataReader/yfinance로 대체한다.

    Args:
        ecos_key: 한국은행 ECOS API 키. None이면 환경변수에서 로드.
        fred_key: FRED API 키. None이면 환경변수에서 로드.
    """

    ECOS_BASE_URL = "https://ecos.bok.or.kr/api"
    FRED_BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(
        self,
        ecos_key: Optional[str] = None,
        fred_key: Optional[str] = None,
    ) -> None:
        self.ecos_key = ecos_key or os.getenv("ECOS_API_KEY", "")
        self.fred_key = fred_key or os.getenv("FRED_API_KEY", "")

        if not self.ecos_key:
            logger.info("ECOS_API_KEY 미설정. FinanceDataReader fallback.")
        if not self.fred_key:
            logger.info("FRED_API_KEY 미설정. yfinance/FDR fallback.")

    def get_bok_rate(self) -> Optional[float]:
        """한국은행 기준금리를 조회한다.

        ECOS API 우선, 없으면 FinanceDataReader fallback.

        Returns:
            기준금리(%). 조회 실패 시 None.
        """
        if self.ecos_key:
            rate = self._get_bok_rate_ecos()
            if rate is not None:
                return rate

        return self._get_bok_rate_fdr()

    def get_us_treasury(self, maturity: str = "10Y") -> Optional[float]:
        """미국 국채 금리를 조회한다.

        Args:
            maturity: 만기 ('10Y', '2Y', '30Y' 등).

        Returns:
            금리(%). 조회 실패 시 None.
        """
        series_map = {
            "2Y": "DGS2",
            "5Y": "DGS5",
            "10Y": "DGS10",
            "30Y": "DGS30",
        }
        series_id = series_map.get(maturity, "DGS10")

        if self.fred_key:
            rate = self._get_fred_series(series_id)
            if rate is not None:
                return rate

        return self._get_treasury_yfinance(maturity)

    def get_dollar_index(self) -> Optional[float]:
        """달러 인덱스를 조회한다.

        FRED API 우선, 없으면 yfinance fallback.

        Returns:
            달러 인덱스 값. 조회 실패 시 None.
        """
        if self.fred_key:
            value = self._get_fred_series("DTWEXBGS")
            if value is not None:
                return value

        return self._get_yfinance_price("DX-Y.NYB")

    def get_vix(self) -> Optional[float]:
        """VIX 지수를 조회한다.

        Returns:
            VIX 지수. 조회 실패 시 None.
        """
        return self._get_yfinance_price("^VIX")

    def get_usd_krw(self) -> Optional[float]:
        """USD/KRW 환율을 조회한다.

        Returns:
            환율. 조회 실패 시 None.
        """
        return self._get_yfinance_price("USDKRW=X")

    def get_macro_summary(self) -> dict[str, Any]:
        """매크로 데이터 종합 요약을 반환한다.

        Returns:
            매크로 지표 딕셔너리.
        """
        summary: dict[str, Any] = {}

        # 한국은행 기준금리
        bok_rate = self.get_bok_rate()
        if bok_rate is not None:
            summary["bok_rate"] = bok_rate

        # 미국 국채 10년물
        us_10y = self.get_us_treasury("10Y")
        if us_10y is not None:
            summary["us_treasury_10y"] = us_10y

        # 미국 국채 2년물
        us_2y = self.get_us_treasury("2Y")
        if us_2y is not None:
            summary["us_treasury_2y"] = us_2y

        # 장단기 금리차
        if us_10y is not None and us_2y is not None:
            summary["yield_spread"] = round(us_10y - us_2y, 2)

        # 달러 인덱스
        dxy = self.get_dollar_index()
        if dxy is not None:
            summary["dollar_index"] = dxy

        # VIX
        vix = self.get_vix()
        if vix is not None:
            summary["vix"] = vix

        # USD/KRW
        usd_krw = self.get_usd_krw()
        if usd_krw is not None:
            summary["usd_krw"] = usd_krw

        logger.info("매크로 데이터 수집 완료: %d개 지표", len(summary))
        return summary

    def format_macro_report(self) -> str:
        """텔레그램 발송용 매크로 리포트를 생성한다.

        Returns:
            포매팅된 매크로 리포트 문자열.
        """
        summary = self.get_macro_summary()

        if not summary:
            return "[매크로 리포트] 데이터 수집 실패"

        lines = [
            "[매크로 리포트]",
            "=" * 35,
        ]

        if "bok_rate" in summary:
            lines.append(f"  한은 기준금리: {summary['bok_rate']:.2f}%")

        if "us_treasury_10y" in summary:
            lines.append(f"  미국채 10Y:    {summary['us_treasury_10y']:.2f}%")

        if "us_treasury_2y" in summary:
            lines.append(f"  미국채 2Y:     {summary['us_treasury_2y']:.2f}%")

        if "yield_spread" in summary:
            spread = summary["yield_spread"]
            sign = "+" if spread >= 0 else ""
            inversion = " (역전)" if spread < 0 else ""
            lines.append(
                f"  장단기차(10-2Y): {sign}{spread:.2f}%{inversion}"
            )

        if "dollar_index" in summary:
            lines.append(f"  달러인덱스:    {summary['dollar_index']:.2f}")

        if "vix" in summary:
            vix = summary["vix"]
            vix_status = ""
            if vix > 30:
                vix_status = " (공포)"
            elif vix > 20:
                vix_status = " (경계)"
            else:
                vix_status = " (안정)"
            lines.append(f"  VIX:           {vix:.2f}{vix_status}")

        if "usd_krw" in summary:
            krw = summary["usd_krw"]
            krw_status = ""
            if krw > 1400:
                krw_status = " (원화 약세)"
            elif krw < 1300:
                krw_status = " (원화 강세)"
            lines.append(f"  USD/KRW:       {krw:,.1f}{krw_status}")

        lines.append("")
        lines.append("=" * 35)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ECOS API
    # ------------------------------------------------------------------

    def _get_bok_rate_ecos(self) -> Optional[float]:
        """ECOS API로 한국은행 기준금리를 조회한다."""
        try:
            # 통계코드: 722Y001, 아이템: 0101000 (기준금리)
            url = (
                f"{self.ECOS_BASE_URL}/StatisticSearch/"
                f"{self.ecos_key}/json/kr/1/1/722Y001/M/"
            )
            # 최근 1개월
            from datetime import datetime

            now = datetime.now()
            end = now.strftime("%Y%m")
            start = (now.replace(day=1)).strftime("%Y%m")
            url += f"{start}/{end}/0101000"

            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            rows = data.get("StatisticSearch", {}).get("row", [])
            if rows:
                return float(rows[-1].get("DATA_VALUE", 0))

            return None
        except Exception as e:
            logger.warning("ECOS 기준금리 조회 실패: %s", e)
            return None

    @staticmethod
    def _get_bok_rate_fdr() -> Optional[float]:
        """FinanceDataReader로 한국은행 기준금리를 조회한다."""
        try:
            import FinanceDataReader as fdr

            df = fdr.DataReader("KR1YT=RR")
            if df is not None and not df.empty:
                return round(float(df["Close"].iloc[-1]), 2)
        except ImportError:
            logger.debug("FinanceDataReader 미설치")
        except Exception as e:
            logger.debug("FDR 기준금리 조회 실패: %s", e)
        return None

    # ------------------------------------------------------------------
    # FRED API
    # ------------------------------------------------------------------

    def _get_fred_series(self, series_id: str) -> Optional[float]:
        """FRED API에서 시계열 최신값을 조회한다."""
        try:
            url = f"{self.FRED_BASE_URL}/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.fred_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            }

            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            observations = data.get("observations", [])
            for obs in observations:
                value = obs.get("value", ".")
                if value != ".":
                    return round(float(value), 4)

            return None
        except Exception as e:
            logger.warning("FRED %s 조회 실패: %s", series_id, e)
            return None

    # ------------------------------------------------------------------
    # yfinance / FDR fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _get_treasury_yfinance(maturity: str) -> Optional[float]:
        """yfinance로 미국 국채 금리를 조회한다."""
        symbol_map = {
            "2Y": "^IRX",
            "5Y": "^FVX",
            "10Y": "^TNX",
            "30Y": "^TYX",
        }
        symbol = symbol_map.get(maturity)
        if not symbol:
            return None

        return MacroCollector._get_yfinance_price(symbol)

    @staticmethod
    def _get_yfinance_price(symbol: str) -> Optional[float]:
        """yfinance로 심볼의 현재가를 조회한다."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            price = getattr(info, "last_price", None)
            if price is not None and price > 0:
                return round(float(price), 4)
            return None
        except ImportError:
            logger.debug("yfinance 미설치")
            return None
        except Exception as e:
            logger.debug("yfinance %s 조회 실패: %s", symbol, e)
            return None
