"""ETF 가격 데이터 수집 모듈.

pykrx 또는 KRX Open API를 활용하여 한국 상장 ETF의 가격(OHLCV) 데이터와
종목 리스트를 수집한다. 듀얼 모멘텀 등 ETF 기반 자산배분 전략에서 사용한다.
"""

import time
from typing import Optional

import pandas as pd

from src.data.data_proxy import create_stock_api

pykrx_stock = create_stock_api()

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 듀얼 모멘텀 전략용 기본 ETF 유니버스
ETF_UNIVERSE: dict[str, dict[str, str]] = {
    "domestic_equity": {"ticker": "069500", "name": "KODEX 200"},
    "us_equity": {"ticker": "360750", "name": "TIGER 미국S&P500"},
    "us_nasdaq": {"ticker": "133690", "name": "TIGER 미국나스닥100"},
    "short_bond": {"ticker": "214980", "name": "KODEX 단기채권PLUS"},
    "gov_bond_3y": {"ticker": "114820", "name": "KODEX 국고채3년"},
    "gold": {"ticker": "132030", "name": "KODEX 골드선물(H)"},
    "inverse_200": {"ticker": "114800", "name": "KODEX 인버스"},
    "gov_bond_10y": {"ticker": "148070", "name": "KOSEF 국고채10년"},
}


def _format_date(date) -> str:
    """날짜를 pykrx가 요구하는 'YYYYMMDD' 문자열로 변환한다.

    Args:
        date: 날짜 (str, pd.Timestamp, datetime 등)

    Returns:
        'YYYYMMDD' 형식 문자열
    """
    if isinstance(date, str):
        return date.replace("-", "")
    if isinstance(date, pd.Timestamp):
        return date.strftime("%Y%m%d")
    return date.strftime("%Y%m%d")


def get_etf_price(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """ETF의 일별 OHLCV 데이터를 반환한다.

    pykrx의 get_etf_ohlcv_by_date를 사용하여 ETF 가격 데이터를 조회한다.
    ETF 전용 API가 실패하면 일반 주식 API로 fallback한다.

    Args:
        ticker: ETF 종목코드 (예: '069500')
        start_date: 시작일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
        end_date: 종료일 ('YYYYMMDD' 또는 'YYYY-MM-DD')

    Returns:
        DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
        Index: DatetimeIndex (날짜)
        데이터가 없으면 빈 DataFrame 반환.
    """
    start = _format_date(start_date)
    end = _format_date(end_date)
    logger.info(f"ETF 가격 데이터 조회: {ticker} ({start} ~ {end})")

    try:
        df = pykrx_stock.get_etf_ohlcv_by_date(start, end, ticker)
        time.sleep(0.5)

        if df.empty:
            logger.warning(
                f"ETF 가격 데이터 없음 (ETF API): {ticker}. "
                f"일반 주식 API로 재시도합니다."
            )
            # 일반 주식 API로 fallback
            df = pykrx_stock.get_market_ohlcv_by_date(start, end, ticker)
            time.sleep(0.5)

        if df.empty:
            logger.warning(f"ETF 가격 데이터 없음: {ticker}")
            return pd.DataFrame()

        # 컬럼명 통일 (pykrx ETF API 반환 컬럼명 처리)
        rename_map = {
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
            "NAV": "nav",
            "기초지수": "base_index",
        }
        df = df.rename(columns=rename_map)
        df.index.name = "date"

        # 필수 컬럼만 유지 (있는 것만)
        base_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [c for c in base_cols if c in df.columns]
        if not available_cols:
            logger.warning(f"ETF 데이터에 필수 컬럼 없음: {ticker}")
            return pd.DataFrame()

        df = df[available_cols]

        # 가격이 0인 행 제거 (상장 전/거래정지 등)
        if "close" in df.columns:
            df = df[df["close"] > 0]

        logger.info(f"ETF 가격 데이터 조회 완료: {ticker}, {len(df)}일")
        return df

    except Exception as e:
        logger.error(f"ETF 가격 데이터 조회 실패: {ticker} - {e}")
        return pd.DataFrame()


def get_etf_list() -> pd.DataFrame:
    """한국 상장 ETF 전체 리스트를 반환한다.

    pykrx의 get_etf_ticker_list를 사용하여 현재 상장된 ETF 목록을 조회한다.

    Returns:
        DataFrame with columns: ['ticker', 'name']
        데이터 조회 실패 시 빈 DataFrame 반환.
    """
    logger.info("ETF 리스트 조회 시작")

    try:
        today = pd.Timestamp.now("Asia/Seoul").strftime("%Y%m%d")
        tickers = pykrx_stock.get_etf_ticker_list(today)
        time.sleep(0.5)

        if not tickers:
            logger.warning("ETF 리스트가 비어 있습니다.")
            return pd.DataFrame(columns=["ticker", "name"])

        rows = []
        for ticker in tickers:
            try:
                name = pykrx_stock.get_etf_ticker_name(ticker)
                rows.append({"ticker": ticker, "name": name})
            except Exception:
                # 개별 종목명 조회 실패 시 빈 문자열로 대체
                rows.append({"ticker": ticker, "name": ""})
            time.sleep(0.1)  # 대량 조회 시 rate limiting

        df = pd.DataFrame(rows)
        logger.info(f"ETF 리스트 조회 완료: {len(df)}개 ETF")
        return df

    except Exception as e:
        logger.error(f"ETF 리스트 조회 실패: {e}")
        return pd.DataFrame(columns=["ticker", "name"])
