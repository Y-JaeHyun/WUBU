"""한국 주식 데이터 수집 모듈.

pykrx 또는 KRX Open API를 사용하여 KOSPI/KOSDAQ 종목의 OHLCV, 시가총액, 기본 지표 데이터를 수집한다.
KRX_API_KEY 환경변수 + krx_openapi 플래그 설정 시 KRX Open API 사용, 아니면 pykrx fallback.
"""

import functools
import time
from typing import Optional

import pandas as pd

from src.data import krx_provider as _krx

if _krx.is_available():
    pykrx_stock = _krx  # type: ignore[assignment]
else:
    from src.data import krx_session
    krx_session.init()  # KRX_DATA_ID/PW 설정 시 세션 쿠키 주입
    from pykrx import stock as pykrx_stock

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds


def _retry_on_failure(func):
    """pykrx API 호출 실패 시 재시도하는 데코레이터."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAY * attempt
                    logger.warning(
                        "%s 실패 (시도 %d/%d): %s — %.1f초 후 재시도",
                        func.__name__, attempt, _MAX_RETRIES, e, delay,
                    )
                    time.sleep(delay)
        logger.error("%s 최종 실패: %s", func.__name__, last_exc)
        raise last_exc

    return wrapper


def _format_date(date) -> str:
    """날짜를 pykrx가 요구하는 'YYYYMMDD' 문자열로 변환한다."""
    if isinstance(date, str):
        return date.replace("-", "")
    if isinstance(date, (pd.Timestamp,)):
        return date.strftime("%Y%m%d")
    # datetime.date, datetime.datetime
    return date.strftime("%Y%m%d")


@_retry_on_failure
def get_stock_list(market: str = "ALL") -> pd.DataFrame:
    """KOSPI/KOSDAQ 전체 종목 리스트를 반환한다.

    Args:
        market: "KOSPI", "KOSDAQ", 또는 "ALL" (기본값)

    Returns:
        DataFrame with columns: ['ticker', 'name']
    """
    logger.info(f"종목 리스트 조회 시작 (market={market})")
    try:
        today = pd.Timestamp.now("Asia/Seoul").strftime("%Y%m%d")

        if market.upper() == "ALL":
            markets = ["KOSPI", "KOSDAQ"]
        else:
            markets = [market.upper()]

        rows = []
        for mkt in markets:
            tickers = pykrx_stock.get_market_ticker_list(today, market=mkt)
            for ticker in tickers:
                name = pykrx_stock.get_market_ticker_name(ticker)
                rows.append({"ticker": ticker, "name": name, "market": mkt})
            time.sleep(0.5)

        df = pd.DataFrame(rows)
        logger.info(f"종목 리스트 조회 완료: {len(df)}개 종목")
        return df

    except Exception as e:
        logger.error(f"종목 리스트 조회 실패: {e}")
        raise


@_retry_on_failure
def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """일별 OHLCV 데이터를 반환한다.

    Args:
        ticker: 종목코드 (예: '005930')
        start_date: 시작일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
        end_date: 종료일 ('YYYYMMDD' 또는 'YYYY-MM-DD')

    Returns:
        DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
        Index: DatetimeIndex (날짜)
    """
    start = _format_date(start_date)
    end = _format_date(end_date)
    logger.info(f"가격 데이터 조회: {ticker} ({start} ~ {end})")

    try:
        df = pykrx_stock.get_market_ohlcv_by_date(start, end, ticker)

        if df.empty:
            logger.warning(f"가격 데이터 없음: {ticker}")
            return df

        df = df.rename(columns={
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
            "등락률": "change_pct",
        })
        df.index.name = "date"

        logger.info(f"가격 데이터 조회 완료: {ticker}, {len(df)}일")
        return df

    except Exception as e:
        logger.error(f"가격 데이터 조회 실패: {ticker} - {e}")
        raise


@_retry_on_failure
def get_market_cap(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """시가총액 데이터를 반환한다.

    Args:
        ticker: 종목코드
        start_date: 시작일
        end_date: 종료일

    Returns:
        DataFrame with columns: ['market_cap', 'volume', 'trade_value', 'listed_shares']
        Index: DatetimeIndex (날짜)
    """
    start = _format_date(start_date)
    end = _format_date(end_date)
    logger.info(f"시가총액 데이터 조회: {ticker} ({start} ~ {end})")

    try:
        df = pykrx_stock.get_market_cap_by_date(start, end, ticker)

        if df.empty:
            logger.warning(f"시가총액 데이터 없음: {ticker}")
            return df

        df = df.rename(columns={
            "시가총액": "market_cap",
            "거래량": "volume",
            "거래대금": "trade_value",
            "상장주식수": "listed_shares",
        })
        df.index.name = "date"

        logger.info(f"시가총액 데이터 조회 완료: {ticker}, {len(df)}일")
        return df

    except Exception as e:
        logger.error(f"시가총액 데이터 조회 실패: {ticker} - {e}")
        raise


@_retry_on_failure
def get_fundamental(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """기본 지표(PER, PBR, 배당수익률 등)를 반환한다.

    Args:
        ticker: 종목코드
        start_date: 시작일
        end_date: 종료일

    Returns:
        DataFrame with columns: ['bps', 'per', 'pbr', 'eps', 'div_yield']
        Index: DatetimeIndex (날짜)
    """
    start = _format_date(start_date)
    end = _format_date(end_date)
    logger.info(f"기본 지표 조회: {ticker} ({start} ~ {end})")

    try:
        df = pykrx_stock.get_market_fundamental_by_date(start, end, ticker)

        if df.empty:
            logger.warning(f"기본 지표 데이터 없음: {ticker}")
            return df

        df = df.rename(columns={
            "BPS": "bps",
            "PER": "per",
            "PBR": "pbr",
            "EPS": "eps",
            "DIV": "div_yield",
            "DPS": "dps",
        })
        df.index.name = "date"

        logger.info(f"기본 지표 조회 완료: {ticker}, {len(df)}일")
        return df

    except Exception as e:
        logger.error(f"기본 지표 조회 실패: {ticker} - {e}")
        raise


@_retry_on_failure
def get_all_fundamentals(
    date: str,
    market: str = "ALL",
) -> pd.DataFrame:
    """특정 날짜의 전 종목 기본 지표를 반환한다 (밸류 팩터 스크리닝용).

    Args:
        date: 조회일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
        market: "KOSPI", "KOSDAQ", 또는 "ALL"

    Returns:
        DataFrame with columns: ['ticker', 'name', 'market', 'sector', 'bps', 'per', 'pbr',
                                  'eps', 'div_yield', 'close', 'market_cap', 'volume']
        Index: RangeIndex
    """
    d = _format_date(date)
    logger.info(f"전 종목 기본 지표 조회: {d} (market={market})")

    if market.upper() == "ALL":
        markets = ["KOSPI", "KOSDAQ"]
    else:
        markets = [market.upper()]

    try:
        results = []
        for mkt in markets:
            # 기본 지표
            fund_df = pykrx_stock.get_market_fundamental(d, market=mkt)
            time.sleep(0.5)

            # 시가총액
            cap_df = pykrx_stock.get_market_cap(d, market=mkt)
            time.sleep(0.5)

            if fund_df.empty or cap_df.empty:
                logger.warning(f"{mkt} 데이터 없음 (날짜: {d})")
                continue

            # 컬럼 통일
            fund_df = fund_df.rename(columns={
                "BPS": "bps", "PER": "per", "PBR": "pbr",
                "EPS": "eps", "DIV": "div_yield", "DPS": "dps",
            })
            cap_df = cap_df.rename(columns={
                "종가": "close", "시가총액": "market_cap",
                "거래량": "volume", "거래대금": "trade_value",
                "상장주식수": "listed_shares",
            })

            merged = fund_df.join(cap_df[["close", "market_cap", "volume"]], how="inner")
            merged = merged.reset_index()
            merged = merged.rename(columns={merged.columns[0]: "ticker"})

            # 종목명 추가
            merged["name"] = [pykrx_stock.get_market_ticker_name(t) for t in merged["ticker"]]
            merged["market"] = mkt

            # 업종 분류 추가
            try:
                sector_df = pykrx_stock.get_market_sector_classifications(d, market=mkt)
                time.sleep(0.5)
                if not sector_df.empty and "업종명" in sector_df.columns:
                    sector_map = sector_df["업종명"].to_dict()
                    merged["sector"] = [sector_map.get(t, "") for t in merged["ticker"]]
                    logger.info(f"{mkt} 업종 분류 매핑 완료: {len(sector_map)}개")
                else:
                    merged["sector"] = ""
            except Exception as e:
                logger.warning(f"{mkt} 업종 분류 조회 실패: {e}")
                merged["sector"] = ""

            results.append(merged)

        if not results:
            logger.warning(f"전 종목 기본 지표 데이터 없음: {d}")
            return pd.DataFrame()

        df = pd.concat(results, ignore_index=True)

        # 컬럼 순서 정리
        col_order = [
            "ticker", "name", "market", "sector", "bps", "per", "pbr",
            "eps", "div_yield", "dps", "close", "market_cap", "volume",
        ]
        # dps가 없을 수도 있으므로 존재하는 컬럼만 선택
        col_order = [c for c in col_order if c in df.columns]
        df = df[col_order]

        logger.info(f"전 종목 기본 지표 조회 완료: {len(df)}개 종목")
        return df

    except Exception as e:
        logger.error(f"전 종목 기본 지표 조회 실패: {e}")
        raise
