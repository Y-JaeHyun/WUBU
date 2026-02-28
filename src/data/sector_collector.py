"""섹터(업종) 분류 데이터 수집 모듈.

pykrx를 활용하여 KRX WICS 업종 분류 정보를 수집한다.
"""

from typing import Dict, List, Optional
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_sector_classification(date: str, market: str = "ALL") -> pd.DataFrame:
    """특정 일자의 섹터 분류 정보를 조회한다.

    pykrx의 WICS 업종 분류를 사용.

    Args:
        date: 조회 날짜 (YYYYMMDD 형식)
        market: 시장 구분 ("KOSPI", "KOSDAQ", "ALL")

    Returns:
        DataFrame with columns: [ticker, sector]
    """
    try:
        from pykrx import stock

        # WICS 업종 코드 목록 조회
        sector_codes = stock.get_index_ticker_list(date, market="WICS")

        results = []
        for sector_code in sector_codes:
            sector_name = stock.get_index_ticker_name(sector_code)
            # 해당 업종에 속한 종목들
            tickers = stock.get_index_portfolio_deposit_file(sector_code, date)
            for ticker in tickers:
                results.append({"ticker": ticker, "sector": sector_name})

        df = pd.DataFrame(results)
        if df.empty:
            logger.warning(f"섹터 분류 데이터 없음: {date}")
            return pd.DataFrame(columns=["ticker", "sector"])

        # 중복 제거 (한 종목이 여러 업종에 속할 경우 첫 번째 사용)
        df = df.drop_duplicates(subset="ticker", keep="first")
        logger.info(f"섹터 분류 조회 완료: {date}, {len(df)}개 종목, {df['sector'].nunique()}개 섹터")
        return df

    except Exception as e:
        logger.error(f"섹터 분류 조회 실패: {e}")
        return pd.DataFrame(columns=["ticker", "sector"])


def get_sector_for_tickers(tickers: List[str], date: str) -> Dict[str, str]:
    """종목 리스트에 대한 섹터 분류를 딕셔너리로 반환한다.

    Args:
        tickers: 종목 코드 리스트
        date: 조회 날짜 (YYYYMMDD 형식)

    Returns:
        {ticker: sector_name} 딕셔너리. 미분류 종목은 "기타"로 할당.
    """
    df = get_sector_classification(date)

    if df.empty:
        return {t: "기타" for t in tickers}

    sector_map = dict(zip(df["ticker"], df["sector"]))

    result = {}
    for ticker in tickers:
        result[ticker] = sector_map.get(ticker, "기타")

    unclassified = sum(1 for v in result.values() if v == "기타")
    if unclassified > 0:
        logger.info(f"미분류 종목 {unclassified}개 → '기타' 섹터 할당")

    return result
