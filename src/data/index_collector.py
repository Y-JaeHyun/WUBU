"""KOSPI/KOSDAQ 지수 데이터 수집 모듈.

pykrx 또는 KRX Open API를 활용하여 시장 지수의 OHLCV 데이터를 수집한다.
마켓 타이밍 전략에서 사용되는 지수 가격 데이터를 제공한다.
"""

from typing import Optional

import pandas as pd

from src.data import krx_provider as _krx

if _krx.is_available():
    pykrx_stock = _krx  # type: ignore[assignment]
else:
    from src.data import krx_session
    krx_session.init()
    from pykrx import stock as pykrx_stock

from src.data.collector import _format_date, _retry_on_failure
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 지수 티커 매핑
INDEX_TICKERS: dict[str, str] = {
    "KOSPI": "1001",
    "KOSDAQ": "2001",
}


@_retry_on_failure
def get_index_data(
    index: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """지수의 일별 OHLCV 데이터를 반환한다.

    Args:
        index: 지수명 ("KOSPI" 또는 "KOSDAQ") 또는 직접 티커 코드
        start_date: 시작일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
        end_date: 종료일 ('YYYYMMDD' 또는 'YYYY-MM-DD')

    Returns:
        DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
        Index: DatetimeIndex (날짜)
    """
    start = _format_date(start_date)
    end = _format_date(end_date)

    # 지수명 → 티커 변환
    ticker = INDEX_TICKERS.get(index.upper(), index)
    logger.info(f"지수 데이터 조회: {index} (ticker={ticker}, {start} ~ {end})")

    try:
        df = pykrx_stock.get_index_ohlcv_by_date(start, end, ticker)

        if df.empty:
            logger.warning(f"지수 데이터 없음: {index}")
            return df

        # pykrx 지수 OHLCV 컬럼명 통일
        df = df.rename(columns={
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
        })

        # 필요한 컬럼만 선택
        available_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[available_cols]
        df.index.name = "date"

        logger.info(f"지수 데이터 조회 완료: {index}, {len(df)}일")
        return df

    except Exception as e:
        logger.error(f"지수 데이터 조회 실패: {index} - {e}")
        raise
