"""pykrx 호환 KRX Open API 프로바이더.

pykrx와 동일한 함수 시그니처를 제공하여 기존 collector 코드 변경을 최소화한다.
KRX_API_KEY 환경변수가 설정되고 krx_openapi 플래그가 ON이면 KRX Open API 사용,
아니면 pykrx fallback.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.data.krx_api import (
    ETF_BYDD_TRD,
    MARKET_INDEX_ENDPOINTS,
    MARKET_INFO_ENDPOINTS,
    MARKET_TRADE_ENDPOINTS,
    KRXOpenAPI,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── 모듈 레벨 싱글턴 ───────────────────────────────────
_api: Optional[KRXOpenAPI] = None
_initialized: bool = False

# ── 세션 캐시 (전종목 조회 결과를 하루 1회만 호출) ──────
_ticker_cache: dict[str, dict] = {}   # {ticker: {name, market}}
_etf_ticker_cache: dict[str, str] = {}  # {ticker: name}
_cache_date: Optional[str] = None


def _init_api(auth_key: str) -> bool:
    """API 클라이언트를 초기화한다."""
    global _api, _initialized
    try:
        _api = KRXOpenAPI(auth_key=auth_key)
        _initialized = True
        logger.info("KRX Open API 초기화 완료")
        return True
    except Exception as e:
        logger.warning("KRX Open API 초기화 실패: %s", e)
        _initialized = False
        return False


def is_available() -> bool:
    """KRX Open API 사용 가능 여부를 반환한다."""
    global _initialized
    if _initialized:
        return True

    key = os.environ.get("KRX_API_KEY", "").strip()
    if not key:
        return False

    # feature flag 체크
    try:
        from src.utils.feature_flags import FeatureFlags
        ff = FeatureFlags()
        if not ff.is_enabled("krx_openapi"):
            return False
    except Exception:
        pass

    return _init_api(key)


def _ensure_api() -> KRXOpenAPI:
    """API 클라이언트가 초기화되었는지 확인하고 반환한다."""
    if _api is None:
        raise RuntimeError("KRX Open API가 초기화되지 않았습니다. is_available()을 먼저 호출하세요.")
    return _api


def _format_date(date) -> str:
    """날짜를 'YYYYMMDD' 문자열로 변환한다."""
    if isinstance(date, str):
        return date.replace("-", "")
    if isinstance(date, pd.Timestamp):
        return date.strftime("%Y%m%d")
    return date.strftime("%Y%m%d")


# ── 세션 캐시 관리 ──────────────────────────────────────


def _refresh_ticker_cache(date: str) -> None:
    """전종목 기본정보를 조회하여 세션 캐시를 갱신한다."""
    global _ticker_cache, _cache_date
    d = _format_date(date)

    if _cache_date == d and _ticker_cache:
        return

    api = _ensure_api()
    _ticker_cache.clear()

    for market, endpoint in MARKET_INFO_ENDPOINTS.items():
        rows = api.request_all(endpoint, {"basDd": d})
        for row in rows:
            ticker = row.get("ISU_SRT_CD", "").strip()
            if not ticker:
                continue
            _ticker_cache[ticker] = {
                "name": row.get("ISU_ABBRV", "").strip(),
                "market": market,
            }

    _cache_date = d
    logger.info("KRX 종목 캐시 갱신: %d종목 (기준일: %s)", len(_ticker_cache), d)


def _refresh_etf_ticker_cache(date: str) -> None:
    """ETF 전종목을 조회하여 세션 캐시를 갱신한다."""
    global _etf_ticker_cache
    d = _format_date(date)

    api = _ensure_api()
    _etf_ticker_cache.clear()

    rows = api.request_all(ETF_BYDD_TRD, {"basDd": d})
    for row in rows:
        ticker = row.get("ISU_SRT_CD", "").strip()
        if not ticker:
            continue
        _etf_ticker_cache[ticker] = row.get("ISU_ABBRV", "").strip()

    logger.info("KRX ETF 캐시 갱신: %d종목 (기준일: %s)", len(_etf_ticker_cache), d)


# ── pykrx 호환 함수 ────────────────────────────────────


def get_market_ticker_list(date: str, market: str = "KOSPI") -> list[str]:
    """종목코드 리스트를 반환한다.

    Args:
        date: 기준일 ('YYYYMMDD').
        market: 'KOSPI' 또는 'KOSDAQ'.

    Returns:
        종목코드 리스트.
    """
    d = _format_date(date)
    _refresh_ticker_cache(d)
    return [
        t for t, info in _ticker_cache.items()
        if info["market"] == market.upper()
    ]


def get_market_ticker_name(ticker: str) -> str:
    """종목명을 반환한다.

    Args:
        ticker: 종목코드.

    Returns:
        종목명. 캐시에 없으면 빈 문자열.
    """
    info = _ticker_cache.get(ticker)
    if info:
        return info["name"]
    # ETF 캐시도 확인
    if ticker in _etf_ticker_cache:
        return _etf_ticker_cache[ticker]
    return ""


def get_market_ohlcv_by_date(
    start: str, end: str, ticker: str,
) -> pd.DataFrame:
    """종목의 일별 OHLCV 데이터를 반환한다.

    Args:
        start: 시작일 ('YYYYMMDD').
        end: 종료일 ('YYYYMMDD').
        ticker: 종목코드.

    Returns:
        DataFrame (index=DatetimeIndex, columns=시가/고가/저가/종가/거래량).
    """
    api = _ensure_api()
    s, e = _format_date(start), _format_date(end)

    # 시장 결정
    info = _ticker_cache.get(ticker)
    market = info["market"] if info else "KOSPI"
    endpoint = MARKET_TRADE_ENDPOINTS.get(market, MARKET_TRADE_ENDPOINTS["KOSPI"])

    all_rows = _fetch_date_range(api, endpoint, s, e, isuCd=ticker)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return _to_ohlcv_dataframe(df)


def get_market_cap_by_date(
    start: str, end: str, ticker: str,
) -> pd.DataFrame:
    """종목의 일별 시가총액 데이터를 반환한다.

    Args:
        start: 시작일 ('YYYYMMDD').
        end: 종료일 ('YYYYMMDD').
        ticker: 종목코드.

    Returns:
        DataFrame (index=DatetimeIndex, columns=시가총액/거래량/거래대금/상장주식수).
    """
    api = _ensure_api()
    s, e = _format_date(start), _format_date(end)

    info = _ticker_cache.get(ticker)
    market = info["market"] if info else "KOSPI"
    endpoint = MARKET_TRADE_ENDPOINTS.get(market, MARKET_TRADE_ENDPOINTS["KOSPI"])

    all_rows = _fetch_date_range(api, endpoint, s, e, isuCd=ticker)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return _to_marketcap_dataframe(df)


def get_market_cap(date: str, market: str = "KOSPI") -> pd.DataFrame:
    """전종목 시가총액 스냅샷을 반환한다.

    Args:
        date: 기준일 ('YYYYMMDD').
        market: 'KOSPI' 또는 'KOSDAQ'.

    Returns:
        DataFrame (index=종목코드, columns=종가/시가총액/거래량/거래대금/상장주식수).
    """
    api = _ensure_api()
    d = _format_date(date)
    endpoint = MARKET_TRADE_ENDPOINTS.get(market.upper(), MARKET_TRADE_ENDPOINTS["KOSPI"])

    rows = api.request_all(endpoint, {"basDd": d})
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return _to_allstock_marketcap(df)


def get_market_fundamental(date: str, market: str = "KOSPI") -> pd.DataFrame:
    """전종목 기본 지표 스냅샷을 반환한다.

    Args:
        date: 기준일 ('YYYYMMDD').
        market: 'KOSPI' 또는 'KOSDAQ'.

    Returns:
        DataFrame (index=종목코드, columns=BPS/PER/PBR/EPS/DIV/DPS).
    """
    api = _ensure_api()
    d = _format_date(date)
    endpoint = MARKET_INFO_ENDPOINTS.get(market.upper(), MARKET_INFO_ENDPOINTS["KOSPI"])

    rows = api.request_all(endpoint, {"basDd": d})
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return _to_fundamental_snapshot(df)


def get_market_fundamental_by_date(
    start: str, end: str, ticker: str,
) -> pd.DataFrame:
    """종목의 일별 기본 지표를 반환한다.

    Args:
        start: 시작일 ('YYYYMMDD').
        end: 종료일 ('YYYYMMDD').
        ticker: 종목코드.

    Returns:
        DataFrame (index=DatetimeIndex, columns=BPS/PER/PBR/EPS/DIV/DPS).
    """
    api = _ensure_api()
    s, e = _format_date(start), _format_date(end)

    info = _ticker_cache.get(ticker)
    market = info["market"] if info else "KOSPI"
    endpoint = MARKET_INFO_ENDPOINTS.get(market, MARKET_INFO_ENDPOINTS["KOSPI"])

    all_rows = _fetch_date_range(api, endpoint, s, e, isuCd=ticker)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return _to_fundamental_timeseries(df)


def get_market_sector_classifications(
    date: str, market: str = "KOSPI",
) -> pd.DataFrame:
    """업종 분류 정보를 반환한다. KRX API 미지원이므로 pykrx fallback."""
    try:
        from pykrx import stock as pykrx_stock
        return pykrx_stock.get_market_sector_classifications(
            _format_date(date), market=market,
        )
    except Exception as e:
        logger.warning("업종 분류 조회 실패 (pykrx fallback): %s", e)
        return pd.DataFrame()


def get_index_ohlcv_by_date(
    start: str, end: str, ticker: str,
) -> pd.DataFrame:
    """지수의 일별 OHLCV 데이터를 반환한다.

    Args:
        start: 시작일 ('YYYYMMDD').
        end: 종료일 ('YYYYMMDD').
        ticker: 지수 코드 ('1001'=KOSPI, '2001'=KOSDAQ).

    Returns:
        DataFrame (index=DatetimeIndex, columns=시가/고가/저가/종가/거래량).
    """
    api = _ensure_api()
    s, e = _format_date(start), _format_date(end)

    endpoint = MARKET_INDEX_ENDPOINTS.get(ticker, MARKET_INDEX_ENDPOINTS.get("1001"))

    all_rows = _fetch_date_range(api, endpoint, s, e, idxIndCd=ticker)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return _to_index_ohlcv(df)


def get_etf_ohlcv_by_date(
    start: str, end: str, ticker: str,
) -> pd.DataFrame:
    """ETF의 일별 OHLCV 데이터를 반환한다.

    Args:
        start: 시작일 ('YYYYMMDD').
        end: 종료일 ('YYYYMMDD').
        ticker: ETF 종목코드.

    Returns:
        DataFrame (index=DatetimeIndex, columns=시가/고가/저가/종가/거래량/NAV/기초지수).
    """
    api = _ensure_api()
    s, e = _format_date(start), _format_date(end)

    all_rows = _fetch_date_range(api, ETF_BYDD_TRD, s, e, isuCd=ticker)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return _to_etf_ohlcv(df)


def get_etf_ticker_list(date: str) -> list[str]:
    """ETF 종목코드 리스트를 반환한다.

    Args:
        date: 기준일 ('YYYYMMDD').

    Returns:
        ETF 종목코드 리스트.
    """
    d = _format_date(date)
    _refresh_etf_ticker_cache(d)
    return list(_etf_ticker_cache.keys())


def get_etf_ticker_name(ticker: str) -> str:
    """ETF 종목명을 반환한다.

    Args:
        ticker: ETF 종목코드.

    Returns:
        ETF 종목명. 캐시에 없으면 빈 문자열.
    """
    return _etf_ticker_cache.get(ticker, "")


# WICS 섹터 함수 (KRX Open API 미지원 → pykrx fallback)


def get_index_ticker_list(date: str, market: str = "WICS") -> list[str]:
    """WICS 업종 코드 리스트. pykrx fallback."""
    try:
        from pykrx import stock
        return stock.get_index_ticker_list(_format_date(date), market=market)
    except Exception as e:
        logger.warning("WICS 업종 코드 조회 실패: %s", e)
        return []


def get_index_ticker_name(sector_code: str) -> str:
    """WICS 업종명. pykrx fallback."""
    try:
        from pykrx import stock
        return stock.get_index_ticker_name(sector_code)
    except Exception as e:
        logger.warning("WICS 업종명 조회 실패: %s", e)
        return ""


def get_index_portfolio_deposit_file(sector_code: str, date: str) -> list[str]:
    """WICS 업종 소속 종목 리스트. pykrx fallback."""
    try:
        from pykrx import stock
        return stock.get_index_portfolio_deposit_file(sector_code, _format_date(date))
    except Exception as e:
        logger.warning("WICS 업종 종목 조회 실패: %s", e)
        return []


# ── 내부 헬퍼 ──────────────────────────────────────────


def _fetch_date_range(
    api: KRXOpenAPI,
    endpoint: str,
    start: str,
    end: str,
    **extra_params: str,
) -> list[dict]:
    """날짜 범위를 일별로 순회하며 데이터를 수집한다.

    KRX API는 기준일(basDd) 단위 조회이므로, 날짜 범위를 순회해야 한다.
    단, 전종목 조회(isuCd 없음)는 단일 날짜만 호출한다.
    """
    s_dt = datetime.strptime(start, "%Y%m%d")
    e_dt = datetime.strptime(end, "%Y%m%d")

    all_rows: list[dict] = []
    current = s_dt

    while current <= e_dt:
        d = current.strftime("%Y%m%d")
        params = {"basDd": d, **extra_params}
        rows = api.request_all(endpoint, params)
        all_rows.extend(rows)
        current += timedelta(days=1)

    return all_rows


def _safe_float(value: str, default: float = 0.0) -> float:
    """문자열을 float로 안전하게 변환한다."""
    try:
        # 쉼표 제거 후 변환
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return default


def _safe_int(value: str, default: int = 0) -> int:
    """문자열을 int로 안전하게 변환한다."""
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (ValueError, TypeError):
        return default


def _to_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """KRX API 응답을 pykrx 호환 OHLCV DataFrame으로 변환한다."""
    if df.empty:
        return pd.DataFrame()

    # 날짜 인덱스
    date_col = _find_column(df, ["BAS_DD", "TRD_DD", "basDd", "trdDd"])
    if date_col is None:
        return pd.DataFrame()

    result = pd.DataFrame(index=pd.to_datetime(df[date_col], format="%Y%m%d"))
    result.index.name = None

    col_map = {
        "시가": ["TDD_OPNPRC", "tddOpnprc"],
        "고가": ["TDD_HGPRC", "tddHgprc"],
        "저가": ["TDD_LWPRC", "tddLwprc"],
        "종가": ["TDD_CLSPRC", "tddClsprc"],
        "거래량": ["ACC_TRDVOL", "accTrdvol"],
    }

    for korean_name, candidates in col_map.items():
        col = _find_column(df, candidates)
        if col is not None:
            result[korean_name] = [_safe_int(v) for v in df[col]]

    return result.sort_index()


def _to_marketcap_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """KRX API 응답을 pykrx 호환 시가총액 DataFrame으로 변환한다."""
    if df.empty:
        return pd.DataFrame()

    date_col = _find_column(df, ["BAS_DD", "TRD_DD", "basDd", "trdDd"])
    if date_col is None:
        return pd.DataFrame()

    result = pd.DataFrame(index=pd.to_datetime(df[date_col], format="%Y%m%d"))
    result.index.name = None

    col_map = {
        "시가총액": ["MKTCAP", "mktcap"],
        "거래량": ["ACC_TRDVOL", "accTrdvol"],
        "거래대금": ["ACC_TRDVAL", "accTrdval"],
        "상장주식수": ["LIST_SHRS", "listShrs"],
    }

    for korean_name, candidates in col_map.items():
        col = _find_column(df, candidates)
        if col is not None:
            result[korean_name] = [_safe_int(v) for v in df[col]]

    return result.sort_index()


def _to_allstock_marketcap(df: pd.DataFrame) -> pd.DataFrame:
    """전종목 시가총액 스냅샷을 pykrx 호환 형식으로 변환한다."""
    if df.empty:
        return pd.DataFrame()

    ticker_col = _find_column(df, ["ISU_SRT_CD", "isuSrtCd"])
    if ticker_col is None:
        return pd.DataFrame()

    result = pd.DataFrame()
    result.index = [str(v).strip() for v in df[ticker_col]]
    result.index.name = None

    col_map = {
        "종가": ["TDD_CLSPRC", "tddClsprc"],
        "시가총액": ["MKTCAP", "mktcap"],
        "거래량": ["ACC_TRDVOL", "accTrdvol"],
        "거래대금": ["ACC_TRDVAL", "accTrdval"],
        "상장주식수": ["LIST_SHRS", "listShrs"],
    }

    for korean_name, candidates in col_map.items():
        col = _find_column(df, candidates)
        if col is not None:
            result[korean_name] = [_safe_int(v) for v in df[col]]

    return result


def _to_fundamental_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """전종목 기본지표 스냅샷을 pykrx 호환 형식으로 변환한다."""
    if df.empty:
        return pd.DataFrame()

    ticker_col = _find_column(df, ["ISU_SRT_CD", "isuSrtCd"])
    if ticker_col is None:
        return pd.DataFrame()

    result = pd.DataFrame()
    result.index = [str(v).strip() for v in df[ticker_col]]
    result.index.name = None

    col_map = {
        "BPS": ["BPS", "bps"],
        "PER": ["PER", "per"],
        "PBR": ["PBR", "pbr"],
        "EPS": ["EPS", "eps"],
        "DIV": ["DVD_YLD", "dvdYld", "DIV"],
        "DPS": ["DPS", "dps"],
    }

    for name, candidates in col_map.items():
        col = _find_column(df, candidates)
        if col is not None:
            result[name] = [_safe_float(v) for v in df[col]]

    return result


def _to_fundamental_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """종목별 기본지표 시계열을 pykrx 호환 형식으로 변환한다."""
    if df.empty:
        return pd.DataFrame()

    date_col = _find_column(df, ["BAS_DD", "TRD_DD", "basDd", "trdDd"])
    if date_col is None:
        return pd.DataFrame()

    result = pd.DataFrame(index=pd.to_datetime(df[date_col], format="%Y%m%d"))
    result.index.name = None

    col_map = {
        "BPS": ["BPS", "bps"],
        "PER": ["PER", "per"],
        "PBR": ["PBR", "pbr"],
        "EPS": ["EPS", "eps"],
        "DIV": ["DVD_YLD", "dvdYld", "DIV"],
        "DPS": ["DPS", "dps"],
    }

    for name, candidates in col_map.items():
        col = _find_column(df, candidates)
        if col is not None:
            result[name] = [_safe_float(v) for v in df[col]]

    return result.sort_index()


def _to_index_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """지수 OHLCV를 pykrx 호환 형식으로 변환한다."""
    if df.empty:
        return pd.DataFrame()

    date_col = _find_column(df, ["BAS_DD", "TRD_DD", "basDd", "trdDd"])
    if date_col is None:
        return pd.DataFrame()

    result = pd.DataFrame(index=pd.to_datetime(df[date_col], format="%Y%m%d"))
    result.index.name = None

    # 지수는 float (소수점 포함)
    col_map = {
        "시가": ["OPNPRC_IDX", "opnprcIdx", "TDD_OPNPRC"],
        "고가": ["HGPRC_IDX", "hgprcIdx", "TDD_HGPRC"],
        "저가": ["LWPRC_IDX", "lwprcIdx", "TDD_LWPRC"],
        "종가": ["CLSPRC_IDX", "clsprcIdx", "TDD_CLSPRC"],
        "거래량": ["ACC_TRDVOL", "accTrdvol"],
    }

    for korean_name, candidates in col_map.items():
        col = _find_column(df, candidates)
        if col is not None:
            if korean_name == "거래량":
                result[korean_name] = [_safe_int(v) for v in df[col]]
            else:
                result[korean_name] = [_safe_float(v) for v in df[col]]

    return result.sort_index()


def _to_etf_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """ETF OHLCV를 pykrx 호환 형식으로 변환한다."""
    if df.empty:
        return pd.DataFrame()

    date_col = _find_column(df, ["BAS_DD", "TRD_DD", "basDd", "trdDd"])
    if date_col is None:
        return pd.DataFrame()

    result = pd.DataFrame(index=pd.to_datetime(df[date_col], format="%Y%m%d"))
    result.index.name = None

    col_map = {
        "시가": ["TDD_OPNPRC", "tddOpnprc"],
        "고가": ["TDD_HGPRC", "tddHgprc"],
        "저가": ["TDD_LWPRC", "tddLwprc"],
        "종가": ["TDD_CLSPRC", "tddClsprc"],
        "거래량": ["ACC_TRDVOL", "accTrdvol"],
        "NAV": ["NAV", "nav"],
        "기초지수": ["OBJ_STKPRC_IDX", "objStkprcIdx"],
    }

    for korean_name, candidates in col_map.items():
        col = _find_column(df, candidates)
        if col is not None:
            if korean_name in ("거래량",):
                result[korean_name] = [_safe_int(v) for v in df[col]]
            else:
                result[korean_name] = [_safe_float(v) for v in df[col]]

    return result.sort_index()


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """DataFrame에서 후보 컬럼명 중 존재하는 첫 번째를 반환한다."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
