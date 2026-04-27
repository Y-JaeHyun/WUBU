"""DART OpenAPI 재무제표 수집 모듈.

opendartreader 라이브러리를 활용하여 DART(전자공시시스템)에서
재무제표 데이터를 수집하고, 퀄리티 팩터 산출에 필요한 지표를 제공한다.

pykrx get_market_fundamental() 대체 파이프라인:
- DART fnlttMultiAcnt API로 분기별 재무데이터 배치 수집 (100개사/요청)
- get_all_fundamentals_dart()가 collector.get_all_fundamentals()의 fallback 역할

DART API key가 없을 경우 pykrx 기본 데이터로 퀄리티 근사치를 산출한다.
"""

import io
import os
import time
import zipfile
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DART_REST_URL = "https://opendart.fss.or.kr/api"

# DART API Rate limiting 간격 (초)
_RATE_LIMIT_SLEEP = 0.5


def _get_dart_reader() -> Optional[object]:
    """DART API 리더 객체를 생성한다.

    환경변수 DART_API_KEY가 설정되어 있으면 OpenDartReader 객체를 반환하고,
    없으면 None을 반환한다.

    Returns:
        OpenDartReader 객체 또는 None
    """
    api_key = os.environ.get("DART_API_KEY")
    if not api_key:
        logger.warning(
            "DART_API_KEY 환경변수가 설정되지 않았습니다. "
            "pykrx 기반 근사치를 사용합니다."
        )
        return None

    try:
        import OpenDartReader
        dart = OpenDartReader(api_key)
        logger.info("OpenDartReader 초기화 완료")
        return dart
    except ImportError:
        logger.warning(
            "opendartreader 라이브러리가 설치되지 않았습니다. "
            "pip install opendartreader 로 설치하세요. "
            "pykrx 기반 근사치를 사용합니다."
        )
        return None
    except Exception as e:
        logger.error(f"OpenDartReader 초기화 실패: {e}")
        return None


def get_financial_statements(
    corp_code: str,
    year: int,
    quarter: int = 4,
) -> pd.DataFrame:
    """DART에서 특정 기업의 재무제표를 조회한다.

    ROE, GP/A (매출총이익/총자산), 부채비율, 발생액(accruals) 등
    퀄리티 팩터에 필요한 지표를 산출한다.

    Args:
        corp_code: 기업 고유코드 (종목코드 또는 DART 고유번호)
        year: 사업연도 (예: 2023)
        quarter: 분기 (1, 2, 3, 4). 4는 연간 보고서.

    Returns:
        DataFrame with columns:
            - roe: 자기자본이익률 (%)
            - gp_over_assets: 매출총이익/총자산
            - debt_ratio: 부채비율 (%)
            - accruals: 발생액 비율 (영업이익 - 영업활동현금흐름) / 총자산
        단일 행의 DataFrame. 데이터 조회 실패 시 빈 DataFrame 반환.
    """
    dart = _get_dart_reader()
    if dart is None:
        logger.info(
            f"DART 리더 없음, 빈 DataFrame 반환 (corp_code={corp_code})"
        )
        return pd.DataFrame()

    # 분기 매핑: DART reprt_code
    reprt_code_map = {
        1: "11013",  # 1분기
        2: "11012",  # 반기
        3: "11014",  # 3분기
        4: "11011",  # 사업보고서(연간)
    }
    reprt_code = reprt_code_map.get(quarter, "11011")

    try:
        # 계정과목 추출 헬퍼
        def _extract_amount(df: pd.DataFrame, account_names: list[str]) -> Optional[float]:
            """재무제표에서 특정 계정과목의 금액을 추출한다.

            정확 매칭(==)을 우선 시도하고, 없으면 부분 매칭(contains)으로 fallback.
            "부채총계"가 "자본과부채총계"에 매칭되는 문제 방지.
            """
            # 1차: 정확 매칭
            for name in account_names:
                mask = df["account_nm"].str.strip() == name
                if mask.any():
                    val = df.loc[mask, "thstrm_amount"].iloc[0]
                    if isinstance(val, str):
                        val = val.replace(",", "")
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
            # 2차: 부분 매칭 (fallback)
            for name in account_names:
                mask = df["account_nm"].str.contains(name, na=False)
                if mask.any():
                    val = df.loc[mask, "thstrm_amount"].iloc[0]
                    if isinstance(val, str):
                        val = val.replace(",", "")
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
            return None

        # 상세 재무제표(finstate_all) 우선 시도 — 매출총이익, 현금흐름 포함
        time.sleep(_RATE_LIMIT_SLEEP)
        stmt = None
        try:
            detail = dart.finstate_all(corp_code, year, reprt_code=reprt_code)
            if detail is not None and not detail.empty:
                stmt = detail
        except Exception:
            pass

        # finstate_all 실패 시 finstate(요약) fallback
        if stmt is None or stmt.empty:
            time.sleep(_RATE_LIMIT_SLEEP)
            all_stmt = dart.finstate(corp_code, year, reprt_code=reprt_code)
            if all_stmt is None or all_stmt.empty:
                logger.warning(f"재무제표 데이터 없음: {corp_code}, {year}Q{quarter}")
                return pd.DataFrame()
            # 연결재무제표(CFS) 우선, 없으면 개별(OFS)
            if "fs_div" in all_stmt.columns:
                cfs = all_stmt[all_stmt["fs_div"] == "CFS"]
                ofs = all_stmt[all_stmt["fs_div"] == "OFS"]
                stmt = cfs if not cfs.empty else ofs
            else:
                stmt = all_stmt

        if stmt is None or stmt.empty:
            logger.warning(f"재무제표 데이터 없음: {corp_code}, {year}Q{quarter}")
            return pd.DataFrame()

        # 주요 계정 추출 (IFRS 기업별 계정명 차이 대응)
        net_income = _extract_amount(stmt, ["당기순이익", "당기순손익"])
        equity = _extract_amount(stmt, ["자본총계"])
        total_assets = _extract_amount(stmt, [
            "자산총계", "부채와자본총계", "부채와 자본 총계",
            "자본과부채총계", "부채및자본총계",
        ])
        total_liabilities = _extract_amount(stmt, ["부채총계"])
        # 부채총계가 없으면 자산 - 자본으로 계산
        if total_liabilities is None and total_assets is not None and equity is not None:
            total_liabilities = total_assets - equity
        gross_profit = _extract_amount(stmt, ["매출총이익"])
        operating_income = _extract_amount(stmt, ["영업이익"])
        operating_cf = _extract_amount(stmt, ["영업활동현금흐름", "영업활동으로인한현금흐름"])

        # 지표 산출
        roe = (net_income / equity * 100) if (net_income is not None and equity and equity != 0) else np.nan
        gp_over_assets = (gross_profit / total_assets) if (gross_profit is not None and total_assets and total_assets != 0) else np.nan
        debt_ratio = (total_liabilities / equity * 100) if (total_liabilities is not None and equity and equity != 0) else np.nan

        # 발생액 비율: (영업이익 - 영업활동현금흐름) / 총자산
        if operating_income is not None and operating_cf is not None and total_assets and total_assets != 0:
            accruals = (operating_income - operating_cf) / total_assets
        else:
            accruals = np.nan

        result = pd.DataFrame([{
            "corp_code": corp_code,
            "year": year,
            "quarter": quarter,
            "roe": roe,
            "gp_over_assets": gp_over_assets,
            "debt_ratio": debt_ratio,
            "accruals": accruals,
        }])

        logger.info(
            f"재무제표 조회 완료: {corp_code}, {year}Q{quarter}, "
            f"ROE={roe:.2f}%" if not np.isnan(roe) else f"재무제표 조회 완료: {corp_code}, ROE=N/A"
        )

        return result

    except Exception as e:
        logger.error(f"재무제표 조회 실패: {corp_code}, {year}Q{quarter} - {e}")
        return pd.DataFrame()


def _estimate_quality_from_pykrx(date: str, market: str = "ALL") -> pd.DataFrame:
    """pykrx 데이터로 퀄리티 근사치를 산출한다.

    DART API가 없을 때 사용하는 fallback 방식이다.
    PBR, PER, EPS 등으로 ROE, GP/A 등을 추정한다.

    추정 방식:
    - ROE 추정: EPS / BPS * 100 (ROE = 순이익/자기자본 = EPS/BPS)
    - GP/A 추정: (매출총이익 데이터 없으므로) ROE * 0.5로 근사 (한계 있음)
    - 부채비율 추정: (PBR - 1) * 100 가 아닌, 직접 계산 불가하여 PBR 역수로 근사
    - 발생액: 직접 추정 불가, 0으로 대체

    Args:
        date: 조회일 ('YYYYMMDD')
        market: "KOSPI", "KOSDAQ", 또는 "ALL"

    Returns:
        DataFrame with columns: ['ticker', 'roe', 'gp_over_assets', 'debt_ratio', 'accruals']
    """
    try:
        from src.data.collector import get_all_fundamentals
        fundamentals = get_all_fundamentals(date, market)
    except Exception as e:
        logger.error(f"pykrx 펀더멘탈 데이터 조회 실패: {e}")
        return pd.DataFrame()

    if fundamentals.empty:
        logger.warning(f"pykrx 펀더멘탈 데이터 없음 (date={date})")
        return pd.DataFrame()

    df = fundamentals.copy()

    # ROE 추정: EPS / BPS * 100
    if "eps" in df.columns and "bps" in df.columns:
        # BPS가 0이거나 음수인 경우 NaN 처리
        valid_bps = df["bps"].replace(0, np.nan)
        df["roe"] = (df["eps"] / valid_bps) * 100
    else:
        df["roe"] = np.nan

    # GP/A 추정: 매출총이익 데이터 없으므로 ROE 기반 근사
    # GP/A는 일반적으로 ROE보다 안정적이므로 ROE의 절반값으로 근사한다
    # 이는 정밀하지 않으나 DART API 없는 상황에서의 차선책이다
    df["gp_over_assets"] = df["roe"].abs() / 200  # 대략적 근사

    # 부채비율 추정: PBR을 이용
    # PBR = 시가총액 / 자기자본, 높은 PBR이 반드시 높은 부채비율을 의미하지는 않으나
    # 대안이 없으므로 간접 추정. BPS 대비 close 비율로 추정
    if "pbr" in df.columns:
        # 부채비율은 직접 추정이 어려우므로 중위값(100%)으로 대체
        # 퀄리티 팩터에서 상대적 순위로 사용하므로 PBR 역수를 부채 프록시로 활용
        # PBR이 높을수록 자기자본 대비 시가총액이 높은 것이나, 부채와 직접 관련은 적음
        # fallback이므로 보수적으로 NaN 처리
        df["debt_ratio"] = np.nan
    else:
        df["debt_ratio"] = np.nan

    # 발생액: pykrx에서 추정 불가
    df["accruals"] = np.nan

    # 필요한 컬럼만 선택
    result_cols = ["ticker", "roe", "gp_over_assets", "debt_ratio", "accruals"]
    available_cols = [c for c in result_cols if c in df.columns]
    result = df[available_cols].copy()

    # NaN이 아닌 유효 데이터 수 로깅
    valid_count = result["roe"].notna().sum()
    logger.info(
        f"pykrx 기반 퀄리티 근사치 산출 완료: "
        f"{len(result)}개 종목, 유효 ROE {valid_count}개"
    )

    return result


def get_quality_data(
    date: str,
    market: str = "ALL",
) -> pd.DataFrame:
    """전 종목의 퀄리티 팩터 데이터를 조회한다.

    DART API key가 있으면 DART에서 재무제표를 수집하고,
    없으면 pykrx 데이터로 근사치를 산출한다.

    Args:
        date: 조회일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
        market: "KOSPI", "KOSDAQ", 또는 "ALL" (기본)

    Returns:
        DataFrame with columns:
            - ticker: 종목코드
            - roe: 자기자본이익률 (%)
            - gp_over_assets: 매출총이익/총자산
            - debt_ratio: 부채비율 (%)
            - accruals: 발생액 비율
    """
    formatted_date = date.replace("-", "")
    logger.info(f"퀄리티 데이터 조회 시작: {formatted_date} (market={market})")

    dart = _get_dart_reader()

    if dart is None:
        # DART API 없음: pykrx fallback
        logger.info("DART API 미사용, pykrx fallback으로 퀄리티 근사치 산출")
        return _estimate_quality_from_pykrx(formatted_date, market)

    # DART API 사용 경로
    try:
        from src.data.collector import get_stock_list
        stock_list = get_stock_list(market)
    except Exception as e:
        logger.error(f"종목 리스트 조회 실패: {e}")
        return _estimate_quality_from_pykrx(formatted_date, market)

    if stock_list.empty:
        logger.warning("종목 리스트가 비어 있습니다.")
        return pd.DataFrame()

    # 조회 연도 결정 (date 기준 직전 사업연도)
    year = int(formatted_date[:4])
    month = int(formatted_date[4:6])
    # 3월 말 이전이면 전전년도 데이터 사용 (사업보고서 공시 시점 고려)
    if month <= 3:
        fiscal_year = year - 2
    else:
        fiscal_year = year - 1

    results = []
    total = len(stock_list)
    success_count = 0

    for idx, row in stock_list.iterrows():
        ticker = row["ticker"]

        try:
            fs = get_financial_statements(ticker, fiscal_year)
            if not fs.empty:
                fs["ticker"] = ticker
                results.append(fs)
                success_count += 1
        except Exception as e:
            logger.debug(f"재무제표 조회 실패 (무시): {ticker} - {e}")

        # 진행 상황 로깅 (100개마다)
        if (idx + 1) % 100 == 0:
            logger.info(
                f"퀄리티 데이터 수집 진행: {idx + 1}/{total} "
                f"(성공: {success_count})"
            )

    if not results:
        logger.warning(
            "DART에서 퀄리티 데이터를 수집하지 못했습니다. "
            "pykrx fallback으로 전환합니다."
        )
        return _estimate_quality_from_pykrx(formatted_date, market)

    df = pd.concat(results, ignore_index=True)

    # 필요한 컬럼 정리
    result_cols = ["ticker", "roe", "gp_over_assets", "debt_ratio", "accruals"]
    available_cols = [c for c in result_cols if c in df.columns]
    df = df[available_cols]

    logger.info(
        f"퀄리티 데이터 조회 완료: {len(df)}개 종목 "
        f"(DART 성공률: {success_count}/{total})"
    )

    return df


# ─── DART 배치 펀더멘탈 파이프라인 ─────────────────────────────────────────────


def _get_dart_api_key() -> str:
    """DART API 키를 환경변수에서 읽는다."""
    return os.environ.get("DART_API_KEY", "").strip()


def _build_corp_code_map() -> dict[str, str]:
    """DART corp_code.xml에서 상장종목 ticker → corp_code 매핑을 만든다.

    Returns:
        {stock_code: corp_code} 딕셔너리 (상장사만 포함)
    """
    api_key = _get_dart_api_key()
    if not api_key:
        logger.warning("DART_API_KEY 미설정 — corp_code 매핑 불가")
        return {}

    try:
        resp = requests.get(
            f"{_DART_REST_URL}/corpCode.xml",
            params={"crtfc_key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        with zf.open("CORPCODE.xml") as f:
            root = ET.parse(f).getroot()

        mapping: dict[str, str] = {}
        for item in root.findall("list"):
            stock_code = (item.findtext("stock_code") or "").strip()
            corp_code = (item.findtext("corp_code") or "").strip()
            if stock_code and corp_code:
                mapping[stock_code] = corp_code

        logger.info("DART corp_code 매핑 완료: %d개 상장사", len(mapping))
        return mapping
    except Exception as e:
        logger.error("DART corp_code 매핑 실패: %s", e)
        return {}


def _determine_latest_report(date_str: str) -> tuple[str, str]:
    """기준일 기준 최근 공시 완료된 보고서의 (사업연도, 보고서코드)를 결정한다.

    공시 일정 기준:
      - 사업보고서(11011): 당해 3월 31일 공시 → 4월 이후 사용 가능
      - 1분기(11013): 5월 15일 공시 → 6월 이후 사용 가능
      - 반기(11012): 8월 14일 공시 → 9월 이후 사용 가능
      - 3분기(11014): 11월 14일 공시 → 12월 이후 사용 가능

    Args:
        date_str: 'YYYYMMDD' 형식

    Returns:
        (bsns_year, reprt_code) 튜플
    """
    year = int(date_str[:4])
    month = int(date_str[4:6])

    if month >= 12:
        return str(year), "11014"    # 3분기 보고서
    elif month >= 9:
        return str(year), "11012"    # 반기 보고서
    elif month >= 6:
        return str(year), "11013"    # 1분기 보고서
    elif month >= 4:
        return str(year - 1), "11011"  # 전년도 사업보고서
    else:
        return str(year - 2), "11011"  # 전전년도 사업보고서 (Q4 실적 미공시)


def _fetch_dart_accounts_batch(
    corp_codes: list[str],
    bsns_year: str,
    reprt_code: str,
    batch_size: int = 80,
) -> dict[str, dict[str, float]]:
    """DART fnlttMultiAcnt API로 여러 기업의 재무계정을 배치 조회한다.

    자본총계(equity)와 당기순이익(net_income)을 수집한다.
    연결재무제표(CFS) 우선, 없으면 개별(OFS).

    Args:
        corp_codes: DART 고유번호 리스트
        bsns_year: 사업연도 (e.g., "2025")
        reprt_code: 보고서 코드
        batch_size: 배치당 최대 요청 수 (DART 권장 최대: 100)

    Returns:
        {corp_code: {"equity": float, "net_income": float}} 딕셔너리
    """
    api_key = _get_dart_api_key()
    if not api_key or not corp_codes:
        return {}

    # 관심 계정명 집합 — 한국어 표기 변형 모두 포함
    _EQUITY_NAMES = {"자본총계"}
    _INCOME_NAMES = {"당기순이익", "당기순손익", "당기순이익(손실)", "당기순손익(손실)"}

    results: dict[str, dict[str, float]] = {}

    for i in range(0, len(corp_codes), batch_size):
        batch = corp_codes[i:i + batch_size]
        code_str = ",".join(batch)

        try:
            resp = requests.get(
                f"{_DART_REST_URL}/fnlttMultiAcnt.json",
                params={
                    "crtfc_key": api_key,
                    "corp_code": code_str,
                    "bsns_year": bsns_year,
                    "reprt_code": reprt_code,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("DART fnlttMultiAcnt 배치 실패 (%d~%d): %s", i, i + batch_size, e)
            time.sleep(_RATE_LIMIT_SLEEP)
            continue

        if data.get("status") != "000":
            logger.debug("DART API 응답 이상 (status=%s): %s", data.get("status"), data.get("message"))
            time.sleep(_RATE_LIMIT_SLEEP)
            continue

        # fs_div 기준으로 연결재무(CFS) 우선, 없으면 개별(OFS) 폴백
        cfs_data: dict[str, dict[str, float]] = {}
        ofs_data: dict[str, dict[str, float]] = {}

        for item in data.get("list", []):
            corp = item.get("corp_code", "").strip()
            fs_div = item.get("fs_div", "").strip()   # CFS=연결, OFS=별도
            acct_nm = (item.get("account_nm") or "").strip()
            amount_raw = (item.get("thstrm_amount") or "").replace(",", "").strip()

            if not corp or not amount_raw:
                continue

            try:
                amount = float(amount_raw)
            except ValueError:
                continue

            target = cfs_data if fs_div == "CFS" else ofs_data

            if corp not in target:
                target[corp] = {}

            if acct_nm in _EQUITY_NAMES and "equity" not in target[corp]:
                target[corp]["equity"] = amount
            elif acct_nm in _INCOME_NAMES and "net_income" not in target[corp]:
                target[corp]["net_income"] = amount

        # 연결 우선 병합
        for corp in set(list(cfs_data.keys()) + list(ofs_data.keys())):
            entry = {}
            if corp in cfs_data:
                entry.update(cfs_data[corp])
            if corp in ofs_data:
                for k, v in ofs_data[corp].items():
                    if k not in entry:
                        entry[k] = v
            if entry:
                results[corp] = entry

        time.sleep(_RATE_LIMIT_SLEEP)

    logger.info(
        "DART 재무 배치 수집 완료: %d개 요청 → %d개 기업 데이터",
        len(corp_codes),
        len(results),
    )
    return results


def get_all_fundamentals_dart(
    date: str,
    market: str = "ALL",
) -> pd.DataFrame:
    """DART 재무제표 + KRX 시장데이터 기반으로 전종목 펀더멘탈을 산출한다.

    pykrx get_market_fundamental() 대체 함수.
    BPS/EPS는 DART 최신 분기 재무제표에서 계산하고,
    PBR/PER은 KRX/pykrx 시장가격으로 계산한다.

    반환 형식은 collector.get_all_fundamentals()와 동일하다:
        ticker, name, market, sector, bps, per, pbr, eps, div_yield, close, market_cap, volume

    Args:
        date: 기준일 ('YYYYMMDD' 또는 'YYYY-MM-DD')
        market: "KOSPI", "KOSDAQ", "ALL" (기본)

    Returns:
        전종목 펀더멘탈 DataFrame (실패 시 빈 DataFrame)
    """
    formatted_date = date.replace("-", "")
    logger.info("DART 기반 전종목 펀더멘탈 수집 시작: %s (market=%s)", formatted_date, market)

    api_key = _get_dart_api_key()
    if not api_key:
        logger.warning("DART_API_KEY 미설정 — get_all_fundamentals_dart 불가")
        return pd.DataFrame()

    markets = ["KOSPI", "KOSDAQ"] if market.upper() == "ALL" else [market.upper()]

    # ── 1. ticker → corp_code 매핑 ──────────────────────────────
    corp_map = _build_corp_code_map()
    if not corp_map:
        logger.error("DART corp_code 매핑 실패 — 펀더멘탈 수집 중단")
        return pd.DataFrame()

    # ── 2. pykrx 시장데이터 수집 (가격, 시가총액, 상장주식수) ──────
    # pykrx get_market_cap은 KRX 로그인 후에도 동작한다 (krx_session 패치 적용)
    market_dfs: list[pd.DataFrame] = []
    for mkt in markets:
        try:
            from src.data.data_proxy import create_stock_api
            pykrx = create_stock_api()
            cap_df = pykrx.get_market_cap(formatted_date, market=mkt)
            if cap_df.empty:
                logger.warning("%s 시장데이터 없음 (date=%s)", mkt, formatted_date)
                continue
            cap_df = cap_df.rename(columns={
                "종가": "close",
                "시가총액": "market_cap",
                "거래량": "volume",
                "상장주식수": "listed_shares",
            })
            cap_df["market"] = mkt

            # 종목명 / 업종 보완
            pykrx.get_market_ticker_list(formatted_date, market=mkt)
            cap_df["name"] = [
                pykrx.get_market_ticker_name(t) for t in cap_df.index
            ]
            try:
                sec_df = pykrx.get_market_sector_classifications(formatted_date, market=mkt)
                if not sec_df.empty and "업종명" in sec_df.columns:
                    cap_df["sector"] = cap_df.index.map(sec_df["업종명"]).fillna("")
                else:
                    cap_df["sector"] = ""
            except Exception:
                cap_df["sector"] = ""

            market_dfs.append(cap_df)
        except Exception as e:
            logger.warning("pykrx 시장데이터 수집 실패 (%s): %s", mkt, e)

    if not market_dfs:
        logger.error("모든 시장 데이터 수집 실패 — DART fallback 중단")
        return pd.DataFrame()

    all_market = pd.concat(market_dfs)
    all_tickers = all_market.index.tolist()

    # ── 3. DART 재무 배치 수집 ───────────────────────────────────
    bsns_year, reprt_code = _determine_latest_report(formatted_date)
    logger.info("DART 기준 사업연도: %s, 보고서코드: %s", bsns_year, reprt_code)

    # ticker → corp_code 매핑 (상장종목 필터)
    ticker_to_corp = {t: corp_map[t] for t in all_tickers if t in corp_map}
    corp_codes_needed = list(ticker_to_corp.values())

    financial_data = _fetch_dart_accounts_batch(
        corp_codes_needed, bsns_year, reprt_code
    )

    # corp_code → ticker 역매핑
    corp_to_ticker = {v: k for k, v in ticker_to_corp.items()}

    # ── 4. 지표 계산 ─────────────────────────────────────────────
    rows: list[dict] = []
    for ticker in all_tickers:
        cap_row = all_market.loc[ticker] if ticker in all_market.index else None
        if cap_row is None:
            continue

        close = float(cap_row.get("close", 0) or 0)
        market_cap = float(cap_row.get("market_cap", 0) or 0)
        volume = float(cap_row.get("volume", 0) or 0)
        listed_shares = float(cap_row.get("listed_shares", 0) or 0)

        # DART 재무데이터
        corp_code = ticker_to_corp.get(ticker)
        fin = financial_data.get(corp_code, {}) if corp_code else {}
        equity = fin.get("equity")
        net_income = fin.get("net_income")

        # BPS = 자본총계 / 상장주식수
        bps = (equity / listed_shares) if (equity is not None and listed_shares > 0) else np.nan
        # EPS = 당기순이익 / 상장주식수
        eps = (net_income / listed_shares) if (net_income is not None and listed_shares > 0) else np.nan
        # PBR = 현재가 / BPS
        pbr = (close / bps) if (bps and not np.isnan(bps) and bps != 0) else np.nan
        # PER = 현재가 / EPS
        per = (close / eps) if (eps and not np.isnan(eps) and eps != 0 and eps > 0) else np.nan

        rows.append({
            "ticker": ticker,
            "name": cap_row.get("name", ""),
            "market": cap_row.get("market", ""),
            "sector": cap_row.get("sector", ""),
            "bps": round(bps) if not np.isnan(bps) else 0,
            "per": round(per, 2) if not np.isnan(per) else 0.0,
            "pbr": round(pbr, 4) if not np.isnan(pbr) else 0.0,
            "eps": round(eps) if not np.isnan(eps) else 0,
            "div_yield": 0.0,  # DART에서 배당수익률 미제공
            "close": int(close),
            "market_cap": int(market_cap),
            "volume": int(volume),
        })

    if not rows:
        logger.warning("DART 기반 펀더멘탈 계산 결과 없음")
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    valid_pbr = (result["pbr"] > 0).sum()
    logger.info(
        "DART 기반 펀더멘탈 수집 완료: %d개 종목, 유효 PBR %d개 (%.1f%%)",
        len(result),
        valid_pbr,
        valid_pbr / len(result) * 100,
    )
    return result
