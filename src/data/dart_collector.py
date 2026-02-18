"""DART OpenAPI 재무제표 수집 모듈.

opendartreader 라이브러리를 활용하여 DART(전자공시시스템)에서
재무제표 데이터를 수집하고, 퀄리티 팩터 산출에 필요한 지표를 제공한다.

DART API key가 없을 경우 pykrx 기본 데이터로 퀄리티 근사치를 산출한다.
"""

import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

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
        # 손익계산서 조회
        time.sleep(_RATE_LIMIT_SLEEP)
        income_stmt = dart.finstate(corp_code, year, reprt_code=reprt_code, fs_div="CFS")
        if income_stmt is None or income_stmt.empty:
            # 연결재무제표 없으면 개별재무제표 조회
            time.sleep(_RATE_LIMIT_SLEEP)
            income_stmt = dart.finstate(corp_code, year, reprt_code=reprt_code, fs_div="OFS")

        # 재무상태표 조회
        time.sleep(_RATE_LIMIT_SLEEP)
        balance_sheet = dart.finstate(corp_code, year, reprt_code=reprt_code, fs_div="CFS")
        if balance_sheet is None or balance_sheet.empty:
            time.sleep(_RATE_LIMIT_SLEEP)
            balance_sheet = dart.finstate(corp_code, year, reprt_code=reprt_code, fs_div="OFS")

        if income_stmt is None or income_stmt.empty:
            logger.warning(f"재무제표 데이터 없음: {corp_code}, {year}Q{quarter}")
            return pd.DataFrame()

        # 계정과목 추출 헬퍼
        def _extract_amount(df: pd.DataFrame, account_names: list[str]) -> Optional[float]:
            """재무제표에서 특정 계정과목의 금액을 추출한다."""
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

        # 주요 계정 추출
        net_income = _extract_amount(income_stmt, ["당기순이익", "당기순손익"])
        equity = _extract_amount(balance_sheet, ["자본총계"])
        total_assets = _extract_amount(balance_sheet, ["자산총계"])
        total_liabilities = _extract_amount(balance_sheet, ["부채총계"])
        gross_profit = _extract_amount(income_stmt, ["매출총이익"])
        operating_income = _extract_amount(income_stmt, ["영업이익"])

        # 현금흐름표에서 영업활동현금흐름 조회
        time.sleep(_RATE_LIMIT_SLEEP)
        try:
            cf_stmt = dart.finstate(corp_code, year, reprt_code=reprt_code, fs_div="CFS")
            operating_cf = _extract_amount(cf_stmt, ["영업활동현금흐름", "영업활동으로인한현금흐름"]) if cf_stmt is not None else None
        except Exception:
            operating_cf = None

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
