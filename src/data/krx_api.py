"""KRX Open API HTTP 클라이언트.

한국거래소 공식 Open API (openapi.krx.co.kr)를 호출한다.
AUTH_KEY 기반 인증, 자동 재시도, 속도 제한을 지원한다.
"""

import time
from typing import Any

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"

# ── 엔드포인트 경로 ────────────────────────────────────
# 주식 (Stock)
STK_BYDD_TRD = "/sto/stk_bydd_trd"          # KOSPI 일별매매정보
KSQ_BYDD_TRD = "/sto/ksq_bydd_trd"          # KOSDAQ 일별매매정보
STK_ISU_BASE_INFO = "/sto/stk_isu_base_info"  # KOSPI 종목기본정보
KSQ_ISU_BASE_INFO = "/sto/ksq_isu_base_info"  # KOSDAQ 종목기본정보

# 지수 (Index)
KOSPI_DD_TRD = "/idx/kospi_dd_trd"           # KOSPI 지수 일별시세
KOSDAQ_DD_TRD = "/idx/kosdaq_dd_trd"         # KOSDAQ 지수 일별시세

# ETF
ETF_BYDD_TRD = "/etf/etf_bydd_trd"          # ETF 일별매매정보

# ── 시장별 엔드포인트 매핑 ──────────────────────────────
MARKET_TRADE_ENDPOINTS = {
    "KOSPI": STK_BYDD_TRD,
    "KOSDAQ": KSQ_BYDD_TRD,
}
MARKET_INFO_ENDPOINTS = {
    "KOSPI": STK_ISU_BASE_INFO,
    "KOSDAQ": KSQ_ISU_BASE_INFO,
}
MARKET_INDEX_ENDPOINTS = {
    "KOSPI": KOSPI_DD_TRD,
    "KOSDAQ": KOSDAQ_DD_TRD,
    "1001": KOSPI_DD_TRD,
    "2001": KOSDAQ_DD_TRD,
}

# ── 예외 클래스 ─────────────────────────────────────────


class KRXAPIError(Exception):
    """KRX API 호출 실패."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"KRX API Error {status_code}: {message}")


class KRXQuotaExceeded(KRXAPIError):
    """일일 호출 한도 초과."""

    def __init__(self) -> None:
        super().__init__(429, "일일 API 호출 한도(10,000회) 초과")


# ── Rate Limiter ────────────────────────────────────────


class RateLimiter:
    """초당 호출 제한. 슬라이딩 윈도우 방식."""

    def __init__(self, max_per_second: float = 5.0) -> None:
        self._interval = 1.0 / max_per_second
        self._last_call: float = 0.0

    def acquire(self) -> None:
        """필요 시 sleep하여 속도 제한을 준수한다."""
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.monotonic()


# ── 메인 클래스 ─────────────────────────────────────────


class KRXOpenAPI:
    """KRX Open API HTTP 클라이언트.

    Args:
        auth_key: KRX Open API 인증키.
        calls_per_second: 초당 최대 호출 수. 기본 5.
        max_retries: 실패 시 재시도 횟수. 기본 3.
        timeout: HTTP 타임아웃(초). 기본 30.
    """

    def __init__(
        self,
        auth_key: str,
        calls_per_second: float = 5.0,
        max_retries: int = 3,
        timeout: int = 30,
    ) -> None:
        self._auth_key = auth_key.strip()
        self._session = requests.Session()
        self._session.headers.update({
            "AUTH_KEY": self._auth_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._rate_limiter = RateLimiter(calls_per_second)
        self._max_retries = max_retries
        self._timeout = timeout

    def request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict]:
        """단일 API 호출.

        Args:
            path: 엔드포인트 경로 (예: '/sto/stk_bydd_trd').
            params: POST JSON payload.

        Returns:
            OutBlock_1 데이터 리스트.

        Raises:
            KRXAPIError: HTTP 에러 또는 응답 파싱 실패.
            KRXQuotaExceeded: 일일 한도 초과.
        """
        url = f"{BASE_URL}{path}"
        payload = params or {}
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                self._rate_limiter.acquire()
                resp = self._session.post(
                    url, json=payload, timeout=self._timeout,
                )

                if resp.status_code == 429:
                    raise KRXQuotaExceeded()

                if resp.status_code != 200:
                    raise KRXAPIError(resp.status_code, resp.text[:200])

                data = resp.json()
                return data.get("OutBlock_1", [])

            except KRXQuotaExceeded:
                raise
            except KRXAPIError:
                raise
            except Exception as e:
                last_exc = e
                if attempt < self._max_retries:
                    delay = 2.0 * attempt
                    logger.warning(
                        "KRX API 호출 실패 (시도 %d/%d): %s — %.1f초 후 재시도",
                        attempt, self._max_retries, e, delay,
                    )
                    time.sleep(delay)

        logger.error("KRX API 최종 실패: %s", last_exc)
        raise KRXAPIError(0, str(last_exc))

    def request_all(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        page_size: int = 1000,
    ) -> list[dict]:
        """페이지네이션 자동 처리.

        numOfRows/pageNo 파라미터를 사용하여 모든 데이터를 가져온다.

        Args:
            path: 엔드포인트 경로.
            params: 기본 파라미터.
            page_size: 페이지당 행 수.

        Returns:
            전체 데이터 리스트.
        """
        payload = dict(params or {})
        payload["numOfRows"] = str(page_size)
        all_rows: list[dict] = []
        page_no = 1

        while True:
            payload["pageNo"] = str(page_no)
            rows = self.request(path, payload)
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < page_size:
                break
            page_no += 1

        return all_rows
