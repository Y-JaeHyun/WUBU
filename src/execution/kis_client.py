"""한국투자증권 OpenAPI 클라이언트 모듈.

REST API를 통한 주문/잔고 조회 기능을 제공한다.
모의투자(paper)와 실전 모드를 지원하며,
토큰 자동 갱신, 요청 재시도, 속도 제한 등을 내장한다.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class KISClient:
    """한국투자증권 OpenAPI 클라이언트.

    REST API를 통한 주문/잔고 조회, WebSocket을 통한 실시간 시세를 제공한다.
    모의투자(paper)와 실전 모드를 지원한다.

    Attributes:
        app_key: 한국투자증권 앱 키.
        app_secret: 한국투자증권 앱 시크릿.
        account_no: 계좌번호 (8자리-2자리 형식).
        is_paper: 모의투자 모드 여부.
    """

    REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"
    PAPER_BASE_URL = "https://openapivts.koreainvestment.com:29443"

    REQUEST_TIMEOUT = 10  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    RATE_LIMIT_DELAY = 0.1  # seconds between API calls

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        account_no: Optional[str] = None,
        is_paper: Optional[bool] = None,
    ) -> None:
        """KISClient를 초기화한다.

        Args:
            app_key: 한국투자증권 앱 키. None이면 환경변수 KIS_APP_KEY에서 로드.
            app_secret: 한국투자증권 앱 시크릿. None이면 환경변수 KIS_APP_SECRET에서 로드.
            account_no: 계좌번호. None이면 모드에 따라 환경변수에서 자동 선택.
            is_paper: 모의투자 모드. None이면 환경변수에서 로드.
                우선순위: KIS_TRADING_MODE > KIS_IS_PAPER > 기본값(True).
        """
        self.app_key: str = app_key or os.getenv("KIS_APP_KEY", "")
        self.app_secret: str = app_secret or os.getenv("KIS_APP_SECRET", "")

        # 모드 결정 우선순위: 파라미터 > KIS_TRADING_MODE > KIS_IS_PAPER > 기본값
        if is_paper is not None:
            self.is_paper: bool = is_paper
        else:
            trading_mode = os.getenv("KIS_TRADING_MODE", "").lower()
            if trading_mode == "live":
                self.is_paper = False
            elif trading_mode == "paper":
                self.is_paper = True
            else:
                env_val = os.getenv("KIS_IS_PAPER", "true").lower()
                self.is_paper = env_val in ("true", "1", "yes")

        # 계좌번호: 파라미터 > 모드별 환경변수 > KIS_ACCOUNT_NO
        if account_no:
            self.account_no: str = account_no
        elif self.is_paper:
            self.account_no = os.getenv(
                "KIS_PAPER_ACCOUNT_NO", os.getenv("KIS_ACCOUNT_NO", "")
            )
        else:
            self.account_no = os.getenv(
                "KIS_REAL_ACCOUNT_NO", os.getenv("KIS_ACCOUNT_NO", "")
            )

        self._access_token: str = ""
        self._token_expires_at: Optional[datetime] = None
        self._last_request_time: float = 0.0
        self._token_cache_path = Path(__file__).resolve().parent.parent.parent / ".kis_token.json"

        # 캐시된 토큰 로드 시도
        self._load_cached_token()

        if not self.is_configured():
            logger.warning(
                "KIS API 설정이 불완전합니다. "
                "KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO를 확인하세요."
            )
        else:
            logger.info(
                "KIS 클라이언트 초기화 완료 (모드: %s, 계좌: %s)",
                self.trading_mode,
                self.account_no[:4] + "****" if len(self.account_no) >= 4 else "****",
            )

    @property
    def base_url(self) -> str:
        """현재 모드에 따른 기본 URL을 반환한다."""
        return self.PAPER_BASE_URL if self.is_paper else self.REAL_BASE_URL

    @property
    def trading_mode(self) -> str:
        """현재 트레이딩 모드 문자열을 반환한다."""
        return "모의투자" if self.is_paper else "실전투자"

    @property
    def mode_tag(self) -> str:
        """알림 메시지용 모드 태그를 반환한다."""
        return "[모의]" if self.is_paper else "[실전]"

    def is_configured(self) -> bool:
        """API 키가 모두 설정되어 있는지 확인한다.

        Returns:
            설정이 완료되었으면 True, 아니면 False.
        """
        return bool(self.app_key) and bool(self.app_secret) and bool(self.account_no)

    # ──────────────────────────────────────────────────────────
    # 인증
    # ──────────────────────────────────────────────────────────

    def get_access_token(self) -> str:
        """액세스 토큰을 발급받는다.

        POST /oauth2/tokenP 엔드포인트를 호출하여 토큰을 발급받고,
        만료 시간을 기록한다.

        Returns:
            발급받은 액세스 토큰 문자열.

        Raises:
            RuntimeError: 토큰 발급에 실패한 경우.
        """
        if not self.is_configured():
            logger.error("KIS API 미설정 상태에서 토큰 발급을 시도했습니다.")
            return ""

        path = "/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        try:
            url = f"{self.base_url}{path}"
            response = requests.post(url, json=body, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            self._access_token = data.get("access_token", "")
            # 토큰 만료 시간 설정 (발급 후 약 24시간, 안전하게 23시간으로 설정)
            expires_in = int(data.get("expires_in", 86400))
            self._token_expires_at = datetime.now() + timedelta(
                seconds=max(expires_in - 3600, 0)
            )

            self._save_cached_token()
            logger.info(
                "KIS 액세스 토큰 발급 완료 (만료: %s)",
                self._token_expires_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return self._access_token

        except requests.exceptions.RequestException as e:
            logger.error("KIS 토큰 발급 실패: %s", e)
            raise RuntimeError(f"KIS 토큰 발급 실패: {e}") from e

    def _ensure_token(self) -> None:
        """액세스 토큰이 유효한지 확인하고, 만료되었으면 재발급한다."""
        if not self._access_token or (
            self._token_expires_at is not None
            and datetime.now() >= self._token_expires_at
        ):
            logger.info("액세스 토큰 갱신 필요. 재발급합니다.")
            self.get_access_token()

    def _save_cached_token(self) -> None:
        """발급받은 토큰을 파일에 캐싱한다."""
        try:
            data = {
                "access_token": self._access_token,
                "expires_at": self._token_expires_at.isoformat()
                if self._token_expires_at
                else "",
                "is_paper": self.is_paper,
            }
            self._token_cache_path.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    def _load_cached_token(self) -> None:
        """캐시된 토큰을 파일에서 로드한다."""
        try:
            if not self._token_cache_path.exists():
                return
            data = json.loads(self._token_cache_path.read_text(encoding="utf-8"))
            # 모드가 다르면 캐시 무효
            if data.get("is_paper") != self.is_paper:
                return
            expires_at_str = data.get("expires_at", "")
            if not expires_at_str:
                return
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.now() >= expires_at:
                return
            self._access_token = data["access_token"]
            self._token_expires_at = expires_at
            logger.info(
                "캐시된 KIS 토큰 로드 완료 (만료: %s)",
                expires_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass

    def _headers(self, tr_id: str = "", content_type: str = "application/json") -> dict:
        """공통 요청 헤더를 생성한다.

        Args:
            tr_id: 트랜잭션 ID (API별로 다름).
            content_type: Content-Type 헤더 값.

        Returns:
            요청에 사용할 헤더 딕셔너리.
        """
        self._ensure_token()

        headers = {
            "content-type": content_type,
            "authorization": f"Bearer {self._access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        if tr_id:
            headers["tr_id"] = tr_id
        return headers

    # ──────────────────────────────────────────────────────────
    # HTTP 요청 헬퍼
    # ──────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        headers: Optional[dict] = None,
        body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """API 요청을 실행한다.

        재시도, 속도 제한, 에러 핸들링을 포함한다.

        Args:
            method: HTTP 메서드 ("GET" 또는 "POST").
            path: API 경로 (예: "/uapi/domestic-stock/v1/trading/order-cash").
            headers: 요청 헤더. None이면 기본 헤더 사용.
            body: POST 요청 시 JSON body.
            params: GET 요청 시 query parameters.

        Returns:
            API 응답 JSON 딕셔너리.
        """
        if not self.is_configured():
            logger.warning("KIS API 미설정 상태. 빈 응답을 반환합니다.")
            return {}

        url = f"{self.base_url}{path}"

        for attempt in range(1, self.MAX_RETRIES + 1):
            # 속도 제한
            elapsed = time.time() - self._last_request_time
            if elapsed < self.RATE_LIMIT_DELAY:
                time.sleep(self.RATE_LIMIT_DELAY - elapsed)

            try:
                if method.upper() == "GET":
                    response = requests.get(
                        url,
                        headers=headers,
                        params=params,
                        timeout=self.REQUEST_TIMEOUT,
                    )
                elif method.upper() == "POST":
                    response = requests.post(
                        url,
                        headers=headers,
                        json=body,
                        timeout=self.REQUEST_TIMEOUT,
                    )
                else:
                    logger.error("지원하지 않는 HTTP 메서드: %s", method)
                    return {}

                self._last_request_time = time.time()
                response.raise_for_status()
                data = response.json()

                # KIS API 응답 코드 확인
                rt_cd = data.get("rt_cd")
                if rt_cd is not None and str(rt_cd) != "0":
                    msg = data.get("msg1", "알 수 없는 오류")
                    logger.error(
                        "KIS API 오류 (rt_cd=%s): %s (경로: %s)", rt_cd, msg, path
                    )
                    return data

                return data

            except requests.exceptions.Timeout:
                logger.warning(
                    "KIS API 타임아웃 (시도 %d/%d): %s",
                    attempt,
                    self.MAX_RETRIES,
                    path,
                )
            except requests.exceptions.ConnectionError:
                logger.warning(
                    "KIS API 연결 실패 (시도 %d/%d): %s",
                    attempt,
                    self.MAX_RETRIES,
                    path,
                )
            except requests.exceptions.RequestException as e:
                logger.error(
                    "KIS API 요청 오류 (시도 %d/%d): %s - %s",
                    attempt,
                    self.MAX_RETRIES,
                    path,
                    e,
                )

            if attempt < self.MAX_RETRIES:
                delay = self.RETRY_DELAY * attempt
                logger.info("%.1f초 후 재시도합니다.", delay)
                time.sleep(delay)

        logger.error("KIS API 요청 최종 실패: %s", path)
        return {}

    # ──────────────────────────────────────────────────────────
    # 주문
    # ──────────────────────────────────────────────────────────

    def _get_tr_id(self, action: str) -> str:
        """모드(실전/모의)에 따른 트랜잭션 ID를 반환한다.

        Args:
            action: "buy", "sell", "cancel" 중 하나.

        Returns:
            트랜잭션 ID 문자열.
        """
        tr_ids = {
            "buy": ("VTTC0802U" if self.is_paper else "TTTC0802U"),
            "sell": ("VTTC0801U" if self.is_paper else "TTTC0801U"),
            "cancel": ("VTTC0803U" if self.is_paper else "TTTC0803U"),
        }
        return tr_ids.get(action, "")

    def _order_type_code(self, order_type: str) -> str:
        """주문 유형 문자열을 KIS API 코드로 변환한다.

        Args:
            order_type: "시장가" 또는 "지정가".

        Returns:
            KIS API 주문 유형 코드.
        """
        mapping = {
            "시장가": "01",
            "지정가": "00",
        }
        return mapping.get(order_type, "01")

    def _parse_account(self) -> tuple[str, str]:
        """계좌번호를 CANO(8자리)와 ACNT_PRDT_CD(2자리)로 분리한다.

        Returns:
            (CANO, ACNT_PRDT_CD) 튜플.
        """
        account = self.account_no.replace("-", "")
        if len(account) >= 10:
            return account[:8], account[8:10]
        return account, "01"

    def place_buy_order(
        self,
        ticker: str,
        qty: int,
        price: int = 0,
        order_type: str = "시장가",
    ) -> dict:
        """매수 주문을 실행한다.

        Args:
            ticker: 종목코드 (6자리).
            qty: 주문 수량.
            price: 주문 가격. 시장가일 경우 0.
            order_type: "시장가" 또는 "지정가".

        Returns:
            주문 결과 딕셔너리. 성공 시 order_no 포함.
        """
        return self._place_order("buy", ticker, qty, price, order_type)

    def place_sell_order(
        self,
        ticker: str,
        qty: int,
        price: int = 0,
        order_type: str = "시장가",
    ) -> dict:
        """매도 주문을 실행한다.

        Args:
            ticker: 종목코드 (6자리).
            qty: 주문 수량.
            price: 주문 가격. 시장가일 경우 0.
            order_type: "시장가" 또는 "지정가".

        Returns:
            주문 결과 딕셔너리. 성공 시 order_no 포함.
        """
        return self._place_order("sell", ticker, qty, price, order_type)

    def _place_order(
        self,
        side: str,
        ticker: str,
        qty: int,
        price: int,
        order_type: str,
    ) -> dict:
        """내부 주문 실행 메서드.

        Args:
            side: "buy" 또는 "sell".
            ticker: 종목코드.
            qty: 주문 수량.
            price: 주문 가격.
            order_type: 주문 유형.

        Returns:
            주문 결과 딕셔너리.
        """
        if qty <= 0:
            logger.warning("주문 수량이 0 이하입니다: ticker=%s, qty=%d", ticker, qty)
            return {"error": "주문 수량이 0 이하입니다."}

        tr_id = self._get_tr_id(side)
        cano, acnt_prdt_cd = self._parse_account()
        ord_dvsn = self._order_type_code(order_type)

        path = "/uapi/domestic-stock/v1/trading/order-cash"
        headers = self._headers(tr_id=tr_id)
        body = {
            "CANO": cano,
            "ACNT_PRDT_CD": acnt_prdt_cd,
            "PDNO": ticker,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price),
        }

        side_kr = "매수" if side == "buy" else "매도"
        logger.info(
            "%s 주문 요청: ticker=%s, qty=%d, price=%d, type=%s",
            side_kr,
            ticker,
            qty,
            price,
            order_type,
        )

        result = self._request("POST", path, headers=headers, body=body)

        if result and result.get("rt_cd") == "0":
            output = result.get("output", {})
            order_no = output.get("ODNO", "")
            logger.info(
                "%s 주문 성공: ticker=%s, order_no=%s",
                side_kr,
                ticker,
                order_no,
            )
            return {
                "success": True,
                "order_no": order_no,
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "price": price,
                "order_type": order_type,
            }

        msg = result.get("msg1", "알 수 없는 오류") if result else "응답 없음"
        logger.error("%s 주문 실패: ticker=%s, msg=%s", side_kr, ticker, msg)
        return {
            "success": False,
            "error": msg,
            "ticker": ticker,
            "side": side,
        }

    def cancel_order(self, order_no: str, ticker: str, qty: int) -> dict:
        """주문을 취소한다.

        Args:
            order_no: 취소할 주문 번호.
            ticker: 종목코드.
            qty: 취소 수량.

        Returns:
            취소 결과 딕셔너리.
        """
        tr_id = self._get_tr_id("cancel")
        cano, acnt_prdt_cd = self._parse_account()

        path = "/uapi/domestic-stock/v1/trading/order-rvsecncl"
        headers = self._headers(tr_id=tr_id)
        body = {
            "CANO": cano,
            "ACNT_PRDT_CD": acnt_prdt_cd,
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": order_no,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",  # 02: 취소
            "ORD_QTY": str(qty),
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y",
        }

        logger.info("주문 취소 요청: order_no=%s, ticker=%s", order_no, ticker)
        result = self._request("POST", path, headers=headers, body=body)

        if result and result.get("rt_cd") == "0":
            logger.info("주문 취소 성공: order_no=%s", order_no)
            return {"success": True, "order_no": order_no}

        msg = result.get("msg1", "알 수 없는 오류") if result else "응답 없음"
        logger.error("주문 취소 실패: order_no=%s, msg=%s", order_no, msg)
        return {"success": False, "error": msg}

    def get_order_status(self, order_no: str) -> dict:
        """주문 상태를 조회한다.

        Args:
            order_no: 조회할 주문 번호.

        Returns:
            주문 상태 딕셔너리. status, filled_qty, filled_price 등 포함.
        """
        tr_id = "VTTC8001R" if self.is_paper else "TTTC8001R"
        cano, acnt_prdt_cd = self._parse_account()

        path = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        headers = self._headers(tr_id=tr_id)
        params = {
            "CANO": cano,
            "ACNT_PRDT_CD": acnt_prdt_cd,
            "INQR_STRT_DT": datetime.now().strftime("%Y%m%d"),
            "INQR_END_DT": datetime.now().strftime("%Y%m%d"),
            "SLL_BUY_DVSN_CD": "00",
            "INQR_DVSN": "00",
            "PDNO": "",
            "CCLD_DVSN": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": order_no,
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        result = self._request("GET", path, headers=headers, params=params)

        if not result or result.get("rt_cd") != "0":
            return {"status": "unknown", "order_no": order_no}

        output_list = result.get("output1", [])
        for item in output_list:
            if item.get("ODNO") == order_no:
                filled_qty = int(item.get("TOT_CCLD_QTY", "0"))
                ord_qty = int(item.get("ORD_QTY", "0"))
                filled_price = float(item.get("AVG_PRVS", "0"))

                if filled_qty >= ord_qty and ord_qty > 0:
                    status = "filled"
                elif filled_qty > 0:
                    status = "partial"
                else:
                    status = "submitted"

                return {
                    "status": status,
                    "order_no": order_no,
                    "filled_qty": filled_qty,
                    "filled_price": filled_price,
                    "ord_qty": ord_qty,
                }

        return {"status": "unknown", "order_no": order_no}

    # ──────────────────────────────────────────────────────────
    # 계좌
    # ──────────────────────────────────────────────────────────

    def get_balance(self) -> dict:
        """계좌 잔고를 조회한다.

        Returns:
            잔고 정보 딕셔너리.
            - total_eval: 총 평가금액
            - total_profit: 총 손익
            - total_profit_pct: 총 수익률 (%)
            - cash: 예수금
            - holdings: 보유 종목 리스트
        """
        tr_id = "VTTC8434R" if self.is_paper else "TTTC8434R"
        cano, acnt_prdt_cd = self._parse_account()

        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = self._headers(tr_id=tr_id)
        params = {
            "CANO": cano,
            "ACNT_PRDT_CD": acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        result = self._request("GET", path, headers=headers, params=params)

        if not result:
            logger.warning("잔고 조회 실패: 빈 응답")
            return {
                "total_eval": 0,
                "total_profit": 0,
                "total_profit_pct": 0.0,
                "cash": 0,
                "holdings": [],
            }

        output1 = result.get("output1", [])
        output2 = result.get("output2", [{}])

        summary = output2[0] if output2 else {}
        total_eval = int(float(summary.get("tot_evlu_amt", "0")))
        cash = int(float(summary.get("dnca_tot_amt", "0")))
        total_purchase = int(float(summary.get("pchs_amt_smtl_amt", "0")))
        total_profit = int(float(summary.get("evlu_pfls_smtl_amt", "0")))
        total_profit_pct = (
            float(total_profit / total_purchase * 100)
            if total_purchase > 0
            else 0.0
        )

        holdings = []
        for item in output1:
            qty = int(item.get("hldg_qty", "0"))
            if qty <= 0:
                continue
            holdings.append({
                "ticker": item.get("pdno", ""),
                "name": item.get("prdt_name", ""),
                "qty": qty,
                "avg_price": int(float(item.get("pchs_avg_pric", "0"))),
                "current_price": int(float(item.get("prpr", "0"))),
                "eval_amount": int(float(item.get("evlu_amt", "0"))),
                "pnl": int(float(item.get("evlu_pfls_amt", "0"))),
                "pnl_pct": float(item.get("evlu_pfls_rt", "0")),
            })

        return {
            "total_eval": total_eval,
            "total_profit": total_profit,
            "total_profit_pct": round(total_profit_pct, 2),
            "cash": cash,
            "holdings": holdings,
        }

    def get_positions(self) -> pd.DataFrame:
        """보유 종목을 DataFrame으로 반환한다.

        Returns:
            보유 종목 DataFrame.
            columns: ticker, name, qty, avg_price, current_price, pnl, pnl_pct
        """
        balance = self.get_balance()
        holdings = balance.get("holdings", [])

        if not holdings:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "name",
                    "qty",
                    "avg_price",
                    "current_price",
                    "pnl",
                    "pnl_pct",
                ]
            )

        df = pd.DataFrame(holdings)
        return df[
            ["ticker", "name", "qty", "avg_price", "current_price", "pnl", "pnl_pct"]
        ]

    def get_buyable_cash(self) -> int:
        """매수 가능 금액을 조회한다.

        Returns:
            매수 가능 금액 (원).
        """
        tr_id = "VTTC8908R" if self.is_paper else "TTTC8908R"
        cano, acnt_prdt_cd = self._parse_account()

        path = "/uapi/domestic-stock/v1/trading/inquire-psbl-order"
        headers = self._headers(tr_id=tr_id)
        params = {
            "CANO": cano,
            "ACNT_PRDT_CD": acnt_prdt_cd,
            "PDNO": "005930",  # 더미 종목 (삼성전자)
            "ORD_UNPR": "0",
            "ORD_DVSN": "01",  # 시장가
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "N",
        }

        result = self._request("GET", path, headers=headers, params=params)

        if not result:
            logger.warning("매수 가능 금액 조회 실패. 잔고에서 현금을 조회합니다.")
            balance = self.get_balance()
            return balance.get("cash", 0)

        output = result.get("output", {})
        buyable = int(float(output.get("ord_psbl_cash", "0")))

        return buyable

    # ──────────────────────────────────────────────────────────
    # 시세
    # ──────────────────────────────────────────────────────────

    def get_current_price(self, ticker: str) -> dict:
        """종목의 현재가를 조회한다.

        Args:
            ticker: 종목코드 (6자리).

        Returns:
            현재가 정보 딕셔너리.
            - price: 현재가
            - volume: 거래량
            - change: 전일 대비
            - change_pct: 등락률 (%)
        """
        tr_id = "FHKST01010100"
        path = "/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self._headers(tr_id=tr_id)
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
        }

        result = self._request("GET", path, headers=headers, params=params)

        if not result:
            logger.warning("현재가 조회 실패: ticker=%s", ticker)
            return {"price": 0, "volume": 0, "change": 0, "change_pct": 0.0}

        output = result.get("output", {})

        return {
            "price": int(output.get("stck_prpr", "0")),
            "volume": int(output.get("acml_vol", "0")),
            "change": int(output.get("prdy_vrss", "0")),
            "change_pct": float(output.get("prdy_ctrt", "0")),
        }
