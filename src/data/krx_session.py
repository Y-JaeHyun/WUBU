"""pykrx 세션 쿠키 주입.

KRX data.krx.co.kr에 로그인하여 JSESSIONID를 획득하고,
pykrx의 webio.Post/Get을 monkey-patch하여 세션 쿠키를 자동 전송한다.

사용법:
    1. .env에 KRX_DATA_ID, KRX_DATA_PW 설정
    2. collector import 전에 krx_session.init() 호출
    3. 이후 pykrx 호출 시 세션 쿠키가 자동 포함됨

참고: pykrx issue #276 (https://github.com/sharebook-kr/pykrx/issues/276)
"""

import os
import time

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── 모듈 상태 ──────────────────────────────────────────
_session: requests.Session | None = None
_patched: bool = False
_last_login: float = 0.0
_login_id: str = ""
_login_pw: str = ""

# ── 상수 ───────────────────────────────────────────────
_LOGIN_PAGE = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
_LOGIN_JSP = (
    "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
)
_LOGIN_URL = (
    "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
)
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_SESSION_TTL = 3600  # 1시간마다 재로그인


def init() -> bool:
    """환경변수에서 KRX 로그인 정보를 읽고 세션을 초기화한다.

    환경변수:
        KRX_DATA_ID: KRX 데이터 포털 아이디
        KRX_DATA_PW: KRX 데이터 포털 비밀번호

    Returns:
        로그인 성공 시 True, 환경변수 미설정 또는 실패 시 False.
    """
    global _login_id, _login_pw
    _login_id = os.environ.get("KRX_DATA_ID", "").strip()
    _login_pw = os.environ.get("KRX_DATA_PW", "").strip()
    if not _login_id or not _login_pw:
        logger.debug("KRX_DATA_ID/PW 미설정 — 세션 쿠키 주입 건너뜀")
        return False
    return login(_login_id, _login_pw)


def login(login_id: str, login_pw: str) -> bool:
    """KRX data.krx.co.kr에 로그인한다.

    로그인 흐름:
        1. GET MDCCOMS001.cmd  → 초기 JSESSIONID 발급
        2. GET login.jsp       → iframe 세션 초기화
        3. POST MDCCOMS001D1.cmd → 실제 로그인
        4. CD011(중복 로그인) → skipDup=Y 추가 후 재전송

    Args:
        login_id: KRX 포털 아이디.
        login_pw: KRX 포털 비밀번호.

    Returns:
        로그인 성공 시 True.
    """
    global _session, _last_login, _login_id, _login_pw
    _login_id = login_id
    _login_pw = login_pw

    _session = requests.Session()
    _patch_webio()

    headers = {"User-Agent": _UA}

    try:
        # 1. 초기 세션(JSESSIONID) 발급
        _session.get(_LOGIN_PAGE, headers=headers, timeout=15)

        # 2. iframe 세션 초기화
        _session.get(
            _LOGIN_JSP,
            headers={**headers, "Referer": _LOGIN_PAGE},
            timeout=15,
        )

        # 3. 로그인 POST
        payload = {
            "mbrNm": "",
            "telNo": "",
            "di": "",
            "certType": "",
            "mbrId": login_id,
            "pw": login_pw,
        }
        resp = _session.post(
            _LOGIN_URL,
            data=payload,
            headers={**headers, "Referer": _LOGIN_PAGE},
            timeout=15,
        )
        data = resp.json()
        error_code = data.get("_error_code", "")

        # 4. 중복 로그인 처리
        if error_code == "CD011":
            payload["skipDup"] = "Y"
            resp = _session.post(
                _LOGIN_URL,
                data=payload,
                headers={**headers, "Referer": _LOGIN_PAGE},
                timeout=15,
            )
            data = resp.json()
            error_code = data.get("_error_code", "")

        success = error_code == "CD001"
        if success:
            _last_login = time.monotonic()
            logger.info("KRX 데이터 포털 로그인 성공")
        else:
            logger.warning("KRX 로그인 실패: error_code=%s", error_code)
        return success

    except Exception as e:
        logger.warning("KRX 로그인 중 예외: %s", e)
        return False


def ensure_session() -> None:
    """세션이 만료되었으면 재로그인한다."""
    if _session is None:
        init()
        return
    if time.monotonic() - _last_login > _SESSION_TTL:
        logger.info("KRX 세션 만료 (>%ds), 재로그인", _SESSION_TTL)
        if _login_id and _login_pw:
            login(_login_id, _login_pw)


def _patch_webio() -> None:
    """pykrx의 webio.Post.read / webio.Get.read를 세션 기반으로 교체한다."""
    global _patched
    if _patched:
        return

    from pykrx.website.comm import webio

    def _session_post_read(self, **params):
        ensure_session()
        return _session.post(self.url, headers=self.headers, data=params)

    def _session_get_read(self, **params):
        ensure_session()
        return _session.get(self.url, headers=self.headers, params=params)

    webio.Post.read = _session_post_read
    webio.Get.read = _session_get_read
    _patched = True
    logger.info("pykrx webio monkey-patch 완료 (세션 쿠키 주입)")


def is_logged_in() -> bool:
    """현재 로그인 상태인지 반환한다."""
    return _session is not None and _last_login > 0
