"""pykrx ↔ KRX Open API 자동 fallback 프록시.

Primary 데이터 소스 호출 실패 시 Fallback 소스로 자동 재시도한다.
Collector 모듈에서 ``pykrx_stock.xxx()`` 호출을 변경 없이 유지하면서
장애 시 투명하게 대체 소스를 사용할 수 있다.

데이터 소스 우선순위:
  1. krx_openapi flag ON + KRX_API_KEY → KRX Open API primary, pykrx fallback
  2. KRX_API_KEY만 설정 (flag OFF)     → pykrx primary, KRX Open API fallback
  3. KRX_API_KEY 없음                  → pykrx only (fallback 없음)
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataProxy:
    """pykrx ↔ KRX Open API 자동 fallback 프록시.

    ``__getattr__`` 을 통해 primary 백엔드의 함수를 호출하고,
    예외 발생 시 fallback 백엔드의 동일 함수를 호출한다.
    """

    def __init__(self, primary, fallback=None,
                 primary_name: str = "primary",
                 fallback_name: str = "fallback"):
        # object.__setattr__ 사용 — __getattr__ 재귀 방지
        object.__setattr__(self, "_primary", primary)
        object.__setattr__(self, "_fallback", fallback)
        object.__setattr__(self, "_primary_name", primary_name)
        object.__setattr__(self, "_fallback_name", fallback_name)
        object.__setattr__(self, "_fallback_count", 0)

    def __getattr__(self, name: str):
        primary = object.__getattribute__(self, "_primary")
        fallback = object.__getattribute__(self, "_fallback")
        primary_name = object.__getattribute__(self, "_primary_name")
        fallback_name = object.__getattribute__(self, "_fallback_name")

        primary_fn = getattr(primary, name, None)
        fallback_fn = getattr(fallback, name, None) if fallback else None

        if primary_fn is None and fallback_fn is None:
            raise AttributeError(
                f"'{type(self).__name__}' 에 '{name}' 속성이 없습니다 "
                f"(primary={primary_name}, fallback={fallback_name})"
            )

        # 한쪽에만 있는 함수 → 직접 반환
        if primary_fn is None:
            return fallback_fn
        if fallback_fn is None:
            return primary_fn

        def _with_fallback(*args, **kwargs):
            try:
                return primary_fn(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    "%s.%s 실패 → %s fallback: %s",
                    primary_name, name, fallback_name, e,
                )
                cnt = object.__getattribute__(self, "_fallback_count")
                object.__setattr__(self, "_fallback_count", cnt + 1)
                return fallback_fn(*args, **kwargs)

        _with_fallback.__name__ = name
        return _with_fallback


def create_stock_api():
    """데이터 소스 초기화 + fallback 프록시 생성.

    Returns:
        DataProxy 또는 pykrx 모듈 (fallback 불필요 시).
    """
    from src.data import krx_provider as _krx

    # pykrx 초기화 (항상 — 세션 쿠키 주입 포함)
    from src.data import krx_session
    krx_session.init()
    from pykrx import stock as _pykrx

    # KRX Open API 가능 여부 (feature flag 무관, API key만 확인)
    krx_api_available = _krx._has_api_key()

    if _krx.is_available():
        # krx_openapi flag ON → KRX API primary, pykrx fallback
        logger.info("데이터 소스: KRX Open API (primary) + pykrx (fallback)")
        return DataProxy(_krx, _pykrx, "krx_api", "pykrx")
    elif krx_api_available:
        # flag OFF + API key → pykrx primary, KRX API fallback
        logger.info("데이터 소스: pykrx (primary) + KRX Open API (fallback)")
        return DataProxy(_pykrx, _krx, "pykrx", "krx_api")
    else:
        # API key 없음 → pykrx only
        logger.info("데이터 소스: pykrx only (fallback 없음)")
        return _pykrx
