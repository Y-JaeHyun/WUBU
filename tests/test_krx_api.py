"""KRX Open API 클라이언트 및 프로바이더 테스트.

모든 HTTP 호출은 mock 처리한다.
"""

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.krx_api import (
    KRXAPIError,
    KRXOpenAPI,
    KRXQuotaExceeded,
    RateLimiter,
)


# ── KRXOpenAPI 테스트 ──────────────────────────────────


class TestKRXOpenAPI:
    """KRX Open API HTTP 클라이언트 테스트."""

    def _make_api(self) -> KRXOpenAPI:
        return KRXOpenAPI(auth_key="test_key", calls_per_second=100.0)

    @patch("src.data.krx_api.requests.Session")
    def test_request_success(self, mock_session_cls):
        """정상 응답 시 OutBlock_1 반환."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "OutBlock_1": [
                {"ISU_SRT_CD": "005930", "TDD_CLSPRC": "70000"},
            ]
        }
        session = MagicMock()
        session.post.return_value = mock_resp
        mock_session_cls.return_value = session

        api = self._make_api()
        result = api.request("/sto/stk_bydd_trd", {"basDd": "20260301"})

        assert len(result) == 1
        assert result[0]["ISU_SRT_CD"] == "005930"

    @patch("src.data.krx_api.requests.Session")
    def test_request_http_error(self, mock_session_cls):
        """HTTP 에러 시 KRXAPIError 발생."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        session = MagicMock()
        session.post.return_value = mock_resp
        mock_session_cls.return_value = session

        api = self._make_api()
        with pytest.raises(KRXAPIError) as exc_info:
            api.request("/sto/stk_bydd_trd", {"basDd": "20260301"})
        assert exc_info.value.status_code == 500

    @patch("src.data.krx_api.requests.Session")
    def test_request_quota_exceeded(self, mock_session_cls):
        """429 응답 시 KRXQuotaExceeded 발생."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Too Many Requests"
        session = MagicMock()
        session.post.return_value = mock_resp
        mock_session_cls.return_value = session

        api = self._make_api()
        with pytest.raises(KRXQuotaExceeded):
            api.request("/sto/stk_bydd_trd", {"basDd": "20260301"})

    @patch("src.data.krx_api.requests.Session")
    @patch("src.data.krx_api.time.sleep")
    def test_request_retry_on_network_error(self, mock_sleep, mock_session_cls):
        """네트워크 에러 시 재시도."""
        session = MagicMock()
        # 2회 실패, 3회째 성공
        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.json.return_value = {"OutBlock_1": [{"data": "ok"}]}
        session.post.side_effect = [
            ConnectionError("timeout"),
            ConnectionError("timeout"),
            mock_resp_ok,
        ]
        mock_session_cls.return_value = session

        api = self._make_api()
        result = api.request("/test", {})
        assert result == [{"data": "ok"}]
        # sleep은 재시도 대기 + rate limiter에서도 호출됨
        assert mock_sleep.call_count >= 2  # 최소 2회 재시도 대기

    @patch("src.data.krx_api.requests.Session")
    def test_request_all_pagination(self, mock_session_cls):
        """페이지네이션 자동 처리."""
        session = MagicMock()
        page1 = MagicMock()
        page1.status_code = 200
        page1.json.return_value = {
            "OutBlock_1": [{"id": str(i)} for i in range(1000)]
        }
        page2 = MagicMock()
        page2.status_code = 200
        page2.json.return_value = {
            "OutBlock_1": [{"id": str(i)} for i in range(1000, 1500)]
        }
        session.post.side_effect = [page1, page2]
        mock_session_cls.return_value = session

        api = self._make_api()
        result = api.request_all("/test", {}, page_size=1000)
        assert len(result) == 1500

    @patch("src.data.krx_api.requests.Session")
    def test_request_all_single_page(self, mock_session_cls):
        """단일 페이지(데이터 < page_size)."""
        session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "OutBlock_1": [{"id": "1"}, {"id": "2"}]
        }
        session.post.return_value = mock_resp
        mock_session_cls.return_value = session

        api = self._make_api()
        result = api.request_all("/test", {}, page_size=1000)
        assert len(result) == 2

    @patch("src.data.krx_api.requests.Session")
    def test_request_empty_response(self, mock_session_cls):
        """빈 응답 시 빈 리스트."""
        session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"OutBlock_1": []}
        session.post.return_value = mock_resp
        mock_session_cls.return_value = session

        api = self._make_api()
        result = api.request("/test", {})
        assert result == []

    @patch("src.data.krx_api.requests.Session")
    def test_auth_key_in_headers(self, mock_session_cls):
        """AUTH_KEY가 헤더에 포함되는지 확인."""
        session = MagicMock()
        mock_session_cls.return_value = session

        api = KRXOpenAPI(auth_key="  my_key_123  ", calls_per_second=100.0)
        assert session.headers.update.call_args[0][0]["AUTH_KEY"] == "my_key_123"


class TestRateLimiter:
    """Rate Limiter 테스트."""

    def test_rate_limiting(self):
        """초당 호출 제한이 동작하는지 확인."""
        limiter = RateLimiter(max_per_second=10.0)  # 0.1초 간격
        start = time.monotonic()
        for _ in range(3):
            limiter.acquire()
        elapsed = time.monotonic() - start
        # 3회 호출 → 최소 0.2초 (첫 번째는 즉시)
        assert elapsed >= 0.15


# ── KRX Provider 테스트 ────────────────────────────────


class TestKRXProvider:
    """KRX Open API 프로바이더 테스트."""

    def _reset_provider(self):
        """프로바이더 상태 초기화."""
        import src.data.krx_provider as kp
        kp._api = None
        kp._initialized = False
        kp._ticker_cache.clear()
        kp._etf_ticker_cache.clear()
        kp._cache_date = None

    def test_is_available_no_key(self):
        """KRX_API_KEY 없으면 False."""
        self._reset_provider()
        import src.data.krx_provider as kp
        with patch.dict("os.environ", {}, clear=True):
            assert kp.is_available() is False

    def test_is_available_flag_disabled(self):
        """플래그 OFF이면 False."""
        self._reset_provider()
        import src.data.krx_provider as kp
        mock_ff = MagicMock()
        mock_ff.is_enabled.return_value = False
        with patch.dict("os.environ", {"KRX_API_KEY": "test_key"}):
            with patch("src.utils.feature_flags.FeatureFlags", return_value=mock_ff):
                assert kp.is_available() is False

    def test_format_date(self):
        """날짜 형식 변환."""
        from src.data.krx_provider import _format_date
        assert _format_date("2026-03-01") == "20260301"
        assert _format_date("20260301") == "20260301"
        assert _format_date(pd.Timestamp("2026-03-01")) == "20260301"

    def test_safe_float(self):
        """문자열 → float 변환."""
        from src.data.krx_provider import _safe_float
        assert _safe_float("1,234.56") == 1234.56
        assert _safe_float("0") == 0.0
        assert _safe_float("") == 0.0
        assert _safe_float(None) == 0.0

    def test_safe_int(self):
        """문자열 → int 변환."""
        from src.data.krx_provider import _safe_int
        assert _safe_int("1,234") == 1234
        assert _safe_int("0") == 0
        assert _safe_int("") == 0

    def test_find_column(self):
        """DataFrame에서 후보 컬럼 찾기."""
        from src.data.krx_provider import _find_column
        df = pd.DataFrame({"BAS_DD": [1], "TDD_CLSPRC": [2]})
        assert _find_column(df, ["BAS_DD", "basDd"]) == "BAS_DD"
        assert _find_column(df, ["basDd", "BAS_DD"]) == "BAS_DD"
        assert _find_column(df, ["noexist"]) is None

    def test_to_ohlcv_dataframe(self):
        """OHLCV DataFrame 변환."""
        from src.data.krx_provider import _to_ohlcv_dataframe
        df = pd.DataFrame({
            "BAS_DD": ["20260301", "20260302"],
            "TDD_OPNPRC": ["70000", "71000"],
            "TDD_HGPRC": ["72000", "73000"],
            "TDD_LWPRC": ["69000", "70000"],
            "TDD_CLSPRC": ["71000", "72000"],
            "ACC_TRDVOL": ["1000000", "2000000"],
        })
        result = _to_ohlcv_dataframe(df)
        assert len(result) == 2
        assert "시가" in result.columns
        assert "종가" in result.columns
        assert result.iloc[0]["시가"] == 70000

    def test_to_fundamental_snapshot(self):
        """전종목 펀더멘탈 스냅샷 변환."""
        from src.data.krx_provider import _to_fundamental_snapshot
        df = pd.DataFrame({
            "ISU_SRT_CD": ["005930", "000660"],
            "BPS": ["50000", "30000"],
            "PER": ["10.5", "8.2"],
            "PBR": ["1.5", "1.2"],
            "EPS": ["5000", "3000"],
            "DVD_YLD": ["2.1", "1.5"],
        })
        result = _to_fundamental_snapshot(df)
        assert len(result) == 2
        assert result.index[0] == "005930"
        assert "PER" in result.columns
        assert result.loc["005930", "PER"] == 10.5

    def test_to_marketcap_allstock(self):
        """전종목 시가총액 변환."""
        from src.data.krx_provider import _to_allstock_marketcap
        df = pd.DataFrame({
            "ISU_SRT_CD": ["005930", "000660"],
            "TDD_CLSPRC": ["70000", "80000"],
            "MKTCAP": ["4000000000", "3000000000"],
            "ACC_TRDVOL": ["10000", "20000"],
            "ACC_TRDVAL": ["700000000", "1600000000"],
            "LIST_SHRS": ["5000000", "4000000"],
        })
        result = _to_allstock_marketcap(df)
        assert len(result) == 2
        assert "시가총액" in result.columns
        assert "종가" in result.columns

    def test_to_index_ohlcv(self):
        """지수 OHLCV 변환."""
        from src.data.krx_provider import _to_index_ohlcv
        df = pd.DataFrame({
            "BAS_DD": ["20260301"],
            "OPNPRC_IDX": ["2600.5"],
            "HGPRC_IDX": ["2650.3"],
            "LWPRC_IDX": ["2580.1"],
            "CLSPRC_IDX": ["2640.2"],
            "ACC_TRDVOL": ["500000000"],
        })
        result = _to_index_ohlcv(df)
        assert len(result) == 1
        assert "시가" in result.columns
        # 지수는 float
        assert isinstance(result.iloc[0]["시가"], float)
        assert result.iloc[0]["시가"] == 2600.5

    def test_to_etf_ohlcv(self):
        """ETF OHLCV 변환."""
        from src.data.krx_provider import _to_etf_ohlcv
        df = pd.DataFrame({
            "BAS_DD": ["20260301"],
            "TDD_OPNPRC": ["35000"],
            "TDD_HGPRC": ["36000"],
            "TDD_LWPRC": ["34000"],
            "TDD_CLSPRC": ["35500"],
            "ACC_TRDVOL": ["1000000"],
            "NAV": ["35200.5"],
        })
        result = _to_etf_ohlcv(df)
        assert len(result) == 1
        assert "NAV" in result.columns

    def test_get_market_ticker_list(self):
        """get_market_ticker_list — 세션 캐시 사용."""
        self._reset_provider()
        import src.data.krx_provider as kp
        from src.data.krx_api import STK_ISU_BASE_INFO, KSQ_ISU_BASE_INFO

        mock_api = MagicMock()
        # KOSPI/KOSDAQ 엔드포인트별 다른 데이터 반환
        def side_effect(endpoint, params):
            if endpoint == STK_ISU_BASE_INFO:
                return [
                    {"ISU_SRT_CD": "005930", "ISU_ABBRV": "삼성전자"},
                    {"ISU_SRT_CD": "000660", "ISU_ABBRV": "SK하이닉스"},
                ]
            else:
                return [
                    {"ISU_SRT_CD": "035420", "ISU_ABBRV": "NAVER"},
                ]
        mock_api.request_all.side_effect = side_effect
        kp._api = mock_api
        kp._initialized = True

        result = kp.get_market_ticker_list("20260301", market="KOSPI")
        assert "005930" in result
        assert "000660" in result
        assert "035420" not in result  # KOSDAQ 종목은 포함 안 됨

    def test_get_market_ticker_name_from_cache(self):
        """get_market_ticker_name — 캐시에서 조회."""
        self._reset_provider()
        import src.data.krx_provider as kp

        # 캐시에 직접 주입
        kp._ticker_cache["005930"] = {"name": "삼성전자", "market": "KOSPI"}
        assert kp.get_market_ticker_name("005930") == "삼성전자"

    def test_get_market_ticker_name_not_found(self):
        """캐시에 없는 종목 → 빈 문자열."""
        self._reset_provider()
        import src.data.krx_provider as kp
        assert kp.get_market_ticker_name("999999") == ""

    def test_session_cache_hit(self):
        """같은 날짜 2회 호출 시 API 1회만."""
        self._reset_provider()
        import src.data.krx_provider as kp

        mock_api = MagicMock()
        mock_api.request_all.return_value = [
            {"ISU_SRT_CD": "005930", "ISU_ABBRV": "삼성전자"},
        ]
        kp._api = mock_api
        kp._initialized = True

        kp.get_market_ticker_list("20260301", market="KOSPI")
        kp.get_market_ticker_list("20260301", market="KOSPI")

        # request_all은 초기화 시 시장 수(KOSPI, KOSDAQ)만큼 호출
        # 두 번째 호출에서는 캐시 히트로 추가 호출 없음
        call_count = mock_api.request_all.call_count
        assert call_count <= 2  # KOSPI + KOSDAQ 각 1회

    def test_get_etf_ticker_list(self):
        """ETF 종목 리스트 조회."""
        self._reset_provider()
        import src.data.krx_provider as kp

        mock_api = MagicMock()
        mock_api.request_all.return_value = [
            {"ISU_SRT_CD": "069500", "ISU_ABBRV": "KODEX 200"},
            {"ISU_SRT_CD": "360750", "ISU_ABBRV": "TIGER 미S&P500"},
        ]
        kp._api = mock_api
        kp._initialized = True

        result = kp.get_etf_ticker_list("20260301")
        assert "069500" in result
        assert len(result) == 2

    def test_collector_conditional_import(self):
        """KRX_API_KEY 없을 때 pykrx fallback으로 collector가 정상 import."""
        self._reset_provider()
        # KRX_API_KEY 없으므로 pykrx로 fallback
        with patch.dict("os.environ", {}, clear=False):
            import importlib
            import src.data.collector
            importlib.reload(src.data.collector)
            # collector가 정상적으로 로드되면 성공
            assert hasattr(src.data.collector, "get_price_data")


class TestProviderFallback:
    """WICS 섹터 pykrx fallback 테스트."""

    def test_sector_classifications_fallback(self):
        """업종 분류는 pykrx fallback 사용."""
        from src.data.krx_provider import get_market_sector_classifications
        mock_df = pd.DataFrame({"업종명": {"005930": "전기전자"}})
        with patch("pykrx.stock.get_market_sector_classifications", return_value=mock_df):
            result = get_market_sector_classifications("20260301", "KOSPI")
            assert isinstance(result, pd.DataFrame)

    def test_wics_index_ticker_list_fallback(self):
        """WICS 업종 코드 목록 pykrx fallback."""
        from src.data.krx_provider import get_index_ticker_list
        with patch("pykrx.stock.get_index_ticker_list", return_value=["G10", "G20"]):
            result = get_index_ticker_list("20260301", "WICS")
            assert "G10" in result
