"""매크로 데이터 수집기(src/data/macro_collector.py) 테스트.

MacroCollector 클래스의 기준금리, 국채금리, 달러인덱스, VIX 조회,
매크로 요약 및 리포트 포맷 기능을 검증한다.
외부 API 호출은 mock 처리한다.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.data.macro_collector import MacroCollector


# ===================================================================
# 공통 픽스처
# ===================================================================


@pytest.fixture
def collector():
    """API 키가 설정된 MacroCollector."""
    return MacroCollector(ecos_key="test_ecos", fred_key="test_fred")


@pytest.fixture
def no_key_collector():
    """API 키가 없는 MacroCollector."""
    return MacroCollector(ecos_key="", fred_key="")


# ===================================================================
# 초기화 테스트
# ===================================================================


class TestInit:
    """MacroCollector 초기화 테스트."""

    def test_keys_from_constructor(self, collector):
        """생성자에서 전달한 키가 설정된다."""
        assert collector.ecos_key == "test_ecos"
        assert collector.fred_key == "test_fred"

    @patch.dict("os.environ", {"ECOS_API_KEY": "env_ecos", "FRED_API_KEY": "env_fred"})
    def test_keys_from_env(self):
        """환경변수에서 키를 로드한다."""
        mc = MacroCollector()
        assert mc.ecos_key == "env_ecos"
        assert mc.fred_key == "env_fred"

    def test_no_keys(self, no_key_collector):
        """키 없이 초기화해도 에러가 발생하지 않는다."""
        assert no_key_collector.ecos_key == ""
        assert no_key_collector.fred_key == ""


# ===================================================================
# get_bok_rate() 테스트
# ===================================================================


class TestGetBokRate:
    """get_bok_rate() 한국은행 기준금리 조회 테스트."""

    @patch("src.data.macro_collector.requests.get")
    def test_ecos_success(self, mock_get, collector):
        """ECOS API 성공 시 기준금리를 반환한다."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "StatisticSearch": {
                "row": [{"DATA_VALUE": "3.50"}],
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = collector.get_bok_rate()
        assert result == 3.50

    @patch("src.data.macro_collector.requests.get")
    def test_ecos_failure_tries_fdr(self, mock_get, collector):
        """ECOS 실패 시 FDR fallback을 시도한다."""
        mock_get.side_effect = Exception("ECOS error")

        with patch.object(
            MacroCollector, "_get_bok_rate_fdr", return_value=3.25
        ):
            result = collector.get_bok_rate()

        assert result == 3.25

    def test_no_ecos_key_uses_fdr(self, no_key_collector):
        """ECOS 키가 없으면 FDR을 사용한다."""
        with patch.object(
            MacroCollector, "_get_bok_rate_fdr", return_value=3.00
        ):
            result = no_key_collector.get_bok_rate()
        assert result == 3.00

    def test_all_fail_returns_none(self, no_key_collector):
        """모든 소스 실패 시 None을 반환한다."""
        with patch.object(
            MacroCollector, "_get_bok_rate_fdr", return_value=None
        ):
            result = no_key_collector.get_bok_rate()
        assert result is None


# ===================================================================
# get_us_treasury() 테스트
# ===================================================================


class TestGetUsTreasury:
    """get_us_treasury() 미국 국채 금리 조회 테스트."""

    @patch.object(MacroCollector, "_get_fred_series")
    def test_fred_success_10y(self, mock_fred, collector):
        """FRED 10Y 조회 성공 시 금리를 반환한다."""
        mock_fred.return_value = 4.25
        result = collector.get_us_treasury("10Y")
        assert result == 4.25
        mock_fred.assert_called_with("DGS10")

    @patch.object(MacroCollector, "_get_fred_series")
    def test_fred_success_2y(self, mock_fred, collector):
        """FRED 2Y 조회 성공 시 금리를 반환한다."""
        mock_fred.return_value = 4.50
        result = collector.get_us_treasury("2Y")
        assert result == 4.50
        mock_fred.assert_called_with("DGS2")

    @patch.object(MacroCollector, "_get_fred_series")
    @patch.object(MacroCollector, "_get_treasury_yfinance")
    def test_fred_failure_tries_yfinance(self, mock_yf, mock_fred, collector):
        """FRED 실패 시 yfinance fallback을 시도한다."""
        mock_fred.return_value = None
        mock_yf.return_value = 4.10

        result = collector.get_us_treasury("10Y")
        assert result == 4.10

    def test_no_fred_key_tries_yfinance(self, no_key_collector):
        """FRED 키가 없으면 yfinance를 사용한다."""
        with patch.object(
            MacroCollector, "_get_treasury_yfinance", return_value=4.00
        ):
            result = no_key_collector.get_us_treasury("10Y")
        assert result == 4.00


# ===================================================================
# get_dollar_index() 테스트
# ===================================================================


class TestGetDollarIndex:
    """get_dollar_index() 달러 인덱스 조회 테스트."""

    @patch.object(MacroCollector, "_get_fred_series")
    def test_fred_success(self, mock_fred, collector):
        """FRED 성공 시 달러 인덱스를 반환한다."""
        mock_fred.return_value = 104.5
        result = collector.get_dollar_index()
        assert result == 104.5

    def test_no_key_tries_yfinance(self, no_key_collector):
        """키 없으면 yfinance fallback을 사용한다."""
        with patch.object(
            MacroCollector, "_get_yfinance_price", return_value=104.0
        ):
            result = no_key_collector.get_dollar_index()
        assert result == 104.0


# ===================================================================
# get_vix() / get_usd_krw() 테스트
# ===================================================================


class TestVixAndFx:
    """get_vix(), get_usd_krw() 테스트."""

    @patch.object(MacroCollector, "_get_yfinance_price")
    def test_vix(self, mock_yf, collector):
        """VIX를 조회한다."""
        mock_yf.return_value = 18.5
        result = collector.get_vix()
        assert result == 18.5
        mock_yf.assert_called_with("^VIX")

    @patch.object(MacroCollector, "_get_yfinance_price")
    def test_usd_krw(self, mock_yf, collector):
        """USD/KRW를 조회한다."""
        mock_yf.return_value = 1350.0
        result = collector.get_usd_krw()
        assert result == 1350.0
        mock_yf.assert_called_with("USDKRW=X")


# ===================================================================
# get_macro_summary() 테스트
# ===================================================================


class TestGetMacroSummary:
    """get_macro_summary() 매크로 요약 테스트."""

    @patch.object(MacroCollector, "get_bok_rate", return_value=3.50)
    @patch.object(MacroCollector, "get_us_treasury")
    @patch.object(MacroCollector, "get_dollar_index", return_value=104.5)
    @patch.object(MacroCollector, "get_vix", return_value=18.5)
    @patch.object(MacroCollector, "get_usd_krw", return_value=1350.0)
    def test_full_summary(self, mock_krw, mock_vix, mock_dxy, mock_ust, mock_bok, collector):
        """전체 매크로 요약이 올바르다."""
        mock_ust.side_effect = lambda m: {"10Y": 4.25, "2Y": 4.50}.get(m)

        result = collector.get_macro_summary()

        assert result["bok_rate"] == 3.50
        assert result["us_treasury_10y"] == 4.25
        assert result["us_treasury_2y"] == 4.50
        assert result["yield_spread"] == -0.25
        assert result["dollar_index"] == 104.5
        assert result["vix"] == 18.5
        assert result["usd_krw"] == 1350.0

    @patch.object(MacroCollector, "get_bok_rate", return_value=None)
    @patch.object(MacroCollector, "get_us_treasury", return_value=None)
    @patch.object(MacroCollector, "get_dollar_index", return_value=None)
    @patch.object(MacroCollector, "get_vix", return_value=None)
    @patch.object(MacroCollector, "get_usd_krw", return_value=None)
    def test_empty_summary(self, *mocks):
        """모든 소스 실패 시 빈 요약을 반환한다."""
        mc = MacroCollector(ecos_key="", fred_key="")
        result = mc.get_macro_summary()
        assert result == {}

    @patch.object(MacroCollector, "get_bok_rate", return_value=3.50)
    @patch.object(MacroCollector, "get_us_treasury", return_value=None)
    @patch.object(MacroCollector, "get_dollar_index", return_value=None)
    @patch.object(MacroCollector, "get_vix", return_value=18.0)
    @patch.object(MacroCollector, "get_usd_krw", return_value=None)
    def test_partial_summary(self, *mocks):
        """일부만 성공해도 가능한 지표는 포함한다."""
        mc = MacroCollector(ecos_key="key", fred_key="")
        result = mc.get_macro_summary()
        assert "bok_rate" in result
        assert "vix" in result
        assert "us_treasury_10y" not in result


# ===================================================================
# format_macro_report() 테스트
# ===================================================================


class TestFormatMacroReport:
    """format_macro_report() 매크로 리포트 포맷 테스트."""

    @patch.object(MacroCollector, "get_macro_summary")
    def test_empty_report(self, mock_summary, collector):
        """데이터가 없으면 실패 메시지를 반환한다."""
        mock_summary.return_value = {}
        result = collector.format_macro_report()
        assert "데이터 수집 실패" in result

    @patch.object(MacroCollector, "get_macro_summary")
    def test_full_report(self, mock_summary, collector):
        """전체 데이터가 있으면 포매팅된 리포트를 반환한다."""
        mock_summary.return_value = {
            "bok_rate": 3.50,
            "us_treasury_10y": 4.25,
            "us_treasury_2y": 4.50,
            "yield_spread": -0.25,
            "dollar_index": 104.5,
            "vix": 18.5,
            "usd_krw": 1350.0,
        }
        result = collector.format_macro_report()

        assert "[매크로 리포트]" in result
        assert "3.50%" in result
        assert "4.25%" in result
        assert "역전" in result
        assert "안정" in result

    @patch.object(MacroCollector, "get_macro_summary")
    def test_vix_fear_status(self, mock_summary, collector):
        """VIX > 30이면 '공포' 상태가 표시된다."""
        mock_summary.return_value = {"vix": 35.0}
        result = collector.format_macro_report()
        assert "공포" in result

    @patch.object(MacroCollector, "get_macro_summary")
    def test_vix_caution_status(self, mock_summary, collector):
        """VIX 20~30이면 '경계' 상태가 표시된다."""
        mock_summary.return_value = {"vix": 25.0}
        result = collector.format_macro_report()
        assert "경계" in result

    @patch.object(MacroCollector, "get_macro_summary")
    def test_krw_weak_status(self, mock_summary, collector):
        """USD/KRW > 1400이면 '원화 약세'가 표시된다."""
        mock_summary.return_value = {"usd_krw": 1420.0}
        result = collector.format_macro_report()
        assert "원화 약세" in result

    @patch.object(MacroCollector, "get_macro_summary")
    def test_krw_strong_status(self, mock_summary, collector):
        """USD/KRW < 1300이면 '원화 강세'가 표시된다."""
        mock_summary.return_value = {"usd_krw": 1280.0}
        result = collector.format_macro_report()
        assert "원화 강세" in result


# ===================================================================
# _get_fred_series() 테스트
# ===================================================================


class TestGetFredSeries:
    """_get_fred_series() FRED API 내부 메서드 테스트."""

    @patch("src.data.macro_collector.requests.get")
    def test_success(self, mock_get, collector):
        """성공 시 최신값을 반환한다."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2026-02-26", "value": "4.25"},
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = collector._get_fred_series("DGS10")
        assert result == 4.25

    @patch("src.data.macro_collector.requests.get")
    def test_skip_missing_values(self, mock_get, collector):
        """'.' 값은 건너뛰고 다음 유효 값을 반환한다."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2026-02-26", "value": "."},
                {"date": "2026-02-25", "value": "4.20"},
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = collector._get_fred_series("DGS10")
        assert result == 4.20

    @patch("src.data.macro_collector.requests.get")
    def test_exception_returns_none(self, mock_get, collector):
        """예외 발생 시 None을 반환한다."""
        mock_get.side_effect = Exception("network error")
        result = collector._get_fred_series("DGS10")
        assert result is None


# ===================================================================
# _get_yfinance_price() 테스트
# ===================================================================


class TestGetYfinancePrice:
    """_get_yfinance_price() yfinance 조회 테스트."""

    def test_success(self):
        """yfinance 성공 시 가격을 반환한다."""
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 18.5

        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info

        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = MacroCollector._get_yfinance_price("^VIX")

        assert result == 18.5

    def test_exception_returns_none(self):
        """조회 중 예외 발생 시 None을 반환한다."""
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = Exception("network error")

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = MacroCollector._get_yfinance_price("^VIX")

        assert result is None
