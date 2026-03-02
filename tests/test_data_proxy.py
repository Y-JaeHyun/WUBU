"""DataProxy fallback 테스트.

pykrx ↔ KRX Open API 자동 fallback 프록시의 동작을 검증한다.
"""

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.data.data_proxy import DataProxy


# ── DataProxy 단위 테스트 ──────────────────────────────────


class TestDataProxy:
    """DataProxy __getattr__ fallback 동작 검증."""

    def _make_backends(self):
        """테스트용 primary/fallback 백엔드 생성."""
        primary = MagicMock()
        fallback = MagicMock()
        return primary, fallback

    def test_primary_success(self):
        """primary 성공 시 fallback 미호출."""
        primary, fallback = self._make_backends()
        primary.get_data.return_value = "primary_result"

        proxy = DataProxy(primary, fallback, "p", "f")
        result = proxy.get_data("arg1")

        assert result == "primary_result"
        primary.get_data.assert_called_once_with("arg1")
        fallback.get_data.assert_not_called()

    def test_primary_fail_fallback_success(self):
        """primary 예외 → fallback 성공."""
        primary, fallback = self._make_backends()
        primary.some_func.side_effect = ConnectionError("timeout")
        fallback.some_func.return_value = "fallback_result"

        proxy = DataProxy(primary, fallback, "p", "f")
        result = proxy.some_func("x", key="y")

        assert result == "fallback_result"
        primary.some_func.assert_called_once_with("x", key="y")
        fallback.some_func.assert_called_once_with("x", key="y")

    def test_both_fail(self):
        """양쪽 모두 실패 → fallback 예외 발생."""
        primary, fallback = self._make_backends()
        primary.bad_func.side_effect = ConnectionError("primary down")
        fallback.bad_func.side_effect = RuntimeError("fallback down")

        proxy = DataProxy(primary, fallback, "p", "f")

        with pytest.raises(RuntimeError, match="fallback down"):
            proxy.bad_func()

    def test_primary_only_method(self):
        """primary에만 있는 메서드 → 직접 호출."""
        primary, fallback = self._make_backends()
        primary.unique_method.return_value = "unique"
        # fallback에는 해당 메서드 없음
        del fallback.unique_method

        proxy = DataProxy(primary, fallback, "p", "f")
        result = proxy.unique_method()

        assert result == "unique"

    def test_fallback_only_method(self):
        """fallback에만 있는 메서드 → 직접 호출."""
        primary, fallback = self._make_backends()
        del primary.special_method
        fallback.special_method.return_value = "special"

        proxy = DataProxy(primary, fallback, "p", "f")
        result = proxy.special_method()

        assert result == "special"

    def test_no_fallback(self):
        """fallback=None → primary만 사용, 실패 시 예외."""
        primary = MagicMock()
        primary.risky.side_effect = ValueError("fail")

        proxy = DataProxy(primary, None, "p", "none")

        with pytest.raises(ValueError, match="fail"):
            proxy.risky()

    def test_attribute_error(self):
        """양쪽 모두 없는 메서드 → AttributeError."""
        primary, fallback = self._make_backends()
        del primary.nonexistent
        del fallback.nonexistent

        proxy = DataProxy(primary, fallback, "p", "f")

        with pytest.raises(AttributeError):
            proxy.nonexistent()

    def test_fallback_count(self):
        """fallback 호출 횟수 추적."""
        primary, fallback = self._make_backends()
        primary.flaky.side_effect = Exception("flaky")
        fallback.flaky.return_value = "ok"

        proxy = DataProxy(primary, fallback, "p", "f")
        assert proxy._fallback_count == 0

        proxy.flaky()
        assert proxy._fallback_count == 1

        proxy.flaky()
        assert proxy._fallback_count == 2

    def test_no_fallback_primary_success(self):
        """fallback=None, primary 성공 → 정상 동작."""
        primary = MagicMock()
        primary.ok_func.return_value = 42

        proxy = DataProxy(primary, None, "p", "none")
        assert proxy.ok_func() == 42


# ── create_stock_api 팩토리 테스트 ─────────────────────────


class TestCreateStockApi:
    """create_stock_api() 팩토리 함수 동작 검증."""

    def test_no_api_key_returns_pykrx(self):
        """KRX_API_KEY 없음 → pykrx 모듈 직접 반환 (프록시 아님)."""
        from src.data.data_proxy import create_stock_api

        with patch("src.data.krx_session.init"):
            with patch("src.data.krx_provider.is_available", return_value=False):
                with patch("src.data.krx_provider._has_api_key", return_value=False):
                    result = create_stock_api()

        # pykrx 모듈이 직접 반환됨 (DataProxy가 아님)
        assert not isinstance(result, DataProxy)

    def test_api_key_no_flag_returns_proxy_pykrx_primary(self):
        """KRX_API_KEY 있고 flag OFF → DataProxy(pykrx primary, krx fallback)."""
        from src.data.data_proxy import create_stock_api

        with patch("src.data.krx_session.init"):
            with patch("src.data.krx_provider.is_available", return_value=False):
                with patch("src.data.krx_provider._has_api_key", return_value=True):
                    result = create_stock_api()

        assert isinstance(result, DataProxy)
        assert result._primary_name == "pykrx"
        assert result._fallback_name == "krx_api"

    def test_api_key_flag_on_returns_proxy_krx_primary(self):
        """KRX_API_KEY 있고 flag ON → DataProxy(krx primary, pykrx fallback)."""
        from src.data.data_proxy import create_stock_api

        with patch("src.data.krx_session.init"):
            with patch("src.data.krx_provider.is_available", return_value=True):
                with patch("src.data.krx_provider._has_api_key", return_value=True):
                    result = create_stock_api()

        assert isinstance(result, DataProxy)
        assert result._primary_name == "krx_api"
        assert result._fallback_name == "pykrx"
