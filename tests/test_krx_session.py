"""pykrx 세션 쿠키 주입 테스트.

모든 HTTP 호출은 mock 처리한다.
"""

import time
from unittest.mock import MagicMock, patch

import pytest


def _reset():
    """krx_session 모듈 상태 초기화."""
    import src.data.krx_session as ks
    ks._session = None
    ks._patched = False
    ks._last_login = 0.0
    ks._login_id = ""
    ks._login_pw = ""


class TestLogin:
    """KRX 로그인 테스트."""

    def test_login_success(self):
        """CD001 응답 시 True, 세션 생성."""
        _reset()
        import src.data.krx_session as ks

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"_error_code": "CD001"}
        mock_session.post.return_value = mock_resp
        mock_session.get.return_value = MagicMock()

        with patch("src.data.krx_session.requests.Session", return_value=mock_session):
            with patch("src.data.krx_session._patch_webio"):
                result = ks.login("test_id", "test_pw")

        assert result is True
        assert ks._last_login > 0
        assert ks._session is mock_session

    def test_login_duplicate_session(self):
        """CD011 → skipDup=Y 재전송 → CD001 성공."""
        _reset()
        import src.data.krx_session as ks

        mock_session = MagicMock()
        # 첫 번째 POST: CD011, 두 번째 POST: CD001
        resp_dup = MagicMock()
        resp_dup.json.return_value = {"_error_code": "CD011"}
        resp_ok = MagicMock()
        resp_ok.json.return_value = {"_error_code": "CD001"}
        mock_session.post.side_effect = [resp_dup, resp_ok]
        mock_session.get.return_value = MagicMock()

        with patch("src.data.krx_session.requests.Session", return_value=mock_session):
            with patch("src.data.krx_session._patch_webio"):
                result = ks.login("test_id", "test_pw")

        assert result is True
        # post는 2회 호출 (로그인 + skipDup 재전송)
        assert mock_session.post.call_count == 2
        # 두 번째 호출에 skipDup=Y 포함
        second_call = mock_session.post.call_args_list[1]
        assert second_call[1]["data"]["skipDup"] == "Y"

    def test_login_failure(self):
        """잘못된 error_code → False."""
        _reset()
        import src.data.krx_session as ks

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"_error_code": "CD999"}
        mock_session.post.return_value = mock_resp
        mock_session.get.return_value = MagicMock()

        with patch("src.data.krx_session.requests.Session", return_value=mock_session):
            with patch("src.data.krx_session._patch_webio"):
                result = ks.login("bad_id", "bad_pw")

        assert result is False

    def test_login_network_error(self):
        """네트워크 에러 시 False."""
        _reset()
        import src.data.krx_session as ks

        mock_session = MagicMock()
        mock_session.get.side_effect = ConnectionError("timeout")

        with patch("src.data.krx_session.requests.Session", return_value=mock_session):
            with patch("src.data.krx_session._patch_webio"):
                result = ks.login("test_id", "test_pw")

        assert result is False


class TestInit:
    """환경변수 기반 초기화 테스트."""

    def test_init_no_env(self):
        """환경변수 없으면 False."""
        _reset()
        import src.data.krx_session as ks

        with patch.dict("os.environ", {}, clear=True):
            result = ks.init()
        assert result is False

    def test_init_with_env(self):
        """환경변수 설정 시 login 호출."""
        _reset()
        import src.data.krx_session as ks

        with patch.dict("os.environ", {"KRX_DATA_ID": "my_id", "KRX_DATA_PW": "my_pw"}):
            with patch.object(ks, "login", return_value=True) as mock_login:
                result = ks.init()

        assert result is True
        mock_login.assert_called_once_with("my_id", "my_pw")


class TestPatchWebio:
    """webio monkey-patch 테스트."""

    def test_patch_replaces_read_methods(self):
        """webio.Post.read / Get.read가 교체됨."""
        _reset()
        import src.data.krx_session as ks
        from pykrx.website.comm import webio

        original_post_read = webio.Post.read
        original_get_read = webio.Get.read

        ks._patch_webio()

        assert webio.Post.read is not original_post_read
        assert webio.Get.read is not original_get_read
        assert ks._patched is True

        # 재호출 시 이중 패치 방지
        patched_post = webio.Post.read
        ks._patch_webio()
        assert webio.Post.read is patched_post  # 동일 함수 유지

    def test_patch_idempotent(self):
        """여러 번 호출해도 1회만 패치."""
        _reset()
        import src.data.krx_session as ks

        ks._patch_webio()
        assert ks._patched is True

        ks._patch_webio()  # 두 번째 호출은 no-op
        assert ks._patched is True


class TestEnsureSession:
    """세션 만료/갱신 테스트."""

    def test_ensure_session_no_session(self):
        """세션 없으면 init 호출."""
        _reset()
        import src.data.krx_session as ks

        with patch.object(ks, "init") as mock_init:
            ks.ensure_session()
        mock_init.assert_called_once()

    def test_ensure_session_expired(self):
        """TTL 초과 시 재로그인."""
        _reset()
        import src.data.krx_session as ks

        ks._session = MagicMock()
        ks._last_login = time.monotonic() - ks._SESSION_TTL - 1
        ks._login_id = "test_id"
        ks._login_pw = "test_pw"

        with patch.object(ks, "login", return_value=True) as mock_login:
            ks.ensure_session()
        mock_login.assert_called_once_with("test_id", "test_pw")

    def test_ensure_session_still_valid(self):
        """TTL 이내면 재로그인 안 함."""
        _reset()
        import src.data.krx_session as ks

        ks._session = MagicMock()
        ks._last_login = time.monotonic()  # 방금 로그인

        with patch.object(ks, "init") as mock_init:
            with patch.object(ks, "login") as mock_login:
                ks.ensure_session()
        mock_init.assert_not_called()
        mock_login.assert_not_called()


class TestIsLoggedIn:
    """로그인 상태 확인 테스트."""

    def test_not_logged_in(self):
        """초기 상태."""
        _reset()
        import src.data.krx_session as ks
        assert ks.is_logged_in() is False

    def test_logged_in(self):
        """로그인 후."""
        _reset()
        import src.data.krx_session as ks
        ks._session = MagicMock()
        ks._last_login = time.monotonic()
        assert ks.is_logged_in() is True
