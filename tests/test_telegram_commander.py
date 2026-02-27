"""TelegramCommander 모듈 테스트.

TelegramCommander의 커맨드 파싱, 핸들러 호출, 보안 검증,
폴링 관리 등을 검증한다. 외부 API(Telegram)는 mock 처리한다.
"""

import os
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from src.alert.telegram_commander import TelegramCommander


# ===================================================================
# 공통 픽스처
# ===================================================================


@pytest.fixture
def mock_notifier():
    """mock TelegramNotifier를 생성한다."""
    notifier = MagicMock()
    notifier.is_configured.return_value = True
    notifier.chat_id = "12345"
    notifier._build_url.return_value = (
        "https://api.telegram.org/botTEST_TOKEN/getUpdates"
    )
    notifier.send_message.return_value = True
    return notifier


@pytest.fixture
def mock_feature_flags():
    """mock FeatureFlags를 생성한다."""
    ff = MagicMock()
    ff.get_summary.return_value = (
        "[Feature Flags]\n"
        "------------------------------\n"
        "  ON  | data_cache: 데이터 캐싱\n"
        "  OFF | global_monitor: 글로벌 시장 모니터"
    )
    ff.get_all_status.return_value = {
        "data_cache": True,
        "global_monitor": False,
    }
    ff.toggle.return_value = True
    ff.is_enabled.return_value = True
    ff.get_config.return_value = {"cache_ttl_hours": 24, "max_cache_size_mb": 500}
    ff.set_config.return_value = True
    return ff


@pytest.fixture
def commander(mock_notifier, mock_feature_flags):
    """TelegramCommander 인스턴스를 생성한다."""
    return TelegramCommander(
        notifier=mock_notifier,
        feature_flags=mock_feature_flags,
        polling_interval=0.1,
    )


def _make_update(update_id, chat_id, text):
    """Telegram getUpdates 응답 형식의 update dict를 생성한다."""
    return {
        "update_id": update_id,
        "message": {
            "text": text,
            "chat": {"id": int(chat_id)},
        },
    }


# ===================================================================
# 1. 초기화 및 기본 커맨드 등록 테스트
# ===================================================================


class TestInitialization:
    """TelegramCommander 초기화 검증."""

    def test_default_commands_registered(self, commander):
        """초기화 시 기본 커맨드 13개가 등록되어야 한다."""
        expected_commands = {
            "/help", "/features", "/toggle",
            "/config", "/status", "/portfolio", "/reload",
            "/signals", "/confirm", "/reject",
            "/short_status", "/short_config",
            "/balance",
        }
        assert set(commander._commands.keys()) == expected_commands

    def test_default_attributes(self, commander, mock_notifier, mock_feature_flags):
        """초기화 시 속성이 올바르게 설정되어야 한다."""
        assert commander.notifier is mock_notifier
        assert commander.feature_flags is mock_feature_flags
        assert commander.polling_interval == 0.1
        assert commander._last_update_id == 0
        assert commander._running is False
        assert commander._thread is None

    def test_default_polling_interval(self, mock_notifier, mock_feature_flags):
        """polling_interval 미지정 시 기본값(2초)을 사용해야 한다."""
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
        )
        assert cmd.polling_interval == TelegramCommander.DEFAULT_POLLING_INTERVAL


# ===================================================================
# 2. /help 커맨드 테스트
# ===================================================================


class TestHelpCommand:
    """/help 커맨드 검증."""

    def test_help_returns_command_list(self, commander):
        """/help는 모든 커맨드 설명을 포함한 텍스트를 반환해야 한다."""
        result = commander._cmd_help("")
        assert "[사용 가능한 커맨드]" in result
        assert "/features" in result
        assert "/toggle" in result
        assert "/config" in result
        assert "/status" in result
        assert "/portfolio" in result
        assert "/reload" in result
        assert "/help" in result

    def test_help_via_process_update(self, commander, mock_notifier):
        """/help 메시지 수신 시 send_message가 호출되어야 한다."""
        update = _make_update(1, "12345", "/help")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()
        sent_text = mock_notifier.send_message.call_args[0][0]
        assert "/features" in sent_text


# ===================================================================
# 3. /features 커맨드 테스트
# ===================================================================


class TestFeaturesCommand:
    """/features 커맨드 검증."""

    def test_features_calls_get_summary(self, commander, mock_feature_flags):
        """/features는 feature_flags.get_summary()를 호출해야 한다."""
        result = commander._cmd_features("")
        mock_feature_flags.get_summary.assert_called_once()
        assert "[Feature Flags]" in result

    def test_features_via_process_update(
        self, commander, mock_notifier, mock_feature_flags
    ):
        """/features 메시지 처리 시 get_summary 결과가 전송되어야 한다."""
        update = _make_update(1, "12345", "/features")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()
        sent_text = mock_notifier.send_message.call_args[0][0]
        assert "[Feature Flags]" in sent_text


# ===================================================================
# 4. /toggle 커맨드 테스트
# ===================================================================


class TestToggleCommand:
    """/toggle 커맨드 검증."""

    def test_toggle_without_args_shows_usage(self, commander, mock_feature_flags):
        """/toggle (인자 없음) 시 사용법과 피처 목록을 반환해야 한다."""
        result = commander._cmd_toggle("")
        assert "사용법" in result
        mock_feature_flags.get_summary.assert_called_once()

    def test_toggle_existing_feature(self, commander, mock_feature_flags):
        """/toggle data_cache는 토글 후 상태를 반환해야 한다."""
        mock_feature_flags.toggle.return_value = True
        mock_feature_flags.is_enabled.return_value = True

        result = commander._cmd_toggle("data_cache")

        mock_feature_flags.toggle.assert_called_once_with("data_cache")
        assert "'data_cache'" in result
        assert "ON" in result

    def test_toggle_to_off(self, commander, mock_feature_flags):
        """토글 후 OFF 상태일 때 OFF가 표시되어야 한다."""
        mock_feature_flags.toggle.return_value = True
        mock_feature_flags.is_enabled.return_value = False

        result = commander._cmd_toggle("data_cache")

        assert "OFF" in result

    def test_toggle_unknown_feature(self, commander, mock_feature_flags):
        """알 수 없는 피처를 토글하면 에러 메시지를 반환해야 한다."""
        mock_feature_flags.toggle.return_value = False

        result = commander._cmd_toggle("unknown_feature")

        assert "알 수 없는 피처" in result

    def test_toggle_via_process_update(self, commander, mock_notifier, mock_feature_flags):
        """/toggle data_cache 메시지 처리 검증."""
        mock_feature_flags.toggle.return_value = True
        mock_feature_flags.is_enabled.return_value = True

        update = _make_update(1, "12345", "/toggle data_cache")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()
        sent_text = mock_notifier.send_message.call_args[0][0]
        assert "'data_cache'" in sent_text


# ===================================================================
# 5. /config 커맨드 테스트
# ===================================================================


class TestConfigCommand:
    """/config 커맨드 검증."""

    def test_config_no_args_shows_usage(self, commander):
        """/config (인자 없음) 시 사용법을 반환해야 한다."""
        result = commander._cmd_config("")
        assert "사용법" in result

    def test_config_show_existing_feature(self, commander, mock_feature_flags):
        """/config data_cache는 해당 피처의 설정을 표시해야 한다."""
        result = commander._cmd_config("data_cache")

        mock_feature_flags.get_config.assert_called_once_with("data_cache")
        assert "[data_cache 설정]" in result
        assert "cache_ttl_hours" in result

    def test_config_show_empty_config(self, commander, mock_feature_flags):
        """설정이 없는 피처는 '설정 없음'을 반환해야 한다."""
        mock_feature_flags.get_config.return_value = {}

        result = commander._cmd_config("data_cache")

        assert "설정 없음" in result

    def test_config_unknown_feature(self, commander, mock_feature_flags):
        """알 수 없는 피처에 대해 에러 메시지를 반환해야 한다."""
        result = commander._cmd_config("nonexistent_feature")

        assert "알 수 없는 피처" in result

    def test_config_set_value(self, commander, mock_feature_flags):
        """/config data_cache key=value로 설정을 변경해야 한다."""
        result = commander._cmd_config("data_cache cache_ttl_hours=48")

        mock_feature_flags.set_config.assert_called_once_with(
            "data_cache", "cache_ttl_hours", 48
        )
        assert "data_cache.cache_ttl_hours" in result
        assert "48" in result

    def test_config_set_value_no_equal_sign(self, commander, mock_feature_flags):
        """key=value 형식이 아니면 사용법을 반환해야 한다."""
        result = commander._cmd_config("data_cache cache_ttl_hours")

        assert "사용법" in result

    def test_config_set_value_failure(self, commander, mock_feature_flags):
        """set_config 실패 시 실패 메시지를 반환해야 한다."""
        mock_feature_flags.set_config.return_value = False

        result = commander._cmd_config("data_cache key=value")

        assert "실패" in result

    def test_config_set_boolean_value(self, commander, mock_feature_flags):
        """boolean 값을 설정할 수 있어야 한다."""
        commander._cmd_config("data_cache include_global=true")

        mock_feature_flags.set_config.assert_called_once_with(
            "data_cache", "include_global", True
        )

    def test_config_set_float_value(self, commander, mock_feature_flags):
        """float 값을 설정할 수 있어야 한다."""
        commander._cmd_config("data_cache threshold=3.14")

        mock_feature_flags.set_config.assert_called_once_with(
            "data_cache", "threshold", 3.14
        )

    def test_config_set_string_value(self, commander, mock_feature_flags):
        """일반 문자열 값을 설정할 수 있어야 한다."""
        commander._cmd_config("data_cache mode=aggressive")

        mock_feature_flags.set_config.assert_called_once_with(
            "data_cache", "mode", "aggressive"
        )

    def test_config_via_process_update_show(
        self, commander, mock_notifier, mock_feature_flags
    ):
        """/config data_cache 메시지 처리 시 설정 조회 결과가 전송되어야 한다."""
        update = _make_update(1, "12345", "/config data_cache")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()
        sent_text = mock_notifier.send_message.call_args[0][0]
        assert "data_cache" in sent_text

    def test_config_via_process_update_set(
        self, commander, mock_notifier, mock_feature_flags
    ):
        """/config data_cache key=val 메시지 처리 시 설정 변경이 되어야 한다."""
        update = _make_update(1, "12345", "/config data_cache cache_ttl_hours=12")
        commander._process_update(update)

        mock_feature_flags.set_config.assert_called_once_with(
            "data_cache", "cache_ttl_hours", 12
        )


# ===================================================================
# 6. /status 커맨드 테스트
# ===================================================================


class TestStatusCommand:
    """/status 커맨드 검증."""

    def test_status_returns_summary(self, commander, mock_feature_flags):
        """/status는 feature_flags.get_summary()를 반환해야 한다."""
        result = commander._cmd_status("")
        mock_feature_flags.get_summary.assert_called_once()
        assert "[Feature Flags]" in result

    def test_status_via_process_update(
        self, commander, mock_notifier, mock_feature_flags
    ):
        """/status 메시지 처리 시 요약 정보가 전송되어야 한다."""
        update = _make_update(1, "12345", "/status")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()


# ===================================================================
# 7. /reload 커맨드 테스트
# ===================================================================


class TestReloadCommand:
    """/reload 커맨드 검증."""

    def test_reload_calls_feature_flags_reload(self, commander, mock_feature_flags):
        """/reload는 feature_flags.reload()를 호출해야 한다."""
        result = commander._cmd_reload("")

        mock_feature_flags.reload.assert_called_once()
        assert "리로드 완료" in result

    def test_reload_shows_summary_after(self, commander, mock_feature_flags):
        """/reload 후 get_summary()의 결과도 포함되어야 한다."""
        result = commander._cmd_reload("")

        mock_feature_flags.get_summary.assert_called_once()
        assert "[Feature Flags]" in result

    def test_reload_via_process_update(
        self, commander, mock_notifier, mock_feature_flags
    ):
        """/reload 메시지 처리 시 reload 호출 및 응답 전송이 되어야 한다."""
        update = _make_update(1, "12345", "/reload")
        commander._process_update(update)

        mock_feature_flags.reload.assert_called_once()
        mock_notifier.send_message.assert_called_once()


# ===================================================================
# 8. 알 수 없는 커맨드 처리 테스트
# ===================================================================


class TestUnknownCommand:
    """알 수 없는 커맨드 처리 검증."""

    def test_unknown_command_sends_error(self, commander, mock_notifier):
        """등록되지 않은 커맨드 수신 시 안내 메시지를 발송해야 한다."""
        update = _make_update(1, "12345", "/unknown_cmd")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()
        sent_text = mock_notifier.send_message.call_args[0][0]
        assert "알 수 없는 커맨드" in sent_text
        assert "/help" in sent_text

    def test_unknown_command_includes_cmd_name(self, commander, mock_notifier):
        """에러 메시지에 실제 입력한 커맨드명이 포함되어야 한다."""
        update = _make_update(1, "12345", "/foobar")
        commander._process_update(update)

        sent_text = mock_notifier.send_message.call_args[0][0]
        assert "/foobar" in sent_text

    def test_non_command_text_ignored(self, commander, mock_notifier):
        """'/'로 시작하지 않는 일반 텍스트는 무시해야 한다."""
        update = _make_update(1, "12345", "안녕하세요")
        commander._process_update(update)

        mock_notifier.send_message.assert_not_called()


# ===================================================================
# 9. 보안: 미등록 chat_id 차단 테스트
# ===================================================================


class TestSecurity:
    """미등록 chat_id 보안 검증."""

    def test_unregistered_chat_id_ignored(self, commander, mock_notifier):
        """등록되지 않은 chat_id에서의 메시지는 무시되어야 한다."""
        update = _make_update(1, "99999", "/help")
        commander._process_update(update)

        mock_notifier.send_message.assert_not_called()

    def test_unregistered_chat_id_toggle_blocked(self, commander, mock_notifier, mock_feature_flags):
        """미등록 chat_id의 /toggle 시도는 차단되어야 한다."""
        update = _make_update(1, "99999", "/toggle data_cache")
        commander._process_update(update)

        mock_notifier.send_message.assert_not_called()
        mock_feature_flags.toggle.assert_not_called()

    def test_empty_chat_id_ignored(self, commander, mock_notifier):
        """chat_id가 빈 update는 무시되어야 한다."""
        update = {
            "update_id": 1,
            "message": {
                "text": "/help",
                "chat": {},
            },
        }
        commander._process_update(update)

        mock_notifier.send_message.assert_not_called()

    def test_registered_chat_id_accepted(self, commander, mock_notifier):
        """등록된 chat_id에서의 메시지는 정상 처리되어야 한다."""
        update = _make_update(1, "12345", "/help")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()


# ===================================================================
# 10. register_command() 외부 커맨드 등록 테스트
# ===================================================================


class TestRegisterCommand:
    """register_command()로 외부 커맨드 등록 검증."""

    def test_register_adds_command(self, commander):
        """register_command()로 새 커맨드가 등록되어야 한다."""
        handler = MagicMock(return_value="커스텀 응답")
        commander.register_command("/custom", handler)

        assert "/custom" in commander._commands

    def test_registered_command_invoked(self, commander, mock_notifier):
        """등록된 외부 커맨드가 메시지 수신 시 호출되어야 한다."""
        handler = MagicMock(return_value="커스텀 응답입니다.")
        commander.register_command("/custom", handler)

        update = _make_update(1, "12345", "/custom some_args")
        commander._process_update(update)

        handler.assert_called_once_with("some_args")
        mock_notifier.send_message.assert_called_once()
        sent_text = mock_notifier.send_message.call_args[0][0]
        assert sent_text == "커스텀 응답입니다."

    def test_registered_command_no_args(self, commander, mock_notifier):
        """인자 없이 외부 커맨드를 호출하면 빈 문자열이 args로 전달되어야 한다."""
        handler = MagicMock(return_value="OK")
        commander.register_command("/ping", handler)

        update = _make_update(1, "12345", "/ping")
        commander._process_update(update)

        handler.assert_called_once_with("")

    def test_register_overrides_existing(self, commander):
        """기존 커맨드를 덮어쓸 수 있어야 한다."""
        new_handler = MagicMock(return_value="새로운 도움말")
        commander.register_command("/help", new_handler)

        result = commander._commands["/help"]("")
        assert result == "새로운 도움말"

    def test_registered_command_returns_none(self, commander, mock_notifier):
        """핸들러가 None을 반환하면 send_message가 호출되지 않아야 한다."""
        handler = MagicMock(return_value=None)
        commander.register_command("/silent", handler)

        update = _make_update(1, "12345", "/silent")
        commander._process_update(update)

        handler.assert_called_once()
        mock_notifier.send_message.assert_not_called()

    def test_registered_command_raises_exception(self, commander, mock_notifier):
        """핸들러가 예외를 발생시키면 오류 메시지가 전송되어야 한다."""
        handler = MagicMock(side_effect=RuntimeError("테스트 오류"))
        commander.register_command("/error", handler)

        update = _make_update(1, "12345", "/error")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()
        sent_text = mock_notifier.send_message.call_args[0][0]
        assert "오류" in sent_text


# ===================================================================
# 11. _parse_value() 타입 추론 테스트
# ===================================================================


class TestParseValue:
    """_parse_value() 정적 메서드의 타입 추론 검증."""

    @pytest.mark.parametrize("input_str,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("yes", True),
        ("Yes", True),
        ("YES", True),
    ])
    def test_parse_true_values(self, input_str, expected):
        """true/yes 계열 문자열은 True로 변환되어야 한다."""
        assert TelegramCommander._parse_value(input_str) is expected

    @pytest.mark.parametrize("input_str,expected", [
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("no", False),
        ("No", False),
        ("NO", False),
    ])
    def test_parse_false_values(self, input_str, expected):
        """false/no 계열 문자열은 False로 변환되어야 한다."""
        assert TelegramCommander._parse_value(input_str) is expected

    @pytest.mark.parametrize("input_str,expected", [
        ("0", 0),
        ("1", 1),
        ("42", 42),
        ("-10", -10),
        ("1000", 1000),
    ])
    def test_parse_int_values(self, input_str, expected):
        """정수 문자열은 int로 변환되어야 한다."""
        result = TelegramCommander._parse_value(input_str)
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize("input_str,expected", [
        ("3.14", 3.14),
        ("-0.5", -0.5),
        ("100.0", 100.0),
        ("0.001", 0.001),
    ])
    def test_parse_float_values(self, input_str, expected):
        """소수점이 있는 숫자 문자열은 float로 변환되어야 한다."""
        result = TelegramCommander._parse_value(input_str)
        assert result == pytest.approx(expected)
        assert isinstance(result, float)

    @pytest.mark.parametrize("input_str", [
        "hello",
        "aggressive",
        "some_string",
        "",
        "not_a_bool",
        "123abc",
    ])
    def test_parse_string_values(self, input_str):
        """숫자나 bool이 아닌 문자열은 그대로 문자열로 반환되어야 한다."""
        result = TelegramCommander._parse_value(input_str)
        assert result == input_str
        assert isinstance(result, str)


# ===================================================================
# getUpdates (폴링) 테스트
# ===================================================================


class TestGetUpdates:
    """_get_updates() 메서드 검증."""

    @patch("src.alert.telegram_commander.requests.get")
    def test_get_updates_success(self, mock_get, commander):
        """getUpdates 성공 시 result 리스트를 반환해야 한다."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "result": [_make_update(100, "12345", "/help")],
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        updates = commander._get_updates()

        assert len(updates) == 1
        assert updates[0]["update_id"] == 100
        mock_get.assert_called_once()

    @patch("src.alert.telegram_commander.requests.get")
    def test_get_updates_empty(self, mock_get, commander):
        """업데이트가 없으면 빈 리스트를 반환해야 한다."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        updates = commander._get_updates()

        assert updates == []

    @patch("src.alert.telegram_commander.requests.get")
    def test_get_updates_not_ok(self, mock_get, commander):
        """API 응답의 ok가 False이면 빈 리스트를 반환해야 한다."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        updates = commander._get_updates()

        assert updates == []

    @patch("src.alert.telegram_commander.requests.get")
    def test_get_updates_request_exception(self, mock_get, commander):
        """요청 실패 시 빈 리스트를 반환해야 한다 (에러 없이)."""
        mock_get.side_effect = requests.RequestException("Connection error")

        updates = commander._get_updates()

        assert updates == []

    @patch("src.alert.telegram_commander.requests.get")
    def test_get_updates_json_error(self, mock_get, commander):
        """JSON 파싱 실패 시 빈 리스트를 반환해야 한다."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        updates = commander._get_updates()

        assert updates == []

    def test_get_updates_not_configured(self, mock_feature_flags):
        """notifier가 미설정이면 빈 리스트를 반환해야 한다."""
        notifier = MagicMock()
        notifier.is_configured.return_value = False

        cmd = TelegramCommander(
            notifier=notifier,
            feature_flags=mock_feature_flags,
        )
        updates = cmd._get_updates()

        assert updates == []

    @patch("src.alert.telegram_commander.requests.get")
    def test_get_updates_uses_correct_offset(self, mock_get, commander):
        """offset은 _last_update_id + 1 이어야 한다."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        commander._last_update_id = 50
        commander._get_updates()

        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["offset"] == 51


# ===================================================================
# _process_update 세부 동작 테스트
# ===================================================================


class TestProcessUpdate:
    """_process_update() 세부 동작 검증."""

    def test_update_id_tracking(self, commander):
        """_last_update_id가 처리한 update의 ID로 갱신되어야 한다."""
        update = _make_update(42, "12345", "/help")
        commander._process_update(update)

        assert commander._last_update_id == 42

    def test_update_id_tracking_higher(self, commander):
        """더 높은 update_id만 갱신되어야 한다."""
        commander._last_update_id = 100
        update = _make_update(50, "12345", "/help")
        commander._process_update(update)

        assert commander._last_update_id == 100

    def test_bot_mention_stripped(self, commander, mock_notifier):
        """@botname이 포함된 커맨드도 정상 처리되어야 한다."""
        update = _make_update(1, "12345", "/help@my_quant_bot")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()
        sent_text = mock_notifier.send_message.call_args[0][0]
        assert "[사용 가능한 커맨드]" in sent_text

    def test_case_insensitive_command(self, commander, mock_notifier):
        """대문자 커맨드도 소문자로 변환하여 처리되어야 한다."""
        update = _make_update(1, "12345", "/HELP")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()

    def test_send_message_called_with_empty_parse_mode(self, commander, mock_notifier):
        """커맨드 응답 시 parse_mode=''로 전송해야 한다."""
        update = _make_update(1, "12345", "/help")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()
        call_kwargs = mock_notifier.send_message.call_args
        assert call_kwargs[1]["parse_mode"] == ""

    def test_empty_message_text(self, commander, mock_notifier):
        """빈 텍스트 메시지는 무시해야 한다."""
        update = {
            "update_id": 1,
            "message": {
                "text": "",
                "chat": {"id": 12345},
            },
        }
        commander._process_update(update)

        mock_notifier.send_message.assert_not_called()

    def test_no_message_in_update(self, commander, mock_notifier):
        """message 필드가 없는 update는 에러 없이 무시해야 한다."""
        update = {"update_id": 1}
        commander._process_update(update)

        mock_notifier.send_message.assert_not_called()


# ===================================================================
# /portfolio 커맨드 테스트
# ===================================================================


class TestPortfolioCommand:
    """/portfolio 커맨드 검증."""

    def test_portfolio_default_response(self, commander):
        """/portfolio는 기본 안내 메시지를 반환해야 한다."""
        result = commander._cmd_portfolio("")
        assert "포트폴리오" in result

    def test_portfolio_via_process_update(self, commander, mock_notifier):
        """/portfolio 메시지 처리 시 응답이 전송되어야 한다."""
        update = _make_update(1, "12345", "/portfolio")
        commander._process_update(update)

        mock_notifier.send_message.assert_called_once()


# ===================================================================
# 폴링 스레드 관리 테스트
# ===================================================================


class TestPollingManagement:
    """폴링 스레드 start/stop 검증."""

    def test_start_polling_sets_running(self, commander):
        """start_polling()은 _running을 True로 설정해야 한다."""
        with patch.object(commander, "_polling_loop"):
            commander.start_polling()
            assert commander._running is True
            commander.stop_polling()

    def test_start_polling_creates_thread(self, commander):
        """start_polling()은 데몬 스레드를 생성해야 한다."""
        with patch.object(commander, "_polling_loop"):
            commander.start_polling()
            assert commander._thread is not None
            assert commander._thread.daemon is True
            assert commander._thread.name == "telegram-commander"
            commander.stop_polling()

    def test_start_polling_twice_noop(self, commander):
        """이미 실행 중이면 두 번째 start_polling()은 무시해야 한다."""
        with patch.object(commander, "_polling_loop"):
            commander.start_polling()
            first_thread = commander._thread
            commander.start_polling()
            assert commander._thread is first_thread
            commander.stop_polling()

    def test_start_polling_not_configured(self, mock_feature_flags):
        """notifier 미설정 시 폴링을 시작하지 않아야 한다."""
        notifier = MagicMock()
        notifier.is_configured.return_value = False

        cmd = TelegramCommander(
            notifier=notifier,
            feature_flags=mock_feature_flags,
        )
        cmd.start_polling()

        assert cmd._running is False
        assert cmd._thread is None

    def test_stop_polling_sets_running_false(self, commander):
        """stop_polling()은 _running을 False로 설정해야 한다."""
        commander._running = True
        commander.stop_polling()
        assert commander._running is False


# ===================================================================
# 다수 update 순차 처리 테스트
# ===================================================================


class TestMultipleUpdates:
    """여러 update를 순차 처리하는 시나리오 검증."""

    def test_multiple_updates_processed_sequentially(
        self, commander, mock_notifier, mock_feature_flags
    ):
        """여러 update가 순차적으로 처리되어야 한다."""
        updates = [
            _make_update(1, "12345", "/help"),
            _make_update(2, "12345", "/status"),
            _make_update(3, "12345", "/features"),
        ]

        for update in updates:
            commander._process_update(update)

        assert mock_notifier.send_message.call_count == 3
        assert commander._last_update_id == 3

    def test_mixed_valid_and_invalid_chat_ids(
        self, commander, mock_notifier
    ):
        """유효/무효 chat_id가 섞여 있을 때 유효한 것만 처리되어야 한다."""
        updates = [
            _make_update(1, "12345", "/help"),    # 유효
            _make_update(2, "99999", "/help"),    # 무효
            _make_update(3, "12345", "/status"),  # 유효
        ]

        for update in updates:
            commander._process_update(update)

        assert mock_notifier.send_message.call_count == 2
        assert commander._last_update_id == 3


# ===================================================================
# 단기 트레이딩 커맨드: 초기화 테스트
# ===================================================================


class TestShortTermInit:
    """short_term_trader/short_term_risk 주입 검증."""

    def test_init_with_short_term_modules(self, mock_notifier, mock_feature_flags):
        """short_term_trader, short_term_risk가 주입되면 속성에 저장되어야 한다."""
        trader = MagicMock()
        risk = MagicMock()
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
            short_term_risk=risk,
        )
        assert cmd._short_term_trader is trader
        assert cmd._short_term_risk is risk

    def test_init_without_short_term_modules(self, mock_notifier, mock_feature_flags):
        """short_term_trader, short_term_risk 미지정 시 None이어야 한다."""
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
        )
        assert cmd._short_term_trader is None
        assert cmd._short_term_risk is None


# ===================================================================
# /signals 커맨드 테스트
# ===================================================================


class TestSignalsCommand:
    """/signals 커맨드 검증."""

    def test_signals_no_trader(self, commander):
        """trader가 None이면 비활성화 메시지를 반환해야 한다."""
        result = commander._cmd_signals("")
        assert "비활성화" in result

    def test_signals_empty_list(self, mock_notifier, mock_feature_flags):
        """대기 시그널이 없으면 안내 메시지를 반환해야 한다."""
        trader = MagicMock()
        trader.get_pending_signals.return_value = []
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
        )
        result = cmd._cmd_signals("")
        assert "대기 중인 시그널이 없습니다" in result

    def test_signals_with_pending(self, mock_notifier, mock_feature_flags):
        """대기 시그널이 있으면 목록을 반환해야 한다."""
        trader = MagicMock()
        trader.get_pending_signals.return_value = [
            {
                "id": "SIG001",
                "side": "BUY",
                "ticker": "005930",
                "strategy": "momentum",
                "confidence": 0.85,
                "expires_at": "2026-02-22 15:00",
            },
            {
                "id": "SIG002",
                "side": "SELL",
                "ticker": "000660",
                "strategy": "mean_revert",
                "confidence": 0.72,
            },
        ]
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
        )
        result = cmd._cmd_signals("")
        assert "[대기 시그널: 2개]" in result
        assert "SIG001" in result
        assert "BUY" in result
        assert "005930" in result
        assert "85%" in result
        assert "SIG002" in result
        assert "SELL" in result
        assert "N/A" in result  # SIG002에 expires_at 없음
        assert "/confirm" in result
        assert "/reject" in result

    def test_signals_via_process_update(self, mock_notifier, mock_feature_flags):
        """/signals 메시지 처리 시 응답이 전송되어야 한다."""
        trader = MagicMock()
        trader.get_pending_signals.return_value = []
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
        )
        update = _make_update(1, "12345", "/signals")
        cmd._process_update(update)
        mock_notifier.send_message.assert_called_once()


# ===================================================================
# /confirm 커맨드 테스트
# ===================================================================


class TestConfirmCommand:
    """/confirm 커맨드 검증."""

    def test_confirm_no_args(self, commander):
        """인자 없으면 사용법을 반환해야 한다."""
        result = commander._cmd_confirm("")
        assert "사용법" in result

    def test_confirm_no_trader(self, commander):
        """trader가 None이면 비활성화 메시지를 반환해야 한다."""
        result = commander._cmd_confirm("SIG001")
        assert "비활성화" in result

    def test_confirm_success(self, mock_notifier, mock_feature_flags):
        """승인 성공 시 성공 메시지를 반환해야 한다."""
        trader = MagicMock()
        trader.confirm_signal.return_value = (True, "SIG001 승인 완료")
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
        )
        result = cmd._cmd_confirm("SIG001")
        trader.confirm_signal.assert_called_once_with("SIG001")
        assert "승인 완료" in result

    def test_confirm_via_process_update(self, mock_notifier, mock_feature_flags):
        """/confirm SIG001 메시지 처리 검증."""
        trader = MagicMock()
        trader.confirm_signal.return_value = (True, "SIG001 승인됨")
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
        )
        update = _make_update(1, "12345", "/confirm SIG001")
        cmd._process_update(update)
        trader.confirm_signal.assert_called_once_with("SIG001")
        mock_notifier.send_message.assert_called_once()


# ===================================================================
# /reject 커맨드 테스트
# ===================================================================


class TestRejectCommand:
    """/reject 커맨드 검증."""

    def test_reject_no_args(self, commander):
        """인자 없으면 사용법을 반환해야 한다."""
        result = commander._cmd_reject("")
        assert "사용법" in result

    def test_reject_no_trader(self, commander):
        """trader가 None이면 비활성화 메시지를 반환해야 한다."""
        result = commander._cmd_reject("SIG001")
        assert "비활성화" in result

    def test_reject_success(self, mock_notifier, mock_feature_flags):
        """거절 성공 시 메시지를 반환해야 한다."""
        trader = MagicMock()
        trader.reject_signal.return_value = (True, "SIG001 거절 완료")
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
        )
        result = cmd._cmd_reject("SIG001")
        trader.reject_signal.assert_called_once_with("SIG001")
        assert "거절 완료" in result

    def test_reject_via_process_update(self, mock_notifier, mock_feature_flags):
        """/reject SIG002 메시지 처리 검증."""
        trader = MagicMock()
        trader.reject_signal.return_value = (True, "SIG002 거절됨")
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
        )
        update = _make_update(1, "12345", "/reject SIG002")
        cmd._process_update(update)
        trader.reject_signal.assert_called_once_with("SIG002")
        mock_notifier.send_message.assert_called_once()


# ===================================================================
# /short_status 커맨드 테스트
# ===================================================================


class TestShortStatusCommand:
    """/short_status 커맨드 검증."""

    def test_short_status_no_trader(self, commander):
        """trader가 None이면 비활성화 메시지를 반환해야 한다."""
        result = commander._cmd_short_status("")
        assert "비활성화" in result

    def test_short_status_with_trader(self, mock_notifier, mock_feature_flags):
        """trader가 있으면 현황을 반환해야 한다."""
        trader = MagicMock()
        trader.get_active_signals.return_value = [
            {"id": "SIG001", "state": "active"},
        ]
        trader.get_all_signals.return_value = [
            {"id": "SIG001", "state": "active"},
            {"id": "SIG002", "state": "pending"},
            {"id": "SIG003", "state": "completed"},
        ]
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
        )
        result = cmd._cmd_short_status("")
        assert "[단기 트레이딩 현황]" in result
        assert "전체 시그널: 3개" in result
        assert "활성 시그널: 1개" in result
        assert "active: 1개" in result
        assert "pending: 1개" in result
        assert "completed: 1개" in result

    def test_short_status_with_risk(self, mock_notifier, mock_feature_flags):
        """risk manager가 있으면 리스크 요약도 포함되어야 한다."""
        trader = MagicMock()
        trader.get_active_signals.return_value = []
        trader.get_all_signals.return_value = []
        risk = MagicMock()
        risk.get_daily_summary.return_value = {
            "daily_pnl_pct": 0.0125,
            "trade_count": 3,
            "can_trade": True,
        }
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
            short_term_risk=risk,
        )
        result = cmd._cmd_short_status("")
        assert "[일일 리스크]" in result
        assert "1.25%" in result
        assert "거래 횟수: 3" in result
        assert "거래 가능: 예" in result

    def test_short_status_risk_cannot_trade(self, mock_notifier, mock_feature_flags):
        """거래 불가 상태일 때 '아니오'가 표시되어야 한다."""
        trader = MagicMock()
        trader.get_active_signals.return_value = []
        trader.get_all_signals.return_value = []
        risk = MagicMock()
        risk.get_daily_summary.return_value = {
            "daily_pnl_pct": -0.05,
            "trade_count": 10,
            "can_trade": False,
        }
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
            short_term_risk=risk,
        )
        result = cmd._cmd_short_status("")
        assert "거래 가능: 아니오" in result

    def test_short_status_via_process_update(self, mock_notifier, mock_feature_flags):
        """/short_status 메시지 처리 시 응답이 전송되어야 한다."""
        trader = MagicMock()
        trader.get_active_signals.return_value = []
        trader.get_all_signals.return_value = []
        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
            short_term_trader=trader,
        )
        update = _make_update(1, "12345", "/short_status")
        cmd._process_update(update)
        mock_notifier.send_message.assert_called_once()


# ===================================================================
# /short_config 커맨드 테스트
# ===================================================================


class TestShortConfigCommand:
    """/short_config 커맨드 검증."""

    def test_short_config_delegates_to_config(self, commander, mock_feature_flags):
        """/short_config는 /config short_term_trading으로 위임해야 한다."""
        mock_feature_flags.get_all_status.return_value = {
            "data_cache": True,
            "short_term_trading": True,
        }
        mock_feature_flags.get_config.return_value = {"max_position": 5}

        result = commander._cmd_short_config("")
        mock_feature_flags.get_config.assert_called_with("short_term_trading")
        assert "short_term_trading" in result

    def test_short_config_set_value(self, commander, mock_feature_flags):
        """/short_config key=val로 설정 변경이 위임되어야 한다."""
        mock_feature_flags.get_all_status.return_value = {
            "data_cache": True,
            "short_term_trading": True,
        }

        commander._cmd_short_config("max_position=10")
        mock_feature_flags.set_config.assert_called_once_with(
            "short_term_trading", "max_position", 10
        )

    def test_short_config_via_process_update(
        self, mock_notifier, mock_feature_flags
    ):
        """/short_config 메시지 처리 검증."""
        mock_feature_flags.get_all_status.return_value = {
            "short_term_trading": True,
        }
        mock_feature_flags.get_config.return_value = {"max_position": 5}

        cmd = TelegramCommander(
            notifier=mock_notifier,
            feature_flags=mock_feature_flags,
        )
        update = _make_update(1, "12345", "/short_config")
        cmd._process_update(update)
        mock_notifier.send_message.assert_called_once()


# ===================================================================
# /help 업데이트 검증
# ===================================================================


class TestHelpIncludesNewCommands:
    """/help에 새 커맨드가 포함되는지 검증."""

    def test_help_includes_signals(self, commander):
        """/help 출력에 /signals가 포함되어야 한다."""
        result = commander._cmd_help("")
        assert "/signals" in result

    def test_help_includes_confirm(self, commander):
        """/help 출력에 /confirm가 포함되어야 한다."""
        result = commander._cmd_help("")
        assert "/confirm" in result

    def test_help_includes_reject(self, commander):
        """/help 출력에 /reject가 포함되어야 한다."""
        result = commander._cmd_help("")
        assert "/reject" in result

    def test_help_includes_short_status(self, commander):
        """/help 출력에 /short_status가 포함되어야 한다."""
        result = commander._cmd_help("")
        assert "/short_status" in result

    def test_help_includes_short_config(self, commander):
        """/help 출력에 /short_config가 포함되어야 한다."""
        result = commander._cmd_help("")
        assert "/short_config" in result
