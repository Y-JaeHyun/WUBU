"""알림 시스템 모듈 테스트.

TelegramNotifier, AlertManager, 조건 클래스(Condition)들을 검증한다.
외부 API(텔레그램)는 mock 처리한다.
"""

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.alert.alert_manager import AlertManager
from src.alert.conditions import (
    MddThresholdCondition,
    DailyMoveCondition,
    RebalanceAlertCondition,
)
from src.alert.telegram_bot import TelegramNotifier


# ===================================================================
# TelegramNotifier 테스트
# ===================================================================


class TestTelegramNotifier:
    """TelegramNotifier 기본 동작 검증."""

    def test_telegram_not_configured(self):
        """토큰이 없으면 is_configured()가 False를 반환한다."""
        notifier = TelegramNotifier(bot_token="", chat_id="")

        assert notifier.is_configured() is False, (
            "토큰과 chat_id가 빈 문자열이면 is_configured()는 False여야 합니다."
        )

    def test_telegram_configured(self):
        """토큰과 chat_id가 있으면 is_configured()가 True를 반환한다."""
        notifier = TelegramNotifier(
            bot_token="test_token_12345",
            chat_id="123456789",
        )

        assert notifier.is_configured() is True, (
            "토큰과 chat_id가 설정되면 is_configured()는 True여야 합니다."
        )

    def test_send_message_not_configured(self):
        """미설정 상태에서 send_message는 False를 반환한다."""
        notifier = TelegramNotifier(bot_token="", chat_id="")

        result = notifier.send_message("테스트 메시지")

        assert result is False, (
            "미설정 상태에서 메시지 발송은 False를 반환해야 합니다."
        )

    @patch("src.alert.telegram_bot.requests.post")
    def test_send_message_success(self, mock_post):
        """설정된 상태에서 메시지 발송 성공 시 True를 반환한다."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        notifier = TelegramNotifier(
            bot_token="test_token",
            chat_id="12345",
        )

        result = notifier.send_message("테스트 메시지")

        assert result is True, "발송 성공 시 True를 반환해야 합니다."
        mock_post.assert_called_once()


# ===================================================================
# MddThresholdCondition 테스트
# ===================================================================


class TestMddThresholdCondition:
    """MDD 임계값 조건 검증."""

    def test_mdd_condition_triggered(self):
        """current_mdd가 임계값 미만(더 큰 낙폭)이면 트리거된다."""
        condition = MddThresholdCondition(threshold=-0.10)  # -10%

        # 현재 MDD가 -15% (임계값 -10%보다 심한 낙폭)
        context = {"current_mdd": -0.15}
        result = condition.check(context)

        assert result is True, (
            "MDD -15%는 임계값 -10% 미만이므로 트리거되어야 합니다."
        )

    def test_mdd_condition_not_triggered(self):
        """current_mdd가 임계값 이상(덜한 낙폭)이면 트리거되지 않는다."""
        condition = MddThresholdCondition(threshold=-0.10)  # -10%

        # 현재 MDD가 -5% (임계값 -10%보다 덜한 낙폭)
        context = {"current_mdd": -0.05}
        result = condition.check(context)

        assert result is False, (
            "MDD -5%는 임계값 -10% 미만이 아니므로 트리거되지 않아야 합니다."
        )

    def test_mdd_condition_exact_threshold(self):
        """MDD가 정확히 임계값이면 트리거되지 않는다 (strict less-than)."""
        condition = MddThresholdCondition(threshold=-0.10)

        context = {"current_mdd": -0.10}
        result = condition.check(context)

        # 구현: current_mdd < threshold (strict), -0.10 < -0.10 은 False
        assert result is False, (
            "MDD가 정확히 임계값이면 트리거되지 않아야 합니다 (strict less-than)."
        )


# ===================================================================
# DailyMoveCondition 테스트
# ===================================================================


class TestDailyMoveCondition:
    """일일 변동 조건 검증."""

    def test_daily_move_condition(self):
        """일일 변동이 임계값을 초과하면 트리거된다."""
        condition = DailyMoveCondition(threshold=0.03)  # 3%

        # 보유 종목 중 5% 상승 (임계값 3% 초과)
        context = {"holdings_daily_returns": {"005930": 0.05}}
        result = condition.check(context)

        assert result is True, (
            "일일 변동 5%는 임계값 3%를 초과하므로 트리거되어야 합니다."
        )

    def test_daily_move_condition_negative(self):
        """음수 일일 변동의 절대값이 임계값을 초과하면 트리거된다."""
        condition = DailyMoveCondition(threshold=0.03)

        context = {"holdings_daily_returns": {"005930": -0.04}}
        result = condition.check(context)

        assert result is True, (
            "일일 변동 -4%의 절대값은 임계값 3%를 초과하므로 트리거되어야 합니다."
        )

    def test_daily_move_condition_below_threshold(self):
        """일일 변동이 임계값 미만이면 트리거되지 않는다."""
        condition = DailyMoveCondition(threshold=0.03)

        context = {"holdings_daily_returns": {"005930": 0.01}}
        result = condition.check(context)

        assert result is False, (
            "일일 변동 1%는 임계값 3% 미만이므로 트리거되지 않아야 합니다."
        )


# ===================================================================
# RebalanceAlertCondition 테스트
# ===================================================================


class TestRebalanceAlertCondition:
    """리밸런싱 알림 조건 검증."""

    def test_rebalance_alert_condition(self):
        """리밸런싱 D-day에 트리거된다."""
        condition = RebalanceAlertCondition()

        # days_to_rebalance=0 → 오늘이 리밸런싱 날
        context = {"days_to_rebalance": 0}
        result = condition.check(context)

        assert result is True, (
            "리밸런싱 D-day에 트리거되어야 합니다."
        )

    def test_rebalance_alert_condition_not_triggered(self):
        """리밸런싱까지 충분한 날이 남으면 트리거되지 않는다."""
        condition = RebalanceAlertCondition()  # days_before=3

        # days_to_rebalance=10 → 아직 멀음
        context = {"days_to_rebalance": 10}
        result = condition.check(context)

        assert result is False, (
            "리밸런싱까지 10일 남으면 트리거되지 않아야 합니다."
        )


# ===================================================================
# AlertManager 테스트
# ===================================================================


class TestAlertManager:
    """AlertManager 동작 검증."""

    def test_alert_manager_check_and_alert(self):
        """조건 등록 후 검사 시 트리거된 조건의 알림이 발생한다."""
        mock_notifier = MagicMock()
        mock_notifier.is_configured.return_value = True
        mock_notifier.send_message.return_value = True

        manager = AlertManager()
        manager.add_notifier(mock_notifier)

        # MDD 조건 등록
        condition = MddThresholdCondition(threshold=-0.10)
        manager.add_condition(condition)

        # MDD -15%로 검사 -> 트리거됨
        context = {"current_mdd": -0.15}
        alerts = manager.check_and_alert(context)

        assert len(alerts) > 0, "트리거된 조건이 있으므로 알림이 발생해야 합니다."
        mock_notifier.send_message.assert_called()

    def test_cooldown_prevents_duplicate(self):
        """쿨다운 기간 내에 동일 조건의 중복 알림이 방지된다."""
        mock_notifier = MagicMock()
        mock_notifier.is_configured.return_value = True
        mock_notifier.send_message.return_value = True

        manager = AlertManager()
        manager.add_notifier(mock_notifier)

        # MddThresholdCondition의 cooldown_hours=24 → 즉시 재검사 시 쿨다운 적용
        condition = MddThresholdCondition(threshold=-0.10)
        manager.add_condition(condition)

        context = {"current_mdd": -0.15}

        # 첫 번째 검사 -> 알림 발생
        alerts1 = manager.check_and_alert(context)
        # 두 번째 검사 (쿨다운 내) -> 알림 방지
        alerts2 = manager.check_and_alert(context)

        assert len(alerts1) > 0, "첫 번째 검사에서 알림이 발생해야 합니다."
        assert len(alerts2) == 0, (
            "쿨다운 기간 내 두 번째 검사에서 알림이 방지되어야 합니다."
        )

    def test_alert_manager_empty_state(self):
        """빈 상태(조건 없음)에서 검사하면 알림이 없다."""
        mock_notifier = MagicMock()
        mock_notifier.is_configured.return_value = True

        manager = AlertManager()
        manager.add_notifier(mock_notifier)

        context = {"current_mdd": -0.50}
        alerts = manager.check_and_alert(context)

        assert len(alerts) == 0, "조건이 없으면 알림이 없어야 합니다."
        mock_notifier.send_message.assert_not_called()

    def test_alert_manager_multiple_conditions(self):
        """여러 조건이 등록되어 있을 때 트리거된 조건만 알림이 발생한다."""
        mock_notifier = MagicMock()
        mock_notifier.is_configured.return_value = True
        mock_notifier.send_message.return_value = True

        manager = AlertManager()
        manager.add_notifier(mock_notifier)

        # 두 가지 조건 등록
        mdd_condition = MddThresholdCondition(threshold=-0.10)
        daily_condition = DailyMoveCondition(threshold=0.05)
        manager.add_condition(mdd_condition)
        manager.add_condition(daily_condition)

        # MDD만 트리거, 일일 변동은 트리거되지 않음
        context = {"current_mdd": -0.15, "holdings_daily_returns": {"005930": 0.01}}
        alerts = manager.check_and_alert(context)

        assert len(alerts) == 1, "MDD 조건만 트리거되어야 합니다."

    def test_alert_manager_notifier_not_configured(self):
        """notifier가 미설정이어도 에러 없이 동작한다."""
        mock_notifier = MagicMock()
        mock_notifier.is_configured.return_value = False

        manager = AlertManager()
        manager.add_notifier(mock_notifier)

        condition = MddThresholdCondition(threshold=-0.10)
        manager.add_condition(condition)

        context = {"current_mdd": -0.15}
        # 에러 없이 동작해야 함
        alerts = manager.check_and_alert(context)

        # notifier 미설정이어도 조건 체크 자체는 수행됨
        assert isinstance(alerts, list), "반환값이 리스트여야 합니다."
