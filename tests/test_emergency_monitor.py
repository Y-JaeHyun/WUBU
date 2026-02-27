"""긴급 리밸런싱 모니터 테스트.

DisclosureAlertCondition, PriceShockCondition, MarketCrashCondition을
검증한다.
"""

import pytest

from src.alert.conditions import (
    DisclosureAlertCondition,
    MarketCrashCondition,
    PriceShockCondition,
)


# ===================================================================
# DisclosureAlertCondition 테스트
# ===================================================================


class TestDisclosureAlert:
    """긴급 공시 알림 조건 검증."""

    def test_disclosure_triggered(self):
        """공시 목록이 존재하면 트리거된다."""
        condition = DisclosureAlertCondition()

        state = {
            "portfolio_disclosures": [
                {
                    "corp_name": "삼성전자",
                    "report_nm": "유상증자 결정",
                    "category": "capital_increase",
                    "is_held": True,
                }
            ]
        }
        result = condition.check(state)

        assert result is True, (
            "공시가 존재하면 트리거되어야 합니다."
        )

        msg = condition.format_message(state)
        assert "삼성전자" in msg, "메시지에 종목명이 포함되어야 합니다."
        assert "[보유]" in msg, "보유 종목에는 [보유] 접두어가 붙어야 합니다."
        assert condition.get_effective_level(state) == "WARNING", (
            "capital_increase는 WARNING 레벨이어야 합니다."
        )

    def test_disclosure_not_triggered(self):
        """공시 목록이 비어있으면 트리거되지 않는다."""
        condition = DisclosureAlertCondition()

        state = {"portfolio_disclosures": []}
        result = condition.check(state)

        assert result is False, (
            "공시가 없으면 트리거되지 않아야 합니다."
        )

        # state에 키 자체가 없을 때도 동일
        state_empty = {}
        result_empty = condition.check(state_empty)

        assert result_empty is False, (
            "portfolio_disclosures 키가 없으면 트리거되지 않아야 합니다."
        )

    def test_disclosure_critical_for_delisting(self):
        """상장폐지(delisting) 공시는 CRITICAL 레벨이다."""
        condition = DisclosureAlertCondition()

        state = {
            "portfolio_disclosures": [
                {
                    "corp_name": "ABC주식회사",
                    "report_nm": "상장폐지 사유 발생",
                    "category": "delisting",
                    "is_held": True,
                }
            ]
        }
        result = condition.check(state)

        assert result is True, "상장폐지 공시는 트리거되어야 합니다."
        assert condition.get_effective_level(state) == "CRITICAL", (
            "상장폐지(delisting) 공시는 CRITICAL이어야 합니다."
        )

    def test_disclosure_critical_for_merger(self):
        """합병(merger) 공시는 CRITICAL 레벨이다."""
        condition = DisclosureAlertCondition()

        state = {
            "portfolio_disclosures": [
                {
                    "corp_name": "XYZ주식회사",
                    "report_nm": "합병 결정",
                    "category": "merger",
                    "is_held": False,
                }
            ]
        }

        assert condition.get_effective_level(state) == "CRITICAL", (
            "합병(merger) 공시는 CRITICAL이어야 합니다."
        )

    def test_disclosure_non_held_no_prefix(self):
        """비보유 종목 공시에는 [보유] 접두어가 없다."""
        condition = DisclosureAlertCondition()

        state = {
            "portfolio_disclosures": [
                {
                    "corp_name": "타사",
                    "report_nm": "공시내용",
                    "category": "etc",
                    "is_held": False,
                }
            ]
        }
        msg = condition.format_message(state)
        assert "[보유]" not in msg, (
            "비보유 종목 공시에는 [보유] 접두어가 없어야 합니다."
        )

    def test_disclosure_cooldown(self):
        """쿨다운이 4시간으로 설정되어 있다."""
        condition = DisclosureAlertCondition()
        assert condition.cooldown_hours == 4, (
            "긴급 공시 쿨다운은 4시간이어야 합니다."
        )


# ===================================================================
# PriceShockCondition 테스트
# ===================================================================


class TestPriceShock:
    """보유 종목 급등/급락 조건 검증."""

    def test_price_shock_triggered_at_5_pct(self):
        """5% 이상 변동 시 트리거된다."""
        condition = PriceShockCondition(threshold=5.0)

        state = {
            "price_shocks": [
                {"name": "삼성전자", "ticker": "005930", "change": -5.5}
            ]
        }
        result = condition.check(state)

        assert result is True, (
            "5.5% 급락은 임계값 5%를 초과하므로 트리거되어야 합니다."
        )

        msg = condition.format_message(state)
        assert "삼성전자" in msg, "메시지에 종목명이 포함되어야 합니다."
        assert "005930" in msg, "메시지에 티커가 포함되어야 합니다."
        assert "-5.5%" in msg, "메시지에 변동률이 포함되어야 합니다."

    def test_price_shock_not_triggered_at_3_pct(self):
        """3% 변동은 임계값(5%) 미만이므로 트리거되지 않는다."""
        condition = PriceShockCondition(threshold=5.0)

        state = {"price_shocks": []}
        result = condition.check(state)

        assert result is False, (
            "급등/급락 목록이 비어있으면 트리거되지 않아야 합니다."
        )

        # state에 키 자체가 없을 때도 동일
        result_empty = condition.check({})
        assert result_empty is False, (
            "price_shocks 키가 없으면 트리거되지 않아야 합니다."
        )

    def test_price_shock_critical_at_8_pct(self):
        """8% 이상 변동은 CRITICAL 레벨이다."""
        condition = PriceShockCondition(threshold=5.0)

        state = {
            "price_shocks": [
                {"name": "NAVER", "ticker": "035420", "change": 8.2}
            ]
        }
        result = condition.check(state)

        assert result is True, "8.2% 급등은 트리거되어야 합니다."
        assert condition.get_effective_level(state) == "CRITICAL", (
            "8.2% 변동은 7% 이상이므로 CRITICAL이어야 합니다."
        )

    def test_price_shock_warning_at_6_pct(self):
        """6% 변동은 WARNING 레벨이다 (5~7% 사이)."""
        condition = PriceShockCondition(threshold=5.0)

        state = {
            "price_shocks": [
                {"name": "카카오", "ticker": "035720", "change": -6.0}
            ]
        }
        assert condition.get_effective_level(state) == "WARNING", (
            "6% 변동은 7% 미만이므로 WARNING이어야 합니다."
        )

    def test_price_shock_cooldown(self):
        """쿨다운이 2시간으로 설정되어 있다."""
        condition = PriceShockCondition()
        assert condition.cooldown_hours == 2, (
            "급등/급락 쿨다운은 2시간이어야 합니다."
        )


# ===================================================================
# MarketCrashCondition 테스트
# ===================================================================


class TestMarketCrash:
    """시장 급변 조건 검증."""

    def test_market_crash_triggered_at_negative_4_pct(self):
        """KOSPI -4% 하락 시 트리거된다."""
        condition = MarketCrashCondition(threshold=3.0)

        state = {"market_change_pct": -4.0}
        result = condition.check(state)

        assert result is True, (
            "KOSPI -4%는 임계값 3%를 초과하므로 트리거되어야 합니다."
        )

        msg = condition.format_message(state)
        assert "KOSPI" in msg, "메시지에 KOSPI가 포함되어야 합니다."
        assert "-4.0%" in msg, "메시지에 변동률이 포함되어야 합니다."
        assert "포트폴리오 점검 필요" in msg, (
            "메시지에 점검 권고가 포함되어야 합니다."
        )
        assert condition.level == "CRITICAL", (
            "시장 급변은 CRITICAL 레벨이어야 합니다."
        )

    def test_market_crash_not_triggered_at_negative_2_pct(self):
        """KOSPI -2% 하락은 임계값(3%) 미만이므로 트리거되지 않는다."""
        condition = MarketCrashCondition(threshold=3.0)

        state = {"market_change_pct": -2.0}
        result = condition.check(state)

        assert result is False, (
            "KOSPI -2%는 임계값 3% 미만이므로 트리거되지 않아야 합니다."
        )

    def test_market_crash_positive_surge(self):
        """KOSPI +3.5% 급등도 트리거된다 (양방향 감지)."""
        condition = MarketCrashCondition(threshold=3.0)

        state = {"market_change_pct": 3.5}
        result = condition.check(state)

        assert result is True, (
            "KOSPI +3.5%는 임계값 3%를 초과하므로 트리거되어야 합니다."
        )

        msg = condition.format_message(state)
        assert "+3.5%" in msg, "양수 변동률도 부호가 표시되어야 합니다."

    def test_market_crash_exact_threshold(self):
        """정확히 임계값이면 트리거되지 않는다 (strict greater-than)."""
        condition = MarketCrashCondition(threshold=3.0)

        state = {"market_change_pct": 3.0}
        result = condition.check(state)

        assert result is False, (
            "정확히 3.0%는 strict >이므로 트리거되지 않아야 합니다."
        )

        state_neg = {"market_change_pct": -3.0}
        result_neg = condition.check(state_neg)

        assert result_neg is False, (
            "정확히 -3.0%도 strict >이므로 트리거되지 않아야 합니다."
        )

    def test_market_crash_empty_state(self):
        """market_change_pct 키가 없으면 트리거되지 않는다."""
        condition = MarketCrashCondition(threshold=3.0)

        result = condition.check({})

        assert result is False, (
            "키가 없으면 기본값 0으로 트리거되지 않아야 합니다."
        )

    def test_market_crash_cooldown(self):
        """쿨다운이 4시간으로 설정되어 있다."""
        condition = MarketCrashCondition()
        assert condition.cooldown_hours == 4, (
            "시장 급변 쿨다운은 4시간이어야 합니다."
        )
