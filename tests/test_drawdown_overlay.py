"""드로다운 기반 디레버리징 오버레이 모듈(src/strategy/drawdown_overlay.py) 테스트.

DrawdownOverlay 초기화, update 동작, apply_overlay 적용,
reset 기능, recovery_buffer 로직 등을 검증한다.
"""

import pytest

from src.strategy.drawdown_overlay import DrawdownOverlay


# ===================================================================
# DrawdownOverlay 초기화 테스트
# ===================================================================


class TestDrawdownOverlayInit:
    """DrawdownOverlay 초기화 검증."""

    def test_init_default_thresholds(self):
        """기본 임계값이 올바르게 설정된다."""
        overlay = DrawdownOverlay()
        assert len(overlay.thresholds) == 3, "기본 임계값은 3개여야 합니다."
        # 내림차순(가장 심한 낙폭 먼저) 정렬 확인
        assert overlay.thresholds[0] == (-0.20, 0.25)
        assert overlay.thresholds[1] == (-0.15, 0.50)
        assert overlay.thresholds[2] == (-0.10, 0.75)

    def test_init_default_recovery_buffer(self):
        """기본 recovery_buffer가 0.02로 설정된다."""
        overlay = DrawdownOverlay()
        assert overlay.recovery_buffer == 0.02, (
            "기본 recovery_buffer는 0.02여야 합니다."
        )

    def test_init_default_state(self):
        """초기 내부 상태가 올바르다."""
        overlay = DrawdownOverlay()
        assert overlay._peak == 0.0, "초기 peak은 0.0이어야 합니다."
        assert overlay._current_exposure == 1.0, (
            "초기 exposure는 1.0이어야 합니다."
        )

    def test_init_custom_thresholds(self):
        """커스텀 임계값이 올바르게 반영된다."""
        custom = [(-0.05, 0.80), (-0.25, 0.10), (-0.15, 0.40)]
        overlay = DrawdownOverlay(thresholds=custom)
        assert len(overlay.thresholds) == 3
        # 정렬 확인: drawdown_pct 오름차순 (가장 심한 먼저)
        assert overlay.thresholds[0][0] == -0.25
        assert overlay.thresholds[1][0] == -0.15
        assert overlay.thresholds[2][0] == -0.05

    def test_init_custom_recovery_buffer(self):
        """커스텀 recovery_buffer가 반영된다."""
        overlay = DrawdownOverlay(recovery_buffer=0.05)
        assert overlay.recovery_buffer == 0.05

    def test_init_thresholds_sorted_automatically(self):
        """정렬되지 않은 임계값도 자동으로 정렬된다."""
        unsorted = [(-0.10, 0.75), (-0.20, 0.25), (-0.15, 0.50)]
        overlay = DrawdownOverlay(thresholds=unsorted)
        drawdowns = [dd for dd, _ in overlay.thresholds]
        assert drawdowns == sorted(drawdowns), (
            "임계값은 drawdown_pct 기준 오름차순(가장 심한 먼저)으로 정렬되어야 합니다."
        )

    def test_init_single_threshold(self):
        """임계값이 1개만 있어도 동작한다."""
        overlay = DrawdownOverlay(thresholds=[(-0.10, 0.50)])
        assert len(overlay.thresholds) == 1
        assert overlay.thresholds[0] == (-0.10, 0.50)


# ===================================================================
# update 메서드 테스트
# ===================================================================


class TestUpdate:
    """update() 메서드 검증."""

    def test_rising_portfolio_always_one(self):
        """포트폴리오가 계속 상승하면 항상 1.0을 반환한다."""
        overlay = DrawdownOverlay()
        values = [100, 110, 120, 130, 140, 150]
        for v in values:
            exposure = overlay.update(v)
            assert exposure == 1.0, (
                f"상승 중에는 exposure가 1.0이어야 합니다 (value={v})."
            )
        assert overlay._peak == 150, "peak이 최고가로 갱신되어야 합니다."

    def test_drawdown_12_percent(self):
        """12% 드로다운에서 0.75 노출을 반환한다."""
        overlay = DrawdownOverlay()
        overlay.update(100)  # peak = 100
        exposure = overlay.update(88)  # drawdown = -12%
        assert exposure == 0.75, (
            "-12% 드로다운에서 exposure는 0.75여야 합니다."
        )

    def test_drawdown_16_percent(self):
        """16% 드로다운에서 0.50 노출을 반환한다."""
        overlay = DrawdownOverlay()
        overlay.update(100)  # peak = 100
        exposure = overlay.update(84)  # drawdown = -16%
        assert exposure == 0.50, (
            "-16% 드로다운에서 exposure는 0.50이어야 합니다."
        )

    def test_drawdown_22_percent(self):
        """22% 드로다운에서 0.25 노출을 반환한다."""
        overlay = DrawdownOverlay()
        overlay.update(100)  # peak = 100
        exposure = overlay.update(78)  # drawdown = -22%
        assert exposure == 0.25, (
            "-22% 드로다운에서 exposure는 0.25여야 합니다."
        )

    def test_drawdown_exactly_10_percent(self):
        """정확히 10% 드로다운에서 0.75 노출을 반환한다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        exposure = overlay.update(90)  # drawdown = -10%
        assert exposure == 0.75, (
            "정확히 -10% 드로다운에서 exposure는 0.75여야 합니다."
        )

    def test_drawdown_exactly_15_percent(self):
        """정확히 15% 드로다운에서 0.50 노출을 반환한다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        exposure = overlay.update(85)  # drawdown = -15%
        assert exposure == 0.50

    def test_drawdown_exactly_20_percent(self):
        """정확히 20% 드로다운에서 0.25 노출을 반환한다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        exposure = overlay.update(80)  # drawdown = -20%
        assert exposure == 0.25

    def test_small_drawdown_no_delevering(self):
        """5% 드로다운(임계값 미달)에서는 1.0을 유지한다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        exposure = overlay.update(95)  # drawdown = -5%
        assert exposure == 1.0, (
            "-5% 드로다운은 임계값에 미달하므로 exposure 1.0이어야 합니다."
        )

    def test_progressive_drawdown(self):
        """점진적 하락에서 단계별 디레버리징이 정확히 동작한다."""
        overlay = DrawdownOverlay()
        overlay.update(1000)  # peak = 1000

        # -8% -> 아직 임계값 미달
        assert overlay.update(920) == 1.0

        # -10% -> 0.75
        assert overlay.update(900) == 0.75

        # -12% -> 여전히 0.75 (다음 단계 -15% 미달)
        assert overlay.update(880) == 0.75

        # -15% -> 0.50
        assert overlay.update(850) == 0.50

        # -18% -> 여전히 0.50 (다음 단계 -20% 미달)
        assert overlay.update(820) == 0.50

        # -20% -> 0.25
        assert overlay.update(800) == 0.25

        # -25% -> 여전히 0.25 (더 이상 단계 없음)
        assert overlay.update(750) == 0.25

    def test_recovery_to_full_exposure(self):
        """드로다운 후 완전 회복하면 1.0으로 돌아온다."""
        overlay = DrawdownOverlay()
        overlay.update(100)  # peak = 100
        overlay.update(88)   # drawdown -12% -> exposure 0.75

        # 회복: drawdown이 -8% 초과해야 1.0 복귀
        # -9% -> 아직 recovery_buffer 미충족
        exposure = overlay.update(91)  # dd = -9%
        assert exposure == 0.75, (
            "-9%는 recovery_buffer(-8%) 미충족이므로 0.75 유지."
        )

        # -7% -> recovery_buffer 충족 (-10% + 2% = -8% 초과)
        exposure = overlay.update(93)  # dd = -7%
        assert exposure == 1.0, (
            "-7%는 recovery_buffer를 충족하므로 1.0으로 복귀해야 합니다."
        )

    def test_peak_updates_on_new_high(self):
        """새로운 고점에서 peak이 갱신된다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        assert overlay._peak == 100
        overlay.update(110)
        assert overlay._peak == 110
        overlay.update(105)  # 하락하지만 peak은 그대로
        assert overlay._peak == 110

    def test_zero_peak_returns_one(self):
        """peak이 0이면 1.0을 반환한다."""
        overlay = DrawdownOverlay()
        # peak 초기값 0, portfolio_value도 0
        exposure = overlay.update(0)
        assert exposure == 1.0, "peak이 0이면 기본값 1.0을 반환해야 합니다."

    def test_negative_portfolio_value(self):
        """음수 포트폴리오 가치에서도 안전하게 동작한다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        # 음수 값은 극단적 케이스이지만 크래시되면 안 됨
        exposure = overlay.update(-10)
        # drawdown = (-10 - 100) / 100 = -1.1 -> 가장 심한 단계(0.25)
        assert exposure == 0.25


# ===================================================================
# apply_overlay 메서드 테스트
# ===================================================================


class TestApplyOverlay:
    """apply_overlay() 메서드 검증."""

    def test_normal_no_drawdown(self):
        """드로다운이 없으면 비중이 그대로 유지된다."""
        overlay = DrawdownOverlay()
        weights = {"005930": 0.3, "000660": 0.3, "035420": 0.4}
        result = overlay.apply_overlay(weights, 100)
        assert result == weights, (
            "드로다운이 없으면 비중이 변경되지 않아야 합니다."
        )

    def test_drawdown_15_percent_weights_halved(self):
        """15% 드로다운에서 비중이 0.5배로 줄어든다."""
        overlay = DrawdownOverlay()
        overlay.update(100)  # peak = 100

        weights = {"005930": 0.4, "000660": 0.6}
        result = overlay.apply_overlay(weights, 85)  # -15% drawdown

        assert abs(result["005930"] - 0.20) < 1e-9, (
            "-15% DD에서 0.4 * 0.5 = 0.2 이어야 합니다."
        )
        assert abs(result["000660"] - 0.30) < 1e-9, (
            "-15% DD에서 0.6 * 0.5 = 0.3 이어야 합니다."
        )

    def test_empty_weights_remain_empty(self):
        """빈 비중 dict는 그대로 빈 dict를 반환한다."""
        overlay = DrawdownOverlay()
        result = overlay.apply_overlay({}, 100)
        assert result == {}, "빈 비중은 그대로 빈 dict여야 합니다."

    def test_apply_overlay_updates_state(self):
        """apply_overlay 호출이 내부 상태를 갱신한다."""
        overlay = DrawdownOverlay()
        overlay.apply_overlay({"A": 0.5}, 100)
        assert overlay._peak == 100, (
            "apply_overlay 호출 후 peak이 갱신되어야 합니다."
        )

    def test_apply_overlay_severe_drawdown(self):
        """극심한 드로다운에서 비중이 0.25배로 줄어든다."""
        overlay = DrawdownOverlay()
        overlay.update(1000)

        weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        result = overlay.apply_overlay(weights, 750)  # -25% drawdown

        for ticker, w in weights.items():
            assert abs(result[ticker] - w * 0.25) < 1e-9, (
                f"-25% DD에서 {ticker} 비중은 {w} * 0.25 = {w * 0.25}이어야 합니다."
            )

    def test_apply_overlay_rising_portfolio(self):
        """포트폴리오가 계속 상승하면 비중이 변경되지 않는다."""
        overlay = DrawdownOverlay()
        weights = {"A": 0.5, "B": 0.5}

        for value in [100, 110, 120, 130]:
            result = overlay.apply_overlay(weights, value)
            assert result == weights, (
                f"상승 중(value={value}) 비중 변경 없어야 합니다."
            )


# ===================================================================
# reset 메서드 테스트
# ===================================================================


class TestReset:
    """reset() 메서드 검증."""

    def test_reset_clears_peak(self):
        """reset 후 peak이 0으로 초기화된다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        overlay.update(88)
        assert overlay._peak == 100

        overlay.reset()
        assert overlay._peak == 0.0, "reset 후 peak은 0.0이어야 합니다."

    def test_reset_clears_exposure(self):
        """reset 후 exposure가 1.0으로 초기화된다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        overlay.update(88)  # exposure = 0.75
        assert overlay._current_exposure == 0.75

        overlay.reset()
        assert overlay._current_exposure == 1.0, (
            "reset 후 exposure는 1.0이어야 합니다."
        )

    def test_reset_allows_fresh_start(self):
        """reset 후 새로운 포트폴리오 가치로 깨끗하게 시작된다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        overlay.update(78)  # -22% -> 0.25
        assert overlay._current_exposure == 0.25

        overlay.reset()

        # 새 시작: 200에서 시작하면 peak=200
        exposure = overlay.update(200)
        assert exposure == 1.0, "reset 후 새로운 값으로 시작하면 1.0이어야 합니다."
        assert overlay._peak == 200

    def test_multiple_resets(self):
        """여러 번 reset해도 안전하다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        overlay.update(85)

        overlay.reset()
        overlay.reset()
        overlay.reset()

        assert overlay._peak == 0.0
        assert overlay._current_exposure == 1.0


# ===================================================================
# recovery_buffer 로직 테스트
# ===================================================================


class TestRecoveryBuffer:
    """recovery_buffer를 통한 oscillation 방지 로직 검증."""

    def test_no_premature_recovery(self):
        """recovery_buffer 미충족 시 노출이 복귀하지 않는다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        overlay.update(89)   # -11% -> 0.75

        # -9%: buffer 미충족 (-10% + 2% = -8% 초과 필요)
        exposure = overlay.update(91)
        assert exposure == 0.75, (
            "-9%는 recovery level -8% 미충족이므로 0.75 유지."
        )

    def test_recovery_at_exact_buffer(self):
        """recovery_buffer 경계값에서 노출이 복귀하지 않는다.

        drawdown이 정확히 recovery_level(-8%)이면 '초과'가 아니므로 복귀 안 됨.
        """
        overlay = DrawdownOverlay()
        overlay.update(100)
        overlay.update(89)  # -11% -> 0.75

        # 정확히 -8% (= -10% + 2%): '초과'가 아님
        exposure = overlay.update(92)  # dd = -8%
        assert exposure == 0.75, (
            "정확히 recovery_level에서는 '초과'가 아니므로 0.75 유지."
        )

    def test_recovery_past_buffer(self):
        """recovery_buffer를 초과하면 노출이 한 단계 복귀한다."""
        overlay = DrawdownOverlay()
        overlay.update(100)
        overlay.update(89)  # -11% -> 0.75

        # -7.9% > -8% (recovery_level): 초과 -> 1.0 복귀
        exposure = overlay.update(92.1)  # dd = -7.9%
        assert exposure == 1.0, (
            "recovery_buffer를 초과하면 1.0으로 복귀해야 합니다."
        )

    def test_oscillation_prevention(self):
        """임계값 근처에서 반복적으로 진동해도 빈번한 전환이 발생하지 않는다."""
        overlay = DrawdownOverlay()
        overlay.update(100)

        # -11% -> 0.75
        assert overlay.update(89) == 0.75

        # -9% (아직 buffer 미충족) -> 0.75 유지
        assert overlay.update(91) == 0.75

        # -10.5% -> 0.75 유지 (이미 디레버리징 상태)
        assert overlay.update(89.5) == 0.75

        # -9.5% -> 0.75 유지 (buffer 미충족)
        assert overlay.update(90.5) == 0.75

        # -8.5% -> 0.75 유지 (buffer 미충족: -8% 초과 필요)
        assert overlay.update(91.5) == 0.75

        # -7% -> 1.0 복귀 (buffer 충족)
        assert overlay.update(93) == 1.0

    def test_stepwise_recovery_from_deep_drawdown(self):
        """깊은 드로다운에서 단계별로 회복한다."""
        overlay = DrawdownOverlay()
        overlay.update(100)

        # -22% -> 0.25
        assert overlay.update(78) == 0.25

        # -19% -> 회복 확인
        # 0.25의 threshold = -20%, recovery_level = -20% + 2% = -18%
        # -19% <= -18% -> 아직 미충족
        assert overlay.update(81) == 0.25

        # -17% -> recovery_level -18% 초과 -> 한 단계 위(0.50)로
        assert overlay.update(83) == 0.50

        # -14% -> 0.50의 threshold = -15%, recovery_level = -13%
        # -14% <= -13% -> 미충족
        assert overlay.update(86) == 0.50

        # -12% -> recovery_level -13% 초과 -> 한 단계 위(0.75)로
        assert overlay.update(88) == 0.75

        # -7% -> 0.75의 threshold = -10%, recovery_level = -8%
        # -7% > -8% -> 충족 -> 1.0 복귀
        assert overlay.update(93) == 1.0

    def test_custom_recovery_buffer(self):
        """커스텀 recovery_buffer가 올바르게 적용된다."""
        overlay = DrawdownOverlay(recovery_buffer=0.05)
        overlay.update(100)
        overlay.update(88)  # -12% -> 0.75

        # recovery_level = -10% + 5% = -5%
        # -6% -> 미충족
        assert overlay.update(94) == 0.75

        # -4% -> 충족 (-5% 초과)
        assert overlay.update(96) == 1.0

    def test_zero_recovery_buffer(self):
        """recovery_buffer가 0이면 즉시 복귀한다."""
        overlay = DrawdownOverlay(recovery_buffer=0.0)
        overlay.update(100)
        overlay.update(88)  # -12% -> 0.75

        # -9% -> 임계값 -10%보다 나음 -> 즉시 복귀
        # recovery_level = -10% + 0% = -10%, dd=-9% > -10% -> 충족
        assert overlay.update(91) == 1.0

    def test_immediate_delever_no_buffer_needed(self):
        """디레버리징 방향으로는 recovery_buffer 없이 즉시 적용된다."""
        overlay = DrawdownOverlay()
        overlay.update(100)

        # -10% -> 즉시 0.75
        assert overlay.update(90) == 0.75

        # 약간 회복 후 다시 하락
        overlay.update(91)  # 여전히 0.75 (buffer 미충족)

        # -15% -> 즉시 0.50 (디레버리징 방향은 즉시 적용)
        assert overlay.update(85) == 0.50

    def test_delever_then_recover_then_delever_again(self):
        """디레버리징 -> 회복 -> 재디레버리징 시나리오."""
        overlay = DrawdownOverlay()
        overlay.update(100)

        # 디레버리징
        assert overlay.update(88) == 0.75  # -12%

        # 완전 회복 (새 고점까지)
        assert overlay.update(93) == 1.0   # buffer 충족
        assert overlay.update(105) == 1.0  # 새 고점
        assert overlay._peak == 105

        # 새 고점 기준 재디레버리징
        # -10% of 105 = 94.5
        assert overlay.update(94) == 0.75  # (94-105)/105 = -10.5%
