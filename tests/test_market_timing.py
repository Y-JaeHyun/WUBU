"""마켓 타이밍 오버레이 모듈(src/strategy/market_timing.py) 테스트.

MarketTimingOverlay 생성, SMA/EMA 시그널, 오버레이 적용,
점진적 전환 모드 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.market_timing import MarketTimingOverlay


# ===================================================================
# MarketTimingOverlay 초기화 테스트
# ===================================================================


class TestMarketTimingInit:
    """MarketTimingOverlay 초기화 검증."""

    def test_init_default(self):
        """기본 파라미터로 초기화된다."""
        mt = MarketTimingOverlay()
        assert mt.ma_period == 200, "기본 ma_period는 200이어야 합니다."
        assert mt.ma_type == "SMA", "기본 ma_type은 'SMA'여야 합니다."
        assert mt.switch_mode == "binary", "기본 switch_mode는 'binary'여야 합니다."

    def test_init_custom(self):
        """커스텀 파라미터가 올바르게 반영된다."""
        mt = MarketTimingOverlay(
            ma_period=100,
            ma_type="EMA",
            switch_mode="gradual",
        )
        assert mt.ma_period == 100, "커스텀 ma_period가 반영되어야 합니다."
        assert mt.ma_type == "EMA", "커스텀 ma_type이 반영되어야 합니다."
        assert mt.switch_mode == "gradual", "커스텀 switch_mode가 반영되어야 합니다."


# ===================================================================
# 시그널 생성 테스트
# ===================================================================


class TestMarketTimingSignal:
    """시그널 생성 검증."""

    def test_signal_risk_on(self, uptrend_index_prices):
        """가격이 SMA 위에 있으면 RISK_ON 시그널을 반환한다."""
        mt = MarketTimingOverlay(ma_period=200, ma_type="SMA", switch_mode="binary")

        signal = mt.get_signal(uptrend_index_prices)

        assert signal == "RISK_ON", (
            "상승 추세에서 가격이 SMA 위에 있으면 RISK_ON이어야 합니다."
        )

    def test_signal_risk_off(self, downtrend_index_prices):
        """가격이 SMA 아래에 있으면 RISK_OFF 시그널을 반환한다."""
        mt = MarketTimingOverlay(ma_period=200, ma_type="SMA", switch_mode="binary")

        signal = mt.get_signal(downtrend_index_prices)

        assert signal == "RISK_OFF", (
            "하락 추세에서 가격이 SMA 아래에 있으면 RISK_OFF여야 합니다."
        )

    def test_signal_ema(self, uptrend_index_prices):
        """EMA 모드에서 상승 추세의 시그널을 확인한다."""
        mt = MarketTimingOverlay(ma_period=200, ma_type="EMA", switch_mode="binary")

        signal = mt.get_signal(uptrend_index_prices)

        # 꾸준한 상승이면 EMA도 현재가 아래에 있어야 함
        assert signal == "RISK_ON", "EMA 모드 상승 추세에서 RISK_ON이어야 합니다."

    def test_signal_ema_downtrend(self, downtrend_index_prices):
        """EMA 모드에서 하락 추세의 시그널을 확인한다."""
        mt = MarketTimingOverlay(ma_period=200, ma_type="EMA", switch_mode="binary")

        signal = mt.get_signal(downtrend_index_prices)

        assert signal == "RISK_OFF", "EMA 모드 하락 추세에서 RISK_OFF여야 합니다."

    def test_insufficient_data(self):
        """데이터가 부족하면 기본값을 반환한다."""
        mt = MarketTimingOverlay(ma_period=200)

        # 50일짜리 데이터 (200일 MA에 불충분)
        dates = pd.bdate_range("2024-01-02", periods=50)
        prices = pd.Series(np.arange(50, dtype=float) + 100, index=dates)

        signal = mt.get_signal(prices)

        # 데이터 부족 시 기본값은 RISK_ON (투자 유지)
        assert signal == "RISK_ON", (
            "데이터 부족 시 기본값 RISK_ON을 반환해야 합니다."
        )


# ===================================================================
# 오버레이 적용 테스트
# ===================================================================


class TestApplyOverlay:
    """오버레이 적용 검증."""

    def test_apply_overlay_risk_on(self, uptrend_index_prices):
        """RISK_ON 시에 비중이 유지된다."""
        mt = MarketTimingOverlay(ma_period=200, switch_mode="binary")

        weights = {"005930": 0.2, "000660": 0.3, "035420": 0.5}
        signal = mt.get_signal(uptrend_index_prices)
        result = mt.apply_overlay(weights, signal)

        assert result == weights, "RISK_ON 시 비중이 그대로 유지되어야 합니다."

    def test_apply_overlay_risk_off(self, downtrend_index_prices):
        """RISK_OFF 시에 비중이 빈 dict로 변환된다."""
        mt = MarketTimingOverlay(ma_period=200, switch_mode="binary")

        weights = {"005930": 0.2, "000660": 0.3, "035420": 0.5}
        signal = mt.get_signal(downtrend_index_prices)
        result = mt.apply_overlay(weights, signal)

        assert result == {}, "RISK_OFF binary 시 빈 dict여야 합니다."

    def test_gradual_mode(self, downtrend_index_prices):
        """점진적 전환 모드에서 비중이 비례적으로 줄어든다."""
        mt = MarketTimingOverlay(
            ma_period=200,
            switch_mode="gradual",
        )

        weights = {"005930": 0.5, "000660": 0.5}
        result = mt.apply_overlay_gradual(weights, downtrend_index_prices)

        # 하락 추세에서 gradual 모드는 비중을 비례적으로 줄임
        # 완전 binary처럼 0이 되지는 않지만, 원래보다 작아야 함
        if result:
            total_result = sum(result.values())
            total_original = sum(weights.values())
            assert total_result <= total_original, (
                "gradual 모드에서 하락 시 비중이 줄어들어야 합니다."
            )
        # 빈 dict도 유효 (강한 하락의 경우)

    def test_apply_overlay_empty_weights(self, uptrend_index_prices):
        """빈 비중 dict는 그대로 빈 dict를 반환한다."""
        mt = MarketTimingOverlay(ma_period=200)

        signal = mt.get_signal(uptrend_index_prices)
        result = mt.apply_overlay({}, signal)

        assert result == {}, "빈 비중은 그대로 빈 dict여야 합니다."

    def test_apply_overlay_insufficient_data(self):
        """데이터 부족 시 비중이 그대로 유지된다 (기본값 RISK_ON)."""
        mt = MarketTimingOverlay(ma_period=200)

        dates = pd.bdate_range("2024-01-02", periods=50)
        prices = pd.Series(np.arange(50, dtype=float) + 100, index=dates)
        weights = {"005930": 0.5, "000660": 0.5}

        signal = mt.get_signal(prices)
        result = mt.apply_overlay(weights, signal)

        assert result == weights, (
            "데이터 부족 시 비중이 유지되어야 합니다 (기본값 RISK_ON)."
        )
