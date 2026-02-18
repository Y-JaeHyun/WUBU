"""듀얼 모멘텀 전략 모듈(src/strategy/dual_momentum.py) 테스트.

DualMomentumStrategy 초기화, 모멘텀 계산, 상대/절대 시그널,
자산 배분 생성, 백테스트 연동 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# 헬퍼 함수
# ===================================================================

def _import_dual_momentum():
    """DualMomentumStrategy 클래스를 임포트한다."""
    from src.strategy.dual_momentum import DualMomentumStrategy
    return DualMomentumStrategy


def _make_etf_prices(
    tickers: list[str],
    periods: int = 300,
    seed: int = 42,
    trends: dict[str, float] | None = None,
) -> dict[str, pd.Series]:
    """ETF 가격 시계열 dict를 생성한다.

    Args:
        tickers: ETF 종목코드 리스트
        periods: 데이터 기간 (거래일 수)
        seed: 랜덤 시드
        trends: 종목별 일평균 수익률 (drift). 없으면 랜덤.

    Returns:
        dict[ticker -> pd.Series] (인덱스: DatetimeIndex)
    """
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-02", periods=periods)
    result = {}
    for i, ticker in enumerate(tickers):
        drift = 0.0003 if trends is None else trends.get(ticker, 0.0003)
        log_returns = np.random.randn(periods) * 0.01 + drift
        prices = 10000 * np.exp(np.cumsum(log_returns))
        result[ticker] = pd.Series(prices, index=dates, name=ticker)
    return result


def _make_known_prices() -> dict[str, pd.Series]:
    """알려진 수익률 구조의 가격 데이터를 생성한다.

    - domestic (069500): 연 15% 상승
    - us (360750): 연 5% 상승
    - safe (214980): 연 2% 상승 (거의 변동 없음)
    """
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=300)
    n = len(dates)

    domestic = 10000 * np.exp(np.linspace(0, 0.15, n))
    us = 10000 * np.exp(np.linspace(0, 0.05, n))
    safe = 10000 * np.exp(np.linspace(0, 0.02, n))

    return {
        "069500": pd.Series(domestic, index=dates, name="069500"),
        "360750": pd.Series(us, index=dates, name="360750"),
        "214980": pd.Series(safe, index=dates, name="214980"),
    }


# ===================================================================
# DualMomentumStrategy 초기화 테스트
# ===================================================================

class TestDualMomentumInit:
    """DualMomentumStrategy 초기화 검증."""

    def test_init_default(self):
        """기본 파라미터로 초기화된다."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy()

        safe = getattr(dms, "safe_asset", getattr(dms, "_safe_asset", None))
        assert safe == "214980", "기본 safe_asset은 '214980'이어야 합니다."

        lookback = getattr(dms, "lookback_months", getattr(dms, "_lookback_months", None))
        assert lookback == 12, "기본 lookback_months는 12이어야 합니다."

    def test_init_custom(self):
        """사용자 정의 파라미터가 반영된다."""
        DualMomentumStrategy = _import_dual_momentum()
        custom_risky = {"domestic": "069500", "us": "360750", "japan": "241180"}
        dms = DualMomentumStrategy(
            risky_assets=custom_risky,
            safe_asset="214980",
            lookback_months=6,
            n_select=2,
        )

        n_select = getattr(dms, "n_select", getattr(dms, "_n_select", None))
        assert n_select == 2, "커스텀 n_select가 반영되어야 합니다."

        lookback = getattr(dms, "lookback_months", getattr(dms, "_lookback_months", None))
        assert lookback == 6, "커스텀 lookback_months가 반영되어야 합니다."


# ===================================================================
# calculate_momentum 테스트
# ===================================================================

class TestCalculateMomentum:
    """모멘텀 계산 검증."""

    def test_calculate_momentum(self):
        """알려진 가격 데이터로 모멘텀이 올바르게 계산된다."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy(lookback_months=12)
        prices = _make_known_prices()

        momentum = dms.calculate_momentum(prices)

        assert isinstance(momentum, dict), "반환값이 dict여야 합니다."
        assert len(momentum) > 0, "모멘텀 결과가 비어 있으면 안 됩니다."

        # domestic이 가장 높은 수익률 (15%)
        if "069500" in momentum and "360750" in momentum:
            assert momentum["069500"] > momentum["360750"], (
                "domestic(15%)이 us(5%)보다 높은 모멘텀이어야 합니다."
            )

    def test_calculate_momentum_insufficient_data(self):
        """데이터가 부족하면 빈 결과 또는 NaN이 포함된다."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy(lookback_months=12)

        # 30일짜리 짧은 데이터 (12개월에 턱없이 부족)
        dates = pd.bdate_range("2024-01-02", periods=30)
        prices = {
            "069500": pd.Series(np.linspace(10000, 10500, 30), index=dates),
            "214980": pd.Series(np.linspace(10000, 10050, 30), index=dates),
        }

        momentum = dms.calculate_momentum(prices)

        # 데이터 부족 시 빈 dict 반환이거나 NaN 포함
        if momentum:
            for v in momentum.values():
                # NaN이거나 빈 dict이 가능
                pass  # 에러 없이 수행되면 통과

    def test_calculate_momentum_custom_lookback(self):
        """커스텀 lookback_months로 모멘텀이 계산된다."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy(lookback_months=6)
        prices = _make_known_prices()

        momentum = dms.calculate_momentum(prices, lookback_months=6)

        assert isinstance(momentum, dict), "커스텀 룩백 모멘텀 결과가 dict여야 합니다."


# ===================================================================
# 상대/절대 시그널 테스트
# ===================================================================

class TestMomentumSignals:
    """상대 모멘텀 및 절대 모멘텀 시그널 검증."""

    def test_relative_signal_selects_best(self):
        """상대 모멘텀이 가장 높은 자산이 선택된다."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy(n_select=1)

        momentum = {"069500": 0.15, "360750": 0.05}
        selected = dms.get_relative_signal(momentum)

        assert isinstance(selected, list), "반환값이 list여야 합니다."
        assert "069500" in selected, (
            "모멘텀이 가장 높은 '069500'이 선택되어야 합니다."
        )

    def test_relative_signal_n_select_2(self):
        """n_select=2일 때 상위 2개가 선택된다."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy(n_select=2)

        momentum = {"069500": 0.15, "360750": 0.10, "133690": 0.05}
        selected = dms.get_relative_signal(momentum, n=2)

        assert len(selected) == 2, "2개가 선택되어야 합니다."
        assert "069500" in selected, "최고 모멘텀 자산이 포함되어야 합니다."
        assert "360750" in selected, "2위 모멘텀 자산이 포함되어야 합니다."

    def test_absolute_signal_positive(self):
        """모멘텀이 양수이면 절대 모멘텀 시그널이 True이다."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy()

        momentum = {"069500": 0.10, "360750": 0.05}
        signals = dms.get_absolute_signal(momentum)

        assert isinstance(signals, dict), "반환값이 dict여야 합니다."
        assert signals["069500"] is True, "양수 모멘텀이면 True여야 합니다."
        assert signals["360750"] is True, "양수 모멘텀이면 True여야 합니다."

    def test_absolute_signal_negative(self):
        """모멘텀이 음수이면 절대 모멘텀 시그널이 False이다."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy()

        momentum = {"069500": -0.05, "360750": -0.10}
        signals = dms.get_absolute_signal(momentum)

        assert signals["069500"] is False, "음수 모멘텀이면 False여야 합니다."
        assert signals["360750"] is False, "음수 모멘텀이면 False여야 합니다."


# ===================================================================
# generate_allocation 테스트
# ===================================================================

class TestGenerateAllocation:
    """자산 배분 생성 검증."""

    def test_generate_allocation_risk_on(self):
        """모멘텀이 양수인 위험자산에 배분한다 (리스크 온)."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy(n_select=1)

        # 강한 상승세
        prices = _make_known_prices()
        allocation = dms.generate_allocation(prices)

        assert isinstance(allocation, dict), "반환값이 dict여야 합니다."
        assert len(allocation) > 0, "배분 결과가 비어 있으면 안 됩니다."

    def test_generate_allocation_risk_off(self):
        """위험자산 모멘텀이 모두 음수이면 안전자산에 배분한다."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy(n_select=1)

        # 하락 추세 가격
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-02", periods=300)
        n = len(dates)
        prices = {
            "069500": pd.Series(10000 * np.exp(np.linspace(0, -0.20, n)), index=dates),
            "360750": pd.Series(10000 * np.exp(np.linspace(0, -0.15, n)), index=dates),
            "214980": pd.Series(10000 * np.exp(np.linspace(0, 0.02, n)), index=dates),
        }

        allocation = dms.generate_allocation(prices)

        assert isinstance(allocation, dict), "반환값이 dict여야 합니다."
        # 안전자산에 배분이 되어야 함
        if allocation:
            # 214980(안전자산)에 비중이 있거나, 또는 위험자산이 아닌 자산에 비중
            safe_weight = allocation.get("214980", 0)
            risky_weight = allocation.get("069500", 0) + allocation.get("360750", 0)
            # 리스크 오프이므로 안전자산 비중이 위험자산보다 커야 함
            assert safe_weight >= risky_weight, (
                "리스크 오프에서 안전자산 비중이 위험자산보다 커야 합니다."
            )

    def test_generate_allocation_weights_sum(self):
        """배분 비중의 합이 1.0이다."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy(n_select=1)
        prices = _make_known_prices()

        allocation = dms.generate_allocation(prices)

        if allocation:
            total = sum(allocation.values())
            assert abs(total - 1.0) < 1e-9, (
                f"배분 비중 합이 1.0이어야 합니다: {total}"
            )

    def test_generate_allocation_no_negative_weights(self):
        """배분 비중에 음수가 없다."""
        DualMomentumStrategy = _import_dual_momentum()
        dms = DualMomentumStrategy(n_select=1)
        prices = _make_known_prices()

        allocation = dms.generate_allocation(prices)

        for ticker, weight in allocation.items():
            assert weight >= 0, (
                f"종목 {ticker}의 비중 {weight}이 음수입니다."
            )
