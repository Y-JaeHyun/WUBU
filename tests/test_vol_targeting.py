"""변동성 타겟팅 오버레이 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_targeting import VolTargetingOverlay


# ---------------------------------------------------------------------------
# 초기화 테스트
# ---------------------------------------------------------------------------

class TestVolTargetingInit:
    """VolTargetingOverlay 초기화 테스트."""

    def test_default_params(self):
        vt = VolTargetingOverlay()
        assert vt.target_vol == 0.15
        assert vt.lookback_days == 20
        assert vt.min_vol == 0.05
        assert vt.max_exposure == 1.0
        assert vt.use_downside_only is True

    def test_custom_params(self):
        vt = VolTargetingOverlay(
            target_vol=0.12, lookback_days=60, min_vol=0.03,
            max_exposure=0.8, use_downside_only=False,
        )
        assert vt.target_vol == 0.12
        assert vt.lookback_days == 60
        assert vt.max_exposure == 0.8
        assert vt.use_downside_only is False


# ---------------------------------------------------------------------------
# 실현 변동성 계산 테스트
# ---------------------------------------------------------------------------

class TestComputeRealizedVol:
    """compute_realized_vol 테스트."""

    def test_insufficient_data_returns_target_vol(self):
        """데이터 부족 시 target_vol 반환."""
        vt = VolTargetingOverlay(target_vol=0.15, lookback_days=20)
        short_series = pd.Series(
            [100_000_000] * 10,
            index=pd.bdate_range("2024-01-01", periods=10),
        )
        assert vt.compute_realized_vol(short_series) == 0.15

    def test_low_volatility_series(self):
        """저변동 시계열 → 낮은 실현 변동성."""
        vt = VolTargetingOverlay(lookback_days=20, use_downside_only=False)
        dates = pd.bdate_range("2024-01-01", periods=30)
        # 아주 작은 일별 변동 (0.1%)
        np.random.seed(42)
        values = 100_000_000 * np.exp(np.cumsum(np.random.randn(30) * 0.001))
        pv = pd.Series(values, index=dates)

        vol = vt.compute_realized_vol(pv)
        assert vol < 0.10  # 연율화 10% 미만

    def test_high_volatility_series(self):
        """고변동 시계열 → 높은 실현 변동성."""
        vt = VolTargetingOverlay(lookback_days=20, use_downside_only=False)
        dates = pd.bdate_range("2024-01-01", periods=30)
        # 큰 일별 변동 (3%)
        np.random.seed(42)
        values = 100_000_000 * np.exp(np.cumsum(np.random.randn(30) * 0.03))
        pv = pd.Series(values, index=dates)

        vol = vt.compute_realized_vol(pv)
        assert vol > 0.20  # 연율화 20% 초과

    def test_downside_only_mode(self):
        """하방 변동성만 사용 시 양수 수익률 무시."""
        vt_both = VolTargetingOverlay(lookback_days=20, use_downside_only=False)
        vt_down = VolTargetingOverlay(lookback_days=20, use_downside_only=True)

        dates = pd.bdate_range("2024-01-01", periods=30)
        np.random.seed(42)
        values = 100_000_000 * np.exp(np.cumsum(np.random.randn(30) * 0.02))
        pv = pd.Series(values, index=dates)

        vol_both = vt_both.compute_realized_vol(pv)
        vol_down = vt_down.compute_realized_vol(pv)

        # 두 값이 다를 수 있지만 둘 다 유효한 양수
        assert vol_both > 0
        assert vol_down > 0

    def test_downside_only_few_negatives_returns_target(self):
        """음수 수익률이 3개 미만이면 target_vol 반환."""
        vt = VolTargetingOverlay(target_vol=0.12, lookback_days=20, use_downside_only=True)
        dates = pd.bdate_range("2024-01-01", periods=25)
        # 꾸준히 상승하는 시계열 (음수 수익률 거의 없음)
        values = 100_000_000 + np.arange(25) * 100_000
        pv = pd.Series(values, index=dates, dtype=float)

        vol = vt.compute_realized_vol(pv)
        assert vol == 0.12


# ---------------------------------------------------------------------------
# 노출 비율 계산 테스트
# ---------------------------------------------------------------------------

class TestGetExposure:
    """get_exposure 테스트."""

    def test_low_vol_capped_at_max(self):
        """저변동(5%) + 목표 15% → exposure 1.0 (캡핑)."""
        vt = VolTargetingOverlay(target_vol=0.15, max_exposure=1.0)
        dates = pd.bdate_range("2024-01-01", periods=5)
        pv = pd.Series([100_000_000] * 5, index=dates)
        # 데이터 부족 → realized = target_vol → exposure = 1.0
        exposure = vt.get_exposure(pv)
        assert exposure == pytest.approx(1.0)

    def test_high_vol_reduces_exposure(self):
        """고변동(30%) + 목표 15% → exposure 0.5."""
        vt = VolTargetingOverlay(
            target_vol=0.15, lookback_days=20, use_downside_only=False
        )
        dates = pd.bdate_range("2024-01-01", periods=30)
        np.random.seed(100)
        # 의도적으로 높은 변동성 생성
        values = 100_000_000 * np.exp(np.cumsum(np.random.randn(30) * 0.03))
        pv = pd.Series(values, index=dates)

        exposure = vt.get_exposure(pv)
        assert exposure < 1.0  # 축소됨

    def test_never_exceeds_max_exposure(self):
        """exposure가 max_exposure를 초과하지 않음."""
        vt = VolTargetingOverlay(target_vol=0.15, max_exposure=0.8)
        dates = pd.bdate_range("2024-01-01", periods=5)
        pv = pd.Series([100_000_000] * 5, index=dates)
        exposure = vt.get_exposure(pv)
        assert exposure <= 0.8


# ---------------------------------------------------------------------------
# 비중 조절 테스트
# ---------------------------------------------------------------------------

class TestApply:
    """apply 테스트."""

    def test_normal_conditions_unchanged(self):
        """데이터 부족(exposure=1.0) → 비중 그대로."""
        vt = VolTargetingOverlay()
        weights = {"A": 0.3, "B": 0.3, "C": 0.4}
        pv = pd.Series([100_000_000] * 5, index=pd.bdate_range("2024-01-01", periods=5))

        result = vt.apply(weights, pv)
        assert result == weights

    def test_high_vol_scales_weights(self):
        """고변동 시 비중 축소."""
        vt = VolTargetingOverlay(
            target_vol=0.10, lookback_days=20, use_downside_only=False
        )
        dates = pd.bdate_range("2024-01-01", periods=30)
        np.random.seed(100)
        values = 100_000_000 * np.exp(np.cumsum(np.random.randn(30) * 0.04))
        pv = pd.Series(values, index=dates)

        weights = {"A": 0.5, "B": 0.5}
        result = vt.apply(weights, pv)

        # 비중이 축소되어야 함
        for ticker in result:
            assert result[ticker] < weights[ticker]

    def test_empty_weights(self):
        """빈 비중 → 빈 비중."""
        vt = VolTargetingOverlay()
        pv = pd.Series([100_000_000] * 5, index=pd.bdate_range("2024-01-01", periods=5))
        result = vt.apply({}, pv)
        assert result == {}
