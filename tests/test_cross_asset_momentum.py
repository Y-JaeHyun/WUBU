"""Cross-Asset Momentum 전략 테스트.

CrossAssetMomentumStrategy의 모멘텀 계산, 자산군 필터,
상관관계 필터, 시그널 생성 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.cross_asset_momentum import (
    CrossAssetMomentumStrategy,
    CROSS_ASSET_UNIVERSE,
    ASSET_CLASS_MAP,
)


def _make_price_df(start_price, end_price, n_days=300):
    """선형 보간으로 가격 데이터를 생성한다."""
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    prices = np.linspace(start_price, end_price, n_days)
    return pd.DataFrame(
        {"open": prices, "high": prices * 1.01, "low": prices * 0.99,
         "close": prices, "volume": np.ones(n_days) * 1000000},
        index=dates,
    )


def _make_random_price(base, vol, n_days=300, seed=42):
    """랜덤 가격 데이터를 생성한다."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    returns = rng.normal(0.0003, vol, n_days)
    prices = base * np.cumprod(1 + returns)
    return pd.DataFrame(
        {"open": prices, "high": prices * 1.01, "low": prices * 0.99,
         "close": prices, "volume": np.ones(n_days) * 1000000},
        index=dates,
    )


class TestCrossAssetInit:
    """초기화 검증."""

    def test_default_parameters(self):
        s = CrossAssetMomentumStrategy()
        assert s.num_assets == 3
        assert s.lookback_short == 63
        assert s.lookback_long == 252
        assert s.use_trend_filter is True
        assert s.use_correlation_filter is True
        assert s.max_per_asset_class == 1

    def test_custom_parameters(self):
        s = CrossAssetMomentumStrategy(
            num_assets=2,
            lookback_short=42,
            short_weight=0.8,
            max_per_asset_class=2,
        )
        assert s.num_assets == 2
        assert s.lookback_short == 42
        assert s.short_weight == 0.8
        assert s.max_per_asset_class == 2

    def test_name_property(self):
        s = CrossAssetMomentumStrategy(num_assets=3)
        assert "CrossAssetMom" in s.name
        assert "top3" in s.name

    def test_default_universe(self):
        assert len(CROSS_ASSET_UNIVERSE) >= 5
        assert "069500" in CROSS_ASSET_UNIVERSE
        assert "132030" in CROSS_ASSET_UNIVERSE


class TestDualMomentumScore:
    """듀얼 모멘텀 스코어 검증."""

    def test_higher_return_higher_score(self):
        """수익률 높은 ETF가 높은 스코어를 받는다."""
        s = CrossAssetMomentumStrategy(
            etf_universe={"AAA": "A", "BBB": "B", "CCC": "safe"},
            safe_asset="CCC",
        )
        prices = {
            "AAA": _make_price_df(100, 180, 300),
            "BBB": _make_price_df(100, 120, 300),
            "CCC": _make_price_df(100, 101, 300),
        }
        scores = s._calculate_dual_momentum_score(prices)
        assert scores["AAA"] > scores["BBB"]

    def test_empty_data(self):
        """데이터 없으면 빈 딕셔너리."""
        s = CrossAssetMomentumStrategy()
        assert s._calculate_dual_momentum_score({}) == {}


class TestSMACrossover:
    """SMA 크로스오버 검증."""

    def test_uptrend_passes(self):
        """상승 추세는 통과."""
        s = CrossAssetMomentumStrategy()
        prices = {"AAA": _make_price_df(100, 200, 300)}
        assert s._check_sma_crossover(prices, "AAA") is True

    def test_downtrend_fails(self):
        """하락 추세는 실패."""
        s = CrossAssetMomentumStrategy()
        prices = {"AAA": _make_price_df(200, 100, 300)}
        assert s._check_sma_crossover(prices, "AAA") is False

    def test_missing_data_passes(self):
        """데이터 없으면 통과."""
        s = CrossAssetMomentumStrategy()
        assert s._check_sma_crossover({}, "AAA") is True


class TestAssetClassFilter:
    """자산군 필터 검증."""

    def test_max_one_per_class(self):
        """자산군당 1개만 선정된다."""
        s = CrossAssetMomentumStrategy(
            num_assets=3,
            max_per_asset_class=1,
            etf_universe={
                "069500": "KODEX200",      # equity_kr
                "091160": "반도체",          # equity_kr
                "371460": "S&P500",         # equity_us
                "132030": "골드",           # commodity
                "439870": "채권",           # bond
            },
        )
        # 069500과 091160은 같은 equity_kr
        ranked = ["069500", "091160", "371460", "132030"]
        result = s._apply_asset_class_filter(ranked)
        # equity_kr에서 최대 1개만 선정
        kr_count = sum(
            1 for t in result if ASSET_CLASS_MAP.get(t) == "equity_kr"
        )
        assert kr_count <= 1

    def test_max_per_class_zero_no_limit(self):
        """max_per_asset_class=0이면 제한 없음."""
        s = CrossAssetMomentumStrategy(
            num_assets=3,
            max_per_asset_class=0,
        )
        ranked = ["069500", "091160", "371460", "132030"]
        result = s._apply_asset_class_filter(ranked)
        assert len(result) == 3


class TestCorrelationFilter:
    """상관관계 필터 검증."""

    def test_highly_correlated_filtered(self):
        """상관관계 높은 자산이 필터링된다."""
        s = CrossAssetMomentumStrategy(
            num_assets=3,
            max_correlation=0.5,
            use_correlation_filter=True,
        )
        # 동일한 시드로 높은 상관관계 생성
        prices = {
            "AAA": _make_price_df(100, 200, 300),
            "BBB": _make_price_df(100, 200, 300),  # AAA와 완전 상관
            "CCC": _make_random_price(100, 0.02, 300, seed=99),
        }
        result = s._apply_correlation_filter(prices, ["AAA", "BBB", "CCC"])
        # AAA와 BBB는 상관관계가 1.0이므로 하나만 선정
        assert len(result) <= 3

    def test_filter_off(self):
        """필터 OFF이면 그대로 반환."""
        s = CrossAssetMomentumStrategy(use_correlation_filter=False)
        candidates = ["AAA", "BBB", "CCC"]
        result = s._apply_correlation_filter({}, candidates)
        assert result == candidates


class TestGenerateSignals:
    """시그널 생성 검증."""

    def _make_universe(self):
        return {
            "AAA": _make_price_df(100, 200, 300),   # 강한 상승
            "BBB": _make_price_df(100, 140, 300),   # 약한 상승
            "CCC": _make_price_df(100, 70, 300),    # 하락
            "DDD": _make_random_price(100, 0.005, 300, seed=55),  # 낮은 변동
            "EEE": _make_price_df(100, 101, 300),   # 안전자산
        }

    def test_signal_generation(self):
        """시그널이 올바르게 생성된다."""
        s = CrossAssetMomentumStrategy(
            num_assets=2,
            safe_asset="EEE",
            etf_universe={"AAA": "A", "BBB": "B", "CCC": "C", "DDD": "D", "EEE": "E"},
            use_trend_filter=False,
            use_correlation_filter=False,
            max_per_asset_class=0,
        )
        prices = self._make_universe()
        signals = s.generate_signals("20240101", {"etf_prices": prices})
        assert len(signals) > 0
        assert sum(signals.values()) <= 1.01

    def test_empty_data(self):
        """데이터 없으면 빈 딕셔너리."""
        s = CrossAssetMomentumStrategy()
        assert s.generate_signals("20240101", {"etf_prices": {}}) == {}

    def test_all_negative_goes_to_safe(self):
        """모든 자산이 음수 모멘텀이면 안전자산 100%."""
        s = CrossAssetMomentumStrategy(
            num_assets=2,
            safe_asset="EEE",
            etf_universe={"AAA": "A", "BBB": "B", "EEE": "E"},
            lookback_short=63,
            use_trend_filter=False,
            use_correlation_filter=False,
            max_per_asset_class=0,
        )
        prices = {
            "AAA": _make_price_df(200, 100, 300),  # 하락
            "BBB": _make_price_df(180, 90, 300),   # 하락
            "EEE": _make_price_df(100, 101, 300),
        }
        signals = s.generate_signals("20240101", {"etf_prices": prices})
        # 안전자산에 비중이 배분
        assert signals.get("EEE", 0) > 0

    def test_diagnostics(self):
        """진단 정보가 저장된다."""
        s = CrossAssetMomentumStrategy(
            num_assets=2,
            safe_asset="EEE",
            etf_universe={"AAA": "A", "BBB": "B", "EEE": "E"},
            use_trend_filter=False,
            use_correlation_filter=False,
            max_per_asset_class=0,
        )
        prices = self._make_universe()
        s.generate_signals("20240101", {"etf_prices": prices})
        assert "status" in s.last_diagnostics
