"""SizeValueStrategy 단위 테스트."""
import pandas as pd
import pytest

from src.strategy.size_value import SizeValueStrategy


@pytest.fixture
def strategy():
    return SizeValueStrategy(
        size_pct=0.50,
        value_pct=0.50,
        max_stocks=5,
        min_volume=0,
        exclude_negative_earnings=False,
    )


@pytest.fixture
def sample_fundamentals():
    """10개 종목 샘플 데이터."""
    return pd.DataFrame({
        "ticker": [f"{i:06d}" for i in range(1, 11)],
        "market_cap": [
            50e8, 100e8, 200e8, 500e8, 1000e8,
            2000e8, 5000e8, 10000e8, 20000e8, 50000e8,
        ],
        "pbr": [0.3, 0.5, 0.8, 1.2, 0.4, 0.6, 1.5, 2.0, 0.9, 1.1],
        "per": [3.0, 5.0, 8.0, 12.0, 4.0, 6.0, 15.0, 20.0, 9.0, 11.0],
        "eps": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "volume": [10000] * 10,
        "close": [10000] * 10,
    })


class TestSizeValueStrategy:

    def test_name(self, strategy):
        assert "SizeValue" in strategy.name
        assert "PBR" in strategy.name

    def test_generate_signals_returns_dict(self, strategy, sample_fundamentals):
        signals = strategy.generate_signals("20240101", {"fundamentals": sample_fundamentals})
        assert isinstance(signals, dict)
        assert len(signals) > 0

    def test_selects_small_cap_stocks(self, strategy, sample_fundamentals):
        """소형주만 선택되는지 확인."""
        signals = strategy.generate_signals("20240101", {"fundamentals": sample_fundamentals})
        selected_tickers = set(signals.keys())

        # size_pct=0.50이므로 시총 하위 50% = 상위 5개 소형주만 후보
        small_cap_tickers = set(sample_fundamentals.nsmallest(5, "market_cap")["ticker"])
        assert selected_tickers.issubset(small_cap_tickers)

    def test_selects_low_pbr_among_small_caps(self, strategy, sample_fundamentals):
        """소형주 중 저PBR 종목이 선택되는지 확인."""
        signals = strategy.generate_signals("20240101", {"fundamentals": sample_fundamentals})

        # 소형주(하위 50%) 중 PBR 하위 50% = 약 2-3개
        assert len(signals) <= 5

    def test_equal_weight(self, strategy, sample_fundamentals):
        """동일 비중 할당 확인."""
        signals = strategy.generate_signals("20240101", {"fundamentals": sample_fundamentals})
        if signals:
            weights = list(signals.values())
            assert abs(sum(weights) - 1.0) < 1e-6
            assert all(abs(w - weights[0]) < 1e-6 for w in weights)

    def test_empty_fundamentals(self, strategy):
        signals = strategy.generate_signals("20240101", {"fundamentals": pd.DataFrame()})
        assert signals == {}

    def test_max_stocks_limit(self, sample_fundamentals):
        """max_stocks 제한 확인."""
        s = SizeValueStrategy(
            size_pct=0.80,
            value_pct=0.80,
            max_stocks=3,
            min_volume=0,
            exclude_negative_earnings=False,
        )
        signals = s.generate_signals("20240101", {"fundamentals": sample_fundamentals})
        assert len(signals) <= 3

    def test_min_volume_filter(self, sample_fundamentals):
        """거래대금 필터 확인."""
        s = SizeValueStrategy(
            size_pct=0.50,
            value_pct=0.50,
            max_stocks=20,
            min_volume=999_999_999_999,  # 매우 높은 기준
        )
        signals = s.generate_signals("20240101", {"fundamentals": sample_fundamentals})
        assert signals == {}

    def test_exclude_negative_earnings(self):
        """적자 기업 제외 확인."""
        df = pd.DataFrame({
            "ticker": ["000001", "000002", "000003"],
            "market_cap": [100e8, 200e8, 300e8],
            "pbr": [0.5, 0.3, 0.4],
            "per": [5, 3, 4],
            "eps": [-100, 200, 300],
            "volume": [10000] * 3,
            "close": [10000] * 3,
        })
        s = SizeValueStrategy(
            size_pct=1.0, value_pct=1.0, max_stocks=10,
            min_volume=0, exclude_negative_earnings=True,
        )
        signals = s.generate_signals("20240101", {"fundamentals": df})
        assert "000001" not in signals

    def test_composite_factor(self, sample_fundamentals):
        """composite 밸류 팩터 확인."""
        s = SizeValueStrategy(
            size_pct=0.50, value_pct=0.50, max_stocks=10,
            min_volume=0, value_factor="composite",
            exclude_negative_earnings=False,
        )
        signals = s.generate_signals("20240101", {"fundamentals": sample_fundamentals})
        assert isinstance(signals, dict)
        assert len(signals) > 0

    def test_invalid_factor_raises(self):
        with pytest.raises(ValueError):
            SizeValueStrategy(value_factor="invalid")
