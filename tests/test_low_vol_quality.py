"""저변동성+품질 결합 전략 모듈 테스트.

LowVolQualityStrategy의 초기화, 변동성 계산, 품질 스코어,
generate_signals 반환값 형식 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.low_vol_quality import LowVolQualityStrategy


# ===================================================================
# 초기화 검증
# ===================================================================

class TestLowVolQualityInit:
    """LowVolQualityStrategy 초기화 검증."""

    def test_default_parameters(self):
        """기본 파라미터로 생성된다."""
        s = LowVolQualityStrategy()
        assert s.vol_period == 60
        assert s.vol_percentile == 30
        assert s.num_stocks == 10
        assert s.roe_weight == 0.5
        assert s.gpa_weight == 0.5

    def test_custom_parameters(self):
        """사용자 정의 파라미터가 올바르게 반영된다."""
        s = LowVolQualityStrategy(
            vol_period=90,
            vol_percentile=20,
            num_stocks=5,
            roe_weight=0.7,
            gpa_weight=0.3,
        )
        assert s.vol_period == 90
        assert s.vol_percentile == 20
        assert s.num_stocks == 5

    def test_name_property(self):
        """name 프로퍼티가 올바른 형식을 반환한다."""
        s = LowVolQualityStrategy(vol_percentile=25, num_stocks=15)
        assert s.name == "LowVolQuality(vol25pct, top15)"


# ===================================================================
# 변동성 계산 테스트
# ===================================================================

class TestVolatilityCalculation:
    """_calculate_volatility 메서드 검증."""

    def _make_price_data(self, ticker, n_days=100, vol_level=0.02):
        """테스트용 가격 데이터를 생성한다."""
        np.random.seed(hash(ticker) % 2**31)
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        returns = np.random.normal(0, vol_level, n_days)
        prices = 10000 * np.cumprod(1 + returns)
        return pd.DataFrame(
            {"close": prices, "volume": np.random.randint(100000, 1000000, n_days)},
            index=dates,
        )

    def test_calculate_volatility(self):
        """변동성 계산이 동작한다."""
        s = LowVolQualityStrategy(vol_period=60)

        prices = {
            "A": self._make_price_data("A", vol_level=0.01),  # 저변동
            "B": self._make_price_data("B", vol_level=0.05),  # 고변동
        }

        vol = s._calculate_volatility(prices, {"A", "B"})

        assert len(vol) == 2
        assert vol["A"] < vol["B"]  # A가 더 낮은 변동성

    def test_insufficient_data_excluded(self):
        """데이터가 부족한 종목은 제외된다."""
        s = LowVolQualityStrategy(vol_period=60)

        prices = {
            "A": self._make_price_data("A", n_days=100),
            "B": self._make_price_data("B", n_days=5),  # 데이터 부족
        }

        vol = s._calculate_volatility(prices, {"A", "B"})

        assert "A" in vol.index
        assert "B" not in vol.index

    def test_empty_prices(self):
        """빈 가격 데이터에서 빈 시리즈 반환."""
        s = LowVolQualityStrategy()
        vol = s._calculate_volatility({}, {"A", "B"})
        assert vol.empty

    def test_missing_ticker_ignored(self):
        """prices에 없는 종목은 무시된다."""
        s = LowVolQualityStrategy(vol_period=60)
        prices = {"A": self._make_price_data("A")}
        vol = s._calculate_volatility(prices, {"A", "B", "C"})
        assert len(vol) == 1
        assert "A" in vol.index


# ===================================================================
# 품질 스코어 테스트
# ===================================================================

class TestQualityScore:
    """_calculate_quality_score 메서드 검증."""

    def test_roe_scoring(self):
        """ROE 기반 스코어가 계산된다."""
        s = LowVolQualityStrategy(roe_weight=1.0, gpa_weight=0.0)
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "roe": [20.0, 10.0, 30.0],
        })

        scores = s._calculate_quality_score(fundamentals)

        # ROE 높은 순: C > A > B
        assert scores["C"] > scores["A"]
        assert scores["A"] > scores["B"]

    def test_estimated_roe_from_eps_bps(self):
        """EPS/BPS에서 ROE를 추정한다."""
        s = LowVolQualityStrategy(roe_weight=1.0, gpa_weight=0.0)
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B"],
            "eps": [5000, 2000],
            "bps": [50000, 40000],
        })

        scores = s._calculate_quality_score(fundamentals)

        # A: ROE = 10%, B: ROE = 5%
        assert scores["A"] > scores["B"]

    def test_gpa_scoring(self):
        """GP/A 기반 스코어가 계산된다."""
        s = LowVolQualityStrategy(roe_weight=0.0, gpa_weight=1.0)
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "gp_over_assets": [0.3, 0.1, 0.2],
        })

        scores = s._calculate_quality_score(fundamentals)

        assert scores["A"] > scores["C"]
        assert scores["C"] > scores["B"]

    def test_empty_fundamentals(self):
        """빈 펀더멘탈에서 빈 시리즈 반환."""
        s = LowVolQualityStrategy()
        result = s._calculate_quality_score(pd.DataFrame())
        assert result.empty


# ===================================================================
# generate_signals 테스트
# ===================================================================

class TestLowVolQualitySignals:
    """generate_signals 메서드 검증."""

    def _make_price_data(self, ticker, n_days=100, vol_level=0.02):
        np.random.seed(hash(ticker) % 2**31)
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        returns = np.random.normal(0, vol_level, n_days)
        prices = 10000 * np.cumprod(1 + returns)
        return pd.DataFrame(
            {"close": prices, "volume": np.random.randint(100000, 1000000, n_days)},
            index=dates,
        )

    def _make_data(self):
        tickers = [f"{i:06d}" for i in range(1, 21)]
        fundamentals = pd.DataFrame({
            "ticker": tickers,
            "name": [f"종목{i}" for i in range(1, 21)],
            "eps": np.random.randint(500, 10000, 20),
            "bps": np.random.randint(10000, 100000, 20),
            "close": np.random.randint(10000, 100000, 20),
            "market_cap": [1e12] * 20,
            "volume": [1_000_000] * 20,
        })

        prices = {}
        for i, ticker in enumerate(tickers):
            vol_level = 0.01 + i * 0.002  # 점점 높은 변동성
            prices[ticker] = self._make_price_data(ticker, vol_level=vol_level)

        return {"fundamentals": fundamentals, "prices": prices}

    def test_returns_dict(self):
        """반환값이 dict 타입이다."""
        np.random.seed(42)
        s = LowVolQualityStrategy(
            num_stocks=5, min_market_cap=0, min_volume=0,
        )
        data = self._make_data()
        signals = s.generate_signals("20240102", data)
        assert isinstance(signals, dict)

    def test_weights_sum_to_one(self):
        """비중의 합이 1.0이다."""
        np.random.seed(42)
        s = LowVolQualityStrategy(
            num_stocks=5, min_market_cap=0, min_volume=0,
        )
        data = self._make_data()
        signals = s.generate_signals("20240102", data)
        if signals:
            total = sum(signals.values())
            assert abs(total - 1.0) < 1e-9

    def test_num_stocks_respected(self):
        """num_stocks 이하의 종목이 선택된다."""
        np.random.seed(42)
        s = LowVolQualityStrategy(
            num_stocks=3, min_market_cap=0, min_volume=0,
        )
        data = self._make_data()
        signals = s.generate_signals("20240102", data)
        assert len(signals) <= 3

    def test_empty_fundamentals_returns_empty(self):
        """펀더멘탈 데이터 없으면 빈 dict 반환."""
        s = LowVolQualityStrategy()
        signals = s.generate_signals("20240102", {"fundamentals": pd.DataFrame()})
        assert signals == {}

    def test_no_prices_uses_quality_only(self):
        """가격 데이터 없으면 품질만으로 선정한다."""
        s = LowVolQualityStrategy(
            num_stocks=3, min_market_cap=0, min_volume=0,
        )
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "eps": [5000, 3000, 7000, 1000],
            "bps": [50000, 30000, 70000, 10000],
            "close": [10000] * 4,
            "market_cap": [1e12] * 4,
            "volume": [1_000_000] * 4,
        })

        signals = s.generate_signals("20240102", {"fundamentals": fundamentals})
        assert isinstance(signals, dict)
        assert len(signals) <= 3

    def test_low_vol_stocks_preferred(self):
        """저변동성 종목이 우선 선택된다."""
        np.random.seed(42)
        s = LowVolQualityStrategy(
            vol_period=60,
            vol_percentile=50,
            num_stocks=2,
            min_market_cap=0,
            min_volume=0,
            roe_weight=1.0,
            gpa_weight=0.0,
        )

        fundamentals = pd.DataFrame({
            "ticker": ["LOW1", "LOW2", "HIGH1", "HIGH2"],
            "eps": [5000, 4000, 6000, 7000],
            "bps": [50000, 40000, 60000, 70000],
            "close": [10000] * 4,
            "market_cap": [1e12] * 4,
            "volume": [1_000_000] * 4,
        })

        prices = {
            "LOW1": self._make_price_data("LOW1", vol_level=0.005),
            "LOW2": self._make_price_data("LOW2", vol_level=0.008),
            "HIGH1": self._make_price_data("HIGH1", vol_level=0.05),
            "HIGH2": self._make_price_data("HIGH2", vol_level=0.06),
        }

        signals = s.generate_signals(
            "20240102", {"fundamentals": fundamentals, "prices": prices}
        )

        # 저변동성 종목(LOW1, LOW2)이 선택되어야 함
        for ticker in signals:
            assert ticker.startswith("LOW"), (
                f"고변동성 종목 {ticker}이 선택됨"
            )
