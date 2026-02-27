"""PEAD 전략 모듈 테스트.

PEADStrategy의 초기화, 유니버스 필터링, 서프라이즈 계산,
generate_signals 반환값 형식 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.pead import PEADStrategy


# ===================================================================
# 초기화 검증
# ===================================================================

class TestPEADStrategyInit:
    """PEADStrategy 초기화 검증."""

    def test_default_parameters(self):
        """기본 파라미터로 생성된다."""
        s = PEADStrategy()
        assert s.surprise_threshold == 0.1
        assert s.holding_days == 40
        assert s.num_stocks == 10
        assert s.surprise_metric == "operating_income"

    def test_custom_parameters(self):
        """사용자 정의 파라미터가 올바르게 반영된다."""
        s = PEADStrategy(
            surprise_threshold=0.2,
            holding_days=60,
            num_stocks=5,
            min_market_cap=500_000_000_000,
            surprise_metric="net_income",
        )
        assert s.surprise_threshold == 0.2
        assert s.holding_days == 60
        assert s.num_stocks == 5
        assert s.surprise_metric == "net_income"

    def test_invalid_metric_raises(self):
        """지원하지 않는 서프라이즈 지표는 ValueError를 발생시킨다."""
        with pytest.raises(ValueError, match="지원하지 않는 서프라이즈 지표"):
            PEADStrategy(surprise_metric="revenue")

    def test_name_property(self):
        """name 프로퍼티가 올바른 형식을 반환한다."""
        s = PEADStrategy(surprise_metric="net_income", num_stocks=5)
        assert s.name == "PEAD(net_income, top5)"


# ===================================================================
# 유니버스 필터링 테스트
# ===================================================================

class TestPEADFilterUniverse:
    """_filter_universe 메서드 검증."""

    def _make_universe(self, n=20):
        np.random.seed(42)
        return pd.DataFrame({
            "ticker": [f"{i:06d}" for i in range(1, n + 1)],
            "name": [f"종목{i}" for i in range(1, n + 1)],
            "eps": np.random.randint(100, 10000, n),
            "close": np.random.randint(5000, 500000, n),
            "market_cap": np.random.randint(
                10_000_000_000, 5_000_000_000_000, n
            ),
            "volume": np.random.randint(1000, 5_000_000, n),
        })

    def test_market_cap_filter(self):
        """시가총액 하한 필터링이 동작한다."""
        s = PEADStrategy(min_market_cap=1_000_000_000_000)
        universe = self._make_universe()
        filtered = s._filter_universe(universe)
        assert all(filtered["market_cap"] >= 1_000_000_000_000)

    def test_empty_dataframe(self):
        """빈 DataFrame 입력 시 빈 DataFrame 반환."""
        s = PEADStrategy()
        result = s._filter_universe(pd.DataFrame())
        assert result.empty


# ===================================================================
# 서프라이즈 계산 테스트
# ===================================================================

class TestSurpriseCalculation:
    """서프라이즈 계산 메서드 검증."""

    def test_surprise_from_earnings(self):
        """DART 실적 데이터에서 서프라이즈를 계산한다."""
        s = PEADStrategy(surprise_metric="operating_income")
        earnings = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "operating_income_current": [200, 150, 80],
            "operating_income_prev": [100, 100, 100],
        })

        surprise = s._calculate_surprise_from_earnings(earnings)

        assert len(surprise) == 3
        assert surprise["A"] == pytest.approx(1.0)  # 100% 증가
        assert surprise["B"] == pytest.approx(0.5)  # 50% 증가
        assert surprise["C"] == pytest.approx(-0.2)  # 20% 감소

    def test_surprise_from_earnings_net_income(self):
        """순이익 기반 서프라이즈를 계산한다."""
        s = PEADStrategy(surprise_metric="net_income")
        earnings = pd.DataFrame({
            "ticker": ["A", "B"],
            "net_income_current": [300, 50],
            "net_income_prev": [100, 100],
        })

        surprise = s._calculate_surprise_from_earnings(earnings)

        assert surprise["A"] == pytest.approx(2.0)
        assert surprise["B"] == pytest.approx(-0.5)

    def test_surprise_excludes_zero_prev(self):
        """전분기 값이 0인 경우 제외한다."""
        s = PEADStrategy(surprise_metric="operating_income")
        earnings = pd.DataFrame({
            "ticker": ["A", "B"],
            "operating_income_current": [200, 100],
            "operating_income_prev": [0, 100],
        })

        surprise = s._calculate_surprise_from_earnings(earnings)

        assert "A" not in surprise.index
        assert "B" in surprise.index

    def test_surprise_empty_earnings(self):
        """빈 실적 데이터에서 빈 시리즈 반환."""
        s = PEADStrategy()
        result = s._calculate_surprise_from_earnings(pd.DataFrame())
        assert result.empty

    def test_surprise_from_eps_fallback(self):
        """pykrx EPS 데이터로 서프라이즈를 근사한다."""
        s = PEADStrategy()
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "eps": [5000, 3000, -1000],
            "close": [50000, 60000, 10000],
        })

        surprise = s._calculate_surprise_from_eps(fundamentals)

        # EPS > 0인 종목만
        assert len(surprise) == 2
        assert "C" not in surprise.index
        # E/P ratio: A=0.1, B=0.05
        assert surprise["A"] == pytest.approx(0.1)
        assert surprise["B"] == pytest.approx(0.05)


# ===================================================================
# generate_signals 테스트
# ===================================================================

class TestPEADGenerateSignals:
    """generate_signals 메서드 검증."""

    def _make_fundamentals(self, n=20):
        np.random.seed(42)
        return pd.DataFrame({
            "ticker": [f"{i:06d}" for i in range(1, n + 1)],
            "name": [f"종목{i}" for i in range(1, n + 1)],
            "eps": np.random.randint(500, 10000, n),
            "close": np.random.randint(10000, 100000, n),
            "market_cap": np.random.randint(
                200_000_000_000, 5_000_000_000_000, n
            ),
            "volume": np.random.randint(100_000, 5_000_000, n),
        })

    def test_returns_dict(self):
        """반환값이 dict 타입이다."""
        s = PEADStrategy(
            num_stocks=5,
            min_market_cap=0,
            min_volume=0,
            surprise_threshold=0.0,
        )
        data = {"fundamentals": self._make_fundamentals()}
        signals = s.generate_signals("20240102", data)
        assert isinstance(signals, dict)

    def test_weights_sum_to_one(self):
        """비중의 합이 1.0이다."""
        s = PEADStrategy(
            num_stocks=5,
            min_market_cap=0,
            min_volume=0,
            surprise_threshold=0.0,
        )
        data = {"fundamentals": self._make_fundamentals()}
        signals = s.generate_signals("20240102", data)
        if signals:
            total = sum(signals.values())
            assert abs(total - 1.0) < 1e-9

    def test_num_stocks_respected(self):
        """num_stocks 이하의 종목이 선택된다."""
        s = PEADStrategy(
            num_stocks=3,
            min_market_cap=0,
            min_volume=0,
            surprise_threshold=0.0,
        )
        data = {"fundamentals": self._make_fundamentals(30)}
        signals = s.generate_signals("20240102", data)
        assert len(signals) <= 3

    def test_empty_fundamentals_returns_empty(self):
        """펀더멘탈 데이터 없으면 빈 dict 반환."""
        s = PEADStrategy()
        signals = s.generate_signals("20240102", {"fundamentals": pd.DataFrame()})
        assert signals == {}

    def test_no_data_returns_empty(self):
        """data가 비어있으면 빈 dict 반환."""
        s = PEADStrategy()
        signals = s.generate_signals("20240102", {})
        assert signals == {}

    def test_with_earnings_data(self):
        """DART 실적 데이터로 시그널을 생성한다."""
        s = PEADStrategy(
            num_stocks=2,
            min_market_cap=0,
            min_volume=0,
            surprise_threshold=0.1,
            surprise_metric="operating_income",
        )

        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "eps": [5000, 3000, 2000, 1000],
            "close": [50000, 30000, 20000, 10000],
            "market_cap": [1e12, 1e12, 1e12, 1e12],
            "volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
        })

        earnings = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "operating_income_current": [200, 150, 80, 120],
            "operating_income_prev": [100, 100, 100, 100],
        })

        data = {"fundamentals": fundamentals, "earnings": earnings}
        signals = s.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        # A(100%), B(50%) 가 상위 2개
        if signals:
            assert "A" in signals
            assert "B" in signals
            assert len(signals) == 2

    def test_threshold_filters_correctly(self):
        """서프라이즈 임계값이 정확하게 적용된다."""
        s = PEADStrategy(
            num_stocks=10,
            min_market_cap=0,
            min_volume=0,
            surprise_threshold=0.5,
        )

        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "eps": [5000, 3000, 2000],
            "close": [50000, 30000, 20000],
            "market_cap": [1e12, 1e12, 1e12],
            "volume": [1_000_000, 1_000_000, 1_000_000],
        })

        earnings = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "operating_income_current": [200, 140, 80],
            "operating_income_prev": [100, 100, 100],
        })

        data = {"fundamentals": fundamentals, "earnings": earnings}
        signals = s.generate_signals("20240102", data)

        # A(100%) >= 0.5, B(40%) < 0.5, C(-20%) < 0.5
        assert "A" in signals
        assert "B" not in signals
        assert "C" not in signals
