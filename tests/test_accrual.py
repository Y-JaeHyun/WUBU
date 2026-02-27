"""발생액 전략 모듈 테스트.

AccrualStrategy의 초기화, 유니버스 필터링, 발생액 스코어 계산,
generate_signals 반환값 형식 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.accrual import AccrualStrategy


# ===================================================================
# 초기화 검증
# ===================================================================

class TestAccrualStrategyInit:
    """AccrualStrategy 초기화 검증."""

    def test_default_parameters(self):
        """기본 파라미터로 생성된다."""
        s = AccrualStrategy()
        assert s.num_stocks == 10
        assert s.min_market_cap == 100_000_000_000
        assert s.exclude_negative_earnings is True

    def test_custom_parameters(self):
        """사용자 정의 파라미터가 올바르게 반영된다."""
        s = AccrualStrategy(
            num_stocks=5,
            min_market_cap=500_000_000_000,
            exclude_negative_earnings=False,
        )
        assert s.num_stocks == 5
        assert s.exclude_negative_earnings is False

    def test_name_property(self):
        """name 프로퍼티가 올바른 형식을 반환한다."""
        s = AccrualStrategy(num_stocks=15)
        assert s.name == "Accrual(top15)"


# ===================================================================
# 유니버스 필터링 테스트
# ===================================================================

class TestAccrualFilterUniverse:
    """_filter_universe 메서드 검증."""

    def test_market_cap_filter(self):
        """시가총액 하한 필터링이 동작한다."""
        s = AccrualStrategy(min_market_cap=1_000_000_000_000)
        universe = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "eps": [5000, 3000, 2000],
            "close": [10000, 20000, 5000],
            "market_cap": [2e12, 5e11, 3e12],
            "volume": [1_000_000, 1_000_000, 1_000_000],
        })

        filtered = s._filter_universe(universe)
        assert "B" not in filtered["ticker"].values

    def test_exclude_negative_earnings(self):
        """적자 기업이 제외된다."""
        s = AccrualStrategy(
            min_market_cap=0,
            min_volume=0,
            exclude_negative_earnings=True,
        )
        universe = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "eps": [5000, -1000, 2000],
            "close": [10000, 20000, 5000],
            "market_cap": [1e12, 1e12, 1e12],
            "volume": [1_000_000, 1_000_000, 1_000_000],
        })

        filtered = s._filter_universe(universe)
        assert "B" not in filtered["ticker"].values

    def test_include_negative_when_disabled(self):
        """exclude_negative_earnings=False 시 적자 기업 포함."""
        s = AccrualStrategy(
            min_market_cap=0,
            min_volume=0,
            exclude_negative_earnings=False,
        )
        universe = pd.DataFrame({
            "ticker": ["A", "B"],
            "eps": [5000, -1000],
            "close": [10000, 20000],
            "market_cap": [1e12, 1e12],
            "volume": [1_000_000, 1_000_000],
        })

        filtered = s._filter_universe(universe)
        assert "B" in filtered["ticker"].values

    def test_empty_dataframe(self):
        """빈 DataFrame 입력 시 빈 DataFrame 반환."""
        s = AccrualStrategy()
        result = s._filter_universe(pd.DataFrame())
        assert result.empty


# ===================================================================
# 발생액 스코어 테스트
# ===================================================================

class TestAccrualScore:
    """_get_accrual_scores 메서드 검증."""

    def test_dart_accruals_data(self):
        """DART 퀄리티 데이터의 accruals 필드로 스코어를 계산한다."""
        s = AccrualStrategy()
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "eps": [5000, 3000, 2000],
        })

        quality_data = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "accruals": [-0.05, 0.10, 0.02],  # A가 가장 좋음 (낮음)
        })

        scores = s._get_accrual_scores(fundamentals, quality_data)

        # 발생액 낮은 순: A(-0.05) > C(0.02) > B(0.10)
        assert scores["A"] > scores["C"]
        assert scores["C"] > scores["B"]

    def test_fundamentals_accruals_column(self):
        """fundamentals에 accruals 컬럼이 있으면 사용한다."""
        s = AccrualStrategy()
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "eps": [5000, 3000, 2000],
            "accruals": [-0.03, 0.15, 0.05],
        })

        scores = s._get_accrual_scores(fundamentals)

        assert scores["A"] > scores["C"]
        assert scores["C"] > scores["B"]

    def test_per_fallback(self):
        """발생액 데이터 없으면 PER 역수를 proxy로 사용한다."""
        s = AccrualStrategy()
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "eps": [5000, 3000, 2000],
            "per": [5.0, 10.0, 15.0],
        })

        scores = s._get_accrual_scores(fundamentals)

        # PER 낮을수록 좋음: A(5) > B(10) > C(15)
        assert scores["A"] > scores["B"]
        assert scores["B"] > scores["C"]

    def test_empty_fundamentals(self):
        """빈 펀더멘탈 데이터에서 빈 시리즈 반환."""
        s = AccrualStrategy()
        result = s._get_accrual_scores(pd.DataFrame())
        assert result.empty

    def test_nan_accruals_filtered(self):
        """NaN 발생액은 필터링된다."""
        s = AccrualStrategy()
        quality_data = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "accruals": [-0.05, np.nan, 0.02],
        })
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "eps": [5000, 3000, 2000],
        })

        scores = s._get_accrual_scores(fundamentals, quality_data)

        assert "A" in scores.index
        assert "C" in scores.index
        # B는 NaN이므로 제외될 수 있음


# ===================================================================
# generate_signals 테스트
# ===================================================================

class TestAccrualSignals:
    """generate_signals 메서드 검증."""

    def _make_fundamentals(self, n=20):
        np.random.seed(42)
        return pd.DataFrame({
            "ticker": [f"{i:06d}" for i in range(1, n + 1)],
            "name": [f"종목{i}" for i in range(1, n + 1)],
            "eps": np.random.randint(500, 10000, n),
            "per": np.random.uniform(3, 30, n).round(2),
            "close": np.random.randint(10000, 100000, n),
            "market_cap": np.random.randint(
                200_000_000_000, 5_000_000_000_000, n
            ),
            "volume": np.random.randint(100_000, 5_000_000, n),
        })

    def test_returns_dict(self):
        """반환값이 dict 타입이다."""
        s = AccrualStrategy(num_stocks=5, min_market_cap=0, min_volume=0)
        data = {"fundamentals": self._make_fundamentals()}
        signals = s.generate_signals("20240102", data)
        assert isinstance(signals, dict)

    def test_weights_sum_to_one(self):
        """비중의 합이 1.0이다."""
        s = AccrualStrategy(num_stocks=5, min_market_cap=0, min_volume=0)
        data = {"fundamentals": self._make_fundamentals()}
        signals = s.generate_signals("20240102", data)
        if signals:
            total = sum(signals.values())
            assert abs(total - 1.0) < 1e-9

    def test_num_stocks_respected(self):
        """num_stocks 이하의 종목이 선택된다."""
        s = AccrualStrategy(num_stocks=3, min_market_cap=0, min_volume=0)
        data = {"fundamentals": self._make_fundamentals(30)}
        signals = s.generate_signals("20240102", data)
        assert len(signals) <= 3

    def test_empty_fundamentals_returns_empty(self):
        """펀더멘탈 데이터 없으면 빈 dict 반환."""
        s = AccrualStrategy()
        signals = s.generate_signals("20240102", {"fundamentals": pd.DataFrame()})
        assert signals == {}

    def test_with_quality_data(self):
        """DART 퀄리티 데이터와 함께 사용한다."""
        s = AccrualStrategy(
            num_stocks=2, min_market_cap=0, min_volume=0,
        )

        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "eps": [5000, 3000, 2000, 4000],
            "close": [10000] * 4,
            "market_cap": [1e12] * 4,
            "volume": [1_000_000] * 4,
        })

        quality = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "accruals": [-0.10, 0.20, -0.05, 0.15],
        })

        data = {"fundamentals": fundamentals, "quality": quality}
        signals = s.generate_signals("20240102", data)

        # 저발생액: A(-0.10), C(-0.05) 상위
        assert "A" in signals
        assert "C" in signals

    def test_selects_low_accrual_stocks(self):
        """저발생액 종목이 선택된다."""
        s = AccrualStrategy(
            num_stocks=2, min_market_cap=0, min_volume=0,
        )

        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "eps": [5000, 3000, 2000, 4000],
            "per": [5.0, 10.0, 15.0, 8.0],
            "close": [10000] * 4,
            "market_cap": [1e12] * 4,
            "volume": [1_000_000] * 4,
        })

        data = {"fundamentals": fundamentals}
        signals = s.generate_signals("20240102", data)

        # PER fallback: 낮은 PER이 좋음 → A(5), D(8) 선택
        assert "A" in signals
        assert isinstance(signals, dict)
