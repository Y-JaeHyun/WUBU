"""주주환원 전략 모듈 테스트.

ShareholderYieldStrategy의 초기화, 유니버스 필터링, 스코어 계산,
generate_signals 반환값 형식 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.shareholder_yield import ShareholderYieldStrategy


# ===================================================================
# 초기화 검증
# ===================================================================

class TestShareholderYieldInit:
    """ShareholderYieldStrategy 초기화 검증."""

    def test_default_parameters(self):
        """기본 파라미터로 생성된다."""
        s = ShareholderYieldStrategy()
        assert s.num_stocks == 10
        assert s.div_weight == 0.6
        assert s.buyback_weight == 0.4
        assert s.min_div_yield == 0.0

    def test_custom_parameters(self):
        """사용자 정의 파라미터가 올바르게 반영된다."""
        s = ShareholderYieldStrategy(
            num_stocks=5,
            div_weight=0.7,
            buyback_weight=0.3,
            min_div_yield=1.0,
        )
        assert s.num_stocks == 5
        assert s.div_weight == 0.7
        assert s.buyback_weight == 0.3
        assert s.min_div_yield == 1.0

    def test_name_property(self):
        """name 프로퍼티가 올바른 형식을 반환한다."""
        s = ShareholderYieldStrategy(num_stocks=15)
        assert s.name == "ShareholderYield(top15)"


# ===================================================================
# 유니버스 필터링 테스트
# ===================================================================

class TestShareholderYieldFilter:
    """_filter_universe 메서드 검증."""

    def test_market_cap_filter(self):
        """시가총액 하한 필터링이 동작한다."""
        s = ShareholderYieldStrategy(min_market_cap=1_000_000_000_000)
        universe = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "div_yield": [3.0, 2.0, 1.0],
            "close": [10000, 20000, 5000],
            "market_cap": [2e12, 5e11, 3e12],
            "volume": [1_000_000, 1_000_000, 1_000_000],
        })

        filtered = s._filter_universe(universe)
        assert "B" not in filtered["ticker"].values  # 5000억 < 1조

    def test_min_div_yield_filter(self):
        """최소 배당수익률 필터가 동작한다."""
        s = ShareholderYieldStrategy(
            min_market_cap=0,
            min_volume=0,
            min_div_yield=1.5,
        )
        universe = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "div_yield": [3.0, 1.0, 2.0],
            "close": [10000, 20000, 5000],
            "market_cap": [1e12, 1e12, 1e12],
            "volume": [1_000_000, 1_000_000, 1_000_000],
        })

        filtered = s._filter_universe(universe)
        assert "B" not in filtered["ticker"].values  # 1.0 < 1.5
        assert len(filtered) == 2

    def test_empty_dataframe(self):
        """빈 DataFrame 입력 시 빈 DataFrame 반환."""
        s = ShareholderYieldStrategy()
        result = s._filter_universe(pd.DataFrame())
        assert result.empty


# ===================================================================
# 스코어 계산 테스트
# ===================================================================

class TestShareholderYieldScore:
    """_calculate_shareholder_yield_score 메서드 검증."""

    def test_div_yield_only_scoring(self):
        """배당수익률 단독 스코어링이 동작한다."""
        s = ShareholderYieldStrategy()
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "div_yield": [4.0, 2.0, 1.0, 3.0],
        })

        scores = s._calculate_shareholder_yield_score(fundamentals)

        # 배당수익률 높은 순: A > D > B > C
        assert scores["A"] > scores["D"]
        assert scores["D"] > scores["B"]
        assert scores["B"] > scores["C"]

    def test_with_buyback_data(self):
        """자사주매입 데이터가 결합된다."""
        s = ShareholderYieldStrategy(div_weight=0.6, buyback_weight=0.4)
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "div_yield": [4.0, 2.0, 1.0],
        })

        buyback = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "buyback_ratio": [1.0, 3.0, 2.0],
        })

        scores = s._calculate_shareholder_yield_score(fundamentals, buyback)

        assert len(scores) == 3

    def test_empty_fundamentals(self):
        """빈 펀더멘탈 데이터에서 빈 시리즈 반환."""
        s = ShareholderYieldStrategy()
        result = s._calculate_shareholder_yield_score(pd.DataFrame())
        assert result.empty

    def test_no_div_yield_column(self):
        """div_yield 컬럼이 없으면 빈 시리즈 반환."""
        s = ShareholderYieldStrategy()
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B"],
            "per": [10, 5],
        })
        result = s._calculate_shareholder_yield_score(fundamentals)
        assert result.empty


# ===================================================================
# generate_signals 테스트
# ===================================================================

class TestShareholderYieldSignals:
    """generate_signals 메서드 검증."""

    def _make_fundamentals(self, n=20):
        np.random.seed(42)
        return pd.DataFrame({
            "ticker": [f"{i:06d}" for i in range(1, n + 1)],
            "name": [f"종목{i}" for i in range(1, n + 1)],
            "div_yield": np.random.uniform(0, 6, n).round(2),
            "close": np.random.randint(5000, 500000, n),
            "market_cap": np.random.randint(
                200_000_000_000, 5_000_000_000_000, n
            ),
            "volume": np.random.randint(100_000, 5_000_000, n),
        })

    def test_returns_dict(self):
        """반환값이 dict 타입이다."""
        s = ShareholderYieldStrategy(num_stocks=5, min_market_cap=0, min_volume=0)
        data = {"fundamentals": self._make_fundamentals()}
        signals = s.generate_signals("20240102", data)
        assert isinstance(signals, dict)

    def test_weights_sum_to_one(self):
        """비중의 합이 1.0이다."""
        s = ShareholderYieldStrategy(num_stocks=5, min_market_cap=0, min_volume=0)
        data = {"fundamentals": self._make_fundamentals()}
        signals = s.generate_signals("20240102", data)
        if signals:
            total = sum(signals.values())
            assert abs(total - 1.0) < 1e-9

    def test_num_stocks_respected(self):
        """num_stocks 이하의 종목이 선택된다."""
        s = ShareholderYieldStrategy(num_stocks=3, min_market_cap=0, min_volume=0)
        data = {"fundamentals": self._make_fundamentals(30)}
        signals = s.generate_signals("20240102", data)
        assert len(signals) <= 3

    def test_empty_fundamentals_returns_empty(self):
        """펀더멘탈 데이터 없으면 빈 dict 반환."""
        s = ShareholderYieldStrategy()
        signals = s.generate_signals("20240102", {"fundamentals": pd.DataFrame()})
        assert signals == {}

    def test_selects_high_div_yield(self):
        """배당수익률이 높은 종목이 선택된다."""
        s = ShareholderYieldStrategy(
            num_stocks=2, min_market_cap=0, min_volume=0,
        )
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "div_yield": [5.0, 1.0, 3.0, 4.0],
            "close": [10000] * 4,
            "market_cap": [1e12] * 4,
            "volume": [1_000_000] * 4,
        })

        signals = s.generate_signals("20240102", {"fundamentals": fundamentals})

        # A(5.0), D(4.0) 가 상위 2개
        assert "A" in signals
        assert "D" in signals
        assert "B" not in signals
