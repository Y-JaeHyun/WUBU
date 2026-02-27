"""밸류 팩터 전략 모듈(src/strategy/value.py) 테스트.

ValueStrategy 생성, 팩터 정렬, 유니버스 필터링,
generate_signals 반환값 형식 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.value import ValueStrategy


# ===================================================================
# ValueStrategy 생성 및 설정 검증
# ===================================================================

class TestValueStrategyInit:
    """ValueStrategy 초기화 검증."""

    def test_default_parameters(self):
        """기본 파라미터로 생성된다."""
        vs = ValueStrategy()
        assert vs._factor == "pbr"
        assert vs.num_stocks == 20
        assert vs.min_market_cap == 100_000_000_000
        assert vs.min_volume == 100_000_000
        assert vs.exclude_negative is True

    def test_custom_parameters(self):
        """사용자 정의 파라미터가 올바르게 반영된다."""
        vs = ValueStrategy(
            factor="per",
            num_stocks=10,
            min_market_cap=500_000_000_000,
            min_volume=50_000_000,
            exclude_negative=False,
        )
        assert vs._factor == "per"
        assert vs.num_stocks == 10
        assert vs.min_market_cap == 500_000_000_000
        assert vs.min_volume == 50_000_000
        assert vs.exclude_negative is False

    def test_composite_factor(self):
        """composite 팩터가 지원된다."""
        vs = ValueStrategy(factor="composite")
        assert vs._factor == "composite"

    def test_invalid_factor_raises(self):
        """지원하지 않는 팩터는 ValueError를 발생시킨다."""
        with pytest.raises(ValueError, match="지원하지 않는 팩터"):
            ValueStrategy(factor="momentum")

    def test_factor_case_insensitive(self):
        """팩터명은 대소문자 구분 없이 동작한다."""
        vs = ValueStrategy(factor="PBR")
        assert vs._factor == "pbr"

        vs2 = ValueStrategy(factor="Per")
        assert vs2._factor == "per"

    def test_name_property(self):
        """name 프로퍼티가 올바른 형식을 반환한다."""
        vs = ValueStrategy(factor="pbr", num_stocks=20)
        assert vs.name == "Value(PBR, top20)"

        vs2 = ValueStrategy(factor="per", num_stocks=30)
        assert vs2.name == "Value(PER, top30)"

        vs3 = ValueStrategy(factor="composite", num_stocks=10)
        assert vs3.name == "Value(COMPOSITE, top10)"


# ===================================================================
# 유니버스 필터링 테스트
# ===================================================================

class TestFilterUniverse:
    """_filter_universe 메서드 검증."""

    def _make_universe(self, n=50):
        """테스트용 유니버스 DataFrame을 생성한다."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "ticker": [f"{i:06d}" for i in range(1, n + 1)],
                "name": [f"종목{i}" for i in range(1, n + 1)],
                "market": ["KOSPI"] * n,
                "pbr": np.random.uniform(0.3, 5.0, n).round(2),
                "per": np.random.uniform(-5, 30, n).round(2),
                "close": np.random.randint(5000, 500000, n),
                "market_cap": np.random.randint(
                    10_000_000_000, 5_000_000_000_000, n
                ),
                "volume": np.random.randint(1_000, 5_000_000, n),
            }
        )

    def test_market_cap_filter(self):
        """시가총액 하한 필터링이 동작한다."""
        vs = ValueStrategy(min_market_cap=1_000_000_000_000)
        universe = self._make_universe()

        filtered = vs._filter_universe(universe)

        assert all(filtered["market_cap"] >= 1_000_000_000_000)

    def test_volume_filter(self):
        """거래대금(volume * close) 하한 필터링이 동작한다."""
        vs = ValueStrategy(min_volume=100_000_000)
        universe = self._make_universe()

        filtered = vs._filter_universe(universe)

        trade_values = filtered["volume"] * filtered["close"]
        assert all(trade_values >= 100_000_000)

    def test_exclude_zero_pbr(self):
        """PBR이 0인 종목이 제외된다 (pbr 팩터 사용 시)."""
        vs = ValueStrategy(factor="pbr")
        universe = self._make_universe(10)
        universe.loc[0, "pbr"] = 0

        filtered = vs._filter_universe(universe)

        assert 0 not in filtered["pbr"].values

    def test_exclude_zero_per(self):
        """PER이 0인 종목이 제외된다 (per 팩터 사용 시)."""
        vs = ValueStrategy(factor="per")
        universe = self._make_universe(10)
        universe.loc[0, "per"] = 0

        filtered = vs._filter_universe(universe)

        assert 0 not in filtered["per"].values

    def test_exclude_negative_pbr(self):
        """exclude_negative=True 시 음수 PBR이 제외된다."""
        vs = ValueStrategy(factor="pbr", exclude_negative=True)
        universe = self._make_universe(10)
        universe.loc[0, "pbr"] = -0.5

        filtered = vs._filter_universe(universe)

        assert all(filtered["pbr"] > 0)

    def test_include_negative_when_disabled(self):
        """exclude_negative=False 시 음수 PER 종목이 유지된다.

        단, PER=0 은 관리종목으로 제외되고, 음수만 허용.
        """
        vs = ValueStrategy(
            factor="per",
            exclude_negative=False,
            min_market_cap=0,
            min_volume=0,
        )
        universe = pd.DataFrame(
            {
                "ticker": ["000001", "000002", "000003"],
                "pbr": [1.0, 2.0, 0.5],
                "per": [-5.0, 10.0, 0],
                "close": [10000, 20000, 5000],
                "market_cap": [1_000_000_000_000] * 3,
                "volume": [1_000_000] * 3,
            }
        )

        filtered = vs._filter_universe(universe)

        # PER=0 은 제외, PER=-5 는 포함
        assert "000001" in filtered["ticker"].values  # per=-5 포함
        assert "000003" not in filtered["ticker"].values  # per=0 제외

    def test_empty_dataframe(self):
        """빈 DataFrame 입력 시 빈 DataFrame 반환."""
        vs = ValueStrategy()
        result = vs._filter_universe(pd.DataFrame())
        assert result.empty

    def test_composite_filter_both(self):
        """composite 팩터 사용 시 PBR과 PER 모두 > 0 필터링."""
        vs = ValueStrategy(
            factor="composite",
            min_market_cap=0,
            min_volume=0,
        )
        universe = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "pbr": [1.0, 0, 2.0, 1.5],
                "per": [10.0, 5.0, 0, 8.0],
                "close": [10000] * 4,
                "market_cap": [1_000_000_000_000] * 4,
                "volume": [1_000_000] * 4,
            }
        )

        filtered = vs._filter_universe(universe)

        # B (pbr=0) 과 C (per=0) 모두 제외
        assert set(filtered["ticker"].values) == {"A", "D"}


# ===================================================================
# 팩터 정렬(랭킹) 검증
# ===================================================================

class TestRankStocks:
    """_rank_stocks 메서드 검증."""

    def test_pbr_ascending(self):
        """PBR 팩터 시 PBR이 낮은 순서로 정렬된다."""
        vs = ValueStrategy(factor="pbr")
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "pbr": [2.0, 0.5, 1.5, 0.8],
                "per": [10, 5, 15, 8],
            }
        )

        ranked = vs._rank_stocks(df)

        assert list(ranked["ticker"]) == ["B", "D", "C", "A"]

    def test_per_ascending(self):
        """PER 팩터 시 PER이 낮은 순서로 정렬된다."""
        vs = ValueStrategy(factor="per")
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "pbr": [2.0, 0.5, 1.5, 0.8],
                "per": [10, 5, 15, 8],
            }
        )

        ranked = vs._rank_stocks(df)

        assert list(ranked["ticker"]) == ["B", "D", "A", "C"]

    def test_composite_ranking(self):
        """composite 팩터 시 PBR+PER 합산 랭킹으로 정렬된다."""
        vs = ValueStrategy(factor="composite")
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "pbr": [3.0, 1.0, 2.0],  # rank: 3, 1, 2
                "per": [1.0, 3.0, 2.0],  # rank: 1, 3, 2
                # composite_rank:   4, 4, 4 -> 동점일 경우 원래 순서 유지
            }
        )

        ranked = vs._rank_stocks(df)

        # composite_rank 컬럼이 생성되어야 한다
        assert "composite_rank" in ranked.columns

    def test_composite_ranking_order(self):
        """composite 팩터의 랭킹이 올바르게 합산된다."""
        vs = ValueStrategy(factor="composite")
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "pbr": [3.0, 1.0, 2.0, 4.0],  # rank: 3, 1, 2, 4
                "per": [5.0, 10.0, 3.0, 1.0],  # rank: 2, 3, 1 (after +1 = 4)
                # 위에서 D의 per=1.0이 rank 1
                # composite: A=3+2=5, B=1+3=4, C=2+1(per_rank)=? D=4+1=5
            }
        )

        ranked = vs._rank_stocks(df)

        # B가 composite_rank가 가장 낮을 가능성 높음
        # pbr_rank: B=1, C=2, A=3, D=4
        # per_rank: D=1, C=2, A=3, B=4
        # composite: B=1+4=5, C=2+2=4, A=3+3=6, D=4+1=5
        # 정렬: C(4), B(5), D(5), A(6)
        assert ranked.iloc[0]["ticker"] == "C"

    def test_empty_dataframe(self):
        """빈 DataFrame을 입력하면 빈 DataFrame을 반환한다."""
        vs = ValueStrategy(factor="pbr")
        result = vs._rank_stocks(pd.DataFrame())
        assert result.empty


# ===================================================================
# generate_signals 반환값 형식 검증
# ===================================================================

class TestGenerateSignals:
    """generate_signals 메서드 반환값 검증."""

    def _make_fundamentals(self, n=30):
        """테스트용 전종목 펀더멘탈 데이터를 생성한다."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "ticker": [f"{i:06d}" for i in range(1, n + 1)],
                "name": [f"종목{i}" for i in range(1, n + 1)],
                "market": ["KOSPI"] * n,
                # 모두 양수로 설정
                "pbr": np.random.uniform(0.3, 5.0, n).round(2),
                "per": np.random.uniform(3, 30, n).round(2),
                "close": np.random.randint(5000, 500000, n),
                "market_cap": np.random.randint(
                    200_000_000_000, 5_000_000_000_000, n
                ),
                "volume": np.random.randint(100_000, 5_000_000, n),
            }
        )

    def test_returns_dict(self):
        """반환값이 dict 타입이다."""
        vs = ValueStrategy(factor="pbr", num_stocks=5)
        data = {"fundamentals": self._make_fundamentals()}

        signals = vs.generate_signals("20240102", data)

        assert isinstance(signals, dict)

    def test_weights_sum_to_one(self):
        """비중의 합이 1.0이다 (동일 비중)."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=5,
            min_market_cap=0,
            min_volume=0,
        )
        data = {"fundamentals": self._make_fundamentals()}

        signals = vs.generate_signals("20240102", data)

        if signals:
            total_weight = sum(signals.values())
            assert abs(total_weight - 1.0) < 1e-9, (
                f"비중 합이 1.0이 아닙니다: {total_weight}"
            )

    def test_equal_weight(self):
        """각 종목의 비중이 동일하다."""
        n_stocks = 5
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=n_stocks,
            min_market_cap=0,
            min_volume=0,
        )
        data = {"fundamentals": self._make_fundamentals()}

        signals = vs.generate_signals("20240102", data)

        if signals:
            weights = list(signals.values())
            expected = 1.0 / len(signals)
            for w in weights:
                assert abs(w - expected) < 1e-9

    def test_num_stocks_respected(self):
        """num_stocks 이하의 종목이 선택된다."""
        n_stocks = 10
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=n_stocks,
            min_market_cap=0,
            min_volume=0,
        )
        data = {"fundamentals": self._make_fundamentals(50)}

        signals = vs.generate_signals("20240102", data)

        assert len(signals) <= n_stocks

    def test_empty_fundamentals_returns_empty(self):
        """펀더멘탈 데이터가 없으면 빈 dict를 반환한다."""
        vs = ValueStrategy()
        data = {"fundamentals": pd.DataFrame()}

        signals = vs.generate_signals("20240102", data)

        assert signals == {}

    def test_no_fundamentals_key_returns_empty(self):
        """data에 fundamentals 키가 없으면 빈 dict를 반환한다."""
        vs = ValueStrategy()

        signals = vs.generate_signals("20240102", {})

        assert signals == {}

    def test_signals_contain_tickers(self):
        """반환된 시그널의 키가 유효한 종목코드(ticker)이다."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=5,
            min_market_cap=0,
            min_volume=0,
        )
        fund = self._make_fundamentals()
        data = {"fundamentals": fund}

        signals = vs.generate_signals("20240102", data)

        valid_tickers = set(fund["ticker"].values)
        for ticker in signals:
            assert ticker in valid_tickers, f"'{ticker}'가 유효한 종목이 아닙니다."

    def test_pbr_factor_selects_lowest(self):
        """PBR 팩터 시 PBR이 가장 낮은 종목들이 선택된다."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=3,
            min_market_cap=0,
            min_volume=0,
            exclude_negative=False,
        )
        fund = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "pbr": [0.5, 0.3, 0.8, 1.2, 0.4],
                "per": [10, 5, 15, 8, 12],
                "close": [10000] * 5,
                "market_cap": [1_000_000_000_000] * 5,
                "volume": [1_000_000] * 5,
            }
        )
        data = {"fundamentals": fund}

        signals = vs.generate_signals("20240102", data)

        # B(0.3), E(0.4), A(0.5) 가 선택되어야 함
        assert set(signals.keys()) == {"B", "E", "A"}

    def test_per_factor_selects_lowest(self):
        """PER 팩터 시 PER이 가장 낮은 종목들이 선택된다."""
        vs = ValueStrategy(
            factor="per",
            num_stocks=2,
            min_market_cap=0,
            min_volume=0,
            exclude_negative=False,
        )
        fund = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "pbr": [1.0, 2.0, 0.5, 1.5],
                "per": [10, 3, 15, 5],
                "close": [10000] * 4,
                "market_cap": [1_000_000_000_000] * 4,
                "volume": [1_000_000] * 4,
            }
        )
        data = {"fundamentals": fund}

        signals = vs.generate_signals("20240102", data)

        # B(3), D(5) 가 선택되어야 함
        assert set(signals.keys()) == {"B", "D"}

    def test_filtered_universe_too_small(self):
        """필터링 후 종목 수가 num_stocks보다 적으면 있는 만큼만 선택한다."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=10,
            min_market_cap=0,
            min_volume=0,
        )
        fund = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "pbr": [0.5, 0.3, 0.8],
                "per": [10, 5, 15],
                "close": [10000] * 3,
                "market_cap": [1_000_000_000_000] * 3,
                "volume": [1_000_000] * 3,
            }
        )
        data = {"fundamentals": fund}

        signals = vs.generate_signals("20240102", data)

        assert len(signals) == 3
        total = sum(signals.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_filtered_out_returns_empty(self):
        """모든 종목이 필터링되면 빈 dict를 반환한다."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=5,
            min_market_cap=100_000_000_000_000,  # 100조 (모두 필터)
        )
        fund = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "pbr": [0.5, 0.3],
                "per": [10, 5],
                "close": [10000] * 2,
                "market_cap": [1_000_000_000_000] * 2,  # 1조 (하한 미달)
                "volume": [1_000_000] * 2,
            }
        )
        data = {"fundamentals": fund}

        signals = vs.generate_signals("20240102", data)

        assert signals == {}


# ===================================================================
# 업종중립 Z-Score 테스트
# ===================================================================

class TestIndustryNeutral:
    """industry_neutral 파라미터 검증."""

    def test_default_disabled(self):
        """기본값은 False이다."""
        vs = ValueStrategy()
        assert vs.industry_neutral is False

    def test_enabled_with_sector(self):
        """sector 컬럼이 있으면 업종중립 Z-Score가 적용된다."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=3,
            min_market_cap=0,
            min_volume=0,
            industry_neutral=True,
        )
        fund = pd.DataFrame({
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "pbr": [0.5, 0.3, 0.8, 1.0, 0.4, 1.5],
            "per": [10, 5, 15, 8, 12, 20],
            "close": [10000] * 6,
            "market_cap": [1_000_000_000_000] * 6,
            "volume": [1_000_000] * 6,
            "sector": ["IT", "IT", "IT", "금융", "금융", "금융"],
        })
        data = {"fundamentals": fund}
        signals = vs.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        assert len(signals) == 3
        assert abs(sum(signals.values()) - 1.0) < 1e-9

    def test_enabled_without_sector(self):
        """sector 컬럼이 없으면 전체 Z-Score로 대체한다."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=3,
            min_market_cap=0,
            min_volume=0,
            industry_neutral=True,
        )
        fund = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "pbr": [0.5, 0.3, 0.8, 1.0],
            "per": [10, 5, 15, 8],
            "close": [10000] * 4,
            "market_cap": [1_000_000_000_000] * 4,
            "volume": [1_000_000] * 4,
        })
        data = {"fundamentals": fund}
        signals = vs.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        assert len(signals) == 3

    def test_composite_with_industry_neutral(self):
        """composite + industry_neutral 조합이 동작한다."""
        vs = ValueStrategy(
            factor="composite",
            num_stocks=2,
            min_market_cap=0,
            min_volume=0,
            industry_neutral=True,
        )
        fund = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "pbr": [0.5, 0.3, 0.8, 1.0],
            "per": [10, 5, 15, 8],
            "close": [10000] * 4,
            "market_cap": [1_000_000_000_000] * 4,
            "volume": [1_000_000] * 4,
            "sector": ["IT", "IT", "금융", "금융"],
        })
        data = {"fundamentals": fund}
        signals = vs.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        assert len(signals) == 2


# ===================================================================
# 주주환원 팩터 테스트
# ===================================================================

class TestShareholderYield:
    """shareholder_yield 파라미터 검증."""

    def test_default_disabled(self):
        """기본값은 False이다."""
        vs = ValueStrategy()
        assert vs.shareholder_yield is False

    def test_enabled_with_div_yield(self):
        """div_yield 컬럼이 있으면 배당수익률이 밸류 스코어에 가산된다."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=2,
            min_market_cap=0,
            min_volume=0,
            shareholder_yield=True,
        )
        # A와 C는 PBR이 비슷하지만 A의 배당이 훨씬 높음 → A가 선호되어야 함
        fund = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "pbr": [0.50, 1.50, 0.51],
            "per": [10, 20, 10],
            "close": [10000] * 3,
            "market_cap": [1_000_000_000_000] * 3,
            "volume": [1_000_000] * 3,
            "div_yield": [5.0, 1.0, 0.5],
        })
        data = {"fundamentals": fund}
        signals = vs.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        assert len(signals) == 2
        assert "A" in signals  # 높은 배당 + 낮은 PBR

    def test_enabled_without_div_yield(self):
        """div_yield 컬럼이 없으면 기존 로직으로 동작한다."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=2,
            min_market_cap=0,
            min_volume=0,
            shareholder_yield=True,
        )
        fund = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "pbr": [0.5, 0.3, 0.8],
            "per": [10, 5, 15],
            "close": [10000] * 3,
            "market_cap": [1_000_000_000_000] * 3,
            "volume": [1_000_000] * 3,
        })
        data = {"fundamentals": fund}
        signals = vs.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        # 배당 데이터 없어도 에러 없이 동작
        assert len(signals) == 2


# ===================================================================
# F-Score 필터 테스트
# ===================================================================

class TestFScoreFilter:
    """f_score_filter 파라미터 검증."""

    def test_default_disabled(self):
        """기본값은 0(비활성)이다."""
        vs = ValueStrategy()
        assert vs.f_score_filter == 0

    def test_filter_with_f_score_column(self):
        """f_score 컬럼이 있으면 직접 사용한다."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=5,
            min_market_cap=0,
            min_volume=0,
            f_score_filter=2,
        )
        fund = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "pbr": [0.5, 0.3, 0.8, 1.0],
            "per": [10, 5, 15, 8],
            "close": [10000] * 4,
            "market_cap": [1_000_000_000_000] * 4,
            "volume": [1_000_000] * 4,
            "f_score": [3, 1, 2, 0],  # A(3), C(2) 만 통과
        })
        data = {"fundamentals": fund}
        signals = vs.generate_signals("20240102", data)

        assert "A" in signals
        assert "C" in signals
        assert "B" not in signals  # f_score=1 < 2
        assert "D" not in signals  # f_score=0 < 2

    def test_filter_with_simplified_fscore(self):
        """f_score 컬럼 없을 때 간소화 F-Score를 사용한다."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=5,
            min_market_cap=0,
            min_volume=0,
            f_score_filter=1,
        )
        fund = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "pbr": [0.5, 0.3, 0.8],
            "per": [10, 5, 15],
            "close": [10000] * 3,
            "market_cap": [1_000_000_000_000] * 3,
            "volume": [1_000_000] * 3,
            "roe": [10.0, -5.0, 15.0],  # A, C만 ROA>0
        })
        data = {"fundamentals": fund}
        signals = vs.generate_signals("20240102", data)

        # B는 ROE < 0이므로 간소화 F-Score=0, 필터링됨
        assert "B" not in signals

    def test_zero_filter_no_effect(self):
        """f_score_filter=0이면 필터가 적용되지 않는다."""
        vs = ValueStrategy(
            factor="pbr",
            num_stocks=5,
            min_market_cap=0,
            min_volume=0,
            f_score_filter=0,
        )
        fund = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "pbr": [0.5, 0.3, 0.8],
            "per": [10, 5, 15],
            "close": [10000] * 3,
            "market_cap": [1_000_000_000_000] * 3,
            "volume": [1_000_000] * 3,
            "f_score": [0, 0, 0],
        })
        data = {"fundamentals": fund}
        signals = vs.generate_signals("20240102", data)

        assert len(signals) == 3  # 필터 비활성이므로 전부 통과
