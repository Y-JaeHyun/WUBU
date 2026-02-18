"""모멘텀 팩터 전략 모듈(src/strategy/momentum.py) 테스트.

MomentumStrategy 생성, 모멘텀 스코어 계산, 유니버스 필터링,
generate_signals 반환값 형식 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.momentum import MomentumStrategy, calculate_momentum_score


# ===================================================================
# MomentumStrategy 초기화 테스트
# ===================================================================


class TestMomentumInit:
    """MomentumStrategy 초기화 검증."""

    def test_init_default(self):
        """기본 파라미터로 초기화된다."""
        ms = MomentumStrategy()
        assert ms.lookback_months == [12], "기본 lookback_months는 [12]이어야 합니다."
        assert ms.skip_month is True, "기본 skip_month는 True여야 합니다."
        assert ms.skip_days == 21, "기본 skip_days는 21이어야 합니다."
        assert ms.num_stocks == 20, "기본 num_stocks는 20이어야 합니다."
        assert ms.weighting == "equal", "기본 weighting은 'equal'이어야 합니다."
        assert ms.min_market_cap == 100_000_000_000, "기본 min_market_cap은 1000억이어야 합니다."
        assert ms.min_volume == 100_000_000, "기본 min_volume은 1억이어야 합니다."

    def test_init_custom_params(self):
        """커스텀 파라미터가 올바르게 반영된다."""
        ms = MomentumStrategy(
            lookback_months=[6, 12],
            skip_month=False,
            skip_days=10,
            num_stocks=30,
            weighting="score",
            min_market_cap=500_000_000_000,
            min_volume=500_000_000,
        )
        assert ms.lookback_months == [6, 12], "커스텀 lookback_months가 반영되어야 합니다."
        assert ms.skip_month is False, "커스텀 skip_month가 반영되어야 합니다."
        # skip_month=False일 때 skip_days는 0으로 설정됨
        assert ms.skip_days == 0, "skip_month=False이면 skip_days는 0이어야 합니다."
        assert ms.num_stocks == 30, "커스텀 num_stocks가 반영되어야 합니다."
        assert ms.weighting == "score", "커스텀 weighting이 반영되어야 합니다."
        assert ms.min_market_cap == 500_000_000_000, "커스텀 min_market_cap이 반영되어야 합니다."
        assert ms.min_volume == 500_000_000, "커스텀 min_volume이 반영되어야 합니다."

    def test_name_property(self):
        """name 프로퍼티가 올바른 형식을 반환한다."""
        ms = MomentumStrategy(lookback_months=[12], skip_month=True, num_stocks=20)
        name = ms.name
        assert "Momentum" in name, "이름에 'Momentum'이 포함되어야 합니다."
        assert "12M" in name, "이름에 룩백 기간이 포함되어야 합니다."
        assert "skip1m" in name, "skip_month=True일 때 'skip1m'이 포함되어야 합니다."
        assert "top20" in name, "이름에 종목 수가 포함되어야 합니다."

    def test_name_property_noskip(self):
        """skip_month=False일 때 이름 형식 확인."""
        ms = MomentumStrategy(lookback_months=[6, 12], skip_month=False, num_stocks=10)
        name = ms.name
        assert "noskip" in name, "skip_month=False일 때 'noskip'이 포함되어야 합니다."
        assert "6/12M" in name, "복수 룩백 기간이 포함되어야 합니다."

    def test_invalid_weighting_raises(self):
        """지원하지 않는 가중 방식은 ValueError를 발생시킨다."""
        with pytest.raises(ValueError, match="지원하지 않는 가중 방식"):
            MomentumStrategy(weighting="invalid")


# ===================================================================
# calculate_momentum_score 테스트
# ===================================================================


class TestCalculateMomentumScore:
    """모멘텀 스코어 계산 함수 검증."""

    def test_calculate_momentum_score_basic(self, sample_price_series):
        """알려진 가격 시퀀스에서 모멘텀 스코어가 올바르게 계산된다."""
        # 252거래일 데이터에서 lookback=252, skip=0 이면
        # (마지막 가격 / 첫 가격) - 1
        prices = sample_price_series
        score = calculate_momentum_score(prices, lookback=252, skip=0)

        expected = prices.iloc[-1] / prices.iloc[-252] - 1
        assert not np.isnan(score), "스코어가 NaN이면 안 됩니다."
        assert abs(score - expected) < 1e-9, f"스코어가 {expected}이어야 하는데 {score}입니다."

    def test_calculate_momentum_score_with_skip(self, sample_price_series):
        """skip_days 적용 시 최근 skip일을 건너뛴 스코어가 계산된다."""
        prices = sample_price_series
        skip = 21
        lookback = 252
        score = calculate_momentum_score(prices, lookback=lookback, skip=skip)

        # (P[-1-skip] / P[-lookback]) - 1
        expected = prices.iloc[-1 - skip] / prices.iloc[-lookback] - 1
        assert not np.isnan(score), "스코어가 NaN이면 안 됩니다."
        assert abs(score - expected) < 1e-9, f"skip 적용 스코어가 {expected}이어야 합니다."

    def test_calculate_momentum_score_no_skip(self, sample_price_series):
        """skip=0일 때 전체 기간 수익률이 반환된다."""
        prices = sample_price_series
        score = calculate_momentum_score(prices, lookback=126, skip=0)

        expected = prices.iloc[-1] / prices.iloc[-126] - 1
        assert not np.isnan(score), "스코어가 NaN이면 안 됩니다."
        assert abs(score - expected) < 1e-9, "skip=0 스코어가 올바르지 않습니다."

    def test_insufficient_data_returns_nan(self):
        """데이터가 부족하면 NaN을 반환한다."""
        dates = pd.bdate_range("2024-01-02", periods=10)
        prices = pd.Series(np.arange(10, dtype=float) + 1, index=dates)
        score = calculate_momentum_score(prices, lookback=252, skip=0)
        assert np.isnan(score), "데이터 부족 시 NaN을 반환해야 합니다."

    def test_skip_greater_equal_lookback_returns_nan(self):
        """skip >= lookback이면 NaN을 반환한다."""
        dates = pd.bdate_range("2024-01-02", periods=100)
        prices = pd.Series(np.arange(100, dtype=float) + 1, index=dates)
        score = calculate_momentum_score(prices, lookback=50, skip=50)
        assert np.isnan(score), "skip >= lookback일 때 NaN을 반환해야 합니다."

    def test_known_simple_sequence(self):
        """알려진 단순 시퀀스의 모멘텀 스코어를 검증한다."""
        # 가격: 100 -> 120 (20% 상승)
        dates = pd.bdate_range("2024-01-02", periods=10)
        prices = pd.Series([100.0] * 9 + [120.0], index=dates)
        score = calculate_momentum_score(prices, lookback=10, skip=0)

        assert abs(score - 0.2) < 1e-9, "100->120 은 20% 모멘텀이어야 합니다."

    def test_negative_momentum(self):
        """하락 종목의 음수 모멘텀 스코어를 검증한다."""
        dates = pd.bdate_range("2024-01-02", periods=10)
        prices = pd.Series([100.0] * 9 + [80.0], index=dates)
        score = calculate_momentum_score(prices, lookback=10, skip=0)

        assert score < 0, "하락 종목은 음수 모멘텀이어야 합니다."
        assert abs(score - (-0.2)) < 1e-9, "100->80 은 -20% 모멘텀이어야 합니다."


# ===================================================================
# generate_signals 테스트
# ===================================================================


class TestGenerateSignals:
    """MomentumStrategy.generate_signals 검증."""

    def test_generate_signals_empty_data(self):
        """빈 데이터 입력 시 빈 dict를 반환한다."""
        ms = MomentumStrategy()
        signals = ms.generate_signals("20240102", {"fundamentals": pd.DataFrame()})
        assert signals == {}, "빈 데이터 입력 시 빈 dict여야 합니다."

    def test_generate_signals_no_fundamentals_key(self):
        """data에 fundamentals 키가 없으면 빈 dict를 반환한다."""
        ms = MomentumStrategy()
        signals = ms.generate_signals("20240102", {})
        assert signals == {}, "fundamentals 키 없으면 빈 dict여야 합니다."

    def test_generate_signals_returns_weights(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """시그널의 비중 합이 1.0 이하인지 확인한다."""
        ms = MomentumStrategy(
            num_stocks=5,
            min_market_cap=0,
            min_volume=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        signals = ms.generate_signals("20240102", data)

        if signals:
            total_weight = sum(signals.values())
            assert total_weight <= 1.0 + 1e-9, (
                f"비중 합이 1.0 이하여야 합니다: {total_weight}"
            )

    def test_generate_signals_num_stocks_limit(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """num_stocks 이하만 반환한다."""
        num = 5
        ms = MomentumStrategy(
            num_stocks=num,
            min_market_cap=0,
            min_volume=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        signals = ms.generate_signals("20240102", data)

        assert len(signals) <= num, f"종목 수가 {num}개 이하여야 합니다: {len(signals)}"

    def test_generate_signals_equal_weight(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """동일비중(equal) 모드에서 각 종목 비중이 같은지 확인한다."""
        ms = MomentumStrategy(
            num_stocks=5,
            weighting="equal",
            min_market_cap=0,
            min_volume=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        signals = ms.generate_signals("20240102", data)

        if signals:
            weights = list(signals.values())
            expected_w = 1.0 / len(signals)
            for w in weights:
                assert abs(w - expected_w) < 1e-9, (
                    f"동일비중 모드에서 각 비중이 {expected_w}이어야 합니다: {w}"
                )


# ===================================================================
# 유니버스 필터링 테스트
# ===================================================================


class TestUniverseFiltering:
    """유니버스 필터링 검증."""

    def test_universe_filtering_market_cap(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """시가총액 필터가 올바르게 동작한다."""
        high_cap = 3_000_000_000_000  # 3조
        ms = MomentumStrategy(
            min_market_cap=high_cap,
            min_volume=0,
        )
        # _filter_universe는 fundamentals와 prices를 받음
        filtered = ms._filter_universe(sample_all_fundamentals, sample_prices_dict)

        # 필터링 결과 종목들의 시가총액 확인
        for ticker in filtered:
            row = sample_all_fundamentals[
                sample_all_fundamentals["ticker"] == ticker
            ]
            if not row.empty:
                assert row.iloc[0]["market_cap"] >= high_cap, (
                    f"종목 {ticker}의 시가총액이 하한({high_cap}) 미만입니다."
                )

    def test_universe_filtering_volume(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """거래대금 필터가 올바르게 동작한다."""
        high_vol = 1_000_000_000  # 10억
        ms = MomentumStrategy(
            min_market_cap=0,
            min_volume=high_vol,
        )
        filtered = ms._filter_universe(sample_all_fundamentals, sample_prices_dict)

        for ticker in filtered:
            row = sample_all_fundamentals[
                sample_all_fundamentals["ticker"] == ticker
            ]
            if not row.empty:
                trade_val = row.iloc[0]["volume"] * row.iloc[0]["close"]
                assert trade_val >= high_vol, (
                    f"종목 {ticker}의 거래대금이 하한({high_vol}) 미만입니다."
                )

    def test_universe_filtering_empty_fundamentals(self):
        """빈 펀더멘탈 데이터이면 빈 리스트를 반환한다."""
        ms = MomentumStrategy()
        filtered = ms._filter_universe(pd.DataFrame(), {})
        assert filtered == [], "빈 펀더멘탈이면 빈 리스트여야 합니다."

    def test_universe_filtering_no_price_data(self, sample_all_fundamentals):
        """가격 데이터가 없으면 빈 리스트를 반환한다."""
        ms = MomentumStrategy(min_market_cap=0, min_volume=0)
        # 가격 dict가 비어 있으면 252일 조건 통과 불가
        filtered = ms._filter_universe(sample_all_fundamentals, {})
        assert filtered == [], "가격 데이터 없으면 빈 리스트여야 합니다."
