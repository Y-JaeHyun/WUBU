"""저변동성 팩터 전략 모듈(src/strategy/low_volatility.py) 테스트.

LowVolatilityStrategy 생성, 변동성 계산, 스코어 산출,
generate_signals 반환값 형식, get_scores 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Strategy
from src.strategy.low_volatility import LowVolatilityStrategy


# ===================================================================
# 헬퍼: 테스트용 가격 데이터 생성
# ===================================================================

def _make_price_df(n_days: int, base: float = 50000.0, vol: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """n_days 거래일 분량의 가격 DataFrame을 생성한다.

    Args:
        n_days: 거래일 수
        base: 시작 가격
        vol: 일별 수익률 표준편차
        seed: 난수 시드

    Returns:
        OHLCV DataFrame (DatetimeIndex)
    """
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    close = base * np.exp(np.cumsum(np.random.randn(n_days) * vol))
    df = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(100_000, 5_000_000, n_days),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _make_flat_price_df(n_days: int, price: float = 50000.0) -> pd.DataFrame:
    """변동성이 0인 (고정 가격) DataFrame을 생성한다."""
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    close = np.full(n_days, price)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n_days, 1_000_000),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ===================================================================
# TestLowVolatilityInit: 초기화 및 속성 검증
# ===================================================================

class TestLowVolatilityInit:
    """LowVolatilityStrategy 초기화 및 속성 검증."""

    def test_init_default(self):
        """기본 파라미터로 초기화된다."""
        strat = LowVolatilityStrategy()
        assert strat.vol_period == 60, "기본 vol_period는 60이어야 합니다."
        assert strat.num_stocks == 20, "기본 num_stocks는 20이어야 합니다."
        assert strat.min_market_cap == 100_000_000_000, "기본 min_market_cap은 1000억이어야 합니다."
        assert strat.min_trading_days == 120, "기본 min_trading_days는 120이어야 합니다."
        assert strat.weighting == "equal", "기본 weighting은 'equal'이어야 합니다."

    def test_init_custom_params(self):
        """커스텀 파라미터가 올바르게 반영된다."""
        strat = LowVolatilityStrategy(
            vol_period=120,
            num_stocks=30,
            min_market_cap=500_000_000_000,
            min_trading_days=200,
            weighting="Equal",
        )
        assert strat.vol_period == 120, "커스텀 vol_period가 반영되어야 합니다."
        assert strat.num_stocks == 30, "커스텀 num_stocks가 반영되어야 합니다."
        assert strat.min_market_cap == 500_000_000_000, "커스텀 min_market_cap이 반영되어야 합니다."
        assert strat.min_trading_days == 200, "커스텀 min_trading_days가 반영되어야 합니다."
        assert strat.weighting == "equal", "weighting은 소문자로 저장되어야 합니다."

    def test_name_property(self):
        """name 프로퍼티가 올바른 형식을 반환한다."""
        strat = LowVolatilityStrategy(vol_period=60, num_stocks=20)
        name = strat.name
        assert name == "LowVol(60d, top20)", f"이름이 'LowVol(60d, top20)'이어야 합니다: {name}"

    def test_name_property_custom(self):
        """커스텀 파라미터로 이름이 올바르게 생성된다."""
        strat = LowVolatilityStrategy(vol_period=120, num_stocks=10)
        name = strat.name
        assert name == "LowVol(120d, top10)", f"이름이 'LowVol(120d, top10)'이어야 합니다: {name}"

    def test_is_strategy_subclass(self):
        """LowVolatilityStrategy는 Strategy ABC의 서브클래스이다."""
        strat = LowVolatilityStrategy()
        assert isinstance(strat, Strategy), "Strategy의 인스턴스여야 합니다."


# ===================================================================
# TestComputeVolatility: 변동성 계산 검증
# ===================================================================

class TestComputeVolatility:
    """_compute_volatility 메서드 검증."""

    def test_compute_volatility_known_series(self):
        """알려진 가격 시계열에서 변동성이 합리적으로 계산된다."""
        strat = LowVolatilityStrategy(vol_period=60)
        price_df = _make_price_df(n_days=252, vol=0.02, seed=42)

        vol = strat._compute_volatility("TEST", price_df)

        assert vol is not None, "유효한 데이터에서 None이면 안 됩니다."
        assert isinstance(vol, float), "반환 타입이 float이어야 합니다."
        # 일별 vol=0.02 -> 연율화 약 0.02 * sqrt(252) ≈ 0.317
        # 랜덤 시드에 따라 다를 수 있으므로 넓은 범위로 검증
        assert 0.1 < vol < 0.8, f"연율화 변동성이 합리적 범위여야 합니다: {vol}"

    def test_compute_volatility_high_vol(self):
        """고변동성 종목의 변동성이 저변동성 종목보다 높다."""
        strat = LowVolatilityStrategy(vol_period=60)

        low_vol_df = _make_price_df(n_days=252, vol=0.005, seed=10)
        high_vol_df = _make_price_df(n_days=252, vol=0.05, seed=10)

        vol_low = strat._compute_volatility("LOW", low_vol_df)
        vol_high = strat._compute_volatility("HIGH", high_vol_df)

        assert vol_low is not None and vol_high is not None
        assert vol_high > vol_low, (
            f"고변동성({vol_high:.4f})이 저변동성({vol_low:.4f})보다 커야 합니다."
        )

    def test_compute_volatility_flat_prices_returns_none(self):
        """가격이 일정한 (변동성~0) 종목은 None을 반환한다."""
        strat = LowVolatilityStrategy(vol_period=60)
        flat_df = _make_flat_price_df(n_days=252)

        vol = strat._compute_volatility("FLAT", flat_df)

        assert vol is None, "변동성이 0인 종목은 None을 반환해야 합니다."

    def test_compute_volatility_short_data_returns_none(self):
        """데이터가 vol_period보다 짧으면 None을 반환한다."""
        strat = LowVolatilityStrategy(vol_period=60)
        short_df = _make_price_df(n_days=30)  # 30일 < vol_period(60)

        vol = strat._compute_volatility("SHORT", short_df)

        assert vol is None, "데이터 부족 시 None을 반환해야 합니다."

    def test_compute_volatility_empty_df_returns_none(self):
        """빈 DataFrame이면 None을 반환한다."""
        strat = LowVolatilityStrategy(vol_period=60)
        empty_df = pd.DataFrame()

        vol = strat._compute_volatility("EMPTY", empty_df)

        assert vol is None, "빈 DataFrame이면 None을 반환해야 합니다."

    def test_compute_volatility_exact_period(self):
        """정확히 vol_period+1 길이의 데이터에서도 계산된다.

        pct_change().dropna() 하면 n-1개 수익률이 남으므로,
        vol_period=60일 때 61일 이상 가격 데이터가 필요하다.
        """
        strat = LowVolatilityStrategy(vol_period=60)
        # 61일: pct_change 후 60개 수익률 -> vol_period와 같으므로 계산 가능
        price_df = _make_price_df(n_days=61, vol=0.02)

        vol = strat._compute_volatility("EXACT", price_df)

        assert vol is not None, "vol_period와 정확히 같은 수익률 수에서도 계산되어야 합니다."

    def test_compute_volatility_uses_recent_period(self):
        """최근 vol_period 기간만 사용하여 변동성을 계산한다."""
        strat = LowVolatilityStrategy(vol_period=60)

        # 앞부분 200일은 변동성 작게, 뒤 100일은 변동성 크게
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-02", periods=300)
        part1 = 50000 * np.exp(np.cumsum(np.random.randn(200) * 0.005))
        part2 = part1[-1] * np.exp(np.cumsum(np.random.randn(100) * 0.05))
        close = np.concatenate([part1, part2])

        df = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": np.full(300, 1_000_000),
            },
            index=dates,
        )

        vol = strat._compute_volatility("MIXED", df)

        assert vol is not None
        # 최근 60일이 고변동성 구간이므로 연율화 변동성이 높아야 함
        assert vol > 0.3, (
            f"최근 구간이 고변동성이므로 vol이 0.3 이상이어야 합니다: {vol:.4f}"
        )


# ===================================================================
# TestComputeScores: 스코어 계산 검증
# ===================================================================

class TestComputeScores:
    """_compute_scores 메서드 검증."""

    def test_low_vol_stocks_score_higher(self):
        """변동성이 낮은 종목이 더 높은 스코어를 받는다."""
        strat = LowVolatilityStrategy(vol_period=60)

        # 3개 종목: 저변동성, 중변동성, 고변동성
        prices = {
            "LOW": _make_price_df(n_days=252, vol=0.005, seed=10),
            "MID": _make_price_df(n_days=252, vol=0.02, seed=20),
            "HIGH": _make_price_df(n_days=252, vol=0.05, seed=30),
        }

        scores = strat._compute_scores(["LOW", "MID", "HIGH"], prices)

        assert len(scores) == 3, f"3개 종목 스코어가 있어야 합니다: {len(scores)}"
        assert scores["LOW"] > scores["MID"], "저변동성 종목이 더 높은 스코어를 받아야 합니다."
        assert scores["MID"] > scores["HIGH"], "중변동성 종목이 고변동성보다 높은 스코어를 받아야 합니다."

    def test_scores_are_positive(self):
        """모든 스코어가 양수이다 (1/vol > 0)."""
        strat = LowVolatilityStrategy(vol_period=60)
        prices = {
            f"T{i}": _make_price_df(n_days=252, vol=0.01 + i * 0.005, seed=i)
            for i in range(10)
        }
        tickers = list(prices.keys())

        scores = strat._compute_scores(tickers, prices)

        for ticker, score in scores.items():
            assert score > 0, f"종목 {ticker}의 스코어 {score}가 양수여야 합니다."

    def test_winsorizing_applied(self):
        """극단값이 winsorizing에 의해 클리핑된다."""
        strat = LowVolatilityStrategy(vol_period=60)

        # 50개 종목 생성: 대부분 비슷한 변동성, 1개만 극단적으로 낮은 변동성
        np.random.seed(42)
        prices = {}
        for i in range(50):
            vol = 0.02 + np.random.rand() * 0.01  # 0.02~0.03 범위
            prices[f"T{i:02d}"] = _make_price_df(n_days=252, vol=vol, seed=i + 100)
        # 극단적 저변동성 종목 추가 → winsorizing 전에는 매우 높은 스코어
        prices["EXTREME_LOW"] = _make_price_df(n_days=252, vol=0.001, seed=999)

        tickers = list(prices.keys())

        # winsorizing 전 raw 스코어 계산
        raw_scores = {}
        for ticker in tickers:
            vol = strat._compute_volatility(ticker, prices[ticker])
            if vol is not None:
                raw_scores[ticker] = 1.0 / vol
        raw_series = pd.Series(raw_scores)

        scores = strat._compute_scores(tickers, prices)

        if len(scores) > 2 and "EXTREME_LOW" in raw_series.index:
            # 극단 종목의 raw 스코어가 winsorizing 후 축소되었는지 확인
            raw_extreme = raw_series["EXTREME_LOW"]
            clipped_extreme = scores["EXTREME_LOW"]
            assert clipped_extreme < raw_extreme, (
                "winsorizing 후 극단값이 축소되어야 합니다."
            )
            # 클리핑 값이 99% 분위수(winsorizing 전 기준)와 일치해야 함
            upper_limit = raw_series.quantile(0.99)
            assert abs(clipped_extreme - upper_limit) < 1e-9, (
                f"극단값이 99% 분위수({upper_limit:.4f})로 클리핑되어야 합니다."
            )

    def test_scores_returns_series(self):
        """반환값이 pd.Series 타입이다."""
        strat = LowVolatilityStrategy(vol_period=60)
        prices = {"A": _make_price_df(n_days=252, seed=1)}

        scores = strat._compute_scores(["A"], prices)

        assert isinstance(scores, pd.Series), "반환값이 pd.Series여야 합니다."

    def test_scores_empty_tickers(self):
        """빈 종목 리스트이면 빈 Series를 반환한다."""
        strat = LowVolatilityStrategy(vol_period=60)

        scores = strat._compute_scores([], {})

        assert isinstance(scores, pd.Series), "반환값이 pd.Series여야 합니다."
        assert scores.empty, "빈 종목 리스트이면 빈 Series여야 합니다."

    def test_scores_skip_invalid_tickers(self):
        """가격 데이터가 없거나 부족한 종목은 스코어에서 제외된다."""
        strat = LowVolatilityStrategy(vol_period=60)
        prices = {
            "GOOD": _make_price_df(n_days=252, seed=1),
            "SHORT": _make_price_df(n_days=30, seed=2),  # 데이터 부족
        }

        scores = strat._compute_scores(["GOOD", "SHORT", "MISSING"], prices)

        assert "GOOD" in scores.index, "유효한 종목은 스코어에 포함되어야 합니다."
        assert "SHORT" not in scores.index, "데이터 부족 종목은 제외되어야 합니다."
        assert "MISSING" not in scores.index, "가격 없는 종목은 제외되어야 합니다."


# ===================================================================
# TestGenerateSignals: 시그널 생성 검증
# ===================================================================

class TestGenerateSignals:
    """LowVolatilityStrategy.generate_signals 검증."""

    def test_generate_signals_returns_dict(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """generate_signals는 dict를 반환한다."""
        strat = LowVolatilityStrategy(
            num_stocks=5,
            min_market_cap=0,
            min_trading_days=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        signals = strat.generate_signals("20240102", data)

        assert isinstance(signals, dict), "반환값이 dict여야 합니다."

    def test_generate_signals_num_stocks_limit(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """num_stocks 이하의 종목만 선택된다."""
        num = 5
        strat = LowVolatilityStrategy(
            num_stocks=num,
            min_market_cap=0,
            min_trading_days=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        signals = strat.generate_signals("20240102", data)

        assert len(signals) <= num, f"종목 수가 {num}개 이하여야 합니다: {len(signals)}"

    def test_generate_signals_weight_sum_le_one(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """시그널 비중의 합이 1.0 이하이다."""
        strat = LowVolatilityStrategy(
            num_stocks=10,
            min_market_cap=0,
            min_trading_days=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        signals = strat.generate_signals("20240102", data)

        if signals:
            total_weight = sum(signals.values())
            assert total_weight <= 1.0 + 1e-9, (
                f"비중 합이 1.0 이하여야 합니다: {total_weight}"
            )

    def test_generate_signals_equal_weight(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """동일비중 모드에서 각 종목 비중이 같다."""
        strat = LowVolatilityStrategy(
            num_stocks=5,
            min_market_cap=0,
            min_trading_days=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        signals = strat.generate_signals("20240102", data)

        if signals:
            weights = list(signals.values())
            expected_w = 1.0 / len(signals)
            for w in weights:
                assert abs(w - expected_w) < 1e-9, (
                    f"동일비중 모드에서 각 비중이 {expected_w}이어야 합니다: {w}"
                )

    def test_generate_signals_empty_fundamentals(self):
        """빈 펀더멘탈 데이터 입력 시 빈 dict를 반환한다."""
        strat = LowVolatilityStrategy()
        signals = strat.generate_signals("20240102", {"fundamentals": pd.DataFrame()})
        assert signals == {}, "빈 펀더멘탈이면 빈 dict여야 합니다."

    def test_generate_signals_no_fundamentals_key(self):
        """data에 fundamentals 키가 없으면 빈 dict를 반환한다."""
        strat = LowVolatilityStrategy()
        signals = strat.generate_signals("20240102", {})
        assert signals == {}, "fundamentals 키 없으면 빈 dict여야 합니다."

    def test_generate_signals_no_prices(self, sample_all_fundamentals):
        """가격 데이터가 없으면 빈 dict를 반환한다."""
        strat = LowVolatilityStrategy(min_market_cap=0, min_trading_days=0)
        data = {"fundamentals": sample_all_fundamentals, "prices": {}}
        signals = strat.generate_signals("20240102", data)
        assert signals == {}, "가격 데이터 없으면 빈 dict여야 합니다."

    def test_generate_signals_valid_tickers(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """반환된 시그널의 키가 유효한 종목코드이다."""
        strat = LowVolatilityStrategy(
            num_stocks=5,
            min_market_cap=0,
            min_trading_days=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        signals = strat.generate_signals("20240102", data)

        valid_tickers = set(sample_all_fundamentals["ticker"].values)
        for ticker in signals:
            assert ticker in valid_tickers, f"'{ticker}'가 유효한 종목이 아닙니다."

    def test_generate_signals_positive_weights(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """모든 비중이 양수이다."""
        strat = LowVolatilityStrategy(
            num_stocks=10,
            min_market_cap=0,
            min_trading_days=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        signals = strat.generate_signals("20240102", data)

        for ticker, weight in signals.items():
            assert weight > 0, f"종목 {ticker}의 비중 {weight}이 양수여야 합니다."

    def test_generate_signals_market_cap_filter(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """시가총액 필터가 올바르게 적용된다."""
        high_cap = 3_000_000_000_000  # 3조
        strat = LowVolatilityStrategy(
            num_stocks=5,
            min_market_cap=high_cap,
            min_trading_days=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        signals = strat.generate_signals("20240102", data)

        # 선택된 종목의 시가총액이 기준 이상인지 확인
        for ticker in signals:
            row = sample_all_fundamentals[sample_all_fundamentals["ticker"] == ticker]
            if not row.empty:
                assert row.iloc[0]["market_cap"] >= high_cap, (
                    f"종목 {ticker}의 시가총액이 {high_cap} 이상이어야 합니다."
                )


# ===================================================================
# TestGetScores: 외부 스코어 접근 검증
# ===================================================================

class TestGetScores:
    """LowVolatilityStrategy.get_scores 검증."""

    def test_get_scores_returns_series(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """get_scores는 pd.Series를 반환한다."""
        strat = LowVolatilityStrategy(
            min_market_cap=0,
            min_trading_days=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        scores = strat.get_scores(data)

        assert isinstance(scores, pd.Series), "반환값이 pd.Series여야 합니다."

    def test_get_scores_correct_index(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """get_scores의 인덱스가 유효한 종목 코드이다."""
        strat = LowVolatilityStrategy(
            min_market_cap=0,
            min_trading_days=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        scores = strat.get_scores(data)

        valid_tickers = set(sample_all_fundamentals["ticker"].values)
        for ticker in scores.index:
            assert ticker in valid_tickers, f"'{ticker}'가 유효한 종목이 아닙니다."

    def test_get_scores_not_empty(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """유효 데이터가 있으면 비어 있지 않은 스코어를 반환한다."""
        strat = LowVolatilityStrategy(
            min_market_cap=0,
            min_trading_days=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        scores = strat.get_scores(data)

        assert len(scores) > 0, "유효 데이터가 있으면 스코어가 비어 있으면 안 됩니다."

    def test_get_scores_empty_data(self):
        """빈 데이터이면 빈 Series를 반환한다."""
        strat = LowVolatilityStrategy()
        data = {"fundamentals": pd.DataFrame(), "prices": {}}
        scores = strat.get_scores(data)

        assert isinstance(scores, pd.Series), "반환값이 pd.Series여야 합니다."
        assert scores.empty, "빈 데이터이면 빈 Series여야 합니다."

    def test_get_scores_all_positive(
        self, sample_all_fundamentals, sample_prices_dict
    ):
        """모든 스코어가 양수이다."""
        strat = LowVolatilityStrategy(
            min_market_cap=0,
            min_trading_days=0,
        )
        data = {
            "fundamentals": sample_all_fundamentals,
            "prices": sample_prices_dict,
        }
        scores = strat.get_scores(data)

        for ticker, score in scores.items():
            assert score > 0, f"종목 {ticker}의 스코어 {score}가 양수여야 합니다."
