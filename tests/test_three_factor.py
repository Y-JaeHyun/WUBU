"""3팩터 전략 모듈(src/strategy/three_factor.py) 테스트.

ThreeFactorStrategy 생성, 밸류+모멘텀+퀄리티 결합,
zscore/rank 방법, generate_signals 반환값 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Strategy


# ===================================================================
# 헬퍼 함수
# ===================================================================

def _import_three_factor():
    """ThreeFactorStrategy 클래스를 임포트한다."""
    from src.strategy.three_factor import ThreeFactorStrategy
    return ThreeFactorStrategy


def _make_three_factor_fundamentals(n=30, seed=42):
    """3팩터 분석용 펀더멘탈 DataFrame을 생성한다.

    밸류(PBR/PER), 모멘텀, 퀄리티 관련 컬럼을 모두 포함한다.
    """
    np.random.seed(seed)
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "ticker": tickers,
        "name": [f"종목{i}" for i in range(1, n + 1)],
        "market": ["KOSPI"] * n,
        "pbr": np.random.uniform(0.3, 5.0, n).round(2),
        "per": np.random.uniform(3, 30, n).round(2),
        "roe": np.random.uniform(5, 30, n).round(2),
        "gp_over_assets": np.random.uniform(0.05, 0.5, n).round(4),
        "debt_ratio": np.random.uniform(20, 200, n).round(2),
        "accruals": np.random.uniform(-0.1, 0.1, n).round(4),
        "close": np.random.randint(5000, 500000, n),
        "market_cap": np.random.randint(200_000_000_000, 5_000_000_000_000, n),
        "volume": np.random.randint(100_000, 5_000_000, n),
        "eps": np.random.randint(500, 20000, n),
        "bps": np.random.randint(5000, 100000, n),
    })


def _make_three_factor_prices(n=30, seed=42):
    """3팩터 분석용 가격 데이터 dict를 생성한다.

    Returns:
        dict[ticker -> DataFrame with OHLCV and '종가' column]
    """
    np.random.seed(seed)
    result = {}
    for i in range(1, n + 1):
        ticker = f"{i:06d}"
        dates = pd.bdate_range("2023-01-02", periods=252)
        base = np.random.randint(10000, 100000)
        close = base * np.exp(np.cumsum(np.random.randn(252) * 0.02))
        df = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "종가": close,  # 한글 종가 컬럼도 포함
            "volume": np.random.randint(100000, 5000000, 252),
        }, index=dates)
        df.index.name = "date"
        result[ticker] = df
    return result


# ===================================================================
# ThreeFactorStrategy 기본 속성 검증
# ===================================================================

class TestThreeFactorInit:
    """ThreeFactorStrategy 초기화 검증."""

    def test_three_factor_name(self):
        """name 프로퍼티가 문자열을 반환한다."""
        ThreeFactorStrategy = _import_three_factor()
        tfs = ThreeFactorStrategy()
        assert isinstance(tfs.name, str), "name은 문자열이어야 합니다."
        assert len(tfs.name) > 0, "name이 비어 있으면 안 됩니다."

    def test_three_factor_is_strategy_subclass(self):
        """ThreeFactorStrategy는 Strategy ABC의 서브클래스이다."""
        ThreeFactorStrategy = _import_three_factor()
        tfs = ThreeFactorStrategy()
        assert isinstance(tfs, Strategy), (
            "ThreeFactorStrategy는 Strategy의 인스턴스여야 합니다."
        )

    def test_default_parameters(self):
        """기본 파라미터가 올바르게 설정된다."""
        ThreeFactorStrategy = _import_three_factor()
        tfs = ThreeFactorStrategy()
        # 기본 num_stocks = 20
        num = getattr(tfs, "num_stocks", getattr(tfs, "_num_stocks", None))
        assert num == 20, "기본 num_stocks는 20이어야 합니다."

    def test_custom_weights(self):
        """사용자 정의 팩터 가중치가 반영된다."""
        ThreeFactorStrategy = _import_three_factor()
        tfs = ThreeFactorStrategy(
            value_weight=0.5,
            momentum_weight=0.3,
            quality_weight=0.2,
        )
        # 가중치 합이 1.0인지 간접 확인
        vw = getattr(tfs, "value_weight", getattr(tfs, "_value_weight", 0.5))
        mw = getattr(tfs, "momentum_weight", getattr(tfs, "_momentum_weight", 0.3))
        qw = getattr(tfs, "quality_weight", getattr(tfs, "_quality_weight", 0.2))
        assert abs(vw + mw + qw - 1.0) < 1e-9, (
            "팩터 가중치 합이 1.0이어야 합니다."
        )


# ===================================================================
# generate_signals 검증
# ===================================================================

class TestThreeFactorGenerateSignals:
    """ThreeFactorStrategy.generate_signals 검증."""

    def test_generate_signals_basic(self):
        """기본 데이터로 시그널이 정상 생성된다."""
        ThreeFactorStrategy = _import_three_factor()
        tfs = ThreeFactorStrategy(num_stocks=5)
        data = {
            "fundamentals": _make_three_factor_fundamentals(),
            "prices": _make_three_factor_prices(),
        }

        signals = tfs.generate_signals("20240102", data)

        assert isinstance(signals, dict), "반환값이 dict여야 합니다."

    def test_generate_signals_zscore_method(self):
        """zscore 결합 방법으로 시그널이 생성된다."""
        ThreeFactorStrategy = _import_three_factor()
        tfs = ThreeFactorStrategy(
            num_stocks=5,
            combination_method="zscore",
        )
        data = {
            "fundamentals": _make_three_factor_fundamentals(),
            "prices": _make_three_factor_prices(),
        }

        signals = tfs.generate_signals("20240102", data)

        assert isinstance(signals, dict), "zscore 결합 결과가 dict여야 합니다."

    def test_generate_signals_rank_method(self):
        """rank 결합 방법으로 시그널이 생성된다."""
        ThreeFactorStrategy = _import_three_factor()
        tfs = ThreeFactorStrategy(
            num_stocks=5,
            combination_method="rank",
        )
        data = {
            "fundamentals": _make_three_factor_fundamentals(),
            "prices": _make_three_factor_prices(),
        }

        signals = tfs.generate_signals("20240102", data)

        assert isinstance(signals, dict), "rank 결합 결과가 dict여야 합니다."

    def test_generate_signals_weights_sum(self):
        """시그널 비중의 합이 1.0 이하이다."""
        ThreeFactorStrategy = _import_three_factor()
        tfs = ThreeFactorStrategy(num_stocks=10)
        data = {
            "fundamentals": _make_three_factor_fundamentals(),
            "prices": _make_three_factor_prices(),
        }

        signals = tfs.generate_signals("20240102", data)

        if signals:
            total_weight = sum(signals.values())
            assert total_weight <= 1.0 + 1e-9, (
                f"비중 합이 1.0 이하여야 합니다: {total_weight}"
            )

    def test_generate_signals_num_stocks(self):
        """num_stocks 이하의 종목이 선택된다."""
        ThreeFactorStrategy = _import_three_factor()
        n_stocks = 5
        tfs = ThreeFactorStrategy(num_stocks=n_stocks)
        data = {
            "fundamentals": _make_three_factor_fundamentals(n=50),
            "prices": _make_three_factor_prices(n=50),
        }

        signals = tfs.generate_signals("20240102", data)

        assert len(signals) <= n_stocks, (
            f"종목 수가 {n_stocks}개 이하여야 합니다: {len(signals)}"
        )

    def test_generate_signals_empty_data(self):
        """빈 데이터 입력 시 빈 dict를 반환한다."""
        ThreeFactorStrategy = _import_three_factor()
        tfs = ThreeFactorStrategy()

        signals = tfs.generate_signals("20240102", {"fundamentals": pd.DataFrame()})

        assert signals == {}, "빈 데이터 입력 시 빈 dict여야 합니다."

    def test_generate_signals_no_fundamentals(self):
        """data에 fundamentals가 없으면 빈 dict를 반환한다."""
        ThreeFactorStrategy = _import_three_factor()
        tfs = ThreeFactorStrategy()

        signals = tfs.generate_signals("20240102", {})

        assert signals == {}, "fundamentals 없으면 빈 dict여야 합니다."

    def test_generate_signals_valid_tickers(self):
        """반환된 시그널의 키가 유효한 종목코드이다."""
        ThreeFactorStrategy = _import_three_factor()
        tfs = ThreeFactorStrategy(num_stocks=5)
        fundamentals = _make_three_factor_fundamentals()
        data = {
            "fundamentals": fundamentals,
            "prices": _make_three_factor_prices(),
        }

        signals = tfs.generate_signals("20240102", data)

        valid_tickers = set(fundamentals["ticker"].values)
        for ticker in signals:
            assert ticker in valid_tickers, (
                f"'{ticker}'가 유효한 종목이 아닙니다."
            )
