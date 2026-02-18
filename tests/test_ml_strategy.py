"""ML 팩터 전략 모듈(src/strategy/ml_factor.py) 테스트.

MLFactorStrategy의 이름, 시그널 생성, 종목 수 제한,
비중 검증, 엣지 케이스 처리를 검증한다.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.backtest.engine import Strategy


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _try_import_ml_factor_strategy():
    """MLFactorStrategy가 있으면 임포트한다."""
    try:
        from src.strategy.ml_factor import MLFactorStrategy
        return MLFactorStrategy
    except ImportError:
        return None


def _make_mock_pipeline(predictions=None):
    """MLPipeline mock 객체를 생성한다."""
    mock = MagicMock()
    mock.model_type = "ridge"
    mock._is_trained = True

    if predictions is not None:
        mock.predict.return_value = predictions
    else:
        # 기본: 30종목에 대한 예측 스코어
        np.random.seed(42)
        n = 30
        tickers = [f"{i:06d}" for i in range(1, n + 1)]
        mock.predict.return_value = pd.Series(
            np.random.randn(n),
            index=tickers,
        )
    return mock


def _make_fundamentals(n=30, seed=42):
    """테스트용 펀더멘탈 DataFrame을 생성한다."""
    np.random.seed(seed)
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "ticker": tickers,
        "name": [f"종목{i}" for i in range(1, n + 1)],
        "market": ["KOSPI"] * (n // 2) + ["KOSDAQ"] * (n - n // 2),
        "pbr": np.abs(np.random.randn(n) * 1 + 1.5).round(2),
        "per": np.abs(np.random.randn(n) * 10 + 12).round(2),
        "eps": np.random.randint(500, 20000, n),
        "bps": np.random.randint(5000, 100000, n),
        "div_yield": (np.random.rand(n) * 5).round(2),
        "roe": np.random.uniform(5, 30, n).round(2),
        "close": np.random.randint(5000, 500000, n),
        "market_cap": np.random.randint(200_000_000_000, 5_000_000_000_000, n),
        "volume": np.random.randint(100_000, 5_000_000, n),
    })


def _make_prices_dict(tickers, n_days=300, seed=42):
    """종목별 가격 데이터 dict를 생성한다."""
    np.random.seed(seed)
    prices = {}
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    for ticker in tickers:
        base = np.random.randint(10000, 100000)
        close = base * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))
        df = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": np.random.randint(100000, 5000000, n_days),
            },
            index=dates,
        )
        df.index.name = "date"
        prices[ticker] = df
    return prices


# ===================================================================
# MLFactorStrategy 검증
# ===================================================================

class TestMLFactorStrategy:
    """MLFactorStrategy 검증."""

    def test_name_property(self):
        """name 프로퍼티가 'MLFactor'를 포함한다."""
        MLFactorStrategy = _try_import_ml_factor_strategy()
        if MLFactorStrategy is None:
            pytest.skip("MLFactorStrategy가 아직 구현되지 않았습니다.")

        mock_pipeline = _make_mock_pipeline()
        strategy = MLFactorStrategy(ml_pipeline=mock_pipeline)

        assert "MLFactor" in strategy.name or "ML" in strategy.name, (
            f"이름에 'MLFactor'가 포함되어야 합니다: {strategy.name}"
        )

    def test_generate_signals_basic(self):
        """기본 시그널이 정상적으로 생성된다."""
        MLFactorStrategy = _try_import_ml_factor_strategy()
        if MLFactorStrategy is None:
            pytest.skip("MLFactorStrategy가 아직 구현되지 않았습니다.")

        fundamentals = _make_fundamentals()
        tickers = fundamentals["ticker"].tolist()
        prices = _make_prices_dict(tickers)
        mock_pipeline = _make_mock_pipeline()

        strategy = MLFactorStrategy(ml_pipeline=mock_pipeline)
        data = {"fundamentals": fundamentals, "prices": prices}
        signals = strategy.generate_signals("20240102", data)

        assert isinstance(signals, dict), "반환값이 dict여야 합니다."
        assert len(signals) > 0, "시그널이 비어 있으면 안 됩니다."

    def test_num_stocks_limit(self):
        """종목 수 제한이 적용된다."""
        MLFactorStrategy = _try_import_ml_factor_strategy()
        if MLFactorStrategy is None:
            pytest.skip("MLFactorStrategy가 아직 구현되지 않았습니다.")

        fundamentals = _make_fundamentals(n=50)
        tickers = fundamentals["ticker"].tolist()
        prices = _make_prices_dict(tickers)

        np.random.seed(42)
        predictions = pd.Series(np.random.randn(50), index=tickers)
        mock_pipeline = _make_mock_pipeline(predictions=predictions)

        num_stocks = 10
        strategy = MLFactorStrategy(ml_pipeline=mock_pipeline, num_stocks=num_stocks)
        data = {"fundamentals": fundamentals, "prices": prices}
        signals = strategy.generate_signals("20240102", data)

        if signals:
            assert len(signals) <= num_stocks, (
                f"종목 수가 {num_stocks}개 이하여야 합니다: {len(signals)}"
            )

    def test_weights_equal(self):
        """use_risk_parity=False이면 동일 비중이다."""
        MLFactorStrategy = _try_import_ml_factor_strategy()
        if MLFactorStrategy is None:
            pytest.skip("MLFactorStrategy가 아직 구현되지 않았습니다.")

        fundamentals = _make_fundamentals(n=20)
        tickers = fundamentals["ticker"].tolist()
        prices = _make_prices_dict(tickers)
        mock_pipeline = _make_mock_pipeline()

        strategy = MLFactorStrategy(
            ml_pipeline=mock_pipeline, num_stocks=5, use_risk_parity=False,
        )
        data = {"fundamentals": fundamentals, "prices": prices}
        signals = strategy.generate_signals("20240102", data)

        if signals and len(signals) > 1:
            weights = list(signals.values())
            expected = 1.0 / len(signals)
            for w in weights:
                assert w == pytest.approx(expected, abs=1e-6), (
                    f"동일 비중이어야 합니다: expected={expected}, got={w}"
                )

    def test_empty_data(self):
        """빈 데이터 입력 시 빈 dict를 반환한다."""
        MLFactorStrategy = _try_import_ml_factor_strategy()
        if MLFactorStrategy is None:
            pytest.skip("MLFactorStrategy가 아직 구현되지 않았습니다.")

        mock_pipeline = _make_mock_pipeline()
        strategy = MLFactorStrategy(ml_pipeline=mock_pipeline)
        data = {"fundamentals": pd.DataFrame(), "prices": {}}
        signals = strategy.generate_signals("20240102", data)

        assert signals == {}, "빈 데이터 시 빈 dict여야 합니다."

    def test_market_cap_filter(self):
        """시가총액 필터가 적용된다."""
        MLFactorStrategy = _try_import_ml_factor_strategy()
        if MLFactorStrategy is None:
            pytest.skip("MLFactorStrategy가 아직 구현되지 않았습니다.")

        np.random.seed(42)
        n = 20
        tickers = [f"{i:06d}" for i in range(1, n + 1)]
        fundamentals = pd.DataFrame({
            "ticker": tickers,
            "name": [f"종목{i}" for i in range(1, n + 1)],
            "market": ["KOSPI"] * n,
            "pbr": np.abs(np.random.randn(n) + 1.5).round(2),
            "per": np.abs(np.random.randn(n) * 10 + 12).round(2),
            "eps": np.random.randint(500, 20000, n),
            "bps": np.random.randint(5000, 100000, n),
            "div_yield": (np.random.rand(n) * 5).round(2),
            "roe": np.random.uniform(5, 30, n).round(2),
            "close": np.random.randint(5000, 500000, n),
            # 절반은 시총 1000억 미만
            "market_cap": [50_000_000_000] * 10 + [500_000_000_000] * 10,
            "volume": np.random.randint(100_000, 5_000_000, n),
        })
        prices = _make_prices_dict(tickers)

        predictions = pd.Series(np.random.randn(n), index=tickers)
        mock_pipeline = _make_mock_pipeline(predictions=predictions)

        strategy = MLFactorStrategy(
            ml_pipeline=mock_pipeline,
            num_stocks=10,
            min_market_cap=100_000_000_000,
        )
        data = {"fundamentals": fundamentals, "prices": prices}
        signals = strategy.generate_signals("20240102", data)

        # 시가총액 필터 적용 시 소형주가 제외됨
        assert isinstance(signals, dict), "반환값이 dict여야 합니다."
        # 시총 1000억 이상인 종목만 포함
        small_cap_tickers = {f"{i:06d}" for i in range(1, 11)}  # 시총 500억
        for t in signals.keys():
            assert t not in small_cap_tickers, (
                f"시총 필터에 의해 {t}가 제외되어야 합니다."
            )

    def test_weights_sum_valid(self):
        """비중 합이 1.0에 가깝다."""
        MLFactorStrategy = _try_import_ml_factor_strategy()
        if MLFactorStrategy is None:
            pytest.skip("MLFactorStrategy가 아직 구현되지 않았습니다.")

        fundamentals = _make_fundamentals()
        tickers = fundamentals["ticker"].tolist()
        prices = _make_prices_dict(tickers)
        mock_pipeline = _make_mock_pipeline()

        strategy = MLFactorStrategy(ml_pipeline=mock_pipeline)
        data = {"fundamentals": fundamentals, "prices": prices}
        signals = strategy.generate_signals("20240102", data)

        if signals:
            total = sum(signals.values())
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"비중 합이 1.0이어야 합니다: {total}"
            )

    def test_strategy_abc(self):
        """MLFactorStrategy가 Strategy ABC의 서브클래스이다."""
        MLFactorStrategy = _try_import_ml_factor_strategy()
        if MLFactorStrategy is None:
            pytest.skip("MLFactorStrategy가 아직 구현되지 않았습니다.")

        mock_pipeline = _make_mock_pipeline()
        strategy = MLFactorStrategy(ml_pipeline=mock_pipeline)
        assert isinstance(strategy, Strategy), (
            "MLFactorStrategy는 Strategy의 인스턴스여야 합니다."
        )
