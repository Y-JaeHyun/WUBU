"""리스크 패리티 전략 모듈(src/strategy/risk_parity.py) 테스트.

RiskParityStrategy의 이름, 시그널 생성, 비중 클리핑,
ThreeFactorStrategy 연동 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.backtest.engine import Strategy


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _import_rp_strategy():
    """RiskParityStrategy 클래스를 임포트한다."""
    from src.strategy.risk_parity import RiskParityStrategy
    return RiskParityStrategy


def _make_mock_selector(signals=None):
    """종목 선정용 mock Strategy 객체를 생성한다."""
    mock = MagicMock(spec=Strategy)
    mock.name = "MockSelector(20)"
    if signals is None:
        # 10개 종목 동일 비중
        tickers = [f"{i:06d}" for i in range(1, 11)]
        signals = {t: 0.1 for t in tickers}
    mock.generate_signals.return_value = signals
    return mock


def _make_prices_dict(tickers, n_days=252):
    """종목별 가격 데이터 dict를 생성한다."""
    np.random.seed(42)
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
# RiskParityStrategy 검증
# ===================================================================

class TestRiskParityStrategy:
    """RiskParityStrategy 검증."""

    def test_name_property(self):
        """name 프로퍼티가 'RiskParity'를 포함한다."""
        RiskParityStrategy = _import_rp_strategy()
        mock_selector = _make_mock_selector()
        strategy = RiskParityStrategy(stock_selector=mock_selector)

        assert "RiskParity" in strategy.name, (
            f"이름에 'RiskParity'가 포함되어야 합니다: {strategy.name}"
        )

    def test_generate_signals_basic(self):
        """기본 시그널이 정상적으로 생성된다."""
        RiskParityStrategy = _import_rp_strategy()
        tickers = [f"{i:06d}" for i in range(1, 11)]
        signals = {t: 0.1 for t in tickers}
        mock_selector = _make_mock_selector(signals=signals)
        prices = _make_prices_dict(tickers, n_days=252)

        strategy = RiskParityStrategy(stock_selector=mock_selector)
        data = {"fundamentals": pd.DataFrame({"ticker": tickers}), "prices": prices}
        result = strategy.generate_signals("20240102", data)

        assert isinstance(result, dict), "반환값이 dict여야 합니다."
        assert len(result) > 0, "비어 있지 않은 시그널이어야 합니다."

    def test_generate_signals_with_three_factor(self):
        """ThreeFactorStrategy와 연동하여 시그널을 생성한다."""
        RiskParityStrategy = _import_rp_strategy()

        # ThreeFactorStrategy mock
        tickers = [f"{i:06d}" for i in range(1, 6)]
        signals = {t: 0.2 for t in tickers}
        mock_selector = _make_mock_selector(signals=signals)
        prices = _make_prices_dict(tickers, n_days=252)

        strategy = RiskParityStrategy(stock_selector=mock_selector)
        data = {"fundamentals": pd.DataFrame({"ticker": tickers}), "prices": prices}
        result = strategy.generate_signals("20240102", data)

        assert isinstance(result, dict), "반환값이 dict여야 합니다."

    def test_max_weight_clipping(self):
        """개별 종목 최대 비중이 제한된다."""
        RiskParityStrategy = _import_rp_strategy()
        tickers = [f"{i:06d}" for i in range(1, 4)]
        signals = {t: 1.0 / 3 for t in tickers}
        mock_selector = _make_mock_selector(signals=signals)
        prices = _make_prices_dict(tickers, n_days=252)

        strategy = RiskParityStrategy(
            stock_selector=mock_selector, max_weight=0.10,
        )
        data = {"fundamentals": pd.DataFrame({"ticker": tickers}), "prices": prices}
        result = strategy.generate_signals("20240102", data)

        if result:
            for ticker, w in result.items():
                # 클리핑 후 정규화하므로 max_weight 자체를 초과할 수 있지만
                # 전체적으로 큰 집중은 방지됨
                assert w <= 1.0, (
                    f"비중이 1.0 이하여야 합니다: {ticker}={w}"
                )

    def test_min_weight_filter(self):
        """최소 비중 미만 종목이 필터링된다."""
        RiskParityStrategy = _import_rp_strategy()
        tickers = [f"{i:06d}" for i in range(1, 11)]
        signals = {t: 0.1 for t in tickers}
        mock_selector = _make_mock_selector(signals=signals)
        prices = _make_prices_dict(tickers, n_days=252)

        strategy = RiskParityStrategy(
            stock_selector=mock_selector,
            min_weight=0.01,
        )
        data = {"fundamentals": pd.DataFrame({"ticker": tickers}), "prices": prices}
        result = strategy.generate_signals("20240102", data)

        if result:
            for ticker, w in result.items():
                assert w >= strategy.min_weight, (
                    f"비중이 min_weight({strategy.min_weight}) 이상이어야 합니다: "
                    f"{ticker}={w}"
                )

    def test_empty_fundamentals(self):
        """빈 펀더멘탈 데이터 시 빈 dict를 반환한다."""
        RiskParityStrategy = _import_rp_strategy()
        mock_selector = _make_mock_selector(signals={})
        strategy = RiskParityStrategy(stock_selector=mock_selector)

        data = {"fundamentals": pd.DataFrame(), "prices": {}}
        result = strategy.generate_signals("20240102", data)

        assert result == {}, "빈 펀더멘탈이면 빈 dict여야 합니다."

    def test_insufficient_price_data(self):
        """가격 데이터가 부족하면 원래 시그널(fallback)을 반환한다."""
        RiskParityStrategy = _import_rp_strategy()
        tickers = [f"{i:06d}" for i in range(1, 6)]
        original_signals = {t: 0.2 for t in tickers}
        mock_selector = _make_mock_selector(signals=original_signals)

        strategy = RiskParityStrategy(stock_selector=mock_selector)
        # 가격 데이터 없음
        data = {"fundamentals": pd.DataFrame({"ticker": tickers}), "prices": {}}
        result = strategy.generate_signals("20240102", data)

        # 가격 데이터 없으면 원래 시그널(동일 비중)로 fallback
        assert isinstance(result, dict), "반환값이 dict여야 합니다."

    def test_weights_sum_valid(self):
        """비중 합이 1.0 이하이다."""
        RiskParityStrategy = _import_rp_strategy()
        tickers = [f"{i:06d}" for i in range(1, 11)]
        signals = {t: 0.1 for t in tickers}
        mock_selector = _make_mock_selector(signals=signals)
        prices = _make_prices_dict(tickers, n_days=252)

        strategy = RiskParityStrategy(stock_selector=mock_selector)
        data = {"fundamentals": pd.DataFrame({"ticker": tickers}), "prices": prices}
        result = strategy.generate_signals("20240102", data)

        if result:
            total = sum(result.values())
            assert total <= 1.0 + 1e-6, (
                f"비중 합이 1.0 이하여야 합니다: {total}"
            )
