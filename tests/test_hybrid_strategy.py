"""하이브리드 전략 모듈(src/strategy/hybrid_strategy.py) 테스트.

코어+헤지 비중 합산, ETF 데이터 없을 때 fallback,
Strategy ABC 인터페이스 준수를 검증한다.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Strategy
from src.strategy.hybrid_strategy import HybridStrategy
from src.strategy.three_factor import ThreeFactorStrategy


# ===================================================================
# 헬퍼
# ===================================================================

def _make_core_strategy(**kwargs):
    """테스트용 ThreeFactorStrategy를 생성한다."""
    defaults = {
        "num_stocks": 10,
        "value_weight": 0.33,
        "momentum_weight": 0.33,
        "quality_weight": 0.34,
    }
    defaults.update(kwargs)
    return ThreeFactorStrategy(**defaults)


def _make_price_df(start: str, periods: int, base_price: int = 50000) -> pd.DataFrame:
    """mock 가격 DataFrame을 생성한다."""
    dates = pd.bdate_range(start, periods=periods)
    np.random.seed(42)
    close_arr = base_price + np.cumsum(np.random.randn(periods) * 200).astype(int)
    close_arr = np.maximum(close_arr, 100)

    return pd.DataFrame(
        {
            "open": close_arr - 100,
            "high": close_arr + 200,
            "low": close_arr - 200,
            "close": close_arr,
            "volume": [1_000_000] * periods,
        },
        index=dates,
    )


# ===================================================================
# Strategy ABC 인터페이스 테스트
# ===================================================================

class TestHybridStrategyInterface:
    """HybridStrategy가 Strategy ABC를 올바르게 구현하는지 검증."""

    def test_is_strategy_subclass(self):
        """HybridStrategy는 Strategy의 서브클래스이다."""
        assert issubclass(HybridStrategy, Strategy)

    def test_has_name_property(self):
        """name 프로퍼티가 존재하고 문자열을 반환한다."""
        core = _make_core_strategy()
        hybrid = HybridStrategy(core_strategy=core, core_weight=0.75)
        assert isinstance(hybrid.name, str)
        assert "Hybrid" in hybrid.name

    def test_has_generate_signals_method(self):
        """generate_signals 메서드가 존재한다."""
        core = _make_core_strategy()
        hybrid = HybridStrategy(core_strategy=core)
        assert hasattr(hybrid, "generate_signals")
        assert callable(hybrid.generate_signals)

    def test_name_contains_weights(self):
        """name에 코어/헤지 비중이 포함된다."""
        core = _make_core_strategy()
        hybrid = HybridStrategy(core_strategy=core, core_weight=0.70)
        assert "70/30" in hybrid.name


# ===================================================================
# 비중 합산 테스트
# ===================================================================

class TestHybridWeights:
    """코어+헤지 비중 합산 검증."""

    def test_core_weight_default(self):
        """기본 코어 비중은 0.75이다."""
        core = _make_core_strategy()
        hybrid = HybridStrategy(core_strategy=core)
        assert hybrid.core_weight == 0.75
        assert hybrid.hedge_weight == pytest.approx(0.25)

    def test_custom_core_weight(self):
        """사용자 정의 코어 비중이 올바르게 설정된다."""
        core = _make_core_strategy()
        hybrid = HybridStrategy(core_strategy=core, core_weight=0.60)
        assert hybrid.core_weight == 0.60
        assert hybrid.hedge_weight == pytest.approx(0.40)

    @patch.object(ThreeFactorStrategy, "generate_signals")
    def test_signals_sum_respects_weights(self, mock_core_signals):
        """코어 시그널의 비중이 core_weight로 스케일된다."""
        mock_core_signals.return_value = {"005930": 0.5, "000660": 0.5}

        core = _make_core_strategy()
        hybrid = HybridStrategy(core_strategy=core, core_weight=0.80)

        data = {
            "fundamentals": pd.DataFrame(),
            "prices": {},
            "index_prices": pd.Series(dtype=float),
        }

        signals = hybrid.generate_signals("20240102", data)

        # 코어 종목의 비중은 원래 비중 * 0.80
        assert "005930" in signals
        assert signals["005930"] == pytest.approx(0.5 * 0.80)

        # 헤지 부분(0.20)이 안전자산으로 배분되어야 함
        total = sum(signals.values())
        assert total == pytest.approx(1.0, abs=0.01)


# ===================================================================
# ETF 데이터 없을 때 Safe Asset Fallback
# ===================================================================

class TestHybridFallback:
    """ETF 가격 데이터 없을 때 안전자산 fallback 검증."""

    @patch.object(ThreeFactorStrategy, "generate_signals")
    def test_no_etf_data_falls_back_to_safe(self, mock_core_signals):
        """ETF 가격 데이터가 없으면 헤지 부분이 안전자산으로 배분된다."""
        mock_core_signals.return_value = {"005930": 1.0}

        core = _make_core_strategy()
        hybrid = HybridStrategy(core_strategy=core, core_weight=0.75)

        data = {
            "fundamentals": pd.DataFrame(),
            "prices": {},  # ETF 가격 없음
            "index_prices": pd.Series(dtype=float),
        }

        signals = hybrid.generate_signals("20240102", data)

        # 안전자산(214980)이 0.25 비중으로 배분
        safe_ticker = hybrid._hedge.safe_asset
        assert safe_ticker in signals
        assert signals[safe_ticker] == pytest.approx(0.25)

    @patch.object(ThreeFactorStrategy, "generate_signals")
    def test_core_failure_still_has_hedge(self, mock_core_signals):
        """코어 전략 실패 시에도 헤지 파트는 동작한다."""
        mock_core_signals.side_effect = Exception("코어 전략 오류")

        core = _make_core_strategy()
        hybrid = HybridStrategy(core_strategy=core, core_weight=0.75)

        data = {
            "fundamentals": pd.DataFrame(),
            "prices": {},
            "index_prices": pd.Series(dtype=float),
        }

        signals = hybrid.generate_signals("20240102", data)

        # 코어 실패, 헤지만 안전자산으로 배분
        safe_ticker = hybrid._hedge.safe_asset
        assert safe_ticker in signals
        assert signals[safe_ticker] == pytest.approx(0.25)


# ===================================================================
# update_holdings 전달 테스트
# ===================================================================

class TestHybridUpdateHoldings:
    """update_holdings가 코어 전략에 올바르게 전달되는지 검증."""

    def test_update_holdings_passed_to_core(self):
        """update_holdings 호출이 코어 전략에 전달된다."""
        core = _make_core_strategy()
        hybrid = HybridStrategy(core_strategy=core, core_weight=0.75)

        holdings = {"005930", "000660"}
        hybrid.update_holdings(holdings)

        assert core._current_holdings == holdings
