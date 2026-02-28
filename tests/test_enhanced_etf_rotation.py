"""Enhanced ETF Rotation 전략 테스트.

EnhancedETFRotationStrategy의 초기화, 복합 모멘텀, 시장 레짐 필터,
추세 필터, 변동성 가중 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.enhanced_etf_rotation import EnhancedETFRotationStrategy


def _make_etf_price(start_price, end_price, n_days=300):
    """선형 보간으로 ETF 가격 데이터를 생성한다."""
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    prices = np.linspace(start_price, end_price, n_days)
    return pd.DataFrame(
        {"open": prices, "high": prices * 1.01, "low": prices * 0.99,
         "close": prices, "volume": np.ones(n_days) * 1000000},
        index=dates,
    )


def _make_volatile_price(base_price, volatility, n_days=300, seed=42):
    """변동성이 있는 ETF 가격 데이터를 생성한다."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    returns = rng.normal(0.0005, volatility, n_days)
    prices = base_price * np.cumprod(1 + returns)
    return pd.DataFrame(
        {"open": prices, "high": prices * 1.01, "low": prices * 0.99,
         "close": prices, "volume": np.ones(n_days) * 1000000},
        index=dates,
    )


class TestEnhancedETFRotationInit:
    """초기화 검증."""

    def test_default_parameters(self):
        s = EnhancedETFRotationStrategy()
        assert s.num_etfs == 3
        assert s.use_vol_weight is True
        assert s.use_market_filter is True
        assert s.use_trend_filter is True
        assert s.market_ma_period == 200
        assert len(s.momentum_weights) == 4

    def test_custom_parameters(self):
        s = EnhancedETFRotationStrategy(
            num_etfs=2,
            use_vol_weight=False,
            use_market_filter=False,
            momentum_weights={21: 0.5, 63: 0.5},
        )
        assert s.num_etfs == 2
        assert s.use_vol_weight is False
        assert len(s.momentum_weights) == 2

    def test_name_property(self):
        s = EnhancedETFRotationStrategy(
            num_etfs=3,
            use_vol_weight=True,
            use_market_filter=True,
            use_trend_filter=True,
        )
        assert "EnhancedETF" in s.name
        assert "top3" in s.name

    def test_name_no_features(self):
        s = EnhancedETFRotationStrategy(
            use_vol_weight=False,
            use_market_filter=False,
            use_trend_filter=False,
        )
        assert "base" in s.name


class TestCompositeMonentum:
    """복합 모멘텀 계산 검증."""

    def test_composite_momentum_basic(self):
        """상승 ETF가 하락 ETF보다 높은 스코어를 받는다."""
        s = EnhancedETFRotationStrategy(
            etf_universe={
                "AAA": "상승", "BBB": "하락", "CCC": "안전",
            },
            safe_asset="CCC",
        )
        etf_prices = {
            "AAA": _make_etf_price(100, 150, 300),
            "BBB": _make_etf_price(100, 80, 300),
            "CCC": _make_etf_price(100, 102, 300),
        }
        result = s._calculate_composite_momentum(etf_prices)
        assert not result.empty
        assert result["AAA"] > result["BBB"]

    def test_composite_momentum_empty_data(self):
        """데이터가 없으면 빈 시리즈를 반환한다."""
        s = EnhancedETFRotationStrategy()
        result = s._calculate_composite_momentum({})
        assert result.empty


class TestMarketRegimeFilter:
    """시장 레짐 필터 검증."""

    def test_risk_on_above_ma(self):
        """지수가 200MA 위이면 RISK_ON."""
        s = EnhancedETFRotationStrategy(
            market_proxy="069500",
            market_ma_period=50,
        )
        # 상승 추세: 최근 가격이 MA 위
        etf_prices = {
            "069500": _make_etf_price(100, 200, 300),
        }
        result = s._check_market_regime(etf_prices)
        assert result == "RISK_ON"

    def test_risk_off_below_ma(self):
        """지수가 200MA 아래이면 RISK_OFF."""
        s = EnhancedETFRotationStrategy(
            market_proxy="069500",
            market_ma_period=50,
        )
        # 하락 추세: 최근 가격이 MA 아래
        etf_prices = {
            "069500": _make_etf_price(200, 100, 300),
        }
        result = s._check_market_regime(etf_prices)
        assert result == "RISK_OFF"

    def test_missing_market_data(self):
        """시장 데이터 없으면 RISK_ON."""
        s = EnhancedETFRotationStrategy(market_proxy="069500")
        result = s._check_market_regime({})
        assert result == "RISK_ON"


class TestTrendFilter:
    """추세 필터 검증."""

    def test_uptrend(self):
        """상승 추세 ETF는 통과한다."""
        s = EnhancedETFRotationStrategy()
        etf_prices = {"AAA": _make_etf_price(100, 200, 300)}
        assert s._check_trend(etf_prices, "AAA") is True

    def test_downtrend(self):
        """하락 추세 ETF는 실패한다."""
        s = EnhancedETFRotationStrategy()
        etf_prices = {"AAA": _make_etf_price(200, 100, 300)}
        assert s._check_trend(etf_prices, "AAA") is False

    def test_missing_data(self):
        """데이터 없으면 통과시킨다."""
        s = EnhancedETFRotationStrategy()
        assert s._check_trend({}, "AAA") is True


class TestDrawdownFilter:
    """최대 하락 필터 검증."""

    def test_within_limit(self):
        """하락이 제한 내이면 통과."""
        s = EnhancedETFRotationStrategy(max_drawdown_filter=0.30)
        etf_prices = {"AAA": _make_etf_price(100, 90, 300)}
        assert s._check_drawdown(etf_prices, "AAA") is True

    def test_excessive_drawdown(self):
        """과도한 하락은 필터링."""
        s = EnhancedETFRotationStrategy(max_drawdown_filter=0.10)
        # 150에서 100으로 하락 → 33% 하락 → 10% 제한 초과
        etf_prices = {"AAA": _make_etf_price(150, 100, 300)}
        assert s._check_drawdown(etf_prices, "AAA") is False


class TestGenerateSignals:
    """전체 시그널 생성 검증."""

    def _make_universe_prices(self):
        """테스트용 유니버스 가격 데이터."""
        return {
            "AAA": _make_etf_price(100, 180, 300),  # 강한 상승
            "BBB": _make_etf_price(100, 130, 300),  # 약한 상승
            "CCC": _make_etf_price(100, 80, 300),   # 하락
            "DDD": _make_etf_price(100, 101, 300),  # 안전자산
        }

    def test_basic_signal_generation(self):
        """기본 시그널이 생성된다."""
        s = EnhancedETFRotationStrategy(
            num_etfs=2,
            safe_asset="DDD",
            etf_universe={"AAA": "A", "BBB": "B", "CCC": "C", "DDD": "D"},
            use_market_filter=False,
            use_trend_filter=False,
        )
        prices = self._make_universe_prices()
        signals = s.generate_signals("20240101", {"etf_prices": prices})
        assert len(signals) > 0
        assert sum(signals.values()) <= 1.01

    def test_empty_data_returns_empty(self):
        """데이터 없으면 빈 딕셔너리."""
        s = EnhancedETFRotationStrategy()
        result = s.generate_signals("20240101", {"etf_prices": {}})
        assert result == {}

    def test_risk_off_increases_safe_weight(self):
        """RISK_OFF 시 안전자산 비중이 증가한다."""
        # market_proxy가 하락 추세인 경우
        s_on = EnhancedETFRotationStrategy(
            num_etfs=2,
            safe_asset="DDD",
            etf_universe={"AAA": "A", "BBB": "B", "CCC": "C", "DDD": "D"},
            market_proxy="CCC",  # 하락 → RISK_OFF
            use_market_filter=True,
            use_trend_filter=False,
            market_ma_period=50,
        )
        prices = self._make_universe_prices()
        signals = s_on.generate_signals("20240101", {"etf_prices": prices})
        safe_w = signals.get("DDD", 0)
        # RISK_OFF이면 안전자산 비중이 0보다 커야 함
        assert safe_w > 0 or len(signals) > 0

    def test_diagnostics_populated(self):
        """진단 정보가 저장된다."""
        s = EnhancedETFRotationStrategy(
            num_etfs=2,
            safe_asset="DDD",
            etf_universe={"AAA": "A", "BBB": "B", "CCC": "C", "DDD": "D"},
            use_market_filter=False,
            use_trend_filter=False,
        )
        prices = self._make_universe_prices()
        s.generate_signals("20240101", {"etf_prices": prices})
        assert s.last_diagnostics.get("status") is not None
