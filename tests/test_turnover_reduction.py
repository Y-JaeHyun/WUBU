"""ThreeFactorStrategy 회전율 감소(turnover reduction) 기능 테스트.

turnover_buffer, holding_bonus 파라미터와 update_holdings 메서드를
통한 회전율 감소 로직을 검증한다.
- buffer=0 → 기존 동작과 동일 (하위 호환)
- buffer zone 내 기존 보유 종목 유지
- buffer zone 밖 기존 보유 종목 탈락
- holding_bonus 스코어 가산 효과
- update_holdings 내부 상태 관리
- 빈 보유 종목 → 순수 Top-N 선정
"""

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pytest

from src.strategy.three_factor import ThreeFactorStrategy


# ===================================================================
# 헬퍼 함수
# ===================================================================

def _make_fundamentals(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """회전율 테스트용 펀더멘탈 DataFrame을 생성한다.

    시가총액은 모두 min_market_cap 이상으로 설정하여
    필터링에 의한 제거를 방지한다.
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


def _make_prices(n: int = 30, seed: int = 42) -> dict:
    """회전율 테스트용 가격 데이터 dict를 생성한다.

    Returns:
        dict[ticker -> DataFrame with OHLCV and '종가' column]
    """
    np.random.seed(seed)
    result = {}  # type: Dict[str, pd.DataFrame]
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
            "종가": close,
            "volume": np.random.randint(100000, 5000000, 252),
        }, index=dates)
        df.index.name = "date"
        result[ticker] = df
    return result


def _make_data(n: int = 30, seed: int = 42) -> dict:
    """fundamentals + prices를 합친 data dict를 반환한다."""
    return {
        "fundamentals": _make_fundamentals(n=n, seed=seed),
        "prices": _make_prices(n=n, seed=seed),
    }


# ===================================================================
# buffer=0 → 기존 동작과 동일 (하위 호환)
# ===================================================================

class TestBufferZeroBackwardCompat:
    """turnover_buffer=0일 때 기존 Top-N 동작과 동일함을 검증한다."""

    def test_buffer_zero_same_as_default(self):
        """buffer=0 전략과 기본 전략의 시그널이 동일하다."""
        data = _make_data(n=30, seed=42)

        default_strategy = ThreeFactorStrategy(num_stocks=10)
        buffer_strategy = ThreeFactorStrategy(
            num_stocks=10, turnover_buffer=0, holding_bonus=0.0,
        )

        signals_default = default_strategy.generate_signals("20240102", data)
        signals_buffer = buffer_strategy.generate_signals("20240102", data)

        assert set(signals_default.keys()) == set(signals_buffer.keys()), (
            "buffer=0이면 기본 전략과 동일한 종목을 선택해야 합니다."
        )

    def test_buffer_zero_with_holdings_same_as_default(self):
        """buffer=0이면 holdings를 설정해도 기본 동작과 동일하다."""
        data = _make_data(n=30, seed=42)

        default_strategy = ThreeFactorStrategy(num_stocks=10)
        buffer_strategy = ThreeFactorStrategy(
            num_stocks=10, turnover_buffer=0, holding_bonus=0.0,
        )
        # buffer=0이면 holdings 업데이트가 무의미해야 함
        buffer_strategy.update_holdings({"000001", "000002", "000003"})

        signals_default = default_strategy.generate_signals("20240102", data)
        signals_buffer = buffer_strategy.generate_signals("20240102", data)

        assert set(signals_default.keys()) == set(signals_buffer.keys()), (
            "buffer=0이면 holdings 설정과 무관하게 기본 동작이어야 합니다."
        )


# ===================================================================
# buffer zone 내 보유 종목 유지
# ===================================================================

class TestHoldingRetainedInBufferZone:
    """기존 보유 종목이 buffer zone 내에 있으면 유지됨을 검증한다."""

    def test_holding_within_buffer_kept(self):
        """Top N 밖이지만 N+buffer 내인 보유 종목이 유지된다."""
        data = _make_data(n=50, seed=99)
        num_stocks = 20
        buffer = 5

        # 먼저 buffer 없이 Top-N 결과 확인
        base_strategy = ThreeFactorStrategy(num_stocks=num_stocks)
        base_signals = base_strategy.generate_signals("20240102", data)
        top_n_tickers = set(base_signals.keys())

        # Top N+buffer 범위의 종목을 파악 (더 넓은 범위에서 선택)
        wide_strategy = ThreeFactorStrategy(num_stocks=num_stocks + buffer)
        wide_signals = wide_strategy.generate_signals("20240102", data)
        wide_tickers = set(wide_signals.keys())

        # buffer zone에만 있는 종목 (Top N에는 없지만 Top N+buffer에는 있는)
        buffer_zone_tickers = wide_tickers - top_n_tickers
        if not buffer_zone_tickers:
            pytest.skip("buffer zone에 해당하는 종목이 없습니다.")

        # buffer zone 종목 중 하나를 보유 종목으로 설정
        holding_ticker = next(iter(buffer_zone_tickers))

        strategy = ThreeFactorStrategy(
            num_stocks=num_stocks, turnover_buffer=buffer,
        )
        strategy.update_holdings({holding_ticker})
        signals = strategy.generate_signals("20240102", data)

        assert holding_ticker in signals, (
            f"buffer zone 내 보유 종목 '{holding_ticker}'가 유지되어야 합니다."
        )


# ===================================================================
# buffer zone 밖 보유 종목 탈락
# ===================================================================

class TestHoldingDroppedOutsideBuffer:
    """buffer zone 밖의 보유 종목이 탈락됨을 검증한다."""

    def test_holding_outside_buffer_dropped(self):
        """N+buffer 밖의 보유 종목은 탈락한다."""
        data = _make_data(n=50, seed=99)
        num_stocks = 20
        buffer = 5
        exit_threshold = num_stocks + buffer  # 25

        # exit_threshold 밖의 종목을 파악
        wide_strategy = ThreeFactorStrategy(num_stocks=exit_threshold)
        wide_signals = wide_strategy.generate_signals("20240102", data)
        wide_tickers = set(wide_signals.keys())

        all_tickers = set(data["fundamentals"]["ticker"].tolist())
        outside_tickers = all_tickers - wide_tickers
        if not outside_tickers:
            pytest.skip("buffer zone 밖에 해당하는 종목이 없습니다.")

        # buffer zone 밖의 종목을 보유 종목으로 설정
        dropped_ticker = next(iter(outside_tickers))

        strategy = ThreeFactorStrategy(
            num_stocks=num_stocks, turnover_buffer=buffer,
        )
        strategy.update_holdings({dropped_ticker})
        signals = strategy.generate_signals("20240102", data)

        assert dropped_ticker not in signals, (
            f"buffer zone 밖 보유 종목 '{dropped_ticker}'는 탈락해야 합니다."
        )


# ===================================================================
# holding_bonus 효과
# ===================================================================

class TestHoldingBonusEffect:
    """holding_bonus가 기존 보유 종목 스코어에 가산됨을 검증한다."""

    def test_holding_bonus_promotes_holding(self):
        """holding_bonus로 인해 경계선 보유 종목이 선택될 수 있다."""
        data = _make_data(n=50, seed=99)
        num_stocks = 20

        # 먼저 buffer 없이 Top-N 결과 확인
        base_strategy = ThreeFactorStrategy(num_stocks=num_stocks)
        base_signals = base_strategy.generate_signals("20240102", data)
        top_n_tickers = set(base_signals.keys())

        # Top 25에는 있지만 Top 20에는 없는 종목 (경계선)
        wider_strategy = ThreeFactorStrategy(num_stocks=25)
        wider_signals = wider_strategy.generate_signals("20240102", data)
        marginal_tickers = set(wider_signals.keys()) - top_n_tickers

        if not marginal_tickers:
            pytest.skip("경계선 종목이 없습니다.")

        marginal_ticker = next(iter(marginal_tickers))

        # 큰 holding_bonus로 경계선 종목이 올라오는지 확인
        strategy = ThreeFactorStrategy(
            num_stocks=num_stocks,
            turnover_buffer=5,
            holding_bonus=100.0,  # 매우 큰 보너스
        )
        strategy.update_holdings({marginal_ticker})
        signals = strategy.generate_signals("20240102", data)

        assert marginal_ticker in signals, (
            f"큰 holding_bonus로 경계선 보유 종목 '{marginal_ticker}'가 "
            "선택되어야 합니다."
        )

    def test_holding_bonus_zero_no_effect(self):
        """holding_bonus=0이면 스코어 가산이 없다."""
        data = _make_data(n=30, seed=42)

        strategy_no_bonus = ThreeFactorStrategy(
            num_stocks=10, turnover_buffer=5, holding_bonus=0.0,
        )
        strategy_with_bonus = ThreeFactorStrategy(
            num_stocks=10, turnover_buffer=5, holding_bonus=0.0,
        )

        holdings = {"000001", "000002"}
        strategy_no_bonus.update_holdings(holdings)
        strategy_with_bonus.update_holdings(holdings)

        signals_a = strategy_no_bonus.generate_signals("20240102", data)
        signals_b = strategy_with_bonus.generate_signals("20240102", data)

        assert set(signals_a.keys()) == set(signals_b.keys()), (
            "holding_bonus=0이면 두 전략의 결과가 동일해야 합니다."
        )


# ===================================================================
# update_holdings 메서드 동작
# ===================================================================

class TestUpdateHoldings:
    """update_holdings 메서드가 내부 상태를 올바르게 관리함을 검증한다."""

    def test_update_holdings_sets_state(self):
        """update_holdings 호출 후 내부 _current_holdings가 갱신된다."""
        strategy = ThreeFactorStrategy(turnover_buffer=5)
        holdings = {"000001", "000002", "000003"}

        strategy.update_holdings(holdings)

        assert strategy._current_holdings == holdings, (
            "update_holdings 후 _current_holdings가 설정되어야 합니다."
        )

    def test_update_holdings_replaces_previous(self):
        """update_holdings는 이전 보유 종목을 완전히 교체한다."""
        strategy = ThreeFactorStrategy(turnover_buffer=5)

        strategy.update_holdings({"000001", "000002"})
        strategy.update_holdings({"000003", "000004"})

        assert strategy._current_holdings == {"000003", "000004"}, (
            "update_holdings는 이전 상태를 교체해야 합니다."
        )
        assert "000001" not in strategy._current_holdings, (
            "이전 보유 종목이 남아 있으면 안 됩니다."
        )

    def test_update_holdings_empty_set(self):
        """빈 집합으로 update_holdings를 호출하면 보유 종목이 초기화된다."""
        strategy = ThreeFactorStrategy(turnover_buffer=5)
        strategy.update_holdings({"000001", "000002"})

        strategy.update_holdings(set())

        assert strategy._current_holdings == set(), (
            "빈 집합으로 호출하면 보유 종목이 비어야 합니다."
        )

    def test_update_holdings_creates_copy(self):
        """update_holdings는 입력 집합의 복사본을 저장한다."""
        strategy = ThreeFactorStrategy(turnover_buffer=5)
        holdings = {"000001", "000002"}

        strategy.update_holdings(holdings)
        holdings.add("000003")  # 원본 변경

        assert "000003" not in strategy._current_holdings, (
            "원본 집합 변경이 내부 상태에 영향을 주면 안 됩니다."
        )

    def test_initial_holdings_empty(self):
        """초기 상태에서 _current_holdings는 빈 집합이다."""
        strategy = ThreeFactorStrategy(turnover_buffer=5)

        assert strategy._current_holdings == set(), (
            "초기 _current_holdings는 빈 집합이어야 합니다."
        )


# ===================================================================
# 빈 보유 종목 → 순수 Top-N 선정
# ===================================================================

class TestEmptyHoldingsPureTopN:
    """보유 종목이 없으면 buffer가 있어도 순수 Top-N과 동일함을 검증한다."""

    def test_empty_holdings_same_as_no_buffer(self):
        """빈 holdings + buffer > 0이면 기본 Top-N과 동일하다."""
        data = _make_data(n=30, seed=42)

        base_strategy = ThreeFactorStrategy(num_stocks=10)
        buffer_strategy = ThreeFactorStrategy(
            num_stocks=10, turnover_buffer=5,
        )
        # holdings를 설정하지 않음 (빈 상태)

        signals_base = base_strategy.generate_signals("20240102", data)
        signals_buffer = buffer_strategy.generate_signals("20240102", data)

        assert set(signals_base.keys()) == set(signals_buffer.keys()), (
            "빈 holdings에서는 buffer가 있어도 기본 Top-N과 동일해야 합니다."
        )

    def test_empty_holdings_after_clear(self):
        """holdings를 비운 후에는 기본 Top-N과 동일하다."""
        data = _make_data(n=30, seed=42)

        base_strategy = ThreeFactorStrategy(num_stocks=10)
        buffer_strategy = ThreeFactorStrategy(
            num_stocks=10, turnover_buffer=5,
        )
        buffer_strategy.update_holdings({"000001", "000002"})
        buffer_strategy.update_holdings(set())  # 비우기

        signals_base = base_strategy.generate_signals("20240102", data)
        signals_buffer = buffer_strategy.generate_signals("20240102", data)

        assert set(signals_base.keys()) == set(signals_buffer.keys()), (
            "holdings를 비운 후에는 기본 Top-N과 동일해야 합니다."
        )

    def test_num_stocks_respected_with_buffer(self):
        """buffer가 있어도 최종 포트폴리오 종목 수는 num_stocks 이하이다."""
        data = _make_data(n=50, seed=42)
        num_stocks = 10

        strategy = ThreeFactorStrategy(
            num_stocks=num_stocks, turnover_buffer=10,
        )
        # 많은 종목을 보유 종목으로 설정해도 num_stocks 이하여야 함
        strategy.update_holdings(
            {f"{i:06d}" for i in range(1, 31)}  # 30개 종목
        )
        signals = strategy.generate_signals("20240102", data)

        assert len(signals) <= num_stocks, (
            f"종목 수({len(signals)})가 num_stocks({num_stocks}) 이하여야 합니다."
        )
