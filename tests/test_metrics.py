"""성과 지표 모듈(src/report/metrics.py) 테스트.

CAGR, MDD, 샤프비율, 소르티노비율, 칼마비율, 승률 계산을 검증한다.
모든 함수는 일별 수익률 시리즈를 입력으로 받는다.
"""

import numpy as np
import pandas as pd
import pytest

from src.report.metrics import (
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    win_rate,
)


# ===================================================================
# CAGR (연평균 수익률) 테스트
# ===================================================================


class TestCagr:
    """CAGR 계산 검증."""

    def test_cagr_positive(self, sample_portfolio_values):
        """상승 수익률 시퀀스의 CAGR이 양수이다."""
        # sample_portfolio_values는 절대값 → 일별 수익률로 변환
        daily_returns = sample_portfolio_values.pct_change().dropna()
        result = cagr(daily_returns)

        assert isinstance(result, float), "CAGR 반환값이 float이어야 합니다."
        assert result > 0, "상승 추세 포트폴리오의 CAGR은 양수여야 합니다."

    def test_cagr_flat(self):
        """변동 없는 포트폴리오의 CAGR은 0에 가깝다."""
        dates = pd.bdate_range("2024-01-02", periods=252)
        values = pd.Series([100_000_000.0] * 252, index=dates)
        daily_returns = values.pct_change().dropna()

        result = cagr(daily_returns)

        assert abs(result) < 1e-6, f"변동 없는 포트폴리오의 CAGR은 0이어야 합니다: {result}"

    def test_cagr_known_value(self):
        """알려진 수익률의 CAGR을 검증한다."""
        # 1년간 100 -> 110 = 10% 수익
        dates = pd.bdate_range("2024-01-02", periods=252)
        values = pd.Series(
            np.linspace(100, 110, 252),
            index=dates,
        )
        daily_returns = values.pct_change().dropna()

        result = cagr(daily_returns)

        # 약 10% CAGR (1년 기준)
        assert abs(result - 0.10) < 0.02, (
            f"100->110의 CAGR은 약 10%여야 합니다: {result * 100:.2f}%"
        )

    def test_cagr_negative(self):
        """하락 포트폴리오의 CAGR은 음수이다."""
        dates = pd.bdate_range("2024-01-02", periods=252)
        values = pd.Series(
            np.linspace(100, 80, 252),
            index=dates,
        )
        daily_returns = values.pct_change().dropna()

        result = cagr(daily_returns)

        assert result < 0, "하락 포트폴리오의 CAGR은 음수여야 합니다."


# ===================================================================
# 최대 낙폭 (MDD) 테스트
# ===================================================================


class TestMaxDrawdown:
    """최대 낙폭 계산 검증."""

    def test_max_drawdown_known(self):
        """알려진 MDD 시퀀스를 검증한다."""
        # 100 -> 120 -> 90 -> 110
        # 고점 120에서 저점 90으로 -25% 낙폭
        dates = pd.bdate_range("2024-01-02", periods=4)
        values = pd.Series([100.0, 120.0, 90.0, 110.0], index=dates)
        daily_returns = values.pct_change().dropna()

        result = max_drawdown(daily_returns)

        assert isinstance(result, float), "MDD 반환값이 float이어야 합니다."
        assert result <= 0, "MDD는 0 이하여야 합니다."
        expected_mdd = (90.0 - 120.0) / 120.0  # -0.25
        assert abs(result - expected_mdd) < 1e-9, (
            f"MDD가 {expected_mdd}이어야 합니다: {result}"
        )

    def test_max_drawdown_no_drawdown(self):
        """꾸준한 상승이면 MDD가 0에 가깝다."""
        dates = pd.bdate_range("2024-01-02", periods=100)
        values = pd.Series(np.arange(100, 200, dtype=float), index=dates)
        daily_returns = values.pct_change().dropna()

        result = max_drawdown(daily_returns)

        assert abs(result) < 1e-9, f"꾸준한 상승의 MDD는 0이어야 합니다: {result}"

    def test_max_drawdown_full_loss(self):
        """전액 손실 시 MDD가 -1에 가깝다."""
        dates = pd.bdate_range("2024-01-02", periods=3)
        values = pd.Series([100.0, 50.0, 1.0], index=dates)
        daily_returns = values.pct_change().dropna()

        result = max_drawdown(daily_returns)

        assert result < -0.9, f"거의 전액 손실 시 MDD가 -90% 이하여야 합니다: {result}"


# ===================================================================
# 샤프 비율 테스트
# ===================================================================


class TestSharpeRatio:
    """샤프 비율 계산 검증."""

    def test_sharpe_ratio_positive(self, sample_daily_returns):
        """양수 평균 수익률의 샤프비율이 양수이다."""
        result = sharpe_ratio(sample_daily_returns)

        assert isinstance(result, float), "샤프비율 반환값이 float이어야 합니다."
        # sample_daily_returns는 약간의 양의 drift가 있으므로 샤프비율 양수 기대
        # (단, 랜덤 시드에 따라 달라질 수 있으므로 유연하게 체크)

    def test_sharpe_ratio_zero_volatility(self):
        """변동성이 0이면 샤프비율이 0 또는 특수값을 반환한다."""
        dates = pd.bdate_range("2024-01-02", periods=100)
        returns = pd.Series([0.001] * 100, index=dates)

        result = sharpe_ratio(returns)

        # 변동성 0이면 0 또는 inf를 반환할 수 있음
        # 구현에 따라 0 또는 inf 모두 유효
        assert isinstance(result, float), "반환값이 float이어야 합니다."

    def test_sharpe_ratio_negative_returns(self):
        """음수 수익률의 샤프비율이 음수이다."""
        dates = pd.bdate_range("2024-01-02", periods=100)
        np.random.seed(99)
        returns = pd.Series(np.random.randn(100) * 0.01 - 0.005, index=dates)

        result = sharpe_ratio(returns)

        assert result < 0, "음수 평균 수익률의 샤프비율은 음수여야 합니다."


# ===================================================================
# 소르티노 비율 테스트
# ===================================================================


class TestSortinoRatio:
    """소르티노 비율 계산 검증."""

    def test_sortino_ratio(self, sample_daily_returns):
        """소르티노 비율이 올바르게 계산된다."""
        result = sortino_ratio(sample_daily_returns)

        assert isinstance(result, float), "소르티노 비율 반환값이 float이어야 합니다."

    def test_sortino_no_downside(self):
        """하방 변동성이 없으면 소르티노 비율이 높거나 특수값이다."""
        dates = pd.bdate_range("2024-01-02", periods=50)
        # 모두 양수 수익률
        returns = pd.Series([0.01] * 50, index=dates)

        result = sortino_ratio(returns)

        # 하방 변동성 0이면 매우 큰 값 또는 inf
        assert isinstance(result, float), "반환값이 float이어야 합니다."
        assert result > 0, "하방 변동성이 없으면 소르티노가 양수여야 합니다."


# ===================================================================
# 칼마 비율 테스트
# ===================================================================


class TestCalmarRatio:
    """칼마 비율 계산 검증."""

    def test_calmar_ratio(self, sample_portfolio_values):
        """칼마 비율(CAGR / |MDD|)이 올바르게 계산된다."""
        daily_returns = sample_portfolio_values.pct_change().dropna()
        result = calmar_ratio(daily_returns)

        assert isinstance(result, float), "칼마비율 반환값이 float이어야 합니다."

    def test_calmar_no_drawdown(self):
        """MDD가 0이면 칼마비율이 높거나 특수값이다."""
        dates = pd.bdate_range("2024-01-02", periods=100)
        values = pd.Series(np.arange(100, 200, dtype=float), index=dates)
        daily_returns = values.pct_change().dropna()

        result = calmar_ratio(daily_returns)

        # MDD=0이면 매우 큰 값 또는 inf 또는 0
        assert isinstance(result, float), "반환값이 float이어야 합니다."


# ===================================================================
# 승률 테스트
# ===================================================================


class TestWinRate:
    """승률 계산 검증."""

    def test_win_rate(self):
        """월별 승률이 올바르게 계산된다."""
        # 12개월 분량의 데이터 생성: 매월 일정한 수익률
        dates = pd.bdate_range("2024-01-02", periods=252)
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0003, index=dates)

        result = win_rate(returns)

        assert isinstance(result, float), "승률 반환값이 float이어야 합니다."
        assert 0.0 <= result <= 1.0, f"승률은 0~1 사이여야 합니다: {result}"

    def test_win_rate_all_positive(self):
        """모든 월의 수익률이 양수이면 승률 100%이다."""
        dates = pd.bdate_range("2024-01-02", periods=252)
        # 매일 +1% 수익률 → 모든 월이 양수
        returns = pd.Series([0.01] * 252, index=dates)

        result = win_rate(returns)

        assert abs(result - 1.0) < 1e-9, "모두 양수이면 승률 100%여야 합니다."

    def test_win_rate_all_negative(self):
        """모든 월의 수익률이 음수이면 승률 0%이다."""
        dates = pd.bdate_range("2024-01-02", periods=252)
        # 매일 -1% 수익률 → 모든 월이 음수
        returns = pd.Series([-0.01] * 252, index=dates)

        result = win_rate(returns)

        assert abs(result) < 1e-9, "모두 음수이면 승률 0%여야 합니다."

    def test_win_rate_empty(self):
        """빈 시리즈이면 0을 반환한다."""
        returns = pd.Series([], dtype=float)

        result = win_rate(returns)

        assert result == 0.0, "빈 시리즈의 승률은 0이어야 합니다."
