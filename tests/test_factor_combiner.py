"""팩터 결합 모듈(src/strategy/factor_combiner.py) 테스트.

zscore 결합, rank 결합, 가중치 적용, 클리핑, NaN 처리 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.factor_combiner import combine_zscore, combine_rank


# ===================================================================
# 헬퍼: 테스트용 팩터 데이터 생성
# ===================================================================


def _make_factor_data():
    """테스트용 팩터 DataFrame들을 생성한다."""
    tickers = ["A", "B", "C", "D", "E"]
    factor1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=tickers, name="factor1")
    factor2 = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=tickers, name="factor2")
    return factor1, factor2


def _make_factor_data_with_nan():
    """NaN이 포함된 팩터 데이터를 생성한다."""
    tickers = ["A", "B", "C", "D", "E"]
    factor1 = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0], index=tickers, name="factor1")
    factor2 = pd.Series([5.0, 4.0, np.nan, 2.0, 1.0], index=tickers, name="factor2")
    return factor1, factor2


# ===================================================================
# combine_zscore 테스트
# ===================================================================


class TestCombineZscore:
    """zscore 결합 함수 검증."""

    def test_combine_zscore_equal_weight(self):
        """동일 가중치로 zscore 결합이 올바르게 수행된다."""
        f1, f2 = _make_factor_data()
        result = combine_zscore(f1, f2)

        assert isinstance(result, pd.Series), "반환값이 pd.Series여야 합니다."
        assert len(result) == 5, "결합 결과 길이가 입력과 같아야 합니다."

        # 동일 가중치 zscore 결합이므로, f1과 f2가 정반대 -> 합이 거의 0
        # 각 팩터의 zscore는 대칭적이므로, 결합 결과가 0 근처여야 함
        assert abs(result.mean()) < 1e-9, (
            "대칭 팩터의 동일 가중치 결합 평균이 0이어야 합니다."
        )

    def test_combine_zscore_custom_weight(self):
        """커스텀 가중치로 zscore 결합이 수행된다."""
        f1, f2 = _make_factor_data()
        result = combine_zscore(f1, f2, value_weight=0.7, momentum_weight=0.3)

        assert isinstance(result, pd.Series), "반환값이 pd.Series여야 합니다."
        assert len(result) == 5, "결합 결과 길이가 입력과 같아야 합니다."

        # f1 가중치가 높으므로, f1이 높은 종목(E)이 상위에 있어야 함
        assert result.idxmax() == "E", (
            "factor1 가중치가 높으면 factor1이 큰 종목이 상위여야 합니다."
        )

    def test_combine_zscore_clipping(self):
        """zscore 결합 시 +-3 클리핑이 적용된다."""
        # 극단적인 값을 가진 팩터
        tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        # 하나의 극단값
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]
        factor = pd.Series(values, index=tickers, name="extreme")
        factor2 = pd.Series([1.0] * 10, index=tickers, name="neutral")

        result = combine_zscore(factor, factor2)

        # 클리핑 후 모든 값이 [-3, 3] 범위 내에 있어야 함
        assert result.max() <= 3.0 + 1e-9, (
            f"클리핑 후 최대값이 3.0 이하여야 합니다: {result.max()}"
        )
        assert result.min() >= -3.0 - 1e-9, (
            f"클리핑 후 최소값이 -3.0 이상이어야 합니다: {result.min()}"
        )


# ===================================================================
# combine_rank 테스트
# ===================================================================


class TestCombineRank:
    """rank 결합 함수 검증."""

    def test_combine_rank_equal_weight(self):
        """동일 가중치로 rank 결합이 올바르게 수행된다."""
        f1, f2 = _make_factor_data()
        result = combine_rank(f1, f2)

        assert isinstance(result, pd.Series), "반환값이 pd.Series여야 합니다."
        assert len(result) == 5, "결합 결과 길이가 입력과 같아야 합니다."

    def test_combine_rank_custom_weight(self):
        """커스텀 가중치로 rank 결합이 수행된다."""
        f1, f2 = _make_factor_data()
        result = combine_rank(f1, f2, value_weight=0.8, momentum_weight=0.2)

        assert isinstance(result, pd.Series), "반환값이 pd.Series여야 합니다."
        assert len(result) == 5, "결합 결과 길이가 입력과 같아야 합니다."

        # f1 가중치가 높으므로, f1이 높은 종목(E)의 결합 랭크가 높아야 함
        # rank 결합에서 높은 값 = 높은 rank -> 상위 종목
        assert result.idxmax() == "E", (
            "factor1 가중치가 높으면 factor1이 큰 종목의 결합 랭크가 가장 높아야 합니다."
        )


# ===================================================================
# 공통 속성 테스트
# ===================================================================


class TestCombineCommon:
    """결합 함수 공통 속성 검증."""

    def test_combine_preserves_index(self):
        """결합 함수가 원본 인덱스를 보존한다."""
        f1, f2 = _make_factor_data()

        zscore_result = combine_zscore(f1, f2)
        rank_result = combine_rank(f1, f2)

        assert list(zscore_result.index) == list(f1.index), (
            "zscore 결합이 원본 인덱스를 보존해야 합니다."
        )
        assert list(rank_result.index) == list(f1.index), (
            "rank 결합이 원본 인덱스를 보존해야 합니다."
        )

    def test_combine_with_nan_handling(self):
        """NaN이 포함된 데이터에서 결합이 에러 없이 수행된다."""
        f1, f2 = _make_factor_data_with_nan()

        zscore_result = combine_zscore(f1, f2)
        rank_result = combine_rank(f1, f2)

        assert isinstance(zscore_result, pd.Series), (
            "NaN 포함 데이터의 zscore 결합 결과가 pd.Series여야 합니다."
        )
        assert isinstance(rank_result, pd.Series), (
            "NaN 포함 데이터의 rank 결합 결과가 pd.Series여야 합니다."
        )
        # NaN이 있는 종목도 결과에 포함될 수 있음 (NaN 전파 또는 무시)
        assert len(zscore_result) > 0, "결과가 비어 있으면 안 됩니다."
