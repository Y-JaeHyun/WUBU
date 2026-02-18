"""N-팩터 결합 함수(src/strategy/factor_combiner.py) 테스트.

combine_n_factors_zscore, combine_n_factors_rank 함수 및
기존 combine_zscore/combine_rank 하위 호환성을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.factor_combiner import combine_zscore, combine_rank


# ===================================================================
# 헬퍼 함수
# ===================================================================

def _import_n_factor_functions():
    """N-팩터 결합 함수들을 임포트한다."""
    from src.strategy.factor_combiner import (
        combine_n_factors_zscore,
        combine_n_factors_rank,
    )
    return combine_n_factors_zscore, combine_n_factors_rank


def _make_n_factor_data():
    """N-팩터 테스트용 팩터 데이터를 생성한다."""
    tickers = ["A", "B", "C", "D", "E"]
    value = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=tickers, name="value")
    momentum = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=tickers, name="momentum")
    quality = pd.Series([3.0, 5.0, 1.0, 4.0, 2.0], index=tickers, name="quality")
    return value, momentum, quality


# ===================================================================
# combine_n_factors_zscore 테스트
# ===================================================================

class TestCombineNFactorsZscore:
    """N-팩터 zscore 결합 함수 검증."""

    def test_combine_n_factors_zscore_two_factors(self):
        """2개 팩터의 zscore 결합이 올바르게 수행된다."""
        combine_n_zscore, _ = _import_n_factor_functions()
        value, momentum, _ = _make_n_factor_data()

        factors = {"value": value, "momentum": momentum}
        result = combine_n_zscore(factors)

        assert isinstance(result, pd.Series), "반환값이 pd.Series여야 합니다."
        assert len(result) == 5, "결합 결과 길이가 입력과 같아야 합니다."

    def test_combine_n_factors_zscore_three_factors(self):
        """3개 팩터의 zscore 결합이 올바르게 수행된다."""
        combine_n_zscore, _ = _import_n_factor_functions()
        value, momentum, quality = _make_n_factor_data()

        factors = {"value": value, "momentum": momentum, "quality": quality}
        result = combine_n_zscore(factors)

        assert isinstance(result, pd.Series), "반환값이 pd.Series여야 합니다."
        assert len(result) == 5, "3팩터 결합 결과 길이가 입력과 같아야 합니다."

    def test_combine_n_factors_zscore_equal_weights(self):
        """가중치 None이면 동일 가중치가 적용된다."""
        combine_n_zscore, _ = _import_n_factor_functions()
        value, momentum, _ = _make_n_factor_data()

        factors = {"value": value, "momentum": momentum}
        result = combine_n_zscore(factors, weights=None)

        assert isinstance(result, pd.Series), "반환값이 pd.Series여야 합니다."
        # value와 momentum이 정반대이고 동일 가중치면 합이 0에 근접
        assert abs(result.mean()) < 1e-9, (
            "대칭 팩터의 동일 가중치 결합 평균이 0이어야 합니다."
        )

    def test_combine_n_factors_zscore_custom_weights(self):
        """커스텀 가중치가 올바르게 적용된다."""
        combine_n_zscore, _ = _import_n_factor_functions()
        value, momentum, _ = _make_n_factor_data()

        factors = {"value": value, "momentum": momentum}
        weights = {"value": 0.8, "momentum": 0.2}
        result = combine_n_zscore(factors, weights=weights)

        assert isinstance(result, pd.Series), "반환값이 pd.Series여야 합니다."
        # value 가중치가 높으므로, value가 높은 종목(E)이 상위
        assert result.idxmax() == "E", (
            "value 가중치가 높으면 value가 큰 종목이 상위여야 합니다."
        )


# ===================================================================
# combine_n_factors_rank 테스트
# ===================================================================

class TestCombineNFactorsRank:
    """N-팩터 rank 결합 함수 검증."""

    def test_combine_n_factors_rank_basic(self):
        """2개 팩터의 rank 결합이 올바르게 수행된다."""
        _, combine_n_rank = _import_n_factor_functions()
        value, momentum, _ = _make_n_factor_data()

        factors = {"value": value, "momentum": momentum}
        result = combine_n_rank(factors)

        assert isinstance(result, pd.Series), "반환값이 pd.Series여야 합니다."
        assert len(result) == 5, "결합 결과 길이가 입력과 같아야 합니다."

    def test_combine_n_factors_rank_three_factors(self):
        """3개 팩터의 rank 결합이 올바르게 수행된다."""
        _, combine_n_rank = _import_n_factor_functions()
        value, momentum, quality = _make_n_factor_data()

        factors = {"value": value, "momentum": momentum, "quality": quality}
        result = combine_n_rank(factors)

        assert isinstance(result, pd.Series), "반환값이 pd.Series여야 합니다."
        assert len(result) == 5, "3팩터 rank 결합 결과가 올바릅니다."


# ===================================================================
# 공통 속성 및 엣지 케이스 테스트
# ===================================================================

class TestNFactorCommon:
    """N-팩터 결합 공통 속성 및 엣지 케이스 검증."""

    def test_n_factors_common_tickers_only(self):
        """공통 종목만 결과에 포함된다."""
        combine_n_zscore, _ = _import_n_factor_functions()

        # 부분적으로 겹치는 인덱스
        f1 = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
        f2 = pd.Series([4.0, 5.0, 6.0], index=["B", "C", "D"])

        factors = {"f1": f1, "f2": f2}
        result = combine_n_zscore(factors)

        # 공통 종목 B, C만 결과에 있어야 함
        assert set(result.index).issubset({"B", "C"}), (
            f"공통 종목만 포함되어야 합니다: {list(result.index)}"
        )

    def test_n_factors_empty_input(self):
        """빈 팩터 dict 입력 시 빈 Series를 반환한다."""
        combine_n_zscore, combine_n_rank = _import_n_factor_functions()

        result_z = combine_n_zscore({})
        result_r = combine_n_rank({})

        assert isinstance(result_z, pd.Series), "빈 입력에도 Series를 반환해야 합니다."
        assert len(result_z) == 0, "빈 입력이면 빈 Series여야 합니다."
        assert isinstance(result_r, pd.Series), "빈 입력에도 Series를 반환해야 합니다."
        assert len(result_r) == 0, "빈 입력이면 빈 Series여야 합니다."

    def test_n_factors_single_factor(self):
        """단일 팩터 입력도 정상 처리된다."""
        combine_n_zscore, combine_n_rank = _import_n_factor_functions()
        value, _, _ = _make_n_factor_data()

        factors = {"value": value}
        result_z = combine_n_zscore(factors)
        result_r = combine_n_rank(factors)

        assert isinstance(result_z, pd.Series), "단일 팩터 zscore 결과가 Series여야 합니다."
        assert isinstance(result_r, pd.Series), "단일 팩터 rank 결과가 Series여야 합니다."
        assert len(result_z) == 5, "단일 팩터 결과 길이가 입력과 같아야 합니다."

    def test_n_factors_with_nan(self):
        """NaN이 포함된 팩터에서도 에러 없이 결합된다."""
        combine_n_zscore, combine_n_rank = _import_n_factor_functions()

        tickers = ["A", "B", "C", "D", "E"]
        f1 = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0], index=tickers)
        f2 = pd.Series([5.0, 4.0, np.nan, 2.0, 1.0], index=tickers)

        factors = {"f1": f1, "f2": f2}
        result_z = combine_n_zscore(factors)
        result_r = combine_n_rank(factors)

        assert isinstance(result_z, pd.Series), "NaN 포함 데이터의 zscore 결합 결과가 Series여야 합니다."
        assert isinstance(result_r, pd.Series), "NaN 포함 데이터의 rank 결합 결과가 Series여야 합니다."


# ===================================================================
# 하위 호환성 테스트 (기존 2팩터 combine 함수)
# ===================================================================

class TestBackwardCompatibility:
    """기존 combine_zscore, combine_rank 함수의 하위 호환성 검증."""

    def test_backward_compat_combine_zscore(self):
        """기존 combine_zscore 함수가 여전히 정상 동작한다."""
        tickers = ["A", "B", "C", "D", "E"]
        f1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=tickers)
        f2 = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=tickers)

        result = combine_zscore(f1, f2)

        assert isinstance(result, pd.Series), "기존 combine_zscore 반환값이 Series여야 합니다."
        assert len(result) == 5, "결합 결과 길이가 입력과 같아야 합니다."

    def test_backward_compat_combine_rank(self):
        """기존 combine_rank 함수가 여전히 정상 동작한다."""
        tickers = ["A", "B", "C", "D", "E"]
        f1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=tickers)
        f2 = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=tickers)

        result = combine_rank(f1, f2)

        assert isinstance(result, pd.Series), "기존 combine_rank 반환값이 Series여야 합니다."
        assert len(result) == 5, "결합 결과 길이가 입력과 같아야 합니다."

    def test_backward_compat_combine_zscore_custom_weight(self):
        """기존 combine_zscore에 커스텀 가중치가 동작한다."""
        tickers = ["A", "B", "C", "D", "E"]
        f1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=tickers)
        f2 = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=tickers)

        result = combine_zscore(f1, f2, value_weight=0.7, momentum_weight=0.3)

        assert isinstance(result, pd.Series), "커스텀 가중치 combine_zscore 결과가 Series여야 합니다."
        assert result.idxmax() == "E", "value 가중치가 높으면 E가 최상위여야 합니다."
