"""ML 피처 엔지니어링 모듈(src/ml/features.py) 테스트.

build_factor_features, build_forward_returns, cross_sectional_normalize
함수의 피처 생성, 정규화, 엣지 케이스 처리를 검증한다.
"""

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _import_build_factor_features():
    """build_factor_features 함수를 임포트한다."""
    from src.ml.features import build_factor_features
    return build_factor_features


def _import_build_forward_returns():
    """build_forward_returns 함수를 임포트한다."""
    from src.ml.features import build_forward_returns
    return build_forward_returns


def _import_cross_sectional_normalize():
    """cross_sectional_normalize 함수를 임포트한다."""
    from src.ml.features import cross_sectional_normalize
    return cross_sectional_normalize


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
        "market_cap": np.random.randint(50_000_000_000, 5_000_000_000_000, n),
        "volume": np.random.randint(10_000, 5_000_000, n),
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
# build_factor_features 검증
# ===================================================================

class TestBuildFactorFeatures:
    """build_factor_features 함수 검증."""

    def test_build_features_basic(self):
        """기본 피처가 정상적으로 생성된다."""
        build_factor_features = _import_build_factor_features()
        fundamentals = _make_fundamentals()
        tickers = fundamentals["ticker"].tolist()
        prices = _make_prices_dict(tickers)

        features = build_factor_features(fundamentals, prices)

        assert isinstance(features, pd.DataFrame), "반환값이 DataFrame이어야 합니다."
        assert len(features) > 0, "빈 DataFrame이면 안 됩니다."

    def test_feature_columns(self):
        """주요 피처 컬럼이 생성된다."""
        build_factor_features = _import_build_factor_features()
        fundamentals = _make_fundamentals()
        tickers = fundamentals["ticker"].tolist()
        prices = _make_prices_dict(tickers)

        features = build_factor_features(fundamentals, prices)

        expected_cols = {"inv_pbr", "div_yield"}
        actual_cols = set(features.columns)
        # 최소한 밸류 피처가 포함되어야 함
        assert len(expected_cols & actual_cols) > 0, (
            f"주요 피처 컬럼이 포함되어야 합니다. 실제: {actual_cols}"
        )

    def test_momentum_features(self):
        """모멘텀 피처가 올바르게 생성된다."""
        build_factor_features = _import_build_factor_features()
        fundamentals = _make_fundamentals(n=5)
        tickers = fundamentals["ticker"].tolist()
        prices = _make_prices_dict(tickers, n_days=300)

        features = build_factor_features(fundamentals, prices)

        # 충분한 가격 데이터가 있으면 모멘텀 피처가 생성됨
        momentum_cols = {"mom_12m", "mom_6m", "mom_3m"}
        present_mom = momentum_cols & set(features.columns)
        # 300일이면 mom_3m, mom_6m은 가능, mom_12m은 데이터 부족일 수 있음
        assert len(present_mom) > 0, (
            f"모멘텀 피처가 최소 하나는 있어야 합니다: {features.columns.tolist()}"
        )

    def test_volatility_features_positive(self):
        """변동성 피처가 양수이다."""
        build_factor_features = _import_build_factor_features()
        fundamentals = _make_fundamentals(n=5)
        tickers = fundamentals["ticker"].tolist()
        prices = _make_prices_dict(tickers, n_days=300)

        features = build_factor_features(fundamentals, prices)

        for col in ["vol_20d", "vol_60d"]:
            if col in features.columns:
                valid = features[col].dropna()
                if not valid.empty:
                    assert (valid > 0).all(), (
                        f"{col}은 양수여야 합니다."
                    )

    def test_value_features_direction(self):
        """밸류 피처의 방향성이 올바르다 (PBR 역수: 높을수록 저평가)."""
        build_factor_features = _import_build_factor_features()

        # 명확한 차이가 나는 2종목
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B"],
            "pbr": [0.5, 5.0],  # A가 저평가
            "per": [5.0, 50.0],  # A가 저평가
            "div_yield": [5.0, 0.5],
            "eps": [10000, 1000],
            "bps": [50000, 10000],
            "market_cap": [1_000_000_000_000, 1_000_000_000_000],
        })

        features = build_factor_features(fundamentals, {})

        if "inv_pbr" in features.columns:
            inv_pbr = features["inv_pbr"].dropna()
            if len(inv_pbr) == 2:
                assert inv_pbr.loc["A"] > inv_pbr.loc["B"], (
                    "PBR이 낮은 종목의 inv_pbr이 더 높아야 합니다."
                )

    def test_empty_fundamentals(self):
        """빈 펀더멘탈 데이터 입력 시 빈 DataFrame을 반환한다."""
        build_factor_features = _import_build_factor_features()

        features = build_factor_features(pd.DataFrame(), {})

        assert isinstance(features, pd.DataFrame), "반환값이 DataFrame이어야 합니다."
        assert features.empty, "빈 입력이면 빈 DataFrame이어야 합니다."

    def test_missing_columns(self):
        """일부 컬럼이 누락되어도 에러 없이 처리된다."""
        build_factor_features = _import_build_factor_features()

        # ticker만 있는 최소 데이터
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
        })

        features = build_factor_features(fundamentals, {})

        assert isinstance(features, pd.DataFrame), "반환값이 DataFrame이어야 합니다."

    def test_forward_returns(self):
        """포워드 수익률이 올바르게 계산된다."""
        build_forward_returns = _import_build_forward_returns()

        tickers = ["A", "B"]
        prices = _make_prices_dict(tickers, n_days=100, seed=42)
        # 기준일 = 시작 직후
        date = "2023-01-02"

        result = build_forward_returns(prices, date, forward_days=21)

        assert isinstance(result, pd.Series), "반환값이 Series여야 합니다."
        # 100일 중 21일 후 수익률 계산 가능
        assert len(result) > 0, "포워드 수익률이 계산되어야 합니다."

    def test_cross_sectional_normalize(self, sample_ml_features):
        """Z-Score 정규화가 올바르게 수행된다."""
        cross_sectional_normalize = _import_cross_sectional_normalize()

        normalized = cross_sectional_normalize(sample_ml_features)

        assert isinstance(normalized, pd.DataFrame), "반환값이 DataFrame이어야 합니다."
        assert normalized.shape == sample_ml_features.shape, (
            "정규화 후 형상이 동일해야 합니다."
        )

        # Z-Score 정규화 후 각 열의 평균이 0에 가까워야 함
        for col in normalized.columns:
            mean_val = normalized[col].mean()
            assert abs(mean_val) < 0.5, (
                f"정규화 후 {col}의 평균이 0에 가까워야 합니다: {mean_val:.4f}"
            )

    def test_winsorizing(self, sample_ml_features):
        """극단값 Winsorizing이 적용된다."""
        cross_sectional_normalize = _import_cross_sectional_normalize()

        # 극단적 이상치 추가
        outlier_features = sample_ml_features.copy()
        outlier_features.iloc[0, 0] = 100.0  # 극단적 양수
        outlier_features.iloc[1, 0] = -100.0  # 극단적 음수

        normalized = cross_sectional_normalize(outlier_features)

        if not normalized.empty:
            col = normalized.columns[0]
            # Winsorizing 후 극단값이 억제되어야 함
            assert normalized[col].max() < 50, (
                "Winsorizing으로 극단값이 억제되어야 합니다."
            )
