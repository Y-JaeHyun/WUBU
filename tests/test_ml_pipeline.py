"""ML 파이프라인 모듈(src/ml/pipeline.py) 테스트.

MLPipeline의 모델 생성, 학습, 예측, 피처 중요도,
시계열 교차검증 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _try_import_ml_pipeline():
    """MLPipeline 클래스가 있으면 임포트한다."""
    try:
        from src.ml.pipeline import MLPipeline
        return MLPipeline
    except ImportError:
        return None


def _make_training_data(n_periods=10, n_stocks=30, n_features=5, seed=42):
    """학습용 피처/타겟 시계열 데이터를 생성한다.

    MLPipeline.train()은 list[DataFrame], list[Series]를 입력받으므로
    여러 시점의 데이터를 리스트로 반환한다.
    """
    np.random.seed(seed)
    feature_names = [f"f{i}" for i in range(1, n_features + 1)]
    feature_history = []
    target_history = []

    for t in range(n_periods):
        tickers = [f"{i:06d}" for i in range(1, n_stocks + 1)]
        X = pd.DataFrame(
            np.random.randn(n_stocks, n_features),
            index=tickers,
            columns=feature_names,
        )
        coefs = np.array([0.5, -0.3, 0.2, 0.1, -0.4])[:n_features]
        y = pd.Series(
            X.values @ coefs + np.random.randn(n_stocks) * 0.5,
            index=tickers,
            name="forward_return",
        )
        feature_history.append(X)
        target_history.append(y)

    return feature_history, target_history


# ===================================================================
# MLPipeline 검증
# ===================================================================

class TestMLPipeline:
    """MLPipeline 검증."""

    def test_create_ridge_model(self):
        """Ridge 모델이 정상 생성된다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        pipeline = MLPipeline(model_type="ridge")
        assert pipeline is not None, "Ridge MLPipeline이 생성되어야 합니다."
        assert pipeline.model_type == "ridge"

    def test_create_random_forest_model(self):
        """Random Forest 모델이 정상 생성된다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        pipeline = MLPipeline(model_type="random_forest")
        assert pipeline is not None, "RF MLPipeline이 생성되어야 합니다."
        assert pipeline.model_type == "random_forest"

    def test_create_gradient_boosting_model(self):
        """Gradient Boosting 모델이 정상 생성된다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        pipeline = MLPipeline(model_type="gradient_boosting")
        assert pipeline is not None, "GBR MLPipeline이 생성되어야 합니다."
        assert pipeline.model_type == "gradient_boosting"

    def test_invalid_model_type(self):
        """잘못된 모델 타입에 대해 에러가 발생한다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        with pytest.raises((ValueError, KeyError)):
            MLPipeline(model_type="invalid_model")

    def test_train_basic(self):
        """기본 학습이 정상적으로 수행된다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        feature_history, target_history = _make_training_data(
            n_periods=10, n_stocks=30, seed=42,
        )
        pipeline = MLPipeline(model_type="ridge", min_train_months=5)

        result = pipeline.train(feature_history, target_history)

        assert isinstance(result, dict), "학습 결과가 dict여야 합니다."
        assert pipeline._is_trained is True, "학습 후 _is_trained가 True여야 합니다."

    def test_train_returns_metrics(self):
        """학습 시 IC, Rank IC 등 지표가 반환된다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        feature_history, target_history = _make_training_data(
            n_periods=10, n_stocks=30, seed=42,
        )
        pipeline = MLPipeline(model_type="ridge", min_train_months=5)

        metrics = pipeline.train(feature_history, target_history)

        assert isinstance(metrics, dict), "반환값이 dict여야 합니다."
        assert "ic_mean" in metrics, "ic_mean 키가 있어야 합니다."
        assert "rank_ic_mean" in metrics, "rank_ic_mean 키가 있어야 합니다."
        assert "feature_importance" in metrics, "feature_importance 키가 있어야 합니다."
        assert "n_samples" in metrics, "n_samples 키가 있어야 합니다."

    def test_predict_basic(self):
        """기본 예측이 정상적으로 수행된다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        feature_history, target_history = _make_training_data(
            n_periods=10, n_stocks=30, seed=42,
        )
        pipeline = MLPipeline(model_type="ridge", min_train_months=5)
        pipeline.train(feature_history, target_history)

        # 새로운 데이터로 예측
        np.random.seed(99)
        X_test = pd.DataFrame(
            np.random.randn(20, 5),
            index=[f"{i:06d}" for i in range(1, 21)],
            columns=[f"f{i}" for i in range(1, 6)],
        )
        predictions = pipeline.predict(X_test)

        assert isinstance(predictions, pd.Series), "예측 결과가 Series여야 합니다."
        assert len(predictions) == 20, "예측 수가 입력 수와 같아야 합니다."

    def test_predict_returns_series(self):
        """예측 결과가 Series를 반환한다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        feature_history, target_history = _make_training_data(
            n_periods=10, n_stocks=30, seed=42,
        )
        pipeline = MLPipeline(model_type="ridge", min_train_months=5)
        pipeline.train(feature_history, target_history)

        np.random.seed(99)
        X_test = pd.DataFrame(
            np.random.randn(15, 5),
            index=[f"{i:06d}" for i in range(1, 16)],
            columns=[f"f{i}" for i in range(1, 6)],
        )
        predictions = pipeline.predict(X_test)

        assert isinstance(predictions, pd.Series), (
            "예측 결과가 Series여야 합니다."
        )
        assert predictions.index.name == "ticker", (
            "인덱스명이 'ticker'여야 합니다."
        )

    def test_feature_importance(self):
        """피처 중요도가 반환된다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        feature_history, target_history = _make_training_data(
            n_periods=10, n_stocks=30, n_features=5, seed=42,
        )
        pipeline = MLPipeline(model_type="ridge", min_train_months=5)
        pipeline.train(feature_history, target_history)

        importance = pipeline.get_feature_importance()

        assert isinstance(importance, dict), "피처 중요도가 dict여야 합니다."
        assert len(importance) == 5, (
            f"피처 중요도 수가 피처 수(5)와 같아야 합니다: {len(importance)}"
        )

    def test_untrained_predict(self):
        """학습하지 않은 모델로 예측 시 빈 Series를 반환한다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        np.random.seed(99)
        X_test = pd.DataFrame(
            np.random.randn(20, 5),
            index=[f"{i:06d}" for i in range(1, 21)],
            columns=[f"f{i}" for i in range(1, 6)],
        )
        pipeline = MLPipeline(model_type="ridge")

        result = pipeline.predict(X_test)

        # 미학습 상태에서는 빈 Series 반환
        assert isinstance(result, pd.Series), "반환값이 Series여야 합니다."
        assert result.empty, "미학습 모델은 빈 Series를 반환해야 합니다."

    def test_cv_expanding_window(self):
        """train 내 TimeSeriesSplit CV가 올바르게 수행된다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        feature_history, target_history = _make_training_data(
            n_periods=15, n_stocks=30, seed=42,
        )
        pipeline = MLPipeline(model_type="ridge", n_splits=5, min_train_months=5)

        result = pipeline.train(feature_history, target_history)

        assert isinstance(result, dict), "학습 결과가 dict여야 합니다."
        # CV fold가 수행되었는지 확인
        assert result["n_folds"] >= 1, (
            f"CV fold가 최소 1개 이상이어야 합니다: {result['n_folds']}"
        )

    def test_reproducibility(self):
        """동일 데이터로 동일 결과가 나온다."""
        MLPipeline = _try_import_ml_pipeline()
        if MLPipeline is None:
            pytest.skip("MLPipeline이 아직 구현되지 않았습니다.")

        feature_history, target_history = _make_training_data(
            n_periods=10, n_stocks=30, seed=42,
        )

        np.random.seed(99)
        X_test = pd.DataFrame(
            np.random.randn(20, 5),
            index=[f"{i:06d}" for i in range(1, 21)],
            columns=[f"f{i}" for i in range(1, 6)],
        )

        pipeline1 = MLPipeline(model_type="ridge", min_train_months=5)
        pipeline1.train(feature_history, target_history)
        pred1 = pipeline1.predict(X_test)

        pipeline2 = MLPipeline(model_type="ridge", min_train_months=5)
        pipeline2.train(feature_history, target_history)
        pred2 = pipeline2.predict(X_test)

        np.testing.assert_array_almost_equal(
            pred1.values, pred2.values, decimal=6,
            err_msg="동일 데이터에서 동일 예측이 나와야 합니다.",
        )
