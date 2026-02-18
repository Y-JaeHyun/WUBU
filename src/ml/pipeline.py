"""ML 팩터 모델 학습/예측 파이프라인.

TimeSeriesSplit 교차 검증으로 팩터 모델을 학습하고,
학습된 모델로 종목별 기대 수익률을 예측한다.

지원 모델:
- Ridge Regression (기본, 안정적)
- Random Forest
- Gradient Boosting

평가 지표:
- IC (Information Coefficient): 예측과 실제 수익률의 상관계수
- Rank IC: 예측 순위와 실제 순위의 스피어만 상관계수
- Feature Importance: 피처별 중요도

사용 예시:
    pipeline = MLPipeline(model_type="ridge", n_splits=5)
    results = pipeline.train(feature_history, target_history)
    predictions = pipeline.predict(current_features)
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 지원하는 모델 타입
VALID_MODEL_TYPES = ("ridge", "random_forest", "gradient_boosting")


class MLPipeline:
    """ML 팩터 모델 파이프라인.

    TimeSeriesSplit 교차 검증으로 모델을 학습하고,
    피처 중요도, IC, Rank IC 등을 산출한다.

    Args:
        model_type: 모델 종류 - "ridge" (기본), "random_forest", "gradient_boosting"
        n_splits: CV 분할 수 (기본 5)
        min_train_months: 최소 학습 기간 - 개월 수 (기본 36)
    """

    def __init__(
        self,
        model_type: str = "ridge",
        n_splits: int = 5,
        min_train_months: int = 36,
    ):
        model_type = model_type.lower()
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"지원하지 않는 모델 타입: {model_type}. "
                f"{VALID_MODEL_TYPES} 중 선택하세요."
            )

        self.model_type = model_type
        self.n_splits = n_splits
        self.min_train_months = min_train_months
        self.model = None
        self.feature_names: list[str] = []
        self._feature_importance: dict[str, float] = {}
        self._is_trained = False

        logger.info(
            f"MLPipeline 초기화: model_type={model_type}, "
            f"n_splits={n_splits}, min_train_months={min_train_months}"
        )

    def _create_model(self):
        """model_type에 따라 sklearn 모델 인스턴스를 생성한다.

        Returns:
            sklearn 모델 객체
        """
        if self.model_type == "ridge":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)

        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
            )

        elif self.model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )

        else:
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)

    def _compute_ic(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> tuple[float, float]:
        """IC와 Rank IC를 계산한다.

        Args:
            predictions: 예측값 배열
            actuals: 실제값 배열

        Returns:
            (IC, Rank IC) 튜플
        """
        if len(predictions) < 3 or len(actuals) < 3:
            return 0.0, 0.0

        # IC: 피어슨 상관계수
        try:
            corr_matrix = np.corrcoef(predictions, actuals)
            ic = float(corr_matrix[0, 1])
            if np.isnan(ic):
                ic = 0.0
        except Exception:
            ic = 0.0

        # Rank IC: 스피어만 순위 상관계수
        try:
            from scipy.stats import spearmanr
            rank_ic, _ = spearmanr(predictions, actuals)
            if np.isnan(rank_ic):
                rank_ic = 0.0
            rank_ic = float(rank_ic)
        except ImportError:
            # scipy 없으면 순위 기반 피어슨으로 근사
            pred_rank = pd.Series(predictions).rank().values
            actual_rank = pd.Series(actuals).rank().values
            corr_matrix = np.corrcoef(pred_rank, actual_rank)
            rank_ic = float(corr_matrix[0, 1])
            if np.isnan(rank_ic):
                rank_ic = 0.0
        except Exception:
            rank_ic = 0.0

        return ic, rank_ic

    def _extract_feature_importance(self, model, feature_names: list[str]) -> dict[str, float]:
        """모델에서 피처 중요도를 추출한다.

        Args:
            model: 학습된 sklearn 모델
            feature_names: 피처 이름 리스트

        Returns:
            {피처명: 중요도} 딕셔너리
        """
        importance = {}

        if hasattr(model, "feature_importances_"):
            # RandomForest, GradientBoosting
            for name, imp in zip(feature_names, model.feature_importances_):
                importance[name] = float(imp)

        elif hasattr(model, "coef_"):
            # Ridge (계수의 절대값으로 중요도 추정)
            coef = np.abs(model.coef_)
            coef_sum = coef.sum()
            if coef_sum > 0:
                normalized_coef = coef / coef_sum
            else:
                normalized_coef = np.ones(len(feature_names)) / len(feature_names)
            for name, imp in zip(feature_names, normalized_coef):
                importance[name] = float(imp)

        else:
            # 중요도를 추출할 수 없으면 동일 중요도
            equal_imp = 1.0 / len(feature_names) if feature_names else 0.0
            for name in feature_names:
                importance[name] = equal_imp

        return importance

    def train(
        self,
        feature_history: list[pd.DataFrame],
        target_history: list[pd.Series],
    ) -> dict:
        """TimeSeriesSplit CV로 모델을 학습한다.

        과거 여러 시점의 피처와 타겟을 사용하여 교차 검증 학습을 수행한다.
        최종 모델은 전체 데이터로 학습한다.

        Args:
            feature_history: 시점별 피처 DataFrame 리스트
                각 DataFrame: index=ticker, columns=feature_names
            target_history: 시점별 타겟 Series 리스트
                각 Series: index=ticker, values=forward_return

        Returns:
            딕셔너리: {
                'ic_mean': float,       # CV IC 평균
                'ic_std': float,        # CV IC 표준편차
                'rank_ic_mean': float,  # CV Rank IC 평균
                'feature_importance': dict[str, float],  # 피처 중요도
                'n_samples': int,       # 총 학습 샘플 수
                'n_features': int,      # 피처 수
                'n_folds': int,         # CV fold 수
            }
        """
        if not feature_history or not target_history:
            logger.warning("학습 데이터 없음: 빈 결과 반환")
            return {
                "ic_mean": 0.0,
                "ic_std": 0.0,
                "rank_ic_mean": 0.0,
                "feature_importance": {},
                "n_samples": 0,
                "n_features": 0,
                "n_folds": 0,
            }

        if len(feature_history) != len(target_history):
            raise ValueError(
                f"피처/타겟 리스트 길이 불일치: "
                f"{len(feature_history)} vs {len(target_history)}"
            )

        # 전체 데이터 결합
        all_features_list = []
        all_targets_list = []

        for features, targets in zip(feature_history, target_history):
            if features.empty or targets.empty:
                continue

            # 공통 종목만 사용
            common = features.index.intersection(targets.index)
            if common.empty:
                continue

            all_features_list.append(features.loc[common])
            all_targets_list.append(targets.loc[common])

        if not all_features_list:
            logger.warning("유효한 학습 데이터 없음: 빈 결과 반환")
            return {
                "ic_mean": 0.0,
                "ic_std": 0.0,
                "rank_ic_mean": 0.0,
                "feature_importance": {},
                "n_samples": 0,
                "n_features": 0,
                "n_folds": 0,
            }

        all_features = pd.concat(all_features_list, axis=0)
        all_targets = pd.concat(all_targets_list, axis=0)

        # 공통 피처 정리
        self.feature_names = all_features.columns.tolist()

        # NaN/inf 처리
        all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        all_targets = all_targets.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X = all_features.values
        y = all_targets.values

        n_samples = len(X)
        n_features = X.shape[1] if len(X.shape) > 1 else 0

        logger.info(
            f"학습 데이터: {n_samples}개 샘플, {n_features}개 피처"
        )

        if n_samples < 10:
            logger.warning("학습 샘플 수 부족 (10개 미만)")
            return {
                "ic_mean": 0.0,
                "ic_std": 0.0,
                "rank_ic_mean": 0.0,
                "feature_importance": {},
                "n_samples": n_samples,
                "n_features": n_features,
                "n_folds": 0,
            }

        # TimeSeriesSplit CV
        from sklearn.model_selection import TimeSeriesSplit

        n_splits = min(self.n_splits, n_samples - 1)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        ic_list = []
        rank_ic_list = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            if len(train_idx) < self.min_train_months:
                continue

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self._create_model()

            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)

                ic, rank_ic = self._compute_ic(predictions, y_val)
                ic_list.append(ic)
                rank_ic_list.append(rank_ic)

                logger.info(
                    f"  Fold {fold_idx + 1}: IC={ic:.4f}, Rank IC={rank_ic:.4f}, "
                    f"train={len(train_idx)}, val={len(val_idx)}"
                )
            except Exception as e:
                logger.warning(f"  Fold {fold_idx + 1} 학습 실패: {e}")

        # 전체 데이터로 최종 모델 학습
        self.model = self._create_model()
        try:
            self.model.fit(X, y)
            self._is_trained = True

            # 피처 중요도 추출
            self._feature_importance = self._extract_feature_importance(
                self.model, self.feature_names
            )
        except Exception as e:
            logger.warning(f"최종 모델 학습 실패: {e}")
            self._is_trained = False

        # CV 결과 요약
        ic_mean = float(np.mean(ic_list)) if ic_list else 0.0
        ic_std = float(np.std(ic_list)) if ic_list else 0.0
        rank_ic_mean = float(np.mean(rank_ic_list)) if rank_ic_list else 0.0

        result = {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "rank_ic_mean": rank_ic_mean,
            "feature_importance": self._feature_importance,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_folds": len(ic_list),
        }

        logger.info(
            f"ML 학습 완료: IC={ic_mean:.4f}(+-{ic_std:.4f}), "
            f"Rank IC={rank_ic_mean:.4f}, folds={len(ic_list)}"
        )

        return result

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """학습된 모델로 종목별 기대 수익률을 예측한다.

        Args:
            features: 피처 DataFrame (index=ticker, columns=feature_names)

        Returns:
            Series (index=ticker, values=predicted_score).
            학습 전이거나 빈 데이터이면 빈 Series 반환.
        """
        if not self._is_trained or self.model is None:
            logger.warning("학습되지 않은 모델: 빈 예측 반환")
            return pd.Series(dtype=float)

        if features.empty:
            logger.warning("빈 피처 데이터: 빈 예측 반환")
            return pd.Series(dtype=float)

        # 학습 시 사용한 피처와 동일하게 정렬
        missing_features = [f for f in self.feature_names if f not in features.columns]
        if missing_features:
            logger.warning(
                f"누락 피처 {len(missing_features)}개: {missing_features}. "
                f"0으로 채움."
            )

        # 피처 정렬 및 누락 피처 보충
        aligned_features = pd.DataFrame(
            0.0,
            index=features.index,
            columns=self.feature_names,
        )

        common_features = [f for f in self.feature_names if f in features.columns]
        aligned_features[common_features] = features[common_features]

        # NaN/inf 처리
        aligned_features = aligned_features.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

        X = aligned_features.values

        try:
            predictions = self.model.predict(X)
            result = pd.Series(
                predictions,
                index=features.index,
                dtype=float,
            )
            result.index.name = "ticker"

            logger.info(
                f"ML 예측 완료: {len(result)}개 종목, "
                f"score 범위=[{result.min():.4f}, {result.max():.4f}]"
            )

            return result

        except Exception as e:
            logger.warning(f"ML 예측 실패: {e}")
            return pd.Series(dtype=float)

    def get_feature_importance(self) -> dict[str, float]:
        """피처 중요도를 반환한다.

        학습 후 호출해야 유효한 결과를 얻을 수 있다.

        Returns:
            {피처명: 중요도} 딕셔너리.
            학습 전이면 빈 딕셔너리 반환.
        """
        if not self._feature_importance:
            logger.warning("피처 중요도 없음: 빈 딕셔너리 반환")
            return {}

        # 중요도 내림차순 정렬
        sorted_importance = dict(
            sorted(
                self._feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        return sorted_importance
