"""퀄리티 팩터 전략 모듈(src/strategy/quality.py) 테스트.

QualityStrategy 생성, 퀄리티 스코어 계산, 가중치 적용,
generate_signals 반환값 형식 등을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Strategy


# ===================================================================
# 헬퍼: QualityStrategy 모듈 임포트 (지연 임포트)
# ===================================================================

def _import_quality_strategy():
    """QualityStrategy 클래스를 임포트한다."""
    from src.strategy.quality import QualityStrategy
    return QualityStrategy


def _make_quality_fundamentals(n=30, seed=42):
    """퀄리티 팩터 분석용 펀더멘탈 DataFrame을 생성한다.

    ROE, GP/A, 부채비율, 발생액 등의 컬럼을 포함한다.
    """
    np.random.seed(seed)
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "ticker": tickers,
        "name": [f"종목{i}" for i in range(1, n + 1)],
        "market": ["KOSPI"] * (n // 2) + ["KOSDAQ"] * (n - n // 2),
        "roe": np.random.uniform(5, 30, n).round(2),
        "gp_over_assets": np.random.uniform(0.05, 0.5, n).round(4),
        "debt_ratio": np.random.uniform(20, 200, n).round(2),
        "accruals": np.random.uniform(-0.1, 0.1, n).round(4),
        "close": np.random.randint(5000, 500000, n),
        "market_cap": np.random.randint(200_000_000_000, 5_000_000_000_000, n),
        "volume": np.random.randint(100_000, 5_000_000, n),
        "pbr": np.random.uniform(0.3, 5.0, n).round(2),
        "per": np.random.uniform(3, 30, n).round(2),
    })


# ===================================================================
# QualityStrategy 기본 속성 검증
# ===================================================================

class TestQualityStrategyInit:
    """QualityStrategy 초기화 및 속성 검증."""

    def test_quality_strategy_name(self):
        """name 프로퍼티가 'Quality'를 포함하는 문자열을 반환한다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy()
        assert "Quality" in qs.name or "quality" in qs.name.lower(), (
            f"이름에 'Quality'가 포함되어야 합니다: {qs.name}"
        )

    def test_quality_strategy_is_strategy_subclass(self):
        """QualityStrategy는 Strategy ABC의 서브클래스이다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy()
        assert isinstance(qs, Strategy), (
            "QualityStrategy는 Strategy의 인스턴스여야 합니다."
        )

    def test_default_parameters(self):
        """기본 파라미터가 올바르게 설정된다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy()
        # num_stocks 기본값 20
        assert hasattr(qs, "num_stocks") or hasattr(qs, "_num_stocks"), (
            "num_stocks 속성이 있어야 합니다."
        )

    def test_custom_num_stocks(self):
        """사용자 정의 num_stocks가 반영된다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy(num_stocks=10)
        num = getattr(qs, "num_stocks", getattr(qs, "_num_stocks", None))
        assert num == 10, "num_stocks가 10으로 설정되어야 합니다."

    def test_custom_weights(self):
        """사용자 정의 가중치가 반영된다."""
        QualityStrategy = _import_quality_strategy()
        custom_weights = {"roe": 0.5, "gpa": 0.2, "debt": 0.2, "accrual": 0.1}
        qs = QualityStrategy(weights=custom_weights)
        # 가중치가 올바르게 저장되었는지 확인
        stored_weights = getattr(qs, "weights", getattr(qs, "_weights", None))
        assert stored_weights is not None, "가중치가 저장되어야 합니다."


# ===================================================================
# calculate_quality_scores 테스트
# ===================================================================

class TestCalculateQualityScores:
    """calculate_quality_scores 메서드 검증."""

    def test_calculate_quality_scores_basic(self):
        """기본 펀더멘탈 데이터로 퀄리티 스코어가 계산된다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy()
        fundamentals = _make_quality_fundamentals()

        scores = qs.calculate_quality_scores(fundamentals)

        assert isinstance(scores, pd.Series), "반환값이 pd.Series여야 합니다."
        assert len(scores) > 0, "스코어가 비어 있으면 안 됩니다."

    def test_calculate_quality_scores_custom_weights(self):
        """커스텀 가중치로 퀄리티 스코어가 계산된다."""
        QualityStrategy = _import_quality_strategy()
        custom_weights = {"roe": 0.7, "gpa": 0.1, "debt": 0.1, "accrual": 0.1}
        qs = QualityStrategy(weights=custom_weights)
        fundamentals = _make_quality_fundamentals()

        scores = qs.calculate_quality_scores(fundamentals)

        assert isinstance(scores, pd.Series), "반환값이 pd.Series여야 합니다."
        assert len(scores) > 0, "스코어가 비어 있으면 안 됩니다."

    def test_quality_scores_roe_ranking(self):
        """ROE가 높은 종목이 더 높은 퀄리티 스코어를 받는 경향이 있다."""
        QualityStrategy = _import_quality_strategy()
        # ROE에 100% 가중치를 부여하여 ROE 영향만 테스트
        qs = QualityStrategy(weights={"roe": 1.0, "gpa": 0.0, "debt": 0.0, "accrual": 0.0})

        # ROE가 명확히 차이나는 종목
        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "roe": [30.0, 10.0, 20.0],
            "gp_over_assets": [0.2, 0.2, 0.2],
            "debt_ratio": [50.0, 50.0, 50.0],
            "accruals": [0.0, 0.0, 0.0],
        })

        scores = qs.calculate_quality_scores(fundamentals)

        # ROE 100% 가중치이므로, ROE 순서대로 스코어가 높아야 함
        assert scores.idxmax() == "A", "ROE가 가장 높은 A가 최고 스코어여야 합니다."
        assert scores.idxmin() == "B", "ROE가 가장 낮은 B가 최저 스코어여야 합니다."

    def test_quality_scores_debt_inverse(self):
        """부채비율이 높은 종목이 더 낮은 퀄리티 스코어를 받는다."""
        QualityStrategy = _import_quality_strategy()
        # 부채비율에 100% 가중치
        qs = QualityStrategy(weights={"roe": 0.0, "gpa": 0.0, "debt": 1.0, "accrual": 0.0})

        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "roe": [15.0, 15.0, 15.0],
            "gp_over_assets": [0.2, 0.2, 0.2],
            "debt_ratio": [200.0, 50.0, 100.0],
            "accruals": [0.0, 0.0, 0.0],
        })

        scores = qs.calculate_quality_scores(fundamentals)

        # 부채비율이 낮을수록 좋으므로 B가 최고, A가 최저
        assert scores.idxmax() == "B", "부채비율이 가장 낮은 B가 최고 스코어여야 합니다."
        assert scores.idxmin() == "A", "부채비율이 가장 높은 A가 최저 스코어여야 합니다."

    def test_quality_scores_with_nan(self):
        """NaN이 포함된 데이터에서도 에러 없이 스코어가 계산된다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy()

        fundamentals = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "roe": [15.0, np.nan, 20.0, 10.0],
            "gp_over_assets": [0.2, 0.3, np.nan, 0.1],
            "debt_ratio": [100.0, 50.0, 80.0, np.nan],
            "accruals": [0.01, 0.02, 0.03, 0.0],
        })

        scores = qs.calculate_quality_scores(fundamentals)

        assert isinstance(scores, pd.Series), "NaN 포함 데이터에서도 Series를 반환해야 합니다."
        # NaN이 아닌 결과가 일부 있어야 함 (또는 NaN 전파)
        assert len(scores) > 0, "결과가 비어 있으면 안 됩니다."

    def test_quality_scores_returns_series_indexed_by_ticker(self):
        """반환된 Series의 인덱스가 티커여야 한다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy()
        fundamentals = _make_quality_fundamentals(n=10)

        scores = qs.calculate_quality_scores(fundamentals)

        # 인덱스가 티커 형태인지 확인
        for idx in scores.index:
            assert isinstance(idx, str), f"인덱스 값 '{idx}'가 문자열이어야 합니다."


# ===================================================================
# generate_signals 검증
# ===================================================================

class TestQualityGenerateSignals:
    """QualityStrategy.generate_signals 반환값 검증."""

    def test_generate_signals_returns_dict(self):
        """generate_signals는 dict를 반환한다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy(num_stocks=5)
        data = {"fundamentals": _make_quality_fundamentals()}

        signals = qs.generate_signals("20240102", data)

        assert isinstance(signals, dict), "반환값이 dict여야 합니다."

    def test_generate_signals_num_stocks_limit(self):
        """num_stocks 이하의 종목이 선택된다."""
        QualityStrategy = _import_quality_strategy()
        n_stocks = 5
        qs = QualityStrategy(num_stocks=n_stocks)
        data = {"fundamentals": _make_quality_fundamentals(n=50)}

        signals = qs.generate_signals("20240102", data)

        assert len(signals) <= n_stocks, (
            f"종목 수가 {n_stocks}개 이하여야 합니다: {len(signals)}"
        )

    def test_generate_signals_weights_sum_le_one(self):
        """시그널 비중의 합이 1.0 이하이다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy(num_stocks=10)
        data = {"fundamentals": _make_quality_fundamentals()}

        signals = qs.generate_signals("20240102", data)

        if signals:
            total_weight = sum(signals.values())
            assert total_weight <= 1.0 + 1e-9, (
                f"비중 합이 1.0 이하여야 합니다: {total_weight}"
            )

    def test_generate_signals_empty_data(self):
        """빈 펀더멘탈 데이터 입력 시 빈 dict를 반환한다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy()
        data = {"fundamentals": pd.DataFrame()}

        signals = qs.generate_signals("20240102", data)

        assert signals == {}, "빈 데이터 입력 시 빈 dict여야 합니다."

    def test_generate_signals_no_fundamentals_key(self):
        """data에 fundamentals 키가 없으면 빈 dict를 반환한다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy()

        signals = qs.generate_signals("20240102", {})

        assert signals == {}, "fundamentals 키 없으면 빈 dict여야 합니다."

    def test_generate_signals_valid_tickers(self):
        """반환된 시그널의 키가 유효한 종목코드이다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy(num_stocks=5)
        fundamentals = _make_quality_fundamentals()
        data = {"fundamentals": fundamentals}

        signals = qs.generate_signals("20240102", data)

        valid_tickers = set(fundamentals["ticker"].values)
        for ticker in signals:
            assert ticker in valid_tickers, (
                f"'{ticker}'가 유효한 종목이 아닙니다."
            )

    def test_generate_signals_positive_weights(self):
        """모든 비중이 양수이다."""
        QualityStrategy = _import_quality_strategy()
        qs = QualityStrategy(num_stocks=10)
        data = {"fundamentals": _make_quality_fundamentals()}

        signals = qs.generate_signals("20240102", data)

        for ticker, weight in signals.items():
            assert weight > 0, (
                f"종목 {ticker}의 비중 {weight}이 양수여야 합니다."
            )
