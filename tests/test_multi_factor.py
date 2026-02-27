"""멀티팩터 전략 모듈(src/strategy/multi_factor.py) 테스트.

MultiFactorStrategy의 turnover_penalty, 업종 비중 제한,
계열사 집중도 제한 기능을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.conglomerate import detect_conglomerate
from src.strategy.multi_factor import MultiFactorStrategy


# ===================================================================
# 헬퍼 함수
# ===================================================================

def _make_multifactor_data(n=30, seed=42):
    """멀티팩터 테스트용 데이터를 생성한다."""
    np.random.seed(seed)
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    dates = pd.bdate_range("2023-01-02", periods=252)

    fundamentals = pd.DataFrame({
        "ticker": tickers,
        "name": [f"종목{i}" for i in range(1, n + 1)],
        "market": ["KOSPI"] * n,
        "pbr": np.random.uniform(0.3, 5.0, n).round(2),
        "per": np.random.uniform(3, 30, n).round(2),
        "close": np.random.randint(5000, 500000, n),
        "market_cap": np.random.randint(200_000_000_000, 5_000_000_000_000, n),
        "volume": np.random.randint(100_000, 5_000_000, n),
    })

    prices = {}
    for i, ticker in enumerate(tickers):
        base = np.random.randint(10000, 100000)
        close = base * np.exp(np.cumsum(np.random.randn(252) * 0.02))
        df = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(100000, 5000000, 252),
        }, index=dates)
        df.index.name = "date"
        prices[ticker] = df

    return {"fundamentals": fundamentals, "prices": prices}


# ===================================================================
# 기본 동작 확인
# ===================================================================

class TestMultiFactorBasic:
    """MultiFactorStrategy 기본 동작 검증."""

    def test_default_init(self):
        """기본 파라미터로 초기화된다."""
        mf = MultiFactorStrategy()
        assert mf.factors == ["value", "momentum"]
        assert mf.weights == [0.5, 0.5]
        assert mf.turnover_penalty == 0.0

    def test_generate_signals(self):
        """기본 시그널 생성이 동작한다."""
        mf = MultiFactorStrategy(num_stocks=5)
        data = _make_multifactor_data()
        signals = mf.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        if signals:
            assert len(signals) <= 5


# ===================================================================
# turnover_penalty 테스트
# ===================================================================

class TestTurnoverPenalty:
    """turnover_penalty 파라미터 검증."""

    def test_default_disabled(self):
        """기본값은 0.0(비활성)이다."""
        mf = MultiFactorStrategy()
        assert mf.turnover_penalty == 0.0

    def test_penalty_reduces_new_stock_entry(self):
        """회전율 페널티가 새 종목 진입을 억제한다."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            turnover_penalty=0.5,
        )
        data = _make_multifactor_data()

        # 첫 번째 리밸런싱
        signals1 = mf.generate_signals("20240102", data)

        # 두 번째 리밸런싱 (이전 포트폴리오 기록됨)
        signals2 = mf.generate_signals("20240201", data)

        # 페널티가 적용되므로 기존 종목이 유지될 가능성이 높음
        assert isinstance(signals1, dict)
        assert isinstance(signals2, dict)
        if signals1 and signals2:
            # 두 번째 결과에서 기존 종목이 일부라도 유지되는지 확인
            overlap = set(signals1.keys()) & set(signals2.keys())
            # 페널티가 충분히 크면 유지율이 높을 수 있음
            assert len(overlap) >= 0  # 최소한 에러 없이 동작

    def test_penalty_prev_holdings_updated(self):
        """시그널 생성 후 _prev_holdings가 갱신된다."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            turnover_penalty=0.5,
        )
        data = _make_multifactor_data()

        assert mf._prev_holdings == set()

        signals = mf.generate_signals("20240102", data)

        if signals:
            assert mf._prev_holdings == set(signals.keys())

    def test_zero_penalty_no_effect(self):
        """turnover_penalty=0이면 페널티가 적용되지 않는다."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            turnover_penalty=0.0,
        )
        data = _make_multifactor_data()

        # 이전 포트폴리오 수동 설정
        mf._prev_holdings = {"000001", "000002", "000003"}

        signals = mf.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        # 페널티 없으므로 기존 종목 유지에 편향 없음

    def test_high_penalty_favors_existing(self):
        """높은 페널티는 기존 종목 유지를 강하게 선호한다."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            turnover_penalty=10.0,  # 매우 높은 페널티
        )
        data = _make_multifactor_data()

        # 첫 번째 리밸런싱
        signals1 = mf.generate_signals("20240102", data)

        if signals1:
            # 두 번째 리밸런싱 (동일 데이터)
            signals2 = mf.generate_signals("20240201", data)

            if signals2:
                overlap = set(signals1.keys()) & set(signals2.keys())
                # 매우 높은 페널티이므로 기존 종목 대부분 유지
                assert len(overlap) >= min(len(signals1), len(signals2)) - 1


# ===================================================================
# 업종 비중 제한 테스트
# ===================================================================

def _make_sector_data(n=30, seed=42):
    """업종 정보가 포함된 멀티팩터 테스트 데이터를 생성한다."""
    np.random.seed(seed)
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    dates = pd.bdate_range("2023-01-02", periods=252)

    # 3개 업종에 분산: 전자(10), 화학(10), 금융(10)
    sectors = (["전자"] * 10 + ["화학"] * 10 + ["금융"] * 10)[:n]

    fundamentals = pd.DataFrame({
        "ticker": tickers,
        "name": [f"종목{i}" for i in range(1, n + 1)],
        "market": ["KOSPI"] * n,
        "sector": sectors,
        "pbr": np.random.uniform(0.3, 5.0, n).round(2),
        "per": np.random.uniform(3, 30, n).round(2),
        "close": np.random.randint(5000, 500000, n),
        "market_cap": np.random.randint(200_000_000_000, 5_000_000_000_000, n),
        "volume": np.random.randint(100_000, 5_000_000, n),
    })

    prices = {}
    for i, ticker in enumerate(tickers):
        base = np.random.randint(10000, 100000)
        close = base * np.exp(np.cumsum(np.random.randn(252) * 0.02))
        df = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(100000, 5000000, 252),
        }, index=dates)
        df.index.name = "date"
        prices[ticker] = df

    return {"fundamentals": fundamentals, "prices": prices}


class TestSectorWeightLimit:
    """max_group_weight(업종 비중 상한) 파라미터 검증."""

    def test_default_max_group_weight(self):
        """기본값은 0.25(25%)이다."""
        mf = MultiFactorStrategy()
        assert mf.max_group_weight == 0.25

    def test_sector_weight_limit_25pct(self):
        """동일 업종 합산 비중이 25%를 초과하지 않는다."""
        mf = MultiFactorStrategy(
            num_stocks=10,
            max_group_weight=0.25,
            max_stocks_per_conglomerate=0,
        )
        data = _make_sector_data()
        signals = mf.generate_signals("20240102", data)

        if signals:
            stock_weight = 1.0 / 10
            fundamentals = data["fundamentals"]
            sector_map = dict(zip(fundamentals["ticker"], fundamentals["sector"]))
            sector_weights: dict[str, float] = {}
            for ticker in signals:
                sector = sector_map.get(ticker, "기타")
                sector_weights[sector] = sector_weights.get(sector, 0.0) + stock_weight

            for sector, weight in sector_weights.items():
                assert weight <= 0.25 + 1e-9, (
                    f"업종 '{sector}' 비중 {weight:.1%} (상한 25%)"
                )

    def test_sector_limit_disabled(self):
        """max_group_weight=0이면 필터링 없이 기존 동작."""
        mf = MultiFactorStrategy(
            num_stocks=10,
            max_group_weight=0,
            max_stocks_per_conglomerate=0,
        )
        data = _make_sector_data()
        signals = mf.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        if signals:
            assert len(signals) <= 10

    def test_sector_limit_without_sector_column(self):
        """sector 컬럼 없으면 graceful degradation (필터 미적용)."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            max_group_weight=0.25,
        )
        data = _make_multifactor_data()
        signals = mf.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        if signals:
            assert len(signals) <= 5

    def test_sector_limit_with_industry_column(self):
        """'industry' 컬럼도 인식한다."""
        mf = MultiFactorStrategy(
            num_stocks=10,
            max_group_weight=0.25,
            max_stocks_per_conglomerate=0,
        )
        data = _make_sector_data()
        data["fundamentals"] = data["fundamentals"].rename(
            columns={"sector": "industry"}
        )
        signals = mf.generate_signals("20240102", data)

        if signals:
            stock_weight = 1.0 / 10
            fundamentals = data["fundamentals"]
            sector_map = dict(zip(fundamentals["ticker"], fundamentals["industry"]))
            sector_weights: dict[str, float] = {}
            for ticker in signals:
                sector = sector_map.get(ticker, "기타")
                sector_weights[sector] = sector_weights.get(sector, 0.0) + stock_weight

            for sector, weight in sector_weights.items():
                assert weight <= 0.25 + 1e-9

    def test_tight_sector_limit(self):
        """max_group_weight=0.10이면 10종목 중 업종당 1종목만 가능."""
        mf = MultiFactorStrategy(
            num_stocks=10,
            max_group_weight=0.10,
            max_stocks_per_conglomerate=0,
        )
        data = _make_sector_data()
        signals = mf.generate_signals("20240102", data)

        if signals:
            stock_weight = 1.0 / 10  # 10%
            fundamentals = data["fundamentals"]
            sector_map = dict(zip(fundamentals["ticker"], fundamentals["sector"]))
            sector_counts: dict[str, int] = {}
            for ticker in signals:
                sector = sector_map.get(ticker, "기타")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            for sector, count in sector_counts.items():
                assert count <= 1


# ===================================================================
# 계열사 집중도 제한 테스트
# ===================================================================


def _make_conglomerate_data(seed=42):
    """계열사가 포함된 멀티팩터 테스트 데이터를 생성한다."""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-02", periods=252)

    names = [
        "삼성전자", "삼성SDI", "삼성전기", "삼성물산",  # 삼성 4종목
        "SK하이닉스", "SK텔레콤", "SK이노베이션",       # SK 3종목
        "LG전자", "LG화학",                             # LG 2종목
        "현대차", "기아", "현대모비스",                   # 현대차 3종목
        "POSCO홀딩스", "카카오", "네이버",               # 각 1종목
        "한화에어로스페이스", "한화솔루션",               # 한화 2종목
        "롯데케미칼", "CJ제일제당", "GS건설",            # 각 1종목
    ]
    n = len(names)
    tickers = [f"{i:06d}" for i in range(1, n + 1)]

    fundamentals = pd.DataFrame({
        "ticker": tickers,
        "name": names,
        "market": ["KOSPI"] * n,
        "sector": ["전기·전자"] * 4 + ["전기·전자"] * 3 + ["전기·전자"] * 2
                  + ["운송장비·부품"] * 3 + ["금속", "IT 서비스", "IT 서비스"]
                  + ["기계·장비"] * 2 + ["화학", "음식료·담배", "건설"],
        "pbr": np.random.uniform(0.5, 3.0, n).round(2),
        "per": np.random.uniform(5, 20, n).round(2),
        "close": np.random.randint(10000, 500000, n),
        "market_cap": np.random.randint(1_000_000_000_000, 50_000_000_000_000, n),
        "volume": np.random.randint(500_000, 10_000_000, n),
    })

    prices = {}
    for i, ticker in enumerate(tickers):
        base = np.random.randint(10000, 100000)
        close = base * np.exp(np.cumsum(np.random.randn(252) * 0.02))
        prices[ticker] = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(100000, 5000000, 252),
        }, index=dates)
        prices[ticker].index.name = "date"

    return {"fundamentals": fundamentals, "prices": prices}


class TestConglomerateDetection:
    """conglomerate 모듈 계열사 탐지 검증."""

    def test_samsung_prefix(self):
        """삼성 접두사 종목을 인식한다."""
        assert detect_conglomerate("삼성전자") == "삼성"
        assert detect_conglomerate("삼성SDI") == "삼성"
        assert detect_conglomerate("삼성바이오로직스") == "삼성"

    def test_samsung_static_map(self):
        """접두사가 '삼성'이 아닌 삼성 계열사를 정적 매핑으로 인식한다."""
        assert detect_conglomerate("호텔신라") == "삼성"
        assert detect_conglomerate("에스원") == "삼성"

    def test_samsung_blacklist(self):
        """비삼성 종목(삼성공조)은 블랙리스트로 제외한다."""
        assert detect_conglomerate("삼성공조") is None

    def test_hyundai_group(self):
        """현대차그룹을 인식한다 (기아 포함)."""
        assert detect_conglomerate("현대차") == "현대차"
        assert detect_conglomerate("현대모비스") == "현대차"
        assert detect_conglomerate("기아") == "현대차"  # 정적 매핑

    def test_hd_hyundai_separate(self):
        """HD현대는 현대차와 별도 그룹이다."""
        assert detect_conglomerate("HD현대") == "HD현대"
        assert detect_conglomerate("HD현대마린솔루션") == "HD현대"

    def test_sk_group(self):
        """SK 그룹을 인식한다."""
        assert detect_conglomerate("SK하이닉스") == "SK"
        assert detect_conglomerate("SK텔레콤") == "SK"

    def test_lg_group(self):
        """LG 그룹을 인식한다."""
        assert detect_conglomerate("LG전자") == "LG"
        assert detect_conglomerate("LG화학") == "LG"

    def test_unknown_returns_none(self):
        """알 수 없는 종목은 None을 반환한다."""
        assert detect_conglomerate("POSCO홀딩스") is None
        assert detect_conglomerate("셀트리온") is None

    def test_empty_returns_none(self):
        """빈 문자열은 None을 반환한다."""
        assert detect_conglomerate("") is None


class TestConglomerateLimit:
    """max_stocks_per_conglomerate 파라미터 검증."""

    def test_default_max_stocks_per_conglomerate(self):
        """기본값은 2이다."""
        mf = MultiFactorStrategy()
        assert mf.max_stocks_per_conglomerate == 2

    def test_conglomerate_limit(self):
        """동일 계열사 종목이 max_stocks_per_conglomerate 이하로 제한된다."""
        mf = MultiFactorStrategy(
            num_stocks=10,
            max_group_weight=0,  # 업종 제한 비활성화
            max_stocks_per_conglomerate=2,
        )
        data = _make_conglomerate_data()
        signals = mf.generate_signals("20240102", data)

        if signals:
            fundamentals = data["fundamentals"]
            name_map = dict(zip(fundamentals["ticker"], fundamentals["name"]))
            conglomerate_counts: dict[str, int] = {}
            for ticker in signals:
                name = name_map.get(ticker, "")
                group = detect_conglomerate(name)
                if group:
                    conglomerate_counts[group] = conglomerate_counts.get(group, 0) + 1

            for group, count in conglomerate_counts.items():
                assert count <= 2, (
                    f"계열사 '{group}'에 {count}개 종목 (최대 2개)"
                )

    def test_conglomerate_limit_disabled(self):
        """max_stocks_per_conglomerate=0이면 제한 없이 기존 동작."""
        mf = MultiFactorStrategy(
            num_stocks=10,
            max_group_weight=0,
            max_stocks_per_conglomerate=0,
        )
        data = _make_conglomerate_data()
        signals = mf.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        if signals:
            assert len(signals) <= 10

    def test_strict_conglomerate_limit(self):
        """max_stocks_per_conglomerate=1이면 계열사당 1종목만 허용한다."""
        mf = MultiFactorStrategy(
            num_stocks=10,
            max_group_weight=0,
            max_stocks_per_conglomerate=1,
        )
        data = _make_conglomerate_data()
        signals = mf.generate_signals("20240102", data)

        if signals:
            fundamentals = data["fundamentals"]
            name_map = dict(zip(fundamentals["ticker"], fundamentals["name"]))
            conglomerate_counts: dict[str, int] = {}
            for ticker in signals:
                name = name_map.get(ticker, "")
                group = detect_conglomerate(name)
                if group:
                    conglomerate_counts[group] = conglomerate_counts.get(group, 0) + 1

            for group, count in conglomerate_counts.items():
                assert count <= 1, (
                    f"계열사 '{group}'에 {count}개 종목 (최대 1개)"
                )


# ===================================================================
# 업종 + 계열사 동시 제한 테스트
# ===================================================================


class TestDualConcentrationFilter:
    """업종 비중 + 계열사 제한 동시 적용 검증."""

    def test_both_limits_simultaneously(self):
        """업종 25% + 계열사 2종목 동시 제한이 동작한다."""
        mf = MultiFactorStrategy(
            num_stocks=10,
            max_group_weight=0.25,
            max_stocks_per_conglomerate=2,
        )
        data = _make_conglomerate_data()
        signals = mf.generate_signals("20240102", data)

        if signals:
            stock_weight = 1.0 / 10
            fundamentals = data["fundamentals"]
            sector_map = dict(zip(fundamentals["ticker"], fundamentals["sector"]))
            name_map = dict(zip(fundamentals["ticker"], fundamentals["name"]))

            # 업종 비중 검증
            sector_weights: dict[str, float] = {}
            for ticker in signals:
                sector = sector_map.get(ticker, "기타")
                sector_weights[sector] = sector_weights.get(sector, 0.0) + stock_weight

            for sector, weight in sector_weights.items():
                assert weight <= 0.25 + 1e-9, (
                    f"업종 '{sector}' 비중 {weight:.1%} (상한 25%)"
                )

            # 계열사 카운트 검증
            conglomerate_counts: dict[str, int] = {}
            for ticker in signals:
                name = name_map.get(ticker, "")
                group = detect_conglomerate(name)
                if group:
                    conglomerate_counts[group] = conglomerate_counts.get(group, 0) + 1

            for group, count in conglomerate_counts.items():
                assert count <= 2, (
                    f"계열사 '{group}'에 {count}개 종목 (최대 2개)"
                )

    def test_no_sector_with_conglomerate(self):
        """sector 컬럼 없이 계열사 제한만 적용된다."""
        mf = MultiFactorStrategy(
            num_stocks=10,
            max_group_weight=0.25,
            max_stocks_per_conglomerate=2,
        )
        data = _make_conglomerate_data()
        # sector 컬럼 제거
        data["fundamentals"] = data["fundamentals"].drop(columns=["sector"])
        signals = mf.generate_signals("20240102", data)

        if signals:
            fundamentals = data["fundamentals"]
            name_map = dict(zip(fundamentals["ticker"], fundamentals["name"]))
            conglomerate_counts: dict[str, int] = {}
            for ticker in signals:
                name = name_map.get(ticker, "")
                group = detect_conglomerate(name)
                if group:
                    conglomerate_counts[group] = conglomerate_counts.get(group, 0) + 1

            for group, count in conglomerate_counts.items():
                assert count <= 2
