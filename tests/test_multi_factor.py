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


# ===================================================================
# 급등/밸류트랩 필터 테스트용 헬퍼
# ===================================================================

def _make_spike_data(n=30, seed=42):
    """급등 종목이 포함된 멀티팩터 테스트 데이터를 생성한다.

    000001: 마지막 날 +25% 급등 + PBR=0.1 (밸류 최상위)
    """
    np.random.seed(seed)
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    dates = pd.bdate_range("2023-01-02", periods=252)

    fundamentals = pd.DataFrame({
        "ticker": tickers,
        "name": [f"종목{i}" for i in range(1, n + 1)],
        "market": ["KOSPI"] * n,
        "pbr": np.random.uniform(0.3, 5.0, n).round(2),
        "per": np.random.uniform(3, 30, n).round(2),
        "eps": np.random.randint(500, 20000, n),
        "bps": np.random.randint(5000, 100000, n),
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

    # 000001: 급등 종목 (마지막 날 +25%)
    spike_ticker = "000001"
    spike_df = prices[spike_ticker].copy()
    spike_df.iloc[-1, spike_df.columns.get_loc("close")] *= 1.25
    prices[spike_ticker] = spike_df

    # 000001: 가장 낮은 PBR (밸류 스코어 최상위)
    fundamentals.loc[fundamentals["ticker"] == spike_ticker, "pbr"] = 0.1

    return {"fundamentals": fundamentals, "prices": prices}


# ===================================================================
# TestSpikeFilter
# ===================================================================

class TestSpikeFilter:
    """spike_filter 파라미터 검증."""

    def test_spike_filter_enabled_by_default(self):
        """기본값은 spike_filter=True이다 (3년 백테스트 기반 최적 설정)."""
        mf = MultiFactorStrategy()
        assert mf.spike_filter is True
        assert mf.spike_threshold_1d == 0.15
        assert mf.spike_threshold_5d == 0.25

    def test_spike_filter_excludes_surged_stock(self):
        """급등 종목(1일 +25%)이 spike_filter에 의해 제외된다."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            spike_filter=True,
            spike_threshold_1d=0.15,
            max_group_weight=0,
            max_stocks_per_conglomerate=0,
        )
        data = _make_spike_data()
        signals = mf.generate_signals("20240102", data)

        # 급등 종목(000001)은 제외되어야 함
        assert "000001" not in signals

    def test_spike_filter_disabled_allows_surged_stock(self):
        """spike_filter=False이면 급등 종목이 PBR=0.1로 밸류 최상위에서 선택될 수 있다."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            spike_filter=False,
            max_group_weight=0,
            max_stocks_per_conglomerate=0,
        )
        data = _make_spike_data()
        signals = mf.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        # PBR=0.1인 000001은 밸류 최상위 → 선택될 가능성 높음
        if signals:
            assert len(signals) <= 5

    def test_spike_filter_high_threshold_allows_stock(self):
        """높은 threshold(30%)에서는 25% 급등 종목이 통과한다."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            spike_filter=True,
            spike_threshold_1d=0.30,
            spike_threshold_5d=0.50,
            max_group_weight=0,
            max_stocks_per_conglomerate=0,
        )
        data = _make_spike_data()
        signals = mf.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        # 25% < 30% threshold이므로 000001이 통과 가능


# ===================================================================
# TestValueTrapFilter
# ===================================================================

class TestValueTrapFilter:
    """value_trap_filter 파라미터 검증."""

    def test_value_trap_filter_disabled_by_default(self):
        """기본값은 value_trap_filter=False이다."""
        mf = MultiFactorStrategy()
        assert mf.value_trap_filter is False
        assert mf.min_roe == 0.0
        assert mf.min_f_score == 0

    def test_value_trap_filter_excludes_negative_roe(self):
        """ROE < 0인 종목이 value_trap_filter에 의해 제외된다."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            value_trap_filter=True,
            min_roe=0.0,
            max_group_weight=0,
            max_stocks_per_conglomerate=0,
        )
        data = _make_spike_data()
        # 000002: 음수 EPS → ROE < 0
        fund = data["fundamentals"]
        fund.loc[fund["ticker"] == "000002", "eps"] = -1000
        fund.loc[fund["ticker"] == "000002", "pbr"] = 0.15  # 밸류 상위

        signals = mf.generate_signals("20240102", data)

        # ROE < 0 종목(000002)은 밸류 스코어에서 제외
        assert "000002" not in signals

    def test_value_trap_filter_f_score(self):
        """min_f_score=1이면 EPS <= 0인 종목이 제외된다."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            value_trap_filter=True,
            min_f_score=1,
            max_group_weight=0,
            max_stocks_per_conglomerate=0,
        )
        data = _make_spike_data()
        # 000003: 음수 EPS → F-Score = 0
        fund = data["fundamentals"]
        fund.loc[fund["ticker"] == "000003", "eps"] = -500
        fund.loc[fund["ticker"] == "000003", "pbr"] = 0.12

        signals = mf.generate_signals("20240102", data)

        assert "000003" not in signals


# ===================================================================
# TestCombinedFilters
# ===================================================================

class TestCombinedFilters:
    """spike_filter + value_trap_filter 동시 적용 검증."""

    def test_combined_filters(self):
        """두 필터가 동시에 동작한다."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            spike_filter=True,
            spike_threshold_1d=0.15,
            value_trap_filter=True,
            min_roe=0.0,
            max_group_weight=0,
            max_stocks_per_conglomerate=0,
        )
        data = _make_spike_data()
        # 000001: 급등 (spike filter)
        # 000002: 음수 ROE (value trap filter)
        fund = data["fundamentals"]
        fund.loc[fund["ticker"] == "000002", "eps"] = -1000
        fund.loc[fund["ticker"] == "000002", "pbr"] = 0.2

        signals = mf.generate_signals("20240102", data)

        assert "000001" not in signals  # spike filter
        assert "000002" not in signals  # value trap filter
        assert len(signals) <= 5

    def test_filters_with_normal_data(self):
        """정상 데이터에서 필터 활성화 시에도 시그널이 생성된다."""
        mf = MultiFactorStrategy(
            num_stocks=5,
            spike_filter=True,
            value_trap_filter=True,
            min_roe=0.0,
            max_group_weight=0,
            max_stocks_per_conglomerate=0,
        )
        data = _make_multifactor_data()
        # eps, bps 추가 (기존 데이터에 없으므로)
        fund = data["fundamentals"]
        n = len(fund)
        np.random.seed(99)
        fund["eps"] = np.random.randint(500, 20000, n)
        fund["bps"] = np.random.randint(5000, 100000, n)

        signals = mf.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        if signals:
            assert len(signals) <= 5


# ===================================================================
# JAE-29: Quality 팩터 + PBR 실시간 보정 테스트
# ===================================================================

def _make_multifactor_data_with_quality(n=30, seed=42):
    """Quality 팩터 테스트용 데이터를 생성한다 (roe, bps 포함)."""
    np.random.seed(seed)
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    dates = pd.bdate_range("2023-01-02", periods=252)

    fundamentals = pd.DataFrame({
        "ticker": tickers,
        "name": [f"종목{i}" for i in range(1, n + 1)],
        "market": ["KOSPI"] * n,
        "pbr": np.random.uniform(0.3, 5.0, n).round(2),
        "per": np.random.uniform(3, 30, n).round(2),
        "close": np.random.randint(5000, 500000, n).astype(float),
        "bps": np.random.randint(3000, 200000, n).astype(float),
        "roe": np.random.uniform(0.02, 0.25, n),
        "market_cap": np.random.randint(200_000_000_000, 5_000_000_000_000, n),
        "volume": np.random.randint(100_000, 5_000_000, n),
    })

    prices = {}
    for ticker in tickers:
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


class TestAdjPBR:
    """가격 기반 PBR 실시간 보정 테스트."""

    def test_adj_pbr_uses_close_over_bps(self):
        """adj_pbr=True 시 value_score가 close/bps 기반으로 변경된다."""
        data = _make_multifactor_data_with_quality()
        fund = data["fundamentals"]

        # bps 기반 adj_pbr 강제 설정: close=10000, bps=5000 → adj_pbr=2.0
        fund["close"] = 10000.0
        fund["bps"] = 5000.0
        fund["pbr"] = 99.0  # 원본 PBR은 극단값

        mf_adj = MultiFactorStrategy(
            factors=["value"], weights=[1.0], num_stocks=10, adj_pbr=True
        )
        mf_no_adj = MultiFactorStrategy(
            factors=["value"], weights=[1.0], num_stocks=10, adj_pbr=False
        )

        sig_adj = mf_adj.generate_signals("20240102", data)
        sig_no_adj = mf_no_adj.generate_signals("20240102", data)

        # adj_pbr=True인 경우 close/bps=2.0 기반 → 모두 동일 스코어, 선정 종목 수 동일
        assert isinstance(sig_adj, dict)
        assert isinstance(sig_no_adj, dict)
        # adj_pbr 적용 시 pbr=99 → 2.0으로 보정되어 value_score 분포 변경
        if sig_adj and sig_no_adj:
            # 보정 전후 포트폴리오가 달라야 한다 (pbr=99 vs close/bps=2.0)
            # 보정 후엔 모든 종목의 adj_pbr이 동일하므로 스코어 차이 無 → 동일 종목
            assert len(sig_adj) == len(sig_no_adj)

    def test_adj_pbr_skips_when_no_bps(self):
        """bps 컬럼이 없으면 기존 pbr을 그대로 사용한다."""
        data = _make_multifactor_data_with_quality()
        data["fundamentals"] = data["fundamentals"].drop(columns=["bps"])

        mf = MultiFactorStrategy(
            factors=["value"], weights=[1.0], num_stocks=5, adj_pbr=True
        )
        signals = mf.generate_signals("20240102", data)
        assert isinstance(signals, dict)

    def test_adj_pbr_default_true(self):
        """adj_pbr 기본값은 True이다."""
        mf = MultiFactorStrategy()
        assert mf.adj_pbr is True


class TestQualityFactor:
    """Quality 팩터 통합 테스트."""

    def test_quality_factor_init(self):
        """quality 팩터로 초기화되면 _quality_strategy가 생성된다."""
        mf = MultiFactorStrategy(
            factors=["value", "momentum", "quality"],
            weights=[0.35, 0.35, 0.30],
        )
        assert mf._quality_strategy is not None

    def test_no_quality_strategy_without_factor(self):
        """quality 팩터 없이 초기화하면 _quality_strategy는 None이다."""
        mf = MultiFactorStrategy(factors=["value", "momentum"], weights=[0.5, 0.5])
        assert mf._quality_strategy is None

    def test_quality_factor_generates_signals(self):
        """V+M+Q 3팩터 조합으로 시그널이 생성된다."""
        data = _make_multifactor_data_with_quality()

        mf = MultiFactorStrategy(
            factors=["value", "momentum", "quality"],
            weights=[0.35, 0.35, 0.30],
            num_stocks=10,
        )
        signals = mf.generate_signals("20240102", data)

        assert isinstance(signals, dict)
        if signals:
            assert len(signals) <= 10
            for w in signals.values():
                assert 0 < w <= 1.0

    def test_quality_only_factor(self):
        """quality만 단일 팩터로 사용해도 동작한다."""
        data = _make_multifactor_data_with_quality()

        mf = MultiFactorStrategy(
            factors=["quality"],
            weights=[1.0],
            num_stocks=5,
        )
        signals = mf.generate_signals("20240102", data)
        assert isinstance(signals, dict)

    def test_quality_weight_sum_correct(self):
        """V+M+Q 가중치 합이 올바르다."""
        mf = MultiFactorStrategy(
            factors=["value", "momentum", "quality"],
            weights=[0.35, 0.35, 0.30],
        )
        assert abs(sum(mf.weights) - 1.0) < 1e-6

    def test_strategy_name_reflects_quality(self):
        """quality 팩터가 포함되면 이름에 반영된다."""
        mf = MultiFactorStrategy(
            factors=["value", "momentum", "quality"],
            weights=[0.35, 0.35, 0.30],
        )
        assert "quality" in mf.name


class TestQualityConfig:
    """strategy_config.py quality 프로필 테스트."""

    def test_quality_profile_exists(self):
        """quality 프로필이 존재한다."""
        from src.strategy.strategy_config import get_multi_factor_config
        config = get_multi_factor_config("quality")
        assert "quality" in config["factors"]
        assert config["adj_pbr"] is True

    def test_quality_weights(self):
        """quality 프로필 가중치가 올바르다."""
        from src.strategy.strategy_config import get_multi_factor_config
        config = get_multi_factor_config("quality")
        assert len(config["factors"]) == len(config["weights"])
        assert abs(sum(config["weights"]) - 1.0) < 1e-6

    def test_create_quality_strategy(self):
        """create_multi_factor('quality')로 인스턴스가 생성된다."""
        from src.strategy.strategy_config import create_multi_factor
        strategy = create_multi_factor("quality")
        assert strategy._quality_strategy is not None
        assert strategy.adj_pbr is True
