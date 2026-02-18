"""테스트 공통 fixture 정의.

mock 객체, 샘플 데이터 등을 제공한다.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 샘플 가격 데이터 (OHLCV)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """20 영업일 분량의 샘플 가격 데이터를 반환한다."""
    dates = pd.bdate_range("2024-01-02", periods=20)
    np.random.seed(42)
    base = 70000  # 삼성전자 기준 가격대
    close = base + np.cumsum(np.random.randn(20) * 500).astype(int)
    close = np.maximum(close, 1000)  # 최소 가격 보장

    df = pd.DataFrame(
        {
            "open": close - np.random.randint(0, 500, 20),
            "high": close + np.random.randint(0, 1000, 20),
            "low": close - np.random.randint(0, 1000, 20),
            "close": close,
            "volume": np.random.randint(100_000, 10_000_000, 20),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# 샘플 시가총액 데이터
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_market_cap_data() -> pd.DataFrame:
    """20 영업일 분량의 샘플 시가총액 데이터를 반환한다."""
    dates = pd.bdate_range("2024-01-02", periods=20)
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "market_cap": np.random.randint(1_000_000_000_000, 5_000_000_000_000, 20),
            "volume": np.random.randint(100_000, 10_000_000, 20),
            "trade_value": np.random.randint(1_000_000_000, 100_000_000_000, 20),
            "listed_shares": [5_969_782_550] * 20,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# 샘플 펀더멘탈 데이터 (개별 종목)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_fundamental_data() -> pd.DataFrame:
    """20 영업일 분량의 개별 종목 펀더멘탈 데이터를 반환한다."""
    dates = pd.bdate_range("2024-01-02", periods=20)

    df = pd.DataFrame(
        {
            "bps": [45000] * 20,
            "per": [12.5] * 20,
            "pbr": [1.5] * 20,
            "eps": [5600] * 20,
            "div_yield": [2.1] * 20,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# 샘플 전종목 펀더멘탈 데이터 (스크리닝용)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_all_fundamentals() -> pd.DataFrame:
    """전종목 펀더멘탈 데이터를 반환한다. 30개 종목."""
    np.random.seed(42)
    n = 30
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    names = [f"종목{i}" for i in range(1, n + 1)]

    df = pd.DataFrame(
        {
            "ticker": tickers,
            "name": names,
            "market": ["KOSPI"] * 15 + ["KOSDAQ"] * 15,
            "bps": np.random.randint(5000, 100000, n),
            "per": np.abs(np.random.randn(n) * 10 + 10).round(2),
            "pbr": np.abs(np.random.randn(n) * 1 + 1.5).round(2),
            "eps": np.random.randint(500, 20000, n),
            "div_yield": (np.random.rand(n) * 5).round(2),
            "close": np.random.randint(5000, 500000, n),
            "market_cap": np.random.randint(
                50_000_000_000, 5_000_000_000_000, n
            ),
            "volume": np.random.randint(10_000, 5_000_000, n),
        }
    )
    return df


# ---------------------------------------------------------------------------
# 샘플 종목 리스트
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_stock_list() -> pd.DataFrame:
    """종목 리스트 샘플 (10 종목)."""
    tickers = [
        ("005930", "삼성전자", "KOSPI"),
        ("000660", "SK하이닉스", "KOSPI"),
        ("035720", "카카오", "KOSPI"),
        ("035420", "NAVER", "KOSPI"),
        ("006400", "삼성SDI", "KOSPI"),
        ("373220", "LG에너지솔루션", "KOSPI"),
        ("247540", "에코프로비엠", "KOSDAQ"),
        ("086520", "에코프로", "KOSDAQ"),
        ("403870", "HPSP", "KOSDAQ"),
        ("058470", "리노공업", "KOSDAQ"),
    ]
    return pd.DataFrame(tickers, columns=["ticker", "name", "market"])


# ---------------------------------------------------------------------------
# Phase 2: 모멘텀 전략용 fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_price_series() -> pd.Series:
    """252 거래일 분량의 가격 시계열."""
    dates = pd.bdate_range("2023-01-02", periods=252)
    np.random.seed(42)
    prices = 50000 * np.exp(np.cumsum(np.random.randn(252) * 0.02))
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def sample_prices_dict() -> dict:
    """여러 종목의 가격 데이터 dict (티커 -> DataFrame)."""
    np.random.seed(42)
    result = {}
    for i in range(30):
        ticker = f"{i+1:06d}"
        dates = pd.bdate_range("2023-01-02", periods=252)
        base = np.random.randint(10000, 100000)
        close = base * np.exp(np.cumsum(np.random.randn(252) * 0.02))
        df = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": np.random.randint(100000, 5000000, 252),
            },
            index=dates,
        )
        df.index.name = "date"
        result[ticker] = df
    return result


# ---------------------------------------------------------------------------
# Phase 2: 마켓 타이밍 오버레이용 fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def uptrend_index_prices() -> pd.Series:
    """상승 추세 지수 가격 (300일)."""
    dates = pd.bdate_range("2023-01-02", periods=300)
    prices = 2400 + np.arange(300) * 2  # 꾸준히 상승
    return pd.Series(prices, index=dates, dtype=float)


@pytest.fixture
def downtrend_index_prices() -> pd.Series:
    """하락 추세 지수 가격 (300일)."""
    dates = pd.bdate_range("2023-01-02", periods=300)
    prices = 2800 - np.arange(300) * 2  # 꾸준히 하락
    return pd.Series(prices, index=dates, dtype=float)


# ---------------------------------------------------------------------------
# Phase 2: 성과 지표(metrics) 테스트용 fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_portfolio_values() -> pd.Series:
    """100 거래일 분량의 포트폴리오 가치 시계열 (상승 추세)."""
    dates = pd.bdate_range("2024-01-02", periods=100)
    np.random.seed(42)
    values = 100_000_000 * np.exp(np.cumsum(np.random.randn(100) * 0.005 + 0.002))
    return pd.Series(values, index=dates, name="portfolio_value")


@pytest.fixture
def sample_daily_returns() -> pd.Series:
    """100 거래일 분량의 일별 수익률."""
    dates = pd.bdate_range("2024-01-02", periods=100)
    np.random.seed(42)
    returns = np.random.randn(100) * 0.01 + 0.0003
    return pd.Series(returns, index=dates, name="daily_return")


# ---------------------------------------------------------------------------
# Phase 4: 공분산/최적화 테스트용 fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_returns_matrix() -> pd.DataFrame:
    """20개 종목 x 252일 수익률 행렬."""
    np.random.seed(42)
    n_stocks, n_days = 20, 252
    tickers = [f"{i:06d}" for i in range(1, n_stocks + 1)]
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    returns = np.random.randn(n_days, n_stocks) * 0.02
    return pd.DataFrame(returns, index=dates, columns=tickers)


@pytest.fixture
def sample_covariance_matrix(sample_returns_matrix) -> pd.DataFrame:
    """20x20 표본 공분산 행렬."""
    return sample_returns_matrix.cov()


@pytest.fixture
def sample_ml_features() -> pd.DataFrame:
    """30개 종목 x 12 피처 DataFrame."""
    np.random.seed(42)
    n = 30
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    features = {
        "inv_pbr": np.random.rand(n),
        "inv_per": np.random.rand(n),
        "div_yield": np.random.rand(n) * 5,
        "mom_12m": np.random.randn(n) * 0.3,
        "mom_6m": np.random.randn(n) * 0.2,
        "mom_3m": np.random.randn(n) * 0.15,
        "roe": np.random.rand(n) * 30,
        "gpa": np.random.rand(n) * 0.5,
        "vol_20d": np.abs(np.random.randn(n) * 0.02) + 0.01,
        "vol_60d": np.abs(np.random.randn(n) * 0.02) + 0.01,
        "log_market_cap": np.random.randn(n) * 0.5 + 27,
        "volume_ratio": np.random.rand(n) * 2 + 0.5,
    }
    return pd.DataFrame(features, index=tickers)


@pytest.fixture
def sample_ml_targets() -> pd.Series:
    """30개 종목의 1개월 후 수익률."""
    np.random.seed(42)
    n = 30
    tickers = [f"{i:06d}" for i in range(1, n + 1)]
    returns = np.random.randn(n) * 0.1
    return pd.Series(returns, index=tickers, name="forward_return")
