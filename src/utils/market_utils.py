"""시장/종목 판별 유틸리티."""

# 알려진 ETF 티커 집합 (정적 캐시)
_KNOWN_ETF_TICKERS: set[str] | None = None


def _load_known_etf_tickers() -> set[str]:
    """프로젝트에 정의된 ETF 유니버스에서 티커를 수집한다."""
    tickers: set[str] = set()

    try:
        from src.data.etf_collector import ETF_UNIVERSE

        tickers |= {v["ticker"] for v in ETF_UNIVERSE.values()}
    except Exception:
        pass

    try:
        from src.strategy.etf_rotation import DEFAULT_ETF_UNIVERSE

        tickers |= set(DEFAULT_ETF_UNIVERSE.keys())
    except Exception:
        pass

    try:
        from src.strategy.cross_asset_momentum import CROSS_ASSET_UNIVERSE

        tickers |= set(CROSS_ASSET_UNIVERSE.keys())
    except Exception:
        pass

    return tickers


def is_etf(ticker: str) -> bool:
    """ETF 종목 여부를 판별한다.

    프로젝트에 정의된 ETF 유니버스(etf_collector, etf_rotation,
    cross_asset_momentum)에 포함된 티커이면 True를 반환한다.

    Args:
        ticker: 종목코드 (예: '069500')

    Returns:
        ETF이면 True, 아니면 False
    """
    global _KNOWN_ETF_TICKERS
    if _KNOWN_ETF_TICKERS is None:
        _KNOWN_ETF_TICKERS = _load_known_etf_tickers()
    return ticker in _KNOWN_ETF_TICKERS
