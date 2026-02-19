"""글로벌 시장 데이터 수집 모듈.

yfinance를 사용하여 S&P500, NASDAQ, VIX 등 글로벌 지수 데이터를 수집한다.
Feature Flag 'global_monitor'로 on/off 제어.
"""

from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SYMBOLS: dict[str, str] = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
    "^DJI": "다우존스",
    "^VIX": "VIX",
    "GC=F": "금",
    "CL=F": "원유 WTI",
    "USDKRW=X": "USD/KRW",
}


def _is_yfinance_available() -> bool:
    """yfinance 설치 여부를 확인한다."""
    try:
        import yfinance  # noqa: F401

        return True
    except ImportError:
        return False


def get_global_snapshot(
    symbols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """글로벌 심볼들의 현재가 스냅샷을 반환한다.

    Args:
        symbols: 조회할 심볼 리스트. None이면 DEFAULT_SYMBOLS 사용.

    Returns:
        DataFrame with columns: ['symbol', 'name', 'price', 'change_pct', 'prev_close']
    """
    if not _is_yfinance_available():
        logger.warning("yfinance 미설치. pip install yfinance")
        return pd.DataFrame()

    import yfinance as yf

    target = symbols or list(DEFAULT_SYMBOLS.keys())
    rows: list[dict] = []

    for sym in target:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.fast_info
            price = getattr(info, "last_price", 0) or 0
            prev_close = getattr(info, "previous_close", 0) or 0
            change_pct = (
                ((price / prev_close) - 1) * 100 if prev_close > 0 else 0
            )
            rows.append(
                {
                    "symbol": sym,
                    "name": DEFAULT_SYMBOLS.get(sym, sym),
                    "price": round(price, 2),
                    "change_pct": round(change_pct, 2),
                    "prev_close": round(prev_close, 2),
                }
            )
        except Exception as e:
            logger.warning("글로벌 데이터 조회 실패 (%s): %s", sym, e)

    return pd.DataFrame(rows)


def format_global_snapshot(df: pd.DataFrame) -> str:
    """글로벌 스냅샷을 텍스트로 포매팅한다.

    Args:
        df: get_global_snapshot() 결과 DataFrame.

    Returns:
        포매팅된 텍스트.
    """
    if df.empty:
        return "[글로벌 시장] 데이터 없음"

    lines = ["[글로벌 시장 현황]", "-" * 40]
    for _, row in df.iterrows():
        sign = "+" if row["change_pct"] >= 0 else ""
        lines.append(
            f"  {row['name']:12s} {row['price']:>12,.2f}  "
            f"({sign}{row['change_pct']:.2f}%)"
        )
    return "\n".join(lines)
