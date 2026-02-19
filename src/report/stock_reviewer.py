"""종목 리뷰 모듈.

보유 종목의 52주 고저 비교, 수익률 시그널 등을 분석하여
일일 리뷰를 생성한다. Feature Flag 'stock_review'로 제어.
"""

import pandas as pd

from src.data.collector import get_price_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StockReviewer:
    """보유 종목 리뷰어.

    Args:
        max_stocks: 리뷰할 최대 종목 수.
    """

    def __init__(self, max_stocks: int = 10) -> None:
        self.max_stocks = max_stocks

    def review_holdings(
        self,
        holdings: list[dict],
        lookback_days: int = 252,
    ) -> str:
        """보유 종목 리뷰를 생성한다.

        Args:
            holdings: KIS 잔고 holdings 리스트.
            lookback_days: 52주 고저 계산용 과거 데이터 일수.

        Returns:
            포매팅된 리뷰 텍스트.
        """
        if not holdings:
            return "[종목 리뷰] 보유 종목 없음"

        lines = ["[보유 종목 리뷰]", "=" * 50]
        reviewed = holdings[: self.max_stocks]

        for h in reviewed:
            ticker = h.get("ticker", "")
            name = h.get("name", ticker)
            pnl_pct = h.get("pnl_pct", 0.0)
            current_price = h.get("current_price", 0)

            lines.append(f"\n{name} ({ticker})")
            lines.append(
                f"  수익률: {pnl_pct:+.2f}%  현재가: {current_price:,}"
            )

            # 52주 고저 비교
            try:
                now = pd.Timestamp.now("Asia/Seoul")
                end_date = now.strftime("%Y%m%d")
                start_date = (
                    now - pd.Timedelta(days=lookback_days)
                ).strftime("%Y%m%d")
                price_df = get_price_data(ticker, start_date, end_date)

                if not price_df.empty and "close" in price_df.columns:
                    high_52w = int(price_df["close"].max())
                    low_52w = int(price_df["close"].min())
                    from_high = (
                        (current_price / high_52w - 1) * 100
                        if high_52w > 0
                        else 0
                    )
                    from_low = (
                        (current_price / low_52w - 1) * 100
                        if low_52w > 0
                        else 0
                    )
                    lines.append(
                        f"  52주 고가: {high_52w:,} ({from_high:+.1f}%)"
                    )
                    lines.append(
                        f"  52주 저가: {low_52w:,} ({from_low:+.1f}%)"
                    )

                    # 20일 평균 거래량 대비
                    if "volume" in price_df.columns and len(price_df) >= 20:
                        avg_vol = int(price_df["volume"].tail(20).mean())
                        last_vol = int(price_df["volume"].iloc[-1])
                        vol_ratio = (
                            last_vol / avg_vol if avg_vol > 0 else 0
                        )
                        if vol_ratio > 2.0:
                            lines.append(
                                f"  거래량: {last_vol:,} "
                                f"(20일 평균 대비 {vol_ratio:.1f}배)"
                            )
            except Exception as e:
                logger.debug("52주 데이터 조회 실패 (%s): %s", ticker, e)

            # 시그널 판단
            signal = "HOLD"
            if pnl_pct > 20:
                signal = "익절 검토"
            elif pnl_pct < -15:
                signal = "손절 검토"
            lines.append(f"  시그널: {signal}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)
