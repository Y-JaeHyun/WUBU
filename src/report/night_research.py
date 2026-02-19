"""야간 리서치 모듈.

글로벌 시장 동향 + 국내 전략 시사점을 종합하여 야간 리포트를 생성한다.
Feature Flag 'night_research'로 제어.
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NightResearcher:
    """야간 리서치 생성기.

    Args:
        include_global: 글로벌 시장 데이터 포함 여부.
    """

    def __init__(self, include_global: bool = True) -> None:
        self.include_global = include_global

    def generate_report(
        self,
        global_snapshot: Optional[pd.DataFrame] = None,
        portfolio_state: Optional[dict] = None,
    ) -> str:
        """야간 리서치 리포트를 생성한다.

        Args:
            global_snapshot: get_global_snapshot() 결과 DataFrame.
            portfolio_state: 포트폴리오 현황 dict (total_eval, cash_pct).

        Returns:
            포매팅된 리포트 텍스트.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"[야간 리서치] {now}",
            "=" * 50,
        ]

        # 글로벌 시장 현황
        has_global = (
            self.include_global
            and global_snapshot is not None
            and not global_snapshot.empty
        )

        if has_global:
            lines.append("\n[글로벌 시장]")
            lines.append("-" * 40)
            for _, row in global_snapshot.iterrows():
                sign = "+" if row.get("change_pct", 0) >= 0 else ""
                lines.append(
                    f"  {row['name']:12s} {row['price']:>12,.2f}  "
                    f"({sign}{row['change_pct']:.2f}%)"
                )

            # VIX 기반 시장 심리
            vix_rows = global_snapshot[global_snapshot["symbol"] == "^VIX"]
            if not vix_rows.empty:
                vix = vix_rows.iloc[0]["price"]
                if vix > 30:
                    sentiment = "공포 (VIX > 30)"
                elif vix > 20:
                    sentiment = "경계 (VIX 20-30)"
                else:
                    sentiment = "안정 (VIX < 20)"
                lines.append(f"\n  시장 심리: {sentiment}")

            # S&P500 방향성
            sp_rows = global_snapshot[global_snapshot["symbol"] == "^GSPC"]
            if not sp_rows.empty:
                sp_change = sp_rows.iloc[0]["change_pct"]
                if sp_change > 1:
                    direction = "미국 강세 -> 한국 긍정적"
                elif sp_change < -1:
                    direction = "미국 약세 -> 한국 주의"
                else:
                    direction = "미국 보합 -> 한국 중립"
                lines.append(f"  방향성: {direction}")

            # 환율
            fx_rows = global_snapshot[
                global_snapshot["symbol"] == "USDKRW=X"
            ]
            if not fx_rows.empty:
                usd_krw = fx_rows.iloc[0]["price"]
                fx_change = fx_rows.iloc[0]["change_pct"]
                lines.append(
                    f"\n[환율] USD/KRW: {usd_krw:,.1f} ({fx_change:+.2f}%)"
                )
                if usd_krw > 1400:
                    lines.append(
                        "  원화 약세 -> 수출주 유리, 외국인 매도 주의"
                    )
                elif usd_krw < 1300:
                    lines.append(
                        "  원화 강세 -> 내수주 유리, 외국인 매수 기대"
                    )
        else:
            lines.append("\n[글로벌 시장] 데이터 없음 (yfinance 미설치 또는 기능 비활성)")

        # 내일 전략 시사점
        lines.append("\n[내일 전략 시사점]")
        lines.append("-" * 40)

        if portfolio_state:
            total = portfolio_state.get("total_eval", 0)
            cash_pct = portfolio_state.get("cash_pct", 0)
            mdd = portfolio_state.get("mdd", 0)
            lines.append(
                f"  포트폴리오: {total:,}원 "
                f"(현금 {cash_pct:.1f}%, MDD {mdd:.2%})"
            )

        # 종합 판단
        if has_global:
            sp_rows = global_snapshot[global_snapshot["symbol"] == "^GSPC"]
            vix_rows = global_snapshot[global_snapshot["symbol"] == "^VIX"]
            sp_change = (
                sp_rows.iloc[0]["change_pct"] if not sp_rows.empty else 0
            )
            vix_val = vix_rows.iloc[0]["price"] if not vix_rows.empty else 15

            if sp_change > 1 and vix_val < 20:
                lines.append("  종합: 글로벌 리스크온 -> 공격적 포지션 유지")
            elif sp_change < -1 or vix_val > 25:
                lines.append("  종합: 글로벌 리스크오프 -> 현금 비중 확대 고려")
            else:
                lines.append("  종합: 중립 -> 기존 전략 유지")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)
