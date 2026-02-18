"""일일 리포트 모듈.

포트폴리오 현황, 일일/누적 수익률, 마켓 요약, 마켓 타이밍 시그널을 포함한
일일 리포트를 생성한다.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DailyReport:
    """일일 리포트 생성기.

    포트폴리오 현황과 시장 데이터를 받아 일일 리포트를 생성한다.

    Args:
        report_date: 리포트 날짜 ('YYYYMMDD' 또는 'YYYY-MM-DD'). None이면 오늘 날짜.
    """

    def __init__(self, report_date: Optional[str] = None) -> None:
        if report_date is not None:
            self.report_date = report_date.replace("-", "")
        else:
            self.report_date = datetime.now().strftime("%Y%m%d")

    # ------------------------------------------------------------------
    # 유틸리티
    # ------------------------------------------------------------------
    @staticmethod
    def _format_number(value: float, fmt: str = ",.0f") -> str:
        """숫자를 포매팅한다."""
        try:
            return format(value, fmt)
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _format_change(value: float, fmt: str = ".2f") -> str:
        """변동률을 부호 포함으로 포매팅한다."""
        try:
            sign = "+" if value > 0 else ""
            return f"{sign}{format(value, fmt)}%"
        except (ValueError, TypeError):
            return str(value)

    # ------------------------------------------------------------------
    # 리포트 생성
    # ------------------------------------------------------------------
    def generate(self, portfolio_state: dict, market_data: dict) -> str:
        """일일 리포트를 생성한다.

        Args:
            portfolio_state: 포트폴리오 현황 딕셔너리
                - holdings: {ticker: quantity} — 종목별 보유 수량
                - cash: float — 보유 현금
                - total_value: float — 총 평가 금액
                - daily_return: float (선택) — 일일 수익률 (%)
                - cumulative_return: float (선택) — 누적 수익률 (%)
                - prev_value: float (선택) — 전일 평가 금액
                - ticker_prices: {ticker: price} (선택) — 종목별 현재가
                - ticker_names: {ticker: name} (선택) — 종목별 이름

            market_data: 시장 데이터 딕셔너리
                - kospi: float — KOSPI 지수
                - kosdaq: float — KOSDAQ 지수
                - kospi_change: float — KOSPI 등락률 (%)
                - kosdaq_change: float — KOSDAQ 등락률 (%)
                - kospi_sma200: float (선택) — KOSPI 200일 이동평균
                - usd_krw: float (선택) — 환율

        Returns:
            포매팅된 일일 리포트 문자열
        """
        lines: list[str] = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_formatted = f"{self.report_date[:4]}-{self.report_date[4:6]}-{self.report_date[6:]}"

        lines.append("=" * 70)
        lines.append("  DAILY REPORT")
        lines.append(f"  날짜: {report_formatted}  |  생성: {now_str}")
        lines.append("=" * 70)
        lines.append("")

        # ────────────────────────────────────────────────────
        # 1. 마켓 요약
        # ────────────────────────────────────────────────────
        lines.append("  [ 마켓 요약 ]")
        lines.append("  " + "─" * 50)

        kospi = market_data.get("kospi", 0)
        kosdaq = market_data.get("kosdaq", 0)
        kospi_change = market_data.get("kospi_change", 0)
        kosdaq_change = market_data.get("kosdaq_change", 0)

        lines.append(f"  KOSPI  : {self._format_number(kospi, ',.2f'):>12}  ({self._format_change(kospi_change)})")
        lines.append(f"  KOSDAQ : {self._format_number(kosdaq, ',.2f'):>12}  ({self._format_change(kosdaq_change)})")

        usd_krw = market_data.get("usd_krw")
        if usd_krw is not None:
            lines.append(f"  USD/KRW: {self._format_number(usd_krw, ',.1f'):>12}")

        lines.append("")

        # ────────────────────────────────────────────────────
        # 2. 마켓 타이밍 시그널 (KOSPI vs 200SMA)
        # ────────────────────────────────────────────────────
        kospi_sma200 = market_data.get("kospi_sma200")
        if kospi_sma200 is not None and kospi > 0:
            lines.append("  [ 마켓 타이밍 시그널 ]")
            lines.append("  " + "─" * 50)

            deviation_pct = (kospi / kospi_sma200 - 1) * 100
            if kospi > kospi_sma200:
                signal = "BULLISH (KOSPI > 200SMA)"
                signal_marker = "[++]"
            elif kospi > kospi_sma200 * 0.98:
                signal = "NEUTRAL (KOSPI ~ 200SMA)"
                signal_marker = "[ = ]"
            else:
                signal = "BEARISH (KOSPI < 200SMA)"
                signal_marker = "[--]"

            lines.append(f"  KOSPI       : {self._format_number(kospi, ',.2f')}")
            lines.append(f"  200SMA      : {self._format_number(kospi_sma200, ',.2f')}")
            lines.append(f"  괴리율      : {self._format_change(deviation_pct)}")
            lines.append(f"  시그널      : {signal_marker} {signal}")
            lines.append("")

        # ────────────────────────────────────────────────────
        # 3. 포트폴리오 현황
        # ────────────────────────────────────────────────────
        lines.append("  [ 포트폴리오 현황 ]")
        lines.append("  " + "─" * 50)

        total_value = portfolio_state.get("total_value", 0)
        cash = portfolio_state.get("cash", 0)
        holdings = portfolio_state.get("holdings", {})
        prev_value = portfolio_state.get("prev_value")
        daily_return = portfolio_state.get("daily_return")
        cumulative_return = portfolio_state.get("cumulative_return")
        ticker_prices = portfolio_state.get("ticker_prices", {})
        ticker_names = portfolio_state.get("ticker_names", {})

        lines.append(f"  총 평가금액  : {self._format_number(total_value):>15}원")
        lines.append(f"  보유 현금    : {self._format_number(cash):>15}원")
        cash_ratio = (cash / total_value * 100) if total_value > 0 else 0
        lines.append(f"  현금 비중    : {cash_ratio:>14.1f}%")
        lines.append(f"  보유 종목 수 : {len(holdings):>14}개")

        if daily_return is not None:
            lines.append(f"  일일 수익률  : {self._format_change(daily_return):>15}")
        elif prev_value is not None and prev_value > 0:
            calc_daily = (total_value / prev_value - 1) * 100
            lines.append(f"  일일 수익률  : {self._format_change(calc_daily):>15}")

        if cumulative_return is not None:
            lines.append(f"  누적 수익률  : {self._format_change(cumulative_return):>15}")

        lines.append("")

        # ────────────────────────────────────────────────────
        # 4. 보유 종목 상세
        # ────────────────────────────────────────────────────
        if holdings:
            lines.append("  [ 보유 종목 상세 ]")
            lines.append("  " + "─" * 65)

            # 헤더
            header = (
                f"  {'종목코드':^10}{'종목명':^14}{'수량':>8}"
                f"{'현재가':>12}{'평가금액':>14}{'비중':>8}"
            )
            lines.append(header)
            lines.append("  " + "─" * 65)

            # 종목별 정보
            holding_details = []
            for ticker, qty in holdings.items():
                price = ticker_prices.get(ticker, 0)
                eval_amount = qty * price
                name = ticker_names.get(ticker, ticker)
                weight = (eval_amount / total_value * 100) if total_value > 0 else 0
                holding_details.append((ticker, name, qty, price, eval_amount, weight))

            # 평가금액 기준 정렬 (내림차순)
            holding_details.sort(key=lambda x: x[4], reverse=True)

            for ticker, name, qty, price, eval_amount, weight in holding_details:
                # 종목명이 너무 길면 자르기
                display_name = name[:7] if len(name) > 7 else name
                row = (
                    f"  {ticker:^10}{display_name:^14}{self._format_number(qty):>8}"
                    f"{self._format_number(price):>12}{self._format_number(eval_amount):>14}"
                    f"{weight:>7.1f}%"
                )
                lines.append(row)

            lines.append("")

        # ────────────────────────────────────────────────────
        # 푸터
        # ────────────────────────────────────────────────────
        lines.append("=" * 70)
        lines.append("")

        report = "\n".join(lines)
        logger.info(f"일일 리포트 생성 완료: {self.report_date}")
        return report

    # ------------------------------------------------------------------
    # 파일 저장
    # ------------------------------------------------------------------
    def save_report(self, content: str, output_path: str) -> None:
        """리포트를 파일로 저장한다.

        Args:
            content: 리포트 문자열
            output_path: 저장할 파일 경로
        """
        # 디렉토리 생성
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"일일 리포트 저장 완료: {output_path}")
