"""백테스트 리포트 생성 모듈.

Backtest 객체의 결과를 텍스트 리포트와 차트로 시각화한다.
다중 전략 비교 기능을 포함한다.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestReport:
    """백테스트 리포트 생성기.

    Backtest 객체 하나 또는 여러 개를 받아 성과 리포트를 생성한다.

    Args:
        backtest: Backtest 객체 (단일 또는 리스트)
    """

    def __init__(self, backtest: Union[object, list[object]]) -> None:
        if isinstance(backtest, list):
            self.backtests = backtest
        else:
            self.backtests = [backtest]

        self._primary = self.backtests[0]

    # ------------------------------------------------------------------
    # 유틸리티: 순수 문자열 테이블 포매팅
    # ------------------------------------------------------------------
    @staticmethod
    def _format_table(headers: list[str], rows: list[list[str]], align: Optional[list[str]] = None) -> str:
        """순수 문자열로 테이블을 생성한다.

        Args:
            headers: 헤더 리스트
            rows: 데이터 행 리스트 (각 행은 문자열 리스트)
            align: 각 컬럼의 정렬 방향 ('l', 'r', 'c'). None이면 전부 오른쪽 정렬

        Returns:
            포매팅된 테이블 문자열
        """
        if not rows:
            return ""

        num_cols = len(headers)
        if align is None:
            align = ["r"] * num_cols

        # 컬럼 너비 결정
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # 패딩 추가
        col_widths = [w + 2 for w in col_widths]

        def _align_cell(text: str, width: int, direction: str) -> str:
            if direction == "l":
                return text.ljust(width)
            elif direction == "c":
                return text.center(width)
            else:
                return text.rjust(width)

        # 헤더
        separator = "+" + "+".join("-" * w for w in col_widths) + "+"
        header_line = "|" + "|".join(
            _align_cell(h, col_widths[i], "c") for i, h in enumerate(headers)
        ) + "|"

        lines = [separator, header_line, separator]

        # 데이터 행
        for row in rows:
            row_line = "|" + "|".join(
                _align_cell(row[i] if i < len(row) else "", col_widths[i], align[i])
                for i in range(num_cols)
            ) + "|"
            lines.append(row_line)

        lines.append(separator)
        return "\n".join(lines)

    @staticmethod
    def _format_number(value: float, fmt: str = ",.0f") -> str:
        """숫자를 포매팅한다."""
        try:
            return format(value, fmt)
        except (ValueError, TypeError):
            return str(value)

    # ------------------------------------------------------------------
    # 연도별 수익률 계산
    # ------------------------------------------------------------------
    def _compute_annual_returns(self, history: pd.DataFrame) -> pd.DataFrame:
        """포트폴리오 히스토리에서 연도별 수익률을 계산한다.

        Args:
            history: get_portfolio_history() 결과

        Returns:
            DataFrame with columns: ['year', 'return_pct']
        """
        if "portfolio_value" not in history.columns:
            return pd.DataFrame(columns=["year", "return_pct"])

        values = history["portfolio_value"]
        annual_data = []

        years = sorted(values.index.year.unique())
        for year in years:
            year_values = values[values.index.year == year]
            if len(year_values) < 2:
                continue

            # 해당 연도 첫날과 마지막날 기준 수익률
            start_val = year_values.iloc[0]
            end_val = year_values.iloc[-1]
            if start_val > 0:
                ret = (end_val / start_val - 1) * 100
                annual_data.append({"year": year, "return_pct": round(ret, 2)})

        return pd.DataFrame(annual_data)

    # ------------------------------------------------------------------
    # 텍스트 리포트 생성
    # ------------------------------------------------------------------
    def generate_text_report(self) -> str:
        """터미널 출력용 텍스트 리포트를 생성한다.

        포함 내용:
            - 성과 지표 테이블
            - 연도별 수익률
            - 최종 포트폴리오 현황

        Returns:
            포매팅된 리포트 문자열
        """
        lines: list[str] = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append("=" * 70)
        lines.append("  BACKTEST REPORT")
        lines.append(f"  생성일시: {now_str}")
        lines.append("=" * 70)
        lines.append("")

        for bt in self.backtests:
            results = bt.get_results()
            history = bt.get_portfolio_history()
            trades = bt.get_trades()

            strategy_name = results.get("strategy_name", "Unknown")

            # --- 전략 요약 ---
            lines.append(f"  전략: {strategy_name}")
            lines.append(f"  기간: {results['start_date']} ~ {results['end_date']}")
            lines.append("-" * 70)
            lines.append("")

            # --- 성과 지표 테이블 ---
            lines.append("  [ 성과 지표 ]")
            metrics_headers = ["지표", "값"]
            metrics_rows = [
                ["초기 자본금", f"{self._format_number(results['initial_capital'])}원"],
                ["최종 가치", f"{self._format_number(results['final_value'])}원"],
                ["총 수익률", f"{results['total_return']:.2f}%"],
                ["연평균 수익률 (CAGR)", f"{results['cagr']:.2f}%"],
                ["샤프 비율", f"{results['sharpe_ratio']:.2f}"],
                ["최대 낙폭 (MDD)", f"{results['mdd']:.2f}%"],
                ["승률", f"{results['win_rate']:.1f}%"],
                ["총 거래 횟수", f"{results['total_trades']}건"],
                ["리밸런싱 횟수", f"{results['rebalance_count']}회"],
            ]
            lines.append(self._format_table(metrics_headers, metrics_rows, align=["l", "r"]))
            lines.append("")

            # --- 연도별 수익률 ---
            annual_returns = self._compute_annual_returns(history)
            if not annual_returns.empty:
                lines.append("  [ 연도별 수익률 ]")
                annual_headers = ["연도", "수익률"]
                annual_rows = [
                    [str(int(row["year"])), f"{row['return_pct']:.2f}%"]
                    for _, row in annual_returns.iterrows()
                ]
                lines.append(self._format_table(annual_headers, annual_rows, align=["c", "r"]))
                lines.append("")

            # --- 최종 포트폴리오 현황 ---
            if not history.empty:
                last = history.iloc[-1]
                lines.append("  [ 최종 포트폴리오 현황 ]")
                portfolio_headers = ["항목", "값"]
                portfolio_rows = [
                    ["포트폴리오 가치", f"{self._format_number(last['portfolio_value'])}원"],
                    ["현금", f"{self._format_number(last['cash'])}원"],
                    ["보유 종목 수", f"{int(last['num_holdings'])}개"],
                    [
                        "현금 비중",
                        f"{last['cash'] / last['portfolio_value'] * 100:.1f}%"
                        if last["portfolio_value"] > 0
                        else "N/A",
                    ],
                ]
                lines.append(self._format_table(portfolio_headers, portfolio_rows, align=["l", "r"]))
                lines.append("")

            # --- 최근 거래 내역 (최대 10건) ---
            if not trades.empty:
                lines.append("  [ 최근 거래 내역 (최대 10건) ]")
                trade_headers = ["날짜", "종목", "매매", "수량", "단가", "거래비용"]
                recent_trades = trades.tail(10)
                trade_rows = []
                for _, t in recent_trades.iterrows():
                    trade_rows.append([
                        t["date"].strftime("%Y-%m-%d") if hasattr(t["date"], "strftime") else str(t["date"]),
                        str(t["ticker"]),
                        "매수" if t["action"] == "buy" else "매도",
                        self._format_number(t["quantity"]),
                        f"{self._format_number(t['price'])}원",
                        f"{self._format_number(t['cost'])}원",
                    ])
                lines.append(self._format_table(
                    trade_headers, trade_rows,
                    align=["c", "c", "c", "r", "r", "r"],
                ))
                lines.append("")

            lines.append("=" * 70)
            lines.append("")

        report = "\n".join(lines)
        logger.info(f"텍스트 리포트 생성 완료 ({len(self.backtests)}개 전략)")
        return report

    # ------------------------------------------------------------------
    # 차트 리포트 생성
    # ------------------------------------------------------------------
    def generate_chart_report(self, output_dir: str) -> None:
        """차트들을 디렉토리에 저장한다.

        생성되는 파일:
            - cumulative_returns.png
            - drawdown.png
            - monthly_heatmap.png
            - annual_returns.png

        Args:
            output_dir: 차트 저장 디렉토리 경로
        """
        from src.report.charts import (
            plot_annual_returns_bar,
            plot_cumulative_returns,
            plot_drawdown,
            plot_monthly_returns_heatmap,
        )

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"차트 리포트 생성 시작: {output_dir}")

        # 다중 전략인 경우 모든 포트폴리오 가치를 하나의 DataFrame으로 결합
        if len(self.backtests) > 1:
            combined = pd.DataFrame()
            for bt in self.backtests:
                history = bt.get_portfolio_history()
                results = bt.get_results()
                name = results.get("strategy_name", f"Strategy")
                combined[name] = history["portfolio_value"]
            history_for_chart = combined
        else:
            history_for_chart = self._primary.get_portfolio_history()

        # 1. 누적 수익률 차트
        try:
            strategy_names = ", ".join(
                bt.get_results().get("strategy_name", "?") for bt in self.backtests
            )
            plot_cumulative_returns(
                history_for_chart,
                title=f"누적 수익률 — {strategy_names}",
                save_path=os.path.join(output_dir, "cumulative_returns.png"),
            )
        except Exception as e:
            logger.error(f"누적 수익률 차트 생성 실패: {e}")

        # 2. 드로다운 차트 (주 전략 기준)
        try:
            primary_history = self._primary.get_portfolio_history()
            plot_drawdown(
                primary_history,
                save_path=os.path.join(output_dir, "drawdown.png"),
            )
        except Exception as e:
            logger.error(f"드로다운 차트 생성 실패: {e}")

        # 3. 월별 히트맵 (주 전략 기준)
        try:
            primary_history = self._primary.get_portfolio_history()
            daily_returns = primary_history["portfolio_value"].pct_change().dropna()
            if len(daily_returns) > 0:
                plot_monthly_returns_heatmap(
                    daily_returns,
                    save_path=os.path.join(output_dir, "monthly_heatmap.png"),
                )
        except Exception as e:
            logger.error(f"월별 히트맵 생성 실패: {e}")

        # 4. 연도별 수익률 바 차트 (주 전략 기준)
        try:
            primary_history = self._primary.get_portfolio_history()
            daily_returns = primary_history["portfolio_value"].pct_change().dropna()
            if len(daily_returns) > 0:
                plot_annual_returns_bar(
                    daily_returns,
                    save_path=os.path.join(output_dir, "annual_returns.png"),
                )
        except Exception as e:
            logger.error(f"연도별 수익률 차트 생성 실패: {e}")

        logger.info(f"차트 리포트 생성 완료: {output_dir}")

    # ------------------------------------------------------------------
    # 다중 전략 비교
    # ------------------------------------------------------------------
    def compare_strategies(self, backtests: Optional[list[object]] = None) -> str:
        """여러 전략의 성과를 비교하는 텍스트 리포트를 생성한다.

        Args:
            backtests: 비교할 Backtest 객체 리스트.
                       None이면 생성자에 전달된 리스트를 사용한다.

        Returns:
            비교 리포트 문자열
        """
        targets = backtests if backtests is not None else self.backtests

        if len(targets) < 2:
            return "비교를 위해서는 2개 이상의 전략이 필요합니다."

        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("  STRATEGY COMPARISON REPORT")
        lines.append(f"  생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        # 비교 테이블 구성
        metrics_names = [
            ("총 수익률", "total_return", ".2f", "%"),
            ("연평균 수익률 (CAGR)", "cagr", ".2f", "%"),
            ("샤프 비율", "sharpe_ratio", ".2f", ""),
            ("최대 낙폭 (MDD)", "mdd", ".2f", "%"),
            ("승률", "win_rate", ".1f", "%"),
            ("총 거래 횟수", "total_trades", ",.0f", "건"),
            ("초기 자본금", "initial_capital", ",.0f", "원"),
            ("최종 가치", "final_value", ",.0f", "원"),
        ]

        # 헤더: 지표 + 각 전략명
        all_results = [bt.get_results() for bt in targets]
        strategy_names = [r.get("strategy_name", f"전략{i+1}") for i, r in enumerate(all_results)]
        headers = ["지표"] + strategy_names

        rows = []
        for display_name, key, fmt, suffix in metrics_names:
            row = [display_name]
            for r in all_results:
                val = r.get(key, 0)
                try:
                    row.append(f"{format(val, fmt)}{suffix}")
                except (ValueError, TypeError):
                    row.append(str(val))
            rows.append(row)

        align = ["l"] + ["r"] * len(strategy_names)
        lines.append(self._format_table(headers, rows, align=align))
        lines.append("")

        # --- 승자 요약 ---
        lines.append("  [ 요약 ]")
        # CAGR 최고
        best_cagr_idx = max(range(len(all_results)), key=lambda i: all_results[i].get("cagr", 0))
        lines.append(f"  - CAGR 최고: {strategy_names[best_cagr_idx]} ({all_results[best_cagr_idx]['cagr']:.2f}%)")

        # 샤프 최고
        best_sharpe_idx = max(range(len(all_results)), key=lambda i: all_results[i].get("sharpe_ratio", 0))
        lines.append(
            f"  - 샤프 비율 최고: {strategy_names[best_sharpe_idx]} "
            f"({all_results[best_sharpe_idx]['sharpe_ratio']:.2f})"
        )

        # MDD 최소 (절댓값 기준 — MDD는 음수이므로 max가 가장 낮은 낙폭)
        best_mdd_idx = max(range(len(all_results)), key=lambda i: all_results[i].get("mdd", -999))
        lines.append(
            f"  - MDD 최소: {strategy_names[best_mdd_idx]} ({all_results[best_mdd_idx]['mdd']:.2f}%)"
        )
        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)
        logger.info(f"전략 비교 리포트 생성 완료 ({len(targets)}개 전략)")
        return report

    # ------------------------------------------------------------------
    # HTML 리포트 생성
    # ------------------------------------------------------------------
    def generate_html_report(
        self,
        output_path: str,
        chart_dir: Optional[str] = None,
    ) -> str:
        """Jinja2 기반 HTML 백테스트 리포트를 생성한다.

        Args:
            output_path: HTML 파일 저장 경로
            chart_dir: 차트 이미지 디렉토리. None이면 차트 생략.

        Returns:
            생성된 HTML 문자열
        """
        from jinja2 import Environment, FileSystemLoader

        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
        env.filters["commafy"] = lambda v: f"{v:,.0f}"
        template = env.get_template("backtest.html")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 전략별 데이터 구성
        strategies_data: list[dict] = []
        for bt in self.backtests:
            r = bt.get_results()
            history = bt.get_portfolio_history()
            annual = self._compute_annual_returns(history)

            annual_list = [
                {"year": int(row["year"]), "return_pct": round(row["return_pct"], 2)}
                for _, row in annual.iterrows()
            ] if not annual.empty else []

            strategies_data.append({
                "name": r.get("strategy_name", "Unknown"),
                "total_return": r.get("total_return", 0),
                "cagr": r.get("cagr", 0),
                "sharpe_ratio": r.get("sharpe_ratio", 0),
                "mdd": r.get("mdd", 0),
                "win_rate": r.get("win_rate", 0),
                "total_trades": r.get("total_trades", 0),
                "initial_capital": r.get("initial_capital", 0),
                "final_value": r.get("final_value", 0),
                "start_date": r.get("start_date", ""),
                "end_date": r.get("end_date", ""),
                "rebalance_count": r.get("rebalance_count", 0),
                "annual_returns": annual_list,
            })

        first = strategies_data[0] if strategies_data else {}
        period = f"{first.get('start_date', '')} ~ {first.get('end_date', '')}"

        # 차트 정보
        charts_data: list[dict] = []
        if chart_dir and os.path.isdir(chart_dir):
            chart_files = [
                ("cumulative_returns.png", "Cumulative Returns"),
                ("drawdown.png", "Drawdown"),
                ("monthly_heatmap.png", "Monthly Returns Heatmap"),
                ("annual_returns.png", "Annual Returns"),
            ]
            for fname, chart_title in chart_files:
                fpath = os.path.join(chart_dir, fname)
                if os.path.isfile(fpath):
                    charts_data.append({"path": fpath, "title": chart_title})

        # 비교 데이터 (2개 이상 전략)
        comparison_data = None
        if len(strategies_data) >= 2:
            names = [s["name"] for s in strategies_data]
            metrics_defs = [
                ("Total Return", "total_return", ".2f", "%", True),
                ("CAGR", "cagr", ".2f", "%", True),
                ("Sharpe Ratio", "sharpe_ratio", ".2f", "", True),
                ("MDD", "mdd", ".2f", "%", False),
                ("Win Rate", "win_rate", ".1f", "%", True),
            ]
            comp_rows = []
            for label, key, fmt, suffix, higher_better in metrics_defs:
                vals = [s[key] for s in strategies_data]
                formatted = [f"{format(v, fmt)}{suffix}" for v in vals]
                if higher_better:
                    best_idx = vals.index(max(vals))
                else:
                    best_idx = vals.index(max(vals))  # MDD: max = least negative
                comp_rows.append({
                    "metric": label,
                    "values": formatted,
                    "best_idx": best_idx,
                })
            comparison_data = {"names": names, "rows": comp_rows}

        html = template.render(
            title="Backtest Report",
            generated_at=now_str,
            period=period,
            strategies=strategies_data,
            charts=charts_data,
            comparison=comparison_data,
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML 리포트 생성 완료: {output_path}")
        return html
