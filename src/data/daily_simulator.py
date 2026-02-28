"""일일 리밸런싱 시뮬레이션 모듈.

매일 장 마감 후 가상 포트폴리오 히스토리를 축적한다.
각 전략의 시그널을 계산하고, 선정 종목/스코어를 JSON으로 저장하며,
실제 보유와의 괴리(drift)를 분석한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DailySimulator:
    """매일 장 마감 후 가상 리밸런싱 결과를 기록하는 시뮬레이터.

    Args:
        data_dir: 시뮬레이션 데이터 저장 디렉토리.
        strategies: {전략이름: 전략인스턴스} 딕셔너리.
    """

    def __init__(
        self,
        data_dir: str = "data/simulation",
        strategies: Optional[dict[str, Any]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.strategies: dict[str, Any] = strategies or {}
        self.strategy_data: dict = {}
        self.etf_prices: dict = {}
        self.dry_run_results: dict[str, dict] = {}
        self.integrated_dry_run: Optional[dict] = None
        self.pool_allocation: Optional[dict[str, float]] = None
        self._ticker_names: dict[str, str] = {}

    def run_daily_simulation(self, date: Optional[str] = None) -> dict[str, Any]:
        """당일 기준 모든 전략의 시그널을 계산하고 저장한다.

        Args:
            date: 시뮬레이션 날짜 ('YYYY-MM-DD' 또는 'YYYYMMDD').
                None이면 오늘 날짜.

        Returns:
            {전략이름: 저장된 결과 딕셔너리} 매핑.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        date_str = date.replace("-", "")
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        if not self.strategies:
            logger.warning("등록된 전략이 없습니다.")
            return {}

        # 펀더멘탈에서 종목명 매핑 구축
        fund = self.strategy_data.get("fundamentals")
        if fund is not None and hasattr(fund, "empty") and not fund.empty:
            if "ticker" in fund.columns and "name" in fund.columns:
                self._ticker_names = dict(zip(fund["ticker"], fund["name"]))

        results: dict[str, Any] = {}

        for name, strategy in self.strategies.items():
            try:
                logger.info("시뮬레이션 시작: %s (%s)", name, date_formatted)

                signals = self._generate_signals(strategy, date_str)
                if not signals:
                    logger.warning("전략 '%s': 시그널 없음", name)
                    continue

                # 선정 결과 구성
                selected = self._build_selected(signals)
                factor_scores = self._extract_factor_scores(signals)

                # 전일 대비 변화 분석
                prev = self._load_selection(
                    self._prev_date(date_formatted), name
                )
                turnover = self._calc_turnover(selected, prev)
                change_info = self._mark_changes(selected, prev)

                result = {
                    "date": date_formatted,
                    "strategy": name,
                    "universe_size": len(signals),
                    "selected": change_info,
                    "factor_scores": factor_scores,
                    "turnover_vs_yesterday": round(turnover, 4),
                    "rebalancing_countdown": self.get_rebalancing_countdown(),
                }

                self.save_selection(date_formatted, name, result)
                results[name] = result

                logger.info(
                    "시뮬레이션 완료: %s (%d종목, turnover=%.2f%%)",
                    name,
                    len(selected),
                    turnover * 100,
                )

            except Exception as e:
                logger.error("전략 '%s' 시뮬레이션 실패: %s", name, e)

        return results

    def save_selection(
        self,
        date: str,
        strategy_name: str,
        data: dict[str, Any],
    ) -> None:
        """선정 결과를 JSON으로 저장한다.

        Args:
            date: 날짜 ('YYYY-MM-DD').
            strategy_name: 전략 이름.
            data: 저장할 데이터 딕셔너리.
        """
        date_dir = self.data_dir / date
        date_dir.mkdir(parents=True, exist_ok=True)
        path = date_dir / f"{strategy_name}.json"

        try:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("시뮬레이션 결과 저장: %s", path)
        except OSError as e:
            logger.error("시뮬레이션 결과 저장 실패: %s", e)

    def get_history(
        self,
        strategy_name: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """최근 N일간의 선정 히스토리를 조회한다.

        Args:
            strategy_name: 전략 이름.
            days: 조회할 일수.

        Returns:
            날짜 내림차순 정렬된 결과 리스트.
        """
        results: list[dict[str, Any]] = []

        date_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir()],
            reverse=True,
        )

        for date_dir in date_dirs[:days]:
            path = date_dir / f"{strategy_name}.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    results.append(data)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("히스토리 로드 실패 (%s): %s", path, e)

        return results

    def analyze_drift(
        self,
        actual_holdings: dict[str, float],
        strategy_name: Optional[str] = None,
        date: Optional[str] = None,
    ) -> dict[str, Any]:
        """실제 보유 vs 가상 포트폴리오 괴리율을 분석한다.

        Args:
            actual_holdings: {종목코드: 비중} 실제 보유 현황.
            strategy_name: 비교할 전략 이름. None이면 첫 번째 전략.
            date: 비교 날짜. None이면 가장 최근 시뮬레이션.

        Returns:
            괴리 분석 결과 딕셔너리.
        """
        if strategy_name is None:
            if self.strategies:
                strategy_name = next(iter(self.strategies))
            else:
                return {"drift_pct": 0.0, "details": [], "error": "전략 없음"}

        if date is None:
            history = self.get_history(strategy_name, days=1)
            if not history:
                return {"drift_pct": 0.0, "details": [], "error": "히스토리 없음"}
            sim_data = history[0]
        else:
            sim_data = self._load_selection(date, strategy_name)
            if sim_data is None:
                return {"drift_pct": 0.0, "details": [], "error": "데이터 없음"}

        # 가상 포트폴리오 비중 추출
        sim_weights: dict[str, float] = {}
        for item in sim_data.get("selected", []):
            ticker = item.get("ticker", "")
            weight = item.get("weight", 0.0)
            sim_weights[ticker] = weight

        # 괴리 계산
        all_tickers = set(actual_holdings.keys()) | set(sim_weights.keys())
        total_drift = 0.0
        details: list[dict[str, Any]] = []

        for ticker in all_tickers:
            actual_w = actual_holdings.get(ticker, 0.0)
            sim_w = sim_weights.get(ticker, 0.0)
            diff = abs(actual_w - sim_w)
            total_drift += diff

            if diff > 0.001:
                details.append({
                    "ticker": ticker,
                    "actual_weight": round(actual_w, 4),
                    "sim_weight": round(sim_w, 4),
                    "diff": round(diff, 4),
                })

        details.sort(key=lambda x: x["diff"], reverse=True)

        return {
            "drift_pct": round(total_drift / 2, 4),
            "details": details,
        }

    def format_telegram_report(self, date: Optional[str] = None) -> str:
        """텔레그램 발송용 시뮬레이션 결과를 포맷한다.

        dry_run_results가 있으면 현재 보유 대비 매수/매도 예상을 포함한다.

        Args:
            date: 리포트 날짜. None이면 가장 최근.

        Returns:
            포매팅된 텔레그램 메시지 문자열.
        """
        if date is None:
            date_dirs = sorted(
                [d for d in self.data_dir.iterdir() if d.is_dir()],
                reverse=True,
            )
            if not date_dirs:
                return "[일일 시뮬레이션] 데이터 없음"
            date = date_dirs[0].name

        countdown = self.get_rebalancing_countdown()

        lines = [
            f"[일일 시뮬레이션] {date}",
            "=" * 35,
        ]

        if countdown == 0:
            lines.append("** 오늘은 리밸런싱일입니다 **")
        elif countdown > 0:
            lines.append(f"리밸런싱까지 D-{countdown}")

        date_dir = self.data_dir / date
        if not date_dir.exists():
            lines.append("데이터 없음")
            return "\n".join(lines)

        json_files = sorted(date_dir.glob("*.json"))
        if not json_files:
            lines.append("시뮬레이션 결과 없음")
            return "\n".join(lines)

        for json_file in json_files:
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                strategy = data.get("strategy", json_file.stem)
                selected = data.get("selected", [])
                turnover = data.get("turnover_vs_yesterday", 0)

                lines.append(f"\n[전략: {strategy}]")
                lines.append("-" * 30)

                if selected:
                    lines.append(f"목표 종목 ({len(selected)}개):")
                    for item in selected[:10]:
                        rank = item.get("rank", "?")
                        name = item.get("name", item.get("ticker", "?"))
                        ticker = item.get("ticker", "")
                        weight = item.get("weight", 0)
                        change = item.get("change", "=")
                        change_mark = self._change_mark(change)
                        label = (
                            f"{name}({ticker})" if name != ticker else ticker
                        )
                        lines.append(
                            f"  {rank}. {label} {weight:.1%} {change_mark}"
                        )
                    if len(selected) > 10:
                        lines.append(f"  ... 외 {len(selected) - 10}개")
                else:
                    lines.append("  선정 종목 없음")

                lines.append(f"  Turnover: {turnover:.1%}")

                # dry_run 결과가 있으면 매수/매도 예상 표시
                dry = self.dry_run_results.get(strategy)
                if dry:
                    self._append_dry_run_lines(lines, dry)

            except (json.JSONDecodeError, OSError) as e:
                logger.warning("리포트 포맷 실패 (%s): %s", json_file, e)

        lines.append("")
        lines.append("=" * 35)
        return "\n".join(lines)

    def _append_dry_run_lines(
        self, lines: list[str], dry: dict[str, Any]
    ) -> None:
        """dry_run 결과를 텔레그램 메시지 라인에 추가한다."""
        sell_orders = dry.get("sell_orders", [])
        buy_orders = dry.get("buy_orders", [])
        total_sell = dry.get("total_sell_amount", 0)
        total_buy = dry.get("total_buy_amount", 0)

        if not sell_orders and not buy_orders:
            lines.append("  변경 없음 (현재 = 목표)")
            return

        lines.append("")
        if sell_orders:
            lines.append(f"  매도 예상: {len(sell_orders)}건 ({total_sell:,}원)")
            for order in sell_orders[:5]:
                ticker = order.get("ticker", "?")
                name = self._ticker_names.get(ticker, ticker)
                qty = order.get("qty", 0)
                amount = order.get("amount", 0)
                lines.append(f"    {name}({ticker}) {qty}주 ({amount:,}원)")
            if len(sell_orders) > 5:
                lines.append(f"    ... 외 {len(sell_orders) - 5}건")

        if buy_orders:
            lines.append(f"  매수 예상: {len(buy_orders)}건 ({total_buy:,}원)")
            for order in buy_orders[:5]:
                ticker = order.get("ticker", "?")
                name = self._ticker_names.get(ticker, ticker)
                qty = order.get("qty", 0)
                amount = order.get("amount", 0)
                lines.append(f"    {name}({ticker}) {qty}주 ({amount:,}원)")
            if len(buy_orders) > 5:
                lines.append(f"    ... 외 {len(buy_orders) - 5}건")

    def get_rebalancing_countdown(self) -> int:
        """다음 리밸런싱일까지 남은 영업일을 계산한다.

        Returns:
            남은 영업일 수. 계산 불가 시 -1.
        """
        try:
            from src.scheduler.holidays import KRXHolidays

            holidays = KRXHolidays()
            today = datetime.now().date()
            return holidays.days_to_next_rebalance(today, "monthly")
        except Exception as e:
            logger.debug("리밸런싱 카운트다운 계산 실패: %s", e)
            return -1

    def format_integrated_report(self, date: Optional[str] = None) -> str:
        """통합 포트폴리오 리포트를 포맷한다.

        기존 전략별 리포트에 통합 포트폴리오 섹션을 추가한다.
        풀 배분, 병합 매도/매수 예상을 포함한다.

        Args:
            date: 리포트 날짜. None이면 가장 최근.

        Returns:
            포매팅된 텔레그램 메시지 문자열.
        """
        # 기존 전략별 리포트
        base_report = self.format_telegram_report(date)

        # 통합 정보가 없으면 기존 리포트만 반환
        if not self.integrated_dry_run:
            return base_report

        dry = self.integrated_dry_run
        lines: list[str] = [base_report]
        lines.append("")
        lines.append("[통합 포트폴리오 (병합)]")
        lines.append("=" * 35)

        # 풀 배분 표시
        if self.pool_allocation:
            lines.append("풀 배분:")
            for pool_name, pct in self.pool_allocation.items():
                lines.append(f"  {pool_name}: {pct:.0%}")

        # 풀별 분해 정보
        pool_breakdown = dry.get("pool_breakdown", {})
        if pool_breakdown:
            lines.append("")
            lines.append("풀별 종목:")
            for pool_name, info in pool_breakdown.items():
                count = info.get("original_count", 0)
                total_w = info.get("total_weight", 0)
                lines.append(
                    f"  {pool_name}: {count}종목 (비중 {total_w:.1%})"
                )

        # 통합 매도/매수 예상
        sell_orders = dry.get("sell_orders", [])
        buy_orders = dry.get("buy_orders", [])
        total_sell = dry.get("total_sell_amount", 0)
        total_buy = dry.get("total_buy_amount", 0)
        portfolio_value = dry.get("portfolio_value", 0)

        lines.append("")
        if portfolio_value > 0:
            lines.append(f"포트폴리오: {portfolio_value:,}원")

        if not sell_orders and not buy_orders:
            lines.append("변경 없음 (현재 = 목표)")
        else:
            if sell_orders:
                lines.append(
                    f"매도 예상: {len(sell_orders)}건 ({total_sell:,}원)"
                )
                for order in sell_orders[:5]:
                    ticker = order.get("ticker", "?")
                    name = self._ticker_names.get(ticker, ticker)
                    qty = order.get("qty", 0)
                    amount = order.get("amount", 0)
                    lines.append(
                        f"  {name}({ticker}) {qty}주 ({amount:,}원)"
                    )
                if len(sell_orders) > 5:
                    lines.append(f"  ... 외 {len(sell_orders) - 5}건")

            if buy_orders:
                lines.append(
                    f"매수 예상: {len(buy_orders)}건 ({total_buy:,}원)"
                )
                for order in buy_orders[:5]:
                    ticker = order.get("ticker", "?")
                    name = self._ticker_names.get(ticker, ticker)
                    qty = order.get("qty", 0)
                    amount = order.get("amount", 0)
                    lines.append(
                        f"  {name}({ticker}) {qty}주 ({amount:,}원)"
                    )
                if len(buy_orders) > 5:
                    lines.append(f"  ... 외 {len(buy_orders) - 5}건")

        # 리스크 체크 표시
        risk = dry.get("risk_check", {})
        if not risk.get("passed", True):
            lines.append("")
            lines.append("리스크 경고:")
            for w in risk.get("warnings", []):
                lines.append(f"  - {w}")

        lines.append("")
        lines.append("=" * 35)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _generate_signals(self, strategy: Any, date_str: str) -> dict:
        """전략 시그널을 생성한다.

        strategy_data(fundamentals/prices/index_prices)를 기본으로 전달하고,
        ETF 전략인 경우 etf_prices를 추가 주입한다.
        """
        try:
            data = dict(self.strategy_data)
            if hasattr(strategy, "etf_universe") and self.etf_prices:
                data["etf_prices"] = self.etf_prices
            return strategy.generate_signals(date_str, data)
        except Exception as e:
            logger.warning("시그널 생성 실패: %s", e)
            return {}

    def _build_selected(self, signals: dict) -> list[dict[str, Any]]:
        """시그널 딕셔너리를 선정 리스트로 변환한다."""
        sorted_items = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        selected: list[dict[str, Any]] = []

        for rank, (ticker, weight) in enumerate(sorted_items, 1):
            name = self._ticker_names.get(ticker, ticker)
            selected.append({
                "ticker": ticker,
                "name": name,
                "weight": float(weight),
                "rank": rank,
                "score": float(weight),
            })

        return selected

    @staticmethod
    def _extract_factor_scores(signals: dict) -> dict[str, dict]:
        """시그널에서 팩터 스코어를 추출한다."""
        scores: dict[str, dict] = {}
        for ticker, weight in signals.items():
            scores[ticker] = {"composite": float(weight)}
        return scores

    def _load_selection(
        self,
        date: str,
        strategy_name: str,
    ) -> Optional[dict[str, Any]]:
        """저장된 선정 결과를 로드한다."""
        path = self.data_dir / date / f"{strategy_name}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _prev_date(date: str) -> str:
        """전일 날짜를 반환한다 (단순 -1일)."""
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            prev = dt - pd.Timedelta(days=1)
            return prev.strftime("%Y-%m-%d")
        except Exception:
            return ""

    @staticmethod
    def _calc_turnover(
        current: list[dict],
        prev_data: Optional[dict],
    ) -> float:
        """전일 대비 종목 변화율(turnover)을 계산한다."""
        if prev_data is None:
            return 1.0

        prev_selected = prev_data.get("selected", [])
        prev_tickers = {item.get("ticker") for item in prev_selected}
        curr_tickers = {item.get("ticker") for item in current}

        if not prev_tickers and not curr_tickers:
            return 0.0

        union = prev_tickers | curr_tickers
        if not union:
            return 0.0

        changed = len(prev_tickers.symmetric_difference(curr_tickers))
        return changed / len(union)

    @staticmethod
    def _mark_changes(
        current: list[dict],
        prev_data: Optional[dict],
    ) -> list[dict]:
        """각 종목에 변동 표시(NEW, =, UP, DOWN)를 추가한다."""
        if prev_data is None:
            for item in current:
                item["change"] = "NEW"
            return current

        prev_ranks: dict[str, int] = {}
        for item in prev_data.get("selected", []):
            prev_ranks[item.get("ticker", "")] = item.get("rank", 999)

        for item in current:
            ticker = item.get("ticker", "")
            if ticker not in prev_ranks:
                item["change"] = "NEW"
            else:
                prev_rank = prev_ranks[ticker]
                curr_rank = item.get("rank", 999)
                if curr_rank < prev_rank:
                    item["change"] = "UP"
                elif curr_rank > prev_rank:
                    item["change"] = "DOWN"
                else:
                    item["change"] = "="

        return current

    @staticmethod
    def _change_mark(change: str) -> str:
        """변동 표시를 이모지 없이 텍스트로 반환한다."""
        marks = {
            "NEW": "<- NEW",
            "UP": "^",
            "DOWN": "v",
            "=": "=",
        }
        return marks.get(change, "")
