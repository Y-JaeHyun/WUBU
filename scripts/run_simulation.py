#!/usr/bin/env python3
"""2026-02-27 리밸런싱 시뮬레이션 스크립트.

전략별 시그널을 생성하고, 매수/매도 예상 리스트를 포함한 리뷰 문서를 생성한다.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.collector import get_all_fundamentals, get_price_data
from src.data.index_collector import get_index_data
from src.data.daily_simulator import DailySimulator
from src.strategy.multi_factor import MultiFactorStrategy
from src.strategy.etf_rotation import ETFRotationStrategy
from src.strategy.three_factor import ThreeFactorStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def collect_strategy_data(date_str: str) -> dict:
    """전략 데이터를 수집한다 (스케줄러의 _collect_strategy_data와 동일 구조)."""
    data = {
        "fundamentals": pd.DataFrame(),
        "prices": {},
        "index_prices": pd.Series(dtype=float),
    }

    # 1. Fundamentals
    logger.info("전 종목 기본 지표 수집 시작: %s", date_str)
    try:
        fundamentals = get_all_fundamentals(date_str)
        data["fundamentals"] = fundamentals
        logger.info("펀더멘탈 수집 완료: %d종목", len(fundamentals))
    except Exception as e:
        logger.error("펀더멘탈 수집 실패: %s", e)
        return data

    # 2. Price data (시총 상위 200종목)
    if not fundamentals.empty and "market_cap" in fundamentals.columns:
        df = fundamentals[fundamentals["market_cap"] > 0].copy()
        top_tickers = (
            df.sort_values("market_cap", ascending=False)
            .head(200)["ticker"]
            .tolist()
        )

        start_dt = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=400)
        start_date = start_dt.strftime("%Y%m%d")

        prices = {}
        total = len(top_tickers)
        for i, ticker in enumerate(top_tickers, 1):
            try:
                price_df = get_price_data(ticker, start_date, date_str)
                if not price_df.empty:
                    prices[ticker] = price_df
            except Exception:
                pass
            if i % 50 == 0:
                logger.info("가격 수집 진행: %d/%d", i, total)

        data["prices"] = prices
        logger.info("가격 데이터 수집 완료: %d/%d종목", len(prices), total)

    # 3. KOSPI 지수
    try:
        start_dt = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=400)
        index_df = get_index_data("KOSPI", start_dt.strftime("%Y%m%d"), date_str)
        if not index_df.empty and "close" in index_df.columns:
            data["index_prices"] = index_df["close"]
            logger.info("KOSPI 지수 수집 완료: %d일", len(index_df))
    except Exception as e:
        logger.warning("KOSPI 지수 수집 실패: %s", e)

    return data


def collect_etf_prices(date_str: str, lookback_days: int = 252) -> dict:
    """ETF 가격 데이터를 수집한다."""
    from src.strategy.etf_rotation import ETFRotationStrategy

    strategy = ETFRotationStrategy()
    start_dt = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=lookback_days + 30)
    start_date = start_dt.strftime("%Y%m%d")

    etf_prices = {}
    for ticker, name in strategy.etf_universe.items():
        try:
            price_df = get_price_data(ticker, start_date, date_str)
            if not price_df.empty:
                etf_prices[ticker] = price_df
                logger.info("ETF 수집: %s (%s) %d일", name, ticker, len(price_df))
        except Exception as e:
            logger.warning("ETF 수집 실패: %s - %s", ticker, e)

    return etf_prices


def generate_report(results: dict, strategy_data: dict, date: str) -> str:
    """시뮬레이션 결과를 리뷰 문서로 포맷한다."""
    lines = [
        f"# 리밸런싱 시뮬레이션 리뷰 — {date}",
        "",
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    fund = strategy_data.get("fundamentals", pd.DataFrame())
    if not fund.empty:
        lines.append(f"## 데이터 현황")
        lines.append(f"- 전체 종목 수: {len(fund)}")
        if "market_cap" in fund.columns:
            total_cap = fund["market_cap"].sum()
            lines.append(f"- 총 시가총액: {total_cap / 1e12:.1f}조원")
        prices = strategy_data.get("prices", {})
        lines.append(f"- 가격 데이터 종목 수: {len(prices)}")
        idx = strategy_data.get("index_prices", pd.Series())
        if not idx.empty:
            lines.append(f"- KOSPI 최근 종가: {idx.iloc[-1]:,.0f}")
        lines.append("")

    for name, result in results.items():
        lines.append(f"## 전략: {name}")
        lines.append("")

        selected = result.get("selected", [])
        turnover = result.get("turnover_vs_yesterday", 0)
        countdown = result.get("rebalancing_countdown", -1)

        if countdown >= 0:
            lines.append(f"- 리밸런싱까지: D-{countdown}")
        lines.append(f"- 선정 종목 수: {len(selected)}")
        lines.append(f"- 전일 대비 회전율: {turnover:.1%}")
        lines.append("")

        if selected:
            lines.append("### 선정 종목")
            lines.append("")
            lines.append("| 순위 | 종목명 | 코드 | 비중 | 변동 |")
            lines.append("|------|--------|------|------|------|")
            for item in selected:
                rank = item.get("rank", "?")
                tname = item.get("name", item.get("ticker", "?"))
                ticker = item.get("ticker", "")
                weight = item.get("weight", 0)
                change = item.get("change", "=")
                marks = {"NEW": "NEW", "UP": "^", "DOWN": "v", "=": "="}
                mark = marks.get(change, "")
                lines.append(
                    f"| {rank} | {tname} | {ticker} | {weight:.1%} | {mark} |"
                )
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*이 리포트는 자동 생성되었습니다.*")

    return "\n".join(lines)


def main():
    date_str = "20260227"
    date_formatted = "2026-02-27"

    print(f"=== 리밸런싱 시뮬레이션 시작: {date_formatted} ===\n")

    # 1. 데이터 수집
    print("[1/4] 전략 데이터 수집 중...")
    strategy_data = collect_strategy_data(date_str)

    fund = strategy_data["fundamentals"]
    if fund.empty:
        print("ERROR: 펀더멘탈 데이터를 수집할 수 없습니다.")
        print("       pykrx API가 정상 동작하는지 확인하세요.")
        # 빈 리포트라도 생성
        report = f"# 리밸런싱 시뮬레이션 리뷰 — {date_formatted}\n\n"
        report += "## 결과: 데이터 수집 실패\n\n"
        report += "pykrx API에서 펀더멘탈 데이터를 가져올 수 없었습니다.\n"
        report += "시장 휴장일이거나 네트워크 문제일 수 있습니다.\n"

        docs_dir = Path("/mnt/data/quant/docs")
        docs_dir.mkdir(exist_ok=True)
        output = docs_dir / f"review_{date_formatted}_simulation.md"
        output.write_text(report, encoding="utf-8")
        print(f"\n리포트 저장: {output}")
        return

    # 2. ETF 가격 수집
    print("[2/4] ETF 가격 수집 중...")
    etf_prices = collect_etf_prices(date_str)

    # 3. 전략 생성 및 시뮬레이션
    print("[3/4] 전략 시뮬레이션 실행 중...")

    strategies = {
        "multi_factor": MultiFactorStrategy(
            factors=["value", "momentum"],
            weights=[0.5, 0.5],
            combine_method="zscore",
            num_stocks=7,
            apply_market_timing=True,
            turnover_penalty=0.1,
            max_group_weight=0.25,
            max_stocks_per_conglomerate=2,
        ),
        "three_factor": ThreeFactorStrategy(num_stocks=10),
    }

    # ETF rotation 추가
    if etf_prices:
        strategies["etf_rotation"] = ETFRotationStrategy(
            lookback=252, num_etfs=3, max_same_sector=1,
        )

    simulator = DailySimulator(
        data_dir="data/simulation",
        strategies=strategies,
    )
    simulator.strategy_data = strategy_data
    if etf_prices:
        simulator.etf_prices = etf_prices

    results = simulator.run_daily_simulation(date_formatted)

    # 4. 리포트 생성
    print("[4/4] 리포트 생성 중...")
    report = generate_report(results, strategy_data, date_formatted)

    # Telegram 형식 리포트도 출력
    telegram_report = simulator.format_telegram_report(date_formatted)
    print("\n" + telegram_report)

    # 파일 저장
    docs_dir = Path("/mnt/data/quant/docs")
    docs_dir.mkdir(exist_ok=True)
    output = docs_dir / f"review_{date_formatted}_simulation.md"
    output.write_text(report, encoding="utf-8")
    print(f"\n리포트 저장: {output}")

    # JSON 결과도 저장
    json_path = docs_dir / f"review_{date_formatted}_simulation.json"
    # selected 리스트를 직렬화 가능하게 변환
    serializable = {}
    for name, res in results.items():
        serializable[name] = {
            "date": res.get("date"),
            "strategy": res.get("strategy"),
            "universe_size": res.get("universe_size"),
            "selected": res.get("selected", []),
            "turnover_vs_yesterday": res.get("turnover_vs_yesterday"),
            "rebalancing_countdown": res.get("rebalancing_countdown"),
        }
    json_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON 저장: {json_path}")

    print(f"\n=== 시뮬레이션 완료 ===")


if __name__ == "__main__":
    main()
