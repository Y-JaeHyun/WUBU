#!/usr/bin/env python3
"""필터 비교 시뮬레이션: spike_filter + value_trap_filter 효과 비교.

3/3 리밸런싱 시뮬레이션을 2가지 설정으로 실행하여 종목 차이를 비교한다:
1. Original: 기존 MultiFactor(V+M) - 필터 없음
2. Filtered: spike_filter + value_trap_filter 적용

Usage:
    python scripts/simulate_filter_comparison.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_comparison(
    capital: int = 2_000_000,
    output_path: str = "docs/filter_comparison_march3.md",
):
    """3/3 리밸런싱 필터 비교 시뮬레이션을 실행한다."""
    from src.data.collector import get_all_fundamentals, get_price_data
    from src.data.index_collector import get_index_data
    from src.strategy.strategy_config import create_multi_factor

    print(f"\n{'='*70}")
    print("  MultiFactor 필터 비교 시뮬레이션")
    print(f"  데이터 기준일: 2026-02-27 (T-1)")
    print(f"  자본금: {capital:,}원")
    print(f"{'='*70}\n")

    t0 = time.time()
    data_date = "20260227"

    # ── 1. 데이터 수집 (1회) ──
    print("[1/3] 데이터 수집 중...")
    try:
        fundamentals = get_all_fundamentals(data_date)
        print(f"  -> {len(fundamentals)}종목 펀더멘탈")
    except Exception as e:
        print(f"  -> 실패: {e}")
        fundamentals = pd.DataFrame()

    prices = {}
    if not fundamentals.empty and "ticker" in fundamentals.columns:
        top_tickers = (
            fundamentals[fundamentals["market_cap"] > 0]
            .sort_values("market_cap", ascending=False)
            .head(200)["ticker"]
            .tolist()
        )
        start_dt = datetime.strptime(data_date, "%Y%m%d") - pd.Timedelta(days=400)
        start_date = start_dt.strftime("%Y%m%d")

        import signal as _signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("pykrx request timeout")

        loaded = 0
        for ticker in top_tickers:
            try:
                _signal.signal(_signal.SIGALRM, _timeout_handler)
                _signal.alarm(15)
                df = get_price_data(ticker, start_date, data_date)
                _signal.alarm(0)
                if not df.empty:
                    prices[ticker] = df
                    loaded += 1
            except (TimeoutError, Exception):
                _signal.alarm(0)
        print(f"  -> {loaded}/{len(top_tickers)}종목 가격 로드")

    index_prices = pd.Series(dtype=float)
    try:
        start_dt = datetime.strptime(data_date, "%Y%m%d") - pd.Timedelta(days=400)
        index_df = get_index_data("KOSPI", start_dt.strftime("%Y%m%d"), data_date)
        if not index_df.empty and "close" in index_df.columns:
            index_prices = index_df["close"]
    except Exception:
        pass

    strategy_data = {
        "fundamentals": fundamentals,
        "prices": prices,
        "index_prices": index_prices,
    }

    # 종목명 매핑
    ticker_names = {}
    if not fundamentals.empty:
        ticker_names = dict(zip(fundamentals["ticker"], fundamentals["name"]))

    # ── 2. Original 시그널 ──
    print("[2/3] Original MultiFactor(V+M) 시그널 생성...")
    original_strategy = create_multi_factor("backtest", spike_filter=False)
    original_signals = original_strategy.generate_signals(data_date, strategy_data)
    print(f"  -> {len(original_signals)}종목 선정")

    # ── 3. Filtered 시그널 ──
    print("[3/3] Filtered MultiFactor(V+M) 시그널 생성...")
    filtered_strategy = create_multi_factor(
        "backtest",
        spike_filter=True,
        value_trap_filter=True,
        min_roe=0.0,
        min_f_score=1,
    )
    filtered_signals = filtered_strategy.generate_signals(data_date, strategy_data)
    print(f"  -> {len(filtered_signals)}종목 선정")

    # ── 4. 비교 분석 ──
    orig_set = set(original_signals.keys())
    filt_set = set(filtered_signals.keys())

    common = orig_set & filt_set
    removed = orig_set - filt_set
    added = filt_set - orig_set

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print("  비교 결과")
    print(f"{'='*70}")
    print(f"  공통 종목: {len(common)}개")
    print(f"  제외 종목: {len(removed)}개 (필터에 의해)")
    print(f"  대체 종목: {len(added)}개 (새로 진입)")
    print()

    for ticker in removed:
        name = ticker_names.get(ticker, ticker)
        # 급등 체크
        spike_info = ""
        if ticker in prices and not prices[ticker].empty:
            close = prices[ticker]["close"]
            if len(close) >= 2:
                ret_1d = float(close.iloc[-1] / close.iloc[-2] - 1)
                ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0.0
                spike_info = f" [1d={ret_1d:+.1%}, 5d={ret_5d:+.1%}]"
        # ROE 체크
        roe_info = ""
        if not fundamentals.empty:
            row = fundamentals[fundamentals["ticker"] == ticker]
            if not row.empty and "eps" in row.columns and "bps" in row.columns:
                eps = float(row["eps"].iloc[0])
                bps = float(row["bps"].iloc[0]) if "bps" in row.columns else 0
                if bps > 0:
                    roe = eps / bps
                    roe_info = f" [ROE={roe:.1%}]"
        print(f"  ❌ 제외: {name}({ticker}){spike_info}{roe_info}")

    for ticker in added:
        name = ticker_names.get(ticker, ticker)
        print(f"  ✅ 대체: {name}({ticker})")

    print(f"\n소요시간: {elapsed:.0f}초")

    # ── MD 저장 ──
    output = PROJECT_ROOT / output_path
    output.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# MultiFactor 필터 비교 (3/3 리밸런싱)",
        "",
        f"**실행일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**데이터 기준일**: 2026-02-27 (T-1)",
        "",
        "## 설정 비교",
        "",
        "| 항목 | Original | Filtered |",
        "|------|----------|----------|",
        "| 전략 | MultiFactor(V+M) | MultiFactor(V+M) |",
        "| spike_filter | OFF | ON (1d>15%, 5d>25%) |",
        "| value_trap_filter | OFF | ON (ROE>=0, F-Score>=1) |",
        "| num_stocks | 10 | 10 |",
        "",
        "## Original 종목",
        "",
        "| 순위 | 종목 | 코드 | 비중 |",
        "|------|------|------|------|",
    ]

    for i, (ticker, weight) in enumerate(
        sorted(original_signals.items(), key=lambda x: x[1], reverse=True), 1
    ):
        name = ticker_names.get(ticker, ticker)
        status = "✅" if ticker in common else "❌"
        md_lines.append(f"| {i} | {status} {name} | {ticker} | {weight:.2%} |")

    md_lines.extend([
        "",
        "## Filtered 종목",
        "",
        "| 순위 | 종목 | 코드 | 비중 | 상태 |",
        "|------|------|------|------|------|",
    ])

    for i, (ticker, weight) in enumerate(
        sorted(filtered_signals.items(), key=lambda x: x[1], reverse=True), 1
    ):
        name = ticker_names.get(ticker, ticker)
        status = "유지" if ticker in common else "신규"
        md_lines.append(f"| {i} | {name} | {ticker} | {weight:.2%} | {status} |")

    md_lines.extend([
        "",
        "## 제외된 종목 상세",
        "",
        "| 종목 | 코드 | 1일 수익률 | 5일 수익률 | ROE | 제외 사유 |",
        "|------|------|-----------|-----------|-----|----------|",
    ])

    for ticker in removed:
        name = ticker_names.get(ticker, ticker)
        ret_1d = ret_5d = roe = 0.0
        reason = []
        if ticker in prices and not prices[ticker].empty:
            close = prices[ticker]["close"]
            if len(close) >= 2:
                ret_1d = float(close.iloc[-1] / close.iloc[-2] - 1)
            if len(close) >= 6:
                ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1)
            if ret_1d > 0.15 or ret_5d > 0.25:
                reason.append("급등")
        if not fundamentals.empty:
            row = fundamentals[fundamentals["ticker"] == ticker]
            if not row.empty and "eps" in row.columns and "bps" in row.columns:
                eps = float(row["eps"].iloc[0])
                bps = float(row["bps"].iloc[0]) if "bps" in row.columns else 0
                if bps > 0:
                    roe = eps / bps
                    if roe < 0:
                        reason.append("ROE<0")
                if eps <= 0:
                    reason.append("F-Score<1")
        reason_str = ", ".join(reason) if reason else "복합"
        md_lines.append(
            f"| {name} | {ticker} | {ret_1d:+.1%} | {ret_5d:+.1%} | "
            f"{roe:.1%} | {reason_str} |"
        )

    md_lines.extend([
        "",
        f"---",
        f"*소요시간: {elapsed:.0f}초 | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    output.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n결과 저장: {output}")

    # JSON
    json_result = {
        "data_date": data_date,
        "original_signals": original_signals,
        "filtered_signals": filtered_signals,
        "common": list(common),
        "removed": list(removed),
        "added": list(added),
        "elapsed_sec": round(elapsed, 1),
    }
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(json_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON 저장: {json_path}")

    return json_result


if __name__ == "__main__":
    run_comparison()
