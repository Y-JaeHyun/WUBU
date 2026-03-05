#!/usr/bin/env python3
"""필터 비교 백테스트: spike_filter + value_trap_filter 효과 분석.

MultiFactor(V+M) 전략의 다양한 필터 조합을 3년 기간으로 백테스트한다.

Usage:
    python scripts/backtest_filter_comparison.py
"""

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

RESULT_DIR = PROJECT_ROOT / "data" / "comprehensive_backtest"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL = 2_000_000
START_DATE = "20230301"
END_DATE = "20260227"  # 마지막 영업일 (토요일 제외)


def build_filter_variants():
    """필터 조합별 전략 인스턴스를 생성한다."""
    from src.strategy.strategy_config import create_multi_factor

    return [
        ("1. baseline", create_multi_factor("backtest", spike_filter=False)),
        ("2. +spike(15/25)", create_multi_factor("backtest")),
        ("3. +vtrap(roe>=0)", create_multi_factor(
            "backtest", spike_filter=False, value_trap_filter=True, min_roe=0.0)),
        ("4. +vtrap(f1)", create_multi_factor(
            "backtest", spike_filter=False,
            value_trap_filter=True, min_roe=0.0, min_f_score=1)),
        ("5. +spike+vtrap", create_multi_factor(
            "backtest", value_trap_filter=True, min_roe=0.0)),
        ("6. +spike+vtrap(f1)", create_multi_factor(
            "backtest", value_trap_filter=True, min_roe=0.0, min_f_score=1)),
        ("7. +spike(10/20)", create_multi_factor(
            "backtest", spike_threshold_1d=0.10, spike_threshold_5d=0.20)),
        ("8. +spike(20/30)", create_multi_factor(
            "backtest", spike_threshold_1d=0.20, spike_threshold_5d=0.30)),
    ]


def run_single_backtest(strategy, start_date, end_date, capital):
    """단일 백테스트를 실행한다."""
    from src.backtest.engine import Backtest

    bt = Backtest(
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        rebalance_freq="monthly",
    )
    bt.run()
    return bt.get_results()


def main():
    print(f"\n{'='*70}")
    print("  MultiFactor 필터 비교 백테스트")
    print(f"  기간: {START_DATE} ~ {END_DATE} (3년)")
    print(f"  자본: {CAPITAL:,}원 | 리밸런싱: monthly")
    print(f"{'='*70}\n")

    variants = build_filter_variants()
    results = []
    total_start = time.time()

    for i, (label, strategy) in enumerate(variants, 1):
        print(f"[{i}/{len(variants)}] {label} ({strategy.name})")
        t0 = time.time()

        try:
            result = run_single_backtest(strategy, START_DATE, END_DATE, CAPITAL)
            elapsed = time.time() - t0

            total_return = result.get("total_return", 0.0)
            cagr = result.get("cagr", 0.0)
            sharpe = result.get("sharpe_ratio", 0.0)
            mdd = result.get("mdd", 0.0)

            results.append({
                "label": label,
                "strategy_name": strategy.name,
                "total_return": round(total_return, 2),
                "cagr": round(cagr, 2),
                "sharpe": round(sharpe, 4),
                "mdd": round(mdd, 2),
                "elapsed": round(elapsed, 1),
            })

            print(f"  -> 수익률={total_return:+.1f}%, CAGR={cagr:.1f}%, "
                  f"Sharpe={sharpe:.2f}, MDD={mdd:.1f}% ({elapsed:.0f}초)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> 실패: {e}")
            traceback.print_exc()
            results.append({
                "label": label,
                "strategy_name": strategy.name,
                "total_return": None,
                "cagr": None,
                "sharpe": None,
                "mdd": None,
                "elapsed": round(elapsed, 1),
                "error": str(e),
            })

    total_elapsed = time.time() - total_start

    # ── 결과 정렬 (Sharpe 내림차순) ──
    valid_results = [r for r in results if r.get("sharpe") is not None]
    valid_results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n{'='*70}")
    print("  결과 요약 (Sharpe 순)")
    print(f"{'='*70}")
    print(f"{'설정':<25s} {'수익률':>8s} {'CAGR':>8s} {'Sharpe':>8s} {'MDD':>8s}")
    print("-" * 60)
    for r in valid_results:
        print(f"{r['label']:<25s} {r['total_return']:>+7.1f}% {r['cagr']:>7.1f}% "
              f"{r['sharpe']:>7.2f} {r['mdd']:>7.1f}%")

    # ── 최적 설정 판단 ──
    if valid_results:
        best = valid_results[0]
        print(f"\n{'='*70}")
        print(f"  최적 설정: {best['label']}")
        print(f"  Sharpe={best['sharpe']:.2f}, 수익률={best['total_return']:+.1f}%, "
              f"MDD={best['mdd']:.1f}%")
        print(f"{'='*70}")

    # ── JSON 저장 ──
    json_path = RESULT_DIR / "filter_comparison.json"
    json_data = {
        "period": f"{START_DATE}-{END_DATE}",
        "capital": CAPITAL,
        "results": results,
        "best": valid_results[0] if valid_results else None,
        "total_elapsed_sec": round(total_elapsed, 1),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON 저장: {json_path}")

    # ── MD 리포트 ──
    md_path = PROJECT_ROOT / "docs" / "filter_comparison_results.md"
    md_lines = [
        "# MultiFactor 필터 비교 백테스트 결과",
        "",
        f"**기간**: {START_DATE} ~ {END_DATE} (3년)",
        f"**자본**: {CAPITAL:,}원 | **리밸런싱**: monthly",
        f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 결과 비교 (Sharpe 순)",
        "",
        "| # | 설정 | 수익률 | CAGR | Sharpe | MDD |",
        "|---|------|--------|------|--------|-----|",
    ]

    for i, r in enumerate(valid_results, 1):
        md_lines.append(
            f"| {i} | {r['label']} | {r['total_return']:+.1f}% | "
            f"{r['cagr']:.1f}% | **{r['sharpe']:.2f}** | {r['mdd']:.1f}% |"
        )

    if valid_results:
        best = valid_results[0]
        baseline = next((r for r in results if "baseline" in r["label"]), None)

        md_lines.extend([
            "",
            "## 분석",
            "",
            f"**최적 설정**: {best['label']}",
            f"- Sharpe: {best['sharpe']:.2f}",
            f"- 수익률: {best['total_return']:+.1f}%",
            f"- MDD: {best['mdd']:.1f}%",
        ])

        if baseline and baseline.get("sharpe") is not None:
            sharpe_diff = best["sharpe"] - baseline["sharpe"]
            mdd_diff = best["mdd"] - baseline["mdd"]
            md_lines.extend([
                "",
                f"**baseline 대비**:",
                f"- Sharpe 변화: {sharpe_diff:+.2f}",
                f"- MDD 변화: {mdd_diff:+.1f}%p",
            ])

    md_lines.extend([
        "",
        f"---",
        f"*총 소요시간: {total_elapsed:.0f}초*",
    ])

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"MD 저장: {md_path}")

    return json_data


if __name__ == "__main__":
    main()
