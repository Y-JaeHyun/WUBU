#!/usr/bin/env python3
"""전체 전략 종합 백테스트 스크립트.

모든 장기/ETF 전략을 3/5/10년 기간별, 200만원 기준으로 백테스트한다.
추가로 풀 배분 비율 변화, feature flag 조합도 테스트한다.

사용법:
    python scripts/comprehensive_backtest.py [--mode all/long/etf/pool/overlay]
"""

import argparse
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
END_DATE = "20260228"  # 최근 거래일

PERIODS = {
    "3yr": "20230301",
}

# 상위 전략에 대해 추가 수행할 기간
EXTENDED_PERIODS = {
    "5yr": "20210301",
    "10yr": "20160301",
}


# ──────────────────────────────────────────────────────────
# 전략 빌더
# ──────────────────────────────────────────────────────────

def build_long_strategies():
    """장기 전략 인스턴스 리스트."""
    from src.strategy.value import ValueStrategy
    from src.strategy.momentum import MomentumStrategy
    from src.strategy.quality import QualityStrategy
    from src.strategy.multi_factor import MultiFactorStrategy
    from src.strategy.three_factor import ThreeFactorStrategy
    from src.strategy.risk_parity import RiskParityStrategy
    from src.strategy.shareholder_yield import ShareholderYieldStrategy
    from src.strategy.pead import PEADStrategy
    from src.strategy.low_vol_quality import LowVolQualityStrategy
    from src.strategy.accrual import AccrualStrategy

    return [
        ("Value(PBR)", ValueStrategy(factor="pbr", num_stocks=10, min_market_cap=0)),
        ("Value(Composite)", ValueStrategy(factor="composite", num_stocks=10, min_market_cap=0)),
        ("Momentum(Residual)", MomentumStrategy(
            lookback_months=[12], skip_month=True, num_stocks=10,
            residual=True, min_market_cap=0)),
        ("Quality", QualityStrategy(num_stocks=10, min_market_cap=0, strict_accrual=True)),
        ("MultiFactor(V+M)", MultiFactorStrategy(
            factors=["value", "momentum"], weights=[0.5, 0.5],
            combine_method="zscore", num_stocks=10, turnover_penalty=0.1)),
        ("ThreeFactor(V+M+Q)", ThreeFactorStrategy(
            num_stocks=10, min_market_cap=0,
            value_weight=0.33, momentum_weight=0.33, quality_weight=0.34)),
        ("ShareholderYield", ShareholderYieldStrategy(num_stocks=10, min_market_cap=0)),
        ("PEAD", PEADStrategy(num_stocks=10, min_market_cap=0)),
        ("LowVolQuality", LowVolQualityStrategy(num_stocks=10, min_market_cap=0)),
        ("Accrual", AccrualStrategy(num_stocks=10, min_market_cap=0)),
        ("RiskParity(MF)", RiskParityStrategy(
            stock_selector=MultiFactorStrategy(
                factors=["value", "momentum"], weights=[0.5, 0.5], num_stocks=10),
            max_weight=0.15)),
    ]


def build_etf_strategies():
    """ETF 전략 인스턴스 리스트."""
    from src.strategy.etf_rotation import ETFRotationStrategy
    from src.strategy.enhanced_etf_rotation import EnhancedETFRotationStrategy
    from src.strategy.cross_asset_momentum import CrossAssetMomentumStrategy

    return [
        ("ETFRotation(12M,top3)", ETFRotationStrategy(
            lookback=252, num_etfs=3)),
        ("ETFRotation(3M,top3)", ETFRotationStrategy(
            lookback=63, num_etfs=3)),
        ("EnhancedETF(default)", EnhancedETFRotationStrategy()),
        ("EnhancedETF(strong_riskoff)", EnhancedETFRotationStrategy(
            cash_ratio_risk_off=0.7)),
        ("EnhancedETF(no_trend)", EnhancedETFRotationStrategy(
            use_trend_filter=False)),
        ("CrossAsset(default)", CrossAssetMomentumStrategy()),
    ]


def build_overlay_combos():
    """오버레이 조합 리스트."""
    from src.strategy.market_timing import MarketTimingOverlay
    from src.strategy.drawdown_overlay import DrawdownOverlay
    from src.strategy.vol_targeting import VolTargetingOverlay

    return {
        "no_overlay": {},
        "market_timing": {
            "overlay": MarketTimingOverlay(
                ma_period=200, ma_type="SMA", switch_mode="gradual",
                reference_index="KOSPI"),
        },
        "drawdown_overlay": {
            "drawdown_overlay": DrawdownOverlay(
                thresholds=[(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)],
                recovery_buffer=0.02),
        },
        "vol_targeting": {
            "vol_targeting": VolTargetingOverlay(
                target_vol=0.15, lookback_days=20, use_downside_only=True),
        },
        "dd+mt": {
            "overlay": MarketTimingOverlay(
                ma_period=200, ma_type="SMA", switch_mode="gradual",
                reference_index="KOSPI"),
            "drawdown_overlay": DrawdownOverlay(
                thresholds=[(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)],
                recovery_buffer=0.02),
        },
    }


# ──────────────────────────────────────────────────────────
# 백테스트 러너
# ──────────────────────────────────────────────────────────

def run_single_backtest(strategy, start_date, end_date, capital, **kwargs):
    """단일 백테스트를 실행한다."""
    from src.backtest.engine import Backtest

    bt = Backtest(
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        rebalance_freq="monthly",
        **kwargs,
    )
    bt.run()
    return bt.get_results()


def run_pool_backtest(pool_strategies, start_date, end_date, capital, **kwargs):
    """3-Pool 백테스트를 실행한다."""
    from src.backtest.engine import Backtest
    from src.strategy.multi_factor import MultiFactorStrategy

    # pool_strategies에서 첫 번째 전략을 기본 전략으로 사용
    first_strat = list(pool_strategies.values())[0][0]

    bt = Backtest(
        strategy=first_strat,  # pool_strategies 사용 시 무시됨
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        rebalance_freq="monthly",
        pool_strategies=pool_strategies,
        **kwargs,
    )
    bt.run()
    return bt.get_results()


# ──────────────────────────────────────────────────────────
# Phase 1: 개별 장기 전략 백테스트
# ──────────────────────────────────────────────────────────

def phase1_individual_long(all_results):
    """Phase 1: 장기 전략 개별 백테스트 (3/5/10년)."""
    strategies = build_long_strategies()

    print(f"\n{'#'*70}")
    print(f"  Phase 1: 장기 전략 개별 백테스트 ({len(strategies)}종 x {len(PERIODS)}기간)")
    print(f"{'#'*70}")

    for period_name, start_date in PERIODS.items():
        print(f"\n  ── {period_name} ({start_date} ~ {END_DATE}) ──")

        for i, (name, strategy) in enumerate(strategies):
            print(f"    [{i+1}/{len(strategies)}] {name}...", end=" ", flush=True)
            t0 = time.time()

            try:
                res = run_single_backtest(strategy, start_date, END_DATE, CAPITAL)
                elapsed = time.time() - t0

                entry = {
                    "phase": "1_individual_long",
                    "period": period_name,
                    "strategy": name,
                    "start_date": start_date,
                    "end_date": END_DATE,
                    "capital": CAPITAL,
                    **{k: res[k] for k in [
                        "total_return", "cagr", "sharpe_ratio", "mdd",
                        "win_rate", "total_trades", "final_value",
                    ]},
                    "elapsed_sec": round(elapsed, 1),
                }
                all_results.append(entry)

                print(f"OK ({elapsed:.0f}s) Return={res['total_return']:.1f}% "
                      f"Sharpe={res['sharpe_ratio']:.2f} MDD={res['mdd']:.1f}%")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAIL ({elapsed:.0f}s): {e}")
                all_results.append({
                    "phase": "1_individual_long",
                    "period": period_name,
                    "strategy": name,
                    "start_date": start_date,
                    "end_date": END_DATE,
                    "capital": CAPITAL,
                    "error": str(e),
                })


# ──────────────────────────────────────────────────────────
# Phase 2: ETF 전략 개별 백테스트
# ──────────────────────────────────────────────────────────

def phase2_individual_etf(all_results):
    """Phase 2: ETF 전략 개별 백테스트."""
    strategies = build_etf_strategies()

    print(f"\n{'#'*70}")
    print(f"  Phase 2: ETF 전략 개별 백테스트 ({len(strategies)}종 x {len(PERIODS)}기간)")
    print(f"{'#'*70}")

    for period_name, start_date in PERIODS.items():
        print(f"\n  ── {period_name} ({start_date} ~ {END_DATE}) ──")

        for i, (name, strategy) in enumerate(strategies):
            print(f"    [{i+1}/{len(strategies)}] {name}...", end=" ", flush=True)
            t0 = time.time()

            try:
                res = run_single_backtest(strategy, start_date, END_DATE, CAPITAL)
                elapsed = time.time() - t0

                entry = {
                    "phase": "2_individual_etf",
                    "period": period_name,
                    "strategy": name,
                    "start_date": start_date,
                    "end_date": END_DATE,
                    "capital": CAPITAL,
                    **{k: res[k] for k in [
                        "total_return", "cagr", "sharpe_ratio", "mdd",
                        "win_rate", "total_trades", "final_value",
                    ]},
                    "elapsed_sec": round(elapsed, 1),
                }
                all_results.append(entry)

                print(f"OK ({elapsed:.0f}s) Return={res['total_return']:.1f}% "
                      f"Sharpe={res['sharpe_ratio']:.2f} MDD={res['mdd']:.1f}%")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAIL ({elapsed:.0f}s): {e}")
                all_results.append({
                    "phase": "2_individual_etf",
                    "period": period_name,
                    "strategy": name,
                    "start_date": start_date,
                    "end_date": END_DATE,
                    "capital": CAPITAL,
                    "error": str(e),
                })


# ──────────────────────────────────────────────────────────
# Phase 3: 풀 배분 비율별 백테스트
# ──────────────────────────────────────────────────────────

def phase3_pool_allocation(all_results):
    """Phase 3: 장기+ETF 배분 비율별 백테스트."""
    from src.strategy.multi_factor import MultiFactorStrategy
    from src.strategy.enhanced_etf_rotation import EnhancedETFRotationStrategy

    ratios = [
        (1.0, 0.0, "100/0"),
        (0.8, 0.2, "80/20"),
        (0.7, 0.3, "70/30"),
        (0.6, 0.4, "60/40"),
        (0.5, 0.5, "50/50"),
    ]

    print(f"\n{'#'*70}")
    print(f"  Phase 3: 풀 배분 비율별 백테스트 ({len(ratios)}비율 x {len(PERIODS)}기간)")
    print(f"{'#'*70}")

    long_strategy = MultiFactorStrategy(
        factors=["value", "momentum"], weights=[0.5, 0.5],
        combine_method="zscore", num_stocks=10, turnover_penalty=0.1)
    etf_strategy = EnhancedETFRotationStrategy(
        cash_ratio_risk_off=0.7)

    for period_name, start_date in PERIODS.items():
        print(f"\n  ── {period_name} ({start_date} ~ {END_DATE}) ──")

        for long_pct, etf_pct, label in ratios:
            print(f"    {label}...", end=" ", flush=True)
            t0 = time.time()

            try:
                if etf_pct == 0:
                    # 단일 전략 모드
                    res = run_single_backtest(
                        long_strategy, start_date, END_DATE, CAPITAL)
                else:
                    pool = {
                        "long_term": (long_strategy, long_pct),
                        "etf_rotation": (etf_strategy, etf_pct),
                    }
                    res = run_pool_backtest(pool, start_date, END_DATE, CAPITAL)

                elapsed = time.time() - t0

                entry = {
                    "phase": "3_pool_allocation",
                    "period": period_name,
                    "strategy": f"MF+EnhETF({label})",
                    "long_pct": long_pct,
                    "etf_pct": etf_pct,
                    "start_date": start_date,
                    "end_date": END_DATE,
                    "capital": CAPITAL,
                    **{k: res[k] for k in [
                        "total_return", "cagr", "sharpe_ratio", "mdd",
                        "win_rate", "total_trades", "final_value",
                    ]},
                    "elapsed_sec": round(elapsed, 1),
                }
                all_results.append(entry)

                print(f"OK ({elapsed:.0f}s) Return={res['total_return']:.1f}% "
                      f"Sharpe={res['sharpe_ratio']:.2f} MDD={res['mdd']:.1f}%")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAIL ({elapsed:.0f}s): {e}")
                all_results.append({
                    "phase": "3_pool_allocation",
                    "period": period_name,
                    "strategy": f"MF+EnhETF({label})",
                    "long_pct": long_pct,
                    "etf_pct": etf_pct,
                    "error": str(e),
                })


# ──────────────────────────────────────────────────────────
# Phase 4: 오버레이 조합별 백테스트
# ──────────────────────────────────────────────────────────

def phase4_overlay_combos(all_results):
    """Phase 4: 오버레이 조합별 백테스트."""
    from src.strategy.multi_factor import MultiFactorStrategy

    overlays = build_overlay_combos()
    strategy = MultiFactorStrategy(
        factors=["value", "momentum"], weights=[0.5, 0.5],
        combine_method="zscore", num_stocks=10, turnover_penalty=0.1)

    print(f"\n{'#'*70}")
    print(f"  Phase 4: 오버레이 조합별 백테스트 ({len(overlays)}조합 x {len(PERIODS)}기간)")
    print(f"{'#'*70}")

    for period_name, start_date in PERIODS.items():
        print(f"\n  ── {period_name} ({start_date} ~ {END_DATE}) ──")

        for overlay_name, overlay_kwargs in overlays.items():
            print(f"    {overlay_name}...", end=" ", flush=True)
            t0 = time.time()

            try:
                res = run_single_backtest(
                    strategy, start_date, END_DATE, CAPITAL, **overlay_kwargs)
                elapsed = time.time() - t0

                entry = {
                    "phase": "4_overlay_combo",
                    "period": period_name,
                    "strategy": f"MF(V+M)+{overlay_name}",
                    "overlay": overlay_name,
                    "start_date": start_date,
                    "end_date": END_DATE,
                    "capital": CAPITAL,
                    **{k: res[k] for k in [
                        "total_return", "cagr", "sharpe_ratio", "mdd",
                        "win_rate", "total_trades", "final_value",
                    ]},
                    "elapsed_sec": round(elapsed, 1),
                }
                all_results.append(entry)

                print(f"OK ({elapsed:.0f}s) Return={res['total_return']:.1f}% "
                      f"Sharpe={res['sharpe_ratio']:.2f} MDD={res['mdd']:.1f}%")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAIL ({elapsed:.0f}s): {e}")
                all_results.append({
                    "phase": "4_overlay_combo",
                    "period": period_name,
                    "strategy": f"MF(V+M)+{overlay_name}",
                    "overlay": overlay_name,
                    "error": str(e),
                })


# ──────────────────────────────────────────────────────────
# Phase 5: 풀 배분 + 오버레이 조합 (현재 설정 vs 최적)
# ──────────────────────────────────────────────────────────

def phase5_pool_plus_overlay(all_results):
    """Phase 5: 현재 설정(70/30 + drawdown) vs 주요 조합."""
    from src.strategy.multi_factor import MultiFactorStrategy
    from src.strategy.enhanced_etf_rotation import EnhancedETFRotationStrategy
    from src.strategy.drawdown_overlay import DrawdownOverlay
    from src.strategy.market_timing import MarketTimingOverlay

    long_strat = MultiFactorStrategy(
        factors=["value", "momentum"], weights=[0.5, 0.5],
        combine_method="zscore", num_stocks=10, turnover_penalty=0.1)
    etf_strat = EnhancedETFRotationStrategy(cash_ratio_risk_off=0.7)

    combos = [
        ("70/30+no_overlay", 0.7, 0.3, {}),
        ("70/30+drawdown", 0.7, 0.3, {
            "drawdown_overlay": DrawdownOverlay(
                thresholds=[(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)],
                recovery_buffer=0.02),
        }),
        ("70/30+MT", 0.7, 0.3, {
            "overlay": MarketTimingOverlay(
                ma_period=200, ma_type="SMA", switch_mode="gradual",
                reference_index="KOSPI"),
        }),
        ("70/30+dd+MT", 0.7, 0.3, {
            "overlay": MarketTimingOverlay(
                ma_period=200, ma_type="SMA", switch_mode="gradual",
                reference_index="KOSPI"),
            "drawdown_overlay": DrawdownOverlay(
                thresholds=[(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)],
                recovery_buffer=0.02),
        }),
        ("60/40+drawdown", 0.6, 0.4, {
            "drawdown_overlay": DrawdownOverlay(
                thresholds=[(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)],
                recovery_buffer=0.02),
        }),
        ("50/50+drawdown", 0.5, 0.5, {
            "drawdown_overlay": DrawdownOverlay(
                thresholds=[(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)],
                recovery_buffer=0.02),
        }),
    ]

    print(f"\n{'#'*70}")
    print(f"  Phase 5: 풀+오버레이 조합 백테스트 ({len(combos)}조합 x {len(PERIODS)}기간)")
    print(f"{'#'*70}")

    for period_name, start_date in PERIODS.items():
        print(f"\n  ── {period_name} ({start_date} ~ {END_DATE}) ──")

        for label, long_pct, etf_pct, overlay_kwargs in combos:
            print(f"    {label}...", end=" ", flush=True)
            t0 = time.time()

            try:
                pool = {
                    "long_term": (long_strat, long_pct),
                    "etf_rotation": (etf_strat, etf_pct),
                }
                res = run_pool_backtest(
                    pool, start_date, END_DATE, CAPITAL, **overlay_kwargs)
                elapsed = time.time() - t0

                entry = {
                    "phase": "5_pool_overlay",
                    "period": period_name,
                    "strategy": label,
                    "long_pct": long_pct,
                    "etf_pct": etf_pct,
                    "start_date": start_date,
                    "end_date": END_DATE,
                    "capital": CAPITAL,
                    **{k: res[k] for k in [
                        "total_return", "cagr", "sharpe_ratio", "mdd",
                        "win_rate", "total_trades", "final_value",
                    ]},
                    "elapsed_sec": round(elapsed, 1),
                }
                all_results.append(entry)

                print(f"OK ({elapsed:.0f}s) Return={res['total_return']:.1f}% "
                      f"Sharpe={res['sharpe_ratio']:.2f} MDD={res['mdd']:.1f}%")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAIL ({elapsed:.0f}s): {e}")
                traceback.print_exc()
                all_results.append({
                    "phase": "5_pool_overlay",
                    "period": period_name,
                    "strategy": label,
                    "error": str(e),
                })


# ──────────────────────────────────────────────────────────
# Phase 6: 상위 전략 확장 기간 테스트
# ──────────────────────────────────────────────────────────

def phase6_extended_periods(all_results):
    """Phase 6: 3yr 상위 전략을 5yr/10yr로 확장 테스트."""
    # 3yr 결과에서 Sharpe 상위 5개 전략 선별
    three_yr = [
        r for r in all_results
        if r.get("period") == "3yr" and "error" not in r
        and r.get("phase") in ("1_individual_long", "2_individual_etf")
    ]
    if not three_yr:
        print("  3yr 결과 없음 - Phase 6 스킵")
        return

    three_yr.sort(key=lambda x: x.get("sharpe_ratio", -999), reverse=True)
    top_strategies = three_yr[:5]
    top_names = {r["strategy"] for r in top_strategies}

    print(f"\n{'#'*70}")
    print(f"  Phase 6: 상위 {len(top_names)} 전략 확장 테스트 (5yr/10yr)")
    print(f"  선정: {', '.join(top_names)}")
    print(f"{'#'*70}")

    all_strategies = build_long_strategies() + [
        (n, s) for n, s in build_etf_strategies()
    ]

    for period_name, start_date in EXTENDED_PERIODS.items():
        print(f"\n  ── {period_name} ({start_date} ~ {END_DATE}) ──")

        for name, strategy in all_strategies:
            if name not in top_names:
                continue
            # build_long_strategies returns tuples of (name, strategy, opts)
            # build_etf_strategies returns tuples of (name, strategy)
            opts = {}
            if isinstance(strategy, tuple):
                # Shouldn't happen after the list comprehension above
                continue

            print(f"    {name}...", end=" ", flush=True)
            t0 = time.time()

            try:
                res = run_single_backtest(strategy, start_date, END_DATE, CAPITAL, **opts)
                elapsed = time.time() - t0

                phase_key = "1_individual_long"
                if any(name == n for n, _ in build_etf_strategies()):
                    phase_key = "2_individual_etf"

                entry = {
                    "phase": phase_key,
                    "period": period_name,
                    "strategy": name,
                    "start_date": start_date,
                    "end_date": END_DATE,
                    "capital": CAPITAL,
                    **{k: res[k] for k in [
                        "total_return", "cagr", "sharpe_ratio", "mdd",
                        "win_rate", "total_trades", "final_value",
                    ]},
                    "elapsed_sec": round(elapsed, 1),
                }
                all_results.append(entry)

                print(f"OK ({elapsed:.0f}s) Return={res['total_return']:.1f}% "
                      f"Sharpe={res['sharpe_ratio']:.2f} MDD={res['mdd']:.1f}%")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAIL ({elapsed:.0f}s): {e}")
                all_results.append({
                    "phase": phase_key if 'phase_key' in dir() else "6_extended",
                    "period": period_name,
                    "strategy": name,
                    "error": str(e),
                })

    # 풀+오버레이 조합도 확장
    print(f"\n  ── 풀+오버레이 확장 ──")
    _phase5_extended(all_results)


def _phase5_extended(all_results):
    """Phase 5 결과 중 상위 조합을 5yr/10yr로 확장."""
    from src.strategy.multi_factor import MultiFactorStrategy
    from src.strategy.enhanced_etf_rotation import EnhancedETFRotationStrategy
    from src.strategy.drawdown_overlay import DrawdownOverlay
    from src.strategy.market_timing import MarketTimingOverlay

    long_strat = MultiFactorStrategy(
        factors=["value", "momentum"], weights=[0.5, 0.5],
        combine_method="zscore", num_stocks=10, turnover_penalty=0.1)
    etf_strat = EnhancedETFRotationStrategy(cash_ratio_risk_off=0.7)

    # 핵심 조합만 확장
    combos = [
        ("70/30+no_overlay", 0.7, 0.3, {}),
        ("70/30+drawdown", 0.7, 0.3, {
            "drawdown_overlay": DrawdownOverlay(
                thresholds=[(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)],
                recovery_buffer=0.02),
        }),
    ]

    for period_name, start_date in EXTENDED_PERIODS.items():
        print(f"\n  ── {period_name} ({start_date} ~ {END_DATE}) ──")

        for label, long_pct, etf_pct, overlay_kwargs in combos:
            print(f"    {label}...", end=" ", flush=True)
            t0 = time.time()

            try:
                pool = {
                    "long_term": (long_strat, long_pct),
                    "etf_rotation": (etf_strat, etf_pct),
                }
                res = run_pool_backtest(
                    pool, start_date, END_DATE, CAPITAL, **overlay_kwargs)
                elapsed = time.time() - t0

                entry = {
                    "phase": "5_pool_overlay",
                    "period": period_name,
                    "strategy": label,
                    "long_pct": long_pct,
                    "etf_pct": etf_pct,
                    "start_date": start_date,
                    "end_date": END_DATE,
                    "capital": CAPITAL,
                    **{k: res[k] for k in [
                        "total_return", "cagr", "sharpe_ratio", "mdd",
                        "win_rate", "total_trades", "final_value",
                    ]},
                    "elapsed_sec": round(elapsed, 1),
                }
                all_results.append(entry)

                print(f"OK ({elapsed:.0f}s) Return={res['total_return']:.1f}% "
                      f"Sharpe={res['sharpe_ratio']:.2f} MDD={res['mdd']:.1f}%")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAIL ({elapsed:.0f}s): {e}")


# ──────────────────────────────────────────────────────────
# MD 리포트 생성
# ──────────────────────────────────────────────────────────

def generate_md_report(all_results, elapsed_total):
    """종합 백테스트 결과를 MD 파일로 생성한다."""
    output = PROJECT_ROOT / "docs" / "comprehensive_backtest_results.md"
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 종합 백테스트 결과",
        "",
        f"**실행일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**초기 자본금**: {CAPITAL:,}원",
        f"**종료일**: {END_DATE}",
        f"**총 소요시간**: {elapsed_total:.0f}초",
        f"**총 테스트 수**: {len(all_results)}개",
        "",
    ]

    # Phase별 분류
    phases = {
        "1_individual_long": "Phase 1: 장기 전략 개별",
        "2_individual_etf": "Phase 2: ETF 전략 개별",
        "3_pool_allocation": "Phase 3: 풀 배분 비율별",
        "4_overlay_combo": "Phase 4: 오버레이 조합별",
        "5_pool_overlay": "Phase 5: 풀+오버레이 조합",
    }

    # 모든 기간을 합산 (3yr + extended 5yr/10yr)
    all_periods = dict(PERIODS)
    all_periods.update(EXTENDED_PERIODS)

    for phase_key, phase_title in phases.items():
        phase_results = [r for r in all_results if r.get("phase") == phase_key]
        if not phase_results:
            continue

        lines.append(f"## {phase_title}")
        lines.append("")

        # 해당 phase에 존재하는 기간만
        available_periods = sorted(set(r.get("period") for r in phase_results if r.get("period")))

        for period_name in available_periods:
            period_results = [
                r for r in phase_results
                if r.get("period") == period_name and "error" not in r
            ]
            if not period_results:
                continue

            period_results.sort(key=lambda x: x.get("sharpe_ratio", -999), reverse=True)

            start_d = all_periods.get(period_name, "?")
            lines.append(f"### {period_name} ({start_d} ~ {END_DATE})")
            lines.append("")
            lines.append("| 순위 | 전략 | 수익률 | CAGR | Sharpe | MDD | 승률 | 최종자산 |")
            lines.append("|------|------|--------|------|--------|-----|------|----------|")

            for rank, r in enumerate(period_results, 1):
                lines.append(
                    f"| {rank} | {r['strategy']} | "
                    f"{r['total_return']:.1f}% | {r['cagr']:.1f}% | "
                    f"**{r['sharpe_ratio']:.2f}** | {r['mdd']:.1f}% | "
                    f"{r['win_rate']:.0f}% | {r['final_value']:,.0f}원 |"
                )

            lines.append("")

        # 에러 전략
        errors = [r for r in phase_results if "error" in r]
        if errors:
            lines.append(f"#### 실패 ({len(errors)}건)")
            for e in errors:
                lines.append(f"- {e.get('strategy', '?')} ({e.get('period', '?')}): {e['error']}")
            lines.append("")

    # ── 종합 분석 ──
    lines.append("## 종합 분석")
    lines.append("")

    # 가장 긴 기간 기준 최적 전략 (5yr > 3yr > 10yr)
    ref_period = None
    ref_results = []
    for p in ["5yr", "3yr", "10yr"]:
        candidates = [r for r in all_results if r.get("period") == p and "error" not in r]
        if candidates:
            ref_period = p
            ref_results = candidates
            break

    if ref_results:
        ref_results.sort(key=lambda x: x.get("sharpe_ratio", -999), reverse=True)
        best = ref_results[0]
        lines.append(f"### {ref_period} 기준 최고 Sharpe 전략")
        lines.append(f"- **{best['strategy']}** (Phase: {best['phase']})")
        lines.append(f"- Sharpe: {best['sharpe_ratio']:.2f}, CAGR: {best['cagr']:.1f}%, MDD: {best['mdd']:.1f}%")
        lines.append(f"- 최종 자산: {best['final_value']:,.0f}원 ({best['total_return']:.1f}%)")
        lines.append("")

        # 안정적 전략 (Sharpe > 0, MDD 최소)
        stable = [r for r in ref_results if r.get("sharpe_ratio", 0) > 0]
        if stable:
            least_mdd = min(stable, key=lambda x: abs(x.get("mdd", 0)))
            if least_mdd["strategy"] != best["strategy"]:
                lines.append(f"### {ref_period} 기준 가장 안정적 전략 (최소 MDD)")
                lines.append(f"- **{least_mdd['strategy']}** (Phase: {least_mdd['phase']})")
                lines.append(f"- Sharpe: {least_mdd['sharpe_ratio']:.2f}, MDD: {least_mdd['mdd']:.1f}%")
                lines.append("")

    # 현재 설정 대비 최적 비교
    current_setting = [
        r for r in all_results
        if r.get("strategy") == "70/30+drawdown" and r.get("period") == ref_period
        and "error" not in r
    ]
    if current_setting and ref_results:
        cur = current_setting[0]
        opt = ref_results[0]
        lines.append("### 현재 설정(70/30+drawdown) vs 최적 전략 비교")
        lines.append("")
        lines.append("| 항목 | 현재 설정 | 최적 전략 | 차이 |")
        lines.append("|------|----------|----------|------|")
        lines.append(f"| 전략 | 70/30+drawdown | {opt['strategy']} | - |")
        lines.append(f"| Sharpe | {cur['sharpe_ratio']:.2f} | {opt['sharpe_ratio']:.2f} | {opt['sharpe_ratio'] - cur['sharpe_ratio']:+.2f} |")
        lines.append(f"| CAGR | {cur['cagr']:.1f}% | {opt['cagr']:.1f}% | {opt['cagr'] - cur['cagr']:+.1f}%p |")
        lines.append(f"| MDD | {cur['mdd']:.1f}% | {opt['mdd']:.1f}% | {opt['mdd'] - cur['mdd']:+.1f}%p |")
        lines.append(f"| 최종자산 | {cur['final_value']:,.0f}원 | {opt['final_value']:,.0f}원 | {opt['final_value'] - cur['final_value']:+,.0f}원 |")
        lines.append("")

        if opt["strategy"] != "70/30+drawdown":
            lines.append(f"**결론**: 현재 설정과 최적 전략이 다릅니다. 최적 전략으로 재시뮬레이션이 필요합니다.")
        else:
            lines.append(f"**결론**: 현재 설정이 최적입니다.")

    lines.extend([
        "",
        "---",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMD 리포트 저장: {output}")
    return output


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="종합 백테스트")
    parser.add_argument("--mode", choices=["all", "long", "etf", "pool", "overlay", "combined"],
                        default="all", help="백테스트 모드")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  종합 백테스트 ({args.mode})")
    print(f"  자본금: {CAPITAL:,}원 | 종료: {END_DATE}")
    print(f"  기간: 3년/5년/10년")
    print(f"{'#'*70}")

    all_results = []
    total_start = time.time()

    if args.mode in ("all", "long"):
        phase1_individual_long(all_results)

    if args.mode in ("all", "etf"):
        phase2_individual_etf(all_results)

    if args.mode in ("all", "pool"):
        phase3_pool_allocation(all_results)

    if args.mode in ("all", "overlay"):
        phase4_overlay_combos(all_results)

    if args.mode in ("all", "combined"):
        phase5_pool_plus_overlay(all_results)

    # Phase 6: 상위 전략 5yr/10yr 확장 (3yr 결과 기반)
    if args.mode == "all" and EXTENDED_PERIODS:
        phase6_extended_periods(all_results)

    total_elapsed = time.time() - total_start

    # JSON 결과 저장
    json_output = {
        "config": {
            "capital": CAPITAL,
            "end_date": END_DATE,
            "periods": PERIODS,
            "mode": args.mode,
        },
        "results": all_results,
        "total_elapsed_sec": round(total_elapsed, 1),
    }

    json_path = RESULT_DIR / "comprehensive_results.json"
    json_path.write_text(
        json.dumps(json_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON 결과 저장: {json_path}")

    # MD 리포트 생성
    generate_md_report(all_results, total_elapsed)

    print(f"\n총 소요시간: {total_elapsed:.0f}초")
    print(f"총 테스트: {len(all_results)}개 "
          f"(성공: {len([r for r in all_results if 'error' not in r])}, "
          f"실패: {len([r for r in all_results if 'error' in r])})")


if __name__ == "__main__":
    main()
