"""Phase 7 백테스트 스크립트 — Sharpe 개선 검증.

8개 전략을 비교하며, Phase 6 Baseline 대비 차등 리밸런싱,
대형주 집중, 하이브리드 전략 등의 효과를 측정한다.

전략 목록:
  1. Baseline (현행 최고)  — 4F+Full+Regime, Sharpe 0.28 재현
  2. No SectorNeutral      — Regime + holding_bonus=0.1, SN 제거
  3. Large-cap 30          — min_cap=300B, 30종목
  4. Large-cap 50          — min_cap=500B, 50종목
  5. Smart Rebalance       — 차등 리밸런싱 + threshold=0.005
  6. Hybrid 75/25          — 코어 75% + ETF 헤지 25%
  7. Hybrid 70/30          — 코어 70% + ETF 헤지 30%
  8. Full Optimized        — Large-cap + Regime + Hybrid + SmartRebal

사용법:
  python scripts/run_sharpe_v2_backtest.py
"""
import sys
import os
import time

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.backtest.engine import Backtest
from src.backtest.walk_forward import WalkForwardBacktest
from src.strategy.three_factor import ThreeFactorStrategy
from src.strategy.hybrid_strategy import HybridStrategy
from src.strategy.market_timing import MarketTimingOverlay
from src.strategy.drawdown_overlay import DrawdownOverlay
from src.strategy.vol_targeting import VolTargetingOverlay
from src.ml.regime_model import RuleBasedRegimeModel

# ── 설정 ──────────────────────────────────────────────
START_DATE = "20160101"
END_DATE = "20251231"
INITIAL_CAPITAL = 100_000_000  # 1억원
REBALANCE_FREQ = "monthly"

# Walk-Forward 설정
WF_TRAIN_YEARS = 5
WF_TEST_YEARS = 1
WF_STEP_MONTHS = 12


# ── 공통 오버레이 생성 ─────────────────────────────────

def make_overlays():
    """공통 오버레이 kwargs를 반환한다."""
    return {
        "overlay": MarketTimingOverlay(
            ma_period=200, ma_type="SMA",
            switch_mode="gradual", reference_index="KOSPI",
        ),
        "drawdown_overlay": DrawdownOverlay(
            thresholds=[(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)],
            recovery_buffer=0.02,
        ),
        "vol_targeting": VolTargetingOverlay(
            target_vol=0.15, lookback_days=20, use_downside_only=True,
        ),
    }


def make_mt_overlay():
    """전략 내부용 MarketTimingOverlay를 반환한다."""
    return MarketTimingOverlay(
        ma_period=200, ma_type="SMA",
        switch_mode="gradual", reference_index="KOSPI",
    )


def make_regime_model():
    """RuleBasedRegimeModel을 반환한다."""
    return RuleBasedRegimeModel(
        factor_names=["value", "momentum", "quality", "low_vol"],
    )


# ── 전략 빌더 ─────────────────────────────────────────

def build_strategies():
    """8개 전략과 백테스트 kwargs를 생성하여 반환한다.

    Returns:
        list of (name, strategy, backtest_kwargs) 튜플
    """
    strategies = []

    # ── 1. Baseline (Phase 6 최고: 4F+Full+Regime) ──
    strategies.append((
        "1) Baseline",
        ThreeFactorStrategy(
            num_stocks=20,
            value_weight=0.28,
            momentum_weight=0.28,
            quality_weight=0.29,
            low_vol_weight=0.15,
            market_timing=make_mt_overlay(),
            sector_neutral=True,
            max_sector_pct=0.25,
            turnover_buffer=5,
            holding_bonus=0.5,
            regime_model=make_regime_model(),
        ),
        {**make_overlays()},
    ))

    # ── 2. No SectorNeutral ──
    strategies.append((
        "2) No SN+Regime",
        ThreeFactorStrategy(
            num_stocks=20,
            value_weight=0.28,
            momentum_weight=0.28,
            quality_weight=0.29,
            low_vol_weight=0.15,
            market_timing=make_mt_overlay(),
            sector_neutral=False,
            turnover_buffer=3,
            holding_bonus=0.1,
            regime_model=make_regime_model(),
        ),
        {**make_overlays()},
    ))

    # ── 3. Large-cap 30 ──
    strategies.append((
        "3) LargeCap30",
        ThreeFactorStrategy(
            num_stocks=30,
            value_weight=0.28,
            momentum_weight=0.28,
            quality_weight=0.29,
            low_vol_weight=0.15,
            min_market_cap=300_000_000_000,
            market_timing=make_mt_overlay(),
            sector_neutral=False,
            turnover_buffer=3,
            holding_bonus=0.1,
            regime_model=make_regime_model(),
        ),
        {**make_overlays()},
    ))

    # ── 4. Large-cap 50 ──
    strategies.append((
        "4) LargeCap50",
        ThreeFactorStrategy(
            num_stocks=50,
            value_weight=0.28,
            momentum_weight=0.28,
            quality_weight=0.29,
            low_vol_weight=0.15,
            min_market_cap=500_000_000_000,
            market_timing=make_mt_overlay(),
            sector_neutral=False,
            turnover_buffer=3,
            holding_bonus=0.1,
            regime_model=make_regime_model(),
        ),
        {**make_overlays()},
    ))

    # ── 5. Smart Rebalance ──
    strategies.append((
        "5) SmartRebal",
        ThreeFactorStrategy(
            num_stocks=20,
            value_weight=0.28,
            momentum_weight=0.28,
            quality_weight=0.29,
            low_vol_weight=0.15,
            market_timing=make_mt_overlay(),
            sector_neutral=False,
            turnover_buffer=3,
            holding_bonus=0.1,
            regime_model=make_regime_model(),
        ),
        {**make_overlays(), "min_rebalance_threshold": 0.005},
    ))

    # ── 6. Hybrid 75/25 ──
    core_75 = ThreeFactorStrategy(
        num_stocks=30,
        value_weight=0.28,
        momentum_weight=0.28,
        quality_weight=0.29,
        low_vol_weight=0.15,
        min_market_cap=300_000_000_000,
        market_timing=make_mt_overlay(),
        sector_neutral=False,
        turnover_buffer=3,
        holding_bonus=0.1,
        regime_model=make_regime_model(),
    )
    strategies.append((
        "6) Hybrid75/25",
        HybridStrategy(core_strategy=core_75, core_weight=0.75),
        {**make_overlays()},
    ))

    # ── 7. Hybrid 70/30 ──
    core_70 = ThreeFactorStrategy(
        num_stocks=30,
        value_weight=0.28,
        momentum_weight=0.28,
        quality_weight=0.29,
        low_vol_weight=0.15,
        min_market_cap=300_000_000_000,
        market_timing=make_mt_overlay(),
        sector_neutral=False,
        turnover_buffer=3,
        holding_bonus=0.1,
        regime_model=make_regime_model(),
    )
    strategies.append((
        "7) Hybrid70/30",
        HybridStrategy(core_strategy=core_70, core_weight=0.70),
        {**make_overlays()},
    ))

    # ── 8. Full Optimized ──
    core_full = ThreeFactorStrategy(
        num_stocks=30,
        value_weight=0.28,
        momentum_weight=0.28,
        quality_weight=0.29,
        low_vol_weight=0.15,
        min_market_cap=300_000_000_000,
        market_timing=make_mt_overlay(),
        sector_neutral=False,
        turnover_buffer=3,
        holding_bonus=0.1,
        regime_model=make_regime_model(),
    )
    strategies.append((
        "8) FullOptimized",
        HybridStrategy(core_strategy=core_full, core_weight=0.75),
        {**make_overlays(), "min_rebalance_threshold": 0.005},
    ))

    return strategies


# ── Walk-Forward 전략 팩토리 맵 ─────────────────────────

def _make_factory(strat_kwargs, hybrid_kwargs=None):
    """Walk-Forward용 strategy_factory를 생성한다."""
    def factory(train_start: str, train_end: str):
        strat = ThreeFactorStrategy(**strat_kwargs)
        if hybrid_kwargs is not None:
            return HybridStrategy(core_strategy=strat, **hybrid_kwargs)
        return strat
    return factory


def build_wf_configs():
    """Walk-Forward 전략 설정을 반환한다.

    Returns:
        list of (name, factory, overlay_kwargs) 튜플
    """
    base_4f = {
        "value_weight": 0.28,
        "momentum_weight": 0.28,
        "quality_weight": 0.29,
        "low_vol_weight": 0.15,
    }

    configs = []

    # 1) Baseline
    configs.append((
        "1) Baseline",
        _make_factory({
            **base_4f,
            "num_stocks": 20,
            "sector_neutral": True,
            "max_sector_pct": 0.25,
            "turnover_buffer": 5,
            "holding_bonus": 0.5,
            "regime_model": make_regime_model(),
        }),
        {**make_overlays()},
    ))

    # 2) No SN+Regime
    configs.append((
        "2) No SN+Regime",
        _make_factory({
            **base_4f,
            "num_stocks": 20,
            "sector_neutral": False,
            "turnover_buffer": 3,
            "holding_bonus": 0.1,
            "regime_model": make_regime_model(),
        }),
        {**make_overlays()},
    ))

    # 3) LargeCap30
    configs.append((
        "3) LargeCap30",
        _make_factory({
            **base_4f,
            "num_stocks": 30,
            "min_market_cap": 300_000_000_000,
            "sector_neutral": False,
            "turnover_buffer": 3,
            "holding_bonus": 0.1,
            "regime_model": make_regime_model(),
        }),
        {**make_overlays()},
    ))

    # 4) LargeCap50
    configs.append((
        "4) LargeCap50",
        _make_factory({
            **base_4f,
            "num_stocks": 50,
            "min_market_cap": 500_000_000_000,
            "sector_neutral": False,
            "turnover_buffer": 3,
            "holding_bonus": 0.1,
            "regime_model": make_regime_model(),
        }),
        {**make_overlays()},
    ))

    # 5) SmartRebal
    configs.append((
        "5) SmartRebal",
        _make_factory({
            **base_4f,
            "num_stocks": 20,
            "sector_neutral": False,
            "turnover_buffer": 3,
            "holding_bonus": 0.1,
            "regime_model": make_regime_model(),
        }),
        {**make_overlays(), "min_rebalance_threshold": 0.005},
    ))

    # 6) Hybrid 75/25
    configs.append((
        "6) Hybrid75/25",
        _make_factory(
            {**base_4f, "num_stocks": 30, "min_market_cap": 300_000_000_000,
             "sector_neutral": False, "turnover_buffer": 3, "holding_bonus": 0.1,
             "regime_model": make_regime_model()},
            {"core_weight": 0.75},
        ),
        {**make_overlays()},
    ))

    # 7) Hybrid 70/30
    configs.append((
        "7) Hybrid70/30",
        _make_factory(
            {**base_4f, "num_stocks": 30, "min_market_cap": 300_000_000_000,
             "sector_neutral": False, "turnover_buffer": 3, "holding_bonus": 0.1,
             "regime_model": make_regime_model()},
            {"core_weight": 0.70},
        ),
        {**make_overlays()},
    ))

    # 8) Full Optimized
    configs.append((
        "8) FullOptimized",
        _make_factory(
            {**base_4f, "num_stocks": 30, "min_market_cap": 300_000_000_000,
             "sector_neutral": False, "turnover_buffer": 3, "holding_bonus": 0.1,
             "regime_model": make_regime_model()},
            {"core_weight": 0.75},
        ),
        {**make_overlays(), "min_rebalance_threshold": 0.005},
    ))

    return configs


# ── In-Sample 백테스트 실행 ────────────────────────────

def run_insample(strategies):
    """전체 기간 In-Sample 백테스트를 실행한다."""
    print("=" * 90)
    print(f"  [In-Sample] 백테스트: {START_DATE} ~ {END_DATE}")
    print(f"  초기 자본: {INITIAL_CAPITAL:,}원 | 리밸런싱: {REBALANCE_FREQ}")
    print("=" * 90)
    print()

    results = []

    for i, (name, strategy, bt_kwargs) in enumerate(strategies):
        print(f"  [{i+1}/{len(strategies)}] {name} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            bt = Backtest(
                strategy=strategy,
                start_date=START_DATE,
                end_date=END_DATE,
                initial_capital=INITIAL_CAPITAL,
                rebalance_freq=REBALANCE_FREQ,
                **bt_kwargs,
            )
            bt.run()
            res = bt.get_results()
            elapsed = time.time() - t0

            results.append({"name": name, "results": res, "elapsed": elapsed})
            print(
                f"OK ({elapsed:.0f}s) | "
                f"CAGR={res['cagr']:.2f}% | "
                f"Sharpe={res['sharpe_ratio']:.2f} | "
                f"MDD={res['mdd']:.2f}% | "
                f"거래={res['total_trades']}건"
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAIL ({elapsed:.0f}s): {e}")
            import traceback
            traceback.print_exc()

    return results


# ── Walk-Forward OOS 백테스트 실행 ─────────────────────

def run_walkforward(configs):
    """Walk-Forward OOS 백테스트를 실행한다."""
    print()
    print("=" * 90)
    print(
        f"  [Walk-Forward OOS] 백테스트: {START_DATE} ~ {END_DATE}  "
        f"(학습={WF_TRAIN_YEARS}년, 검증={WF_TEST_YEARS}년, 스텝={WF_STEP_MONTHS}개월)"
    )
    print("=" * 90)
    print()

    results = []

    for i, (name, factory, overlay_kwargs) in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {name} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            wf = WalkForwardBacktest(
                strategy_factory=factory,
                full_start_date=START_DATE,
                full_end_date=END_DATE,
                train_years=WF_TRAIN_YEARS,
                test_years=WF_TEST_YEARS,
                step_months=WF_STEP_MONTHS,
                initial_capital=INITIAL_CAPITAL,
                rebalance_freq=REBALANCE_FREQ,
                **overlay_kwargs,
            )
            wf.run()
            oos = wf.get_oos_results()
            elapsed = time.time() - t0

            results.append({"name": name, "results": oos, "elapsed": elapsed})
            print(
                f"OK ({elapsed:.0f}s) | "
                f"CAGR={oos['cagr']:.2f}% | "
                f"Sharpe={oos['sharpe_ratio']:.2f} | "
                f"MDD={oos['mdd']:.2f}% | "
                f"윈도우={oos['num_windows']}개"
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAIL ({elapsed:.0f}s): {e}")
            import traceback
            traceback.print_exc()

    return results


# ── 결과 테이블 출력 ──────────────────────────────────

def print_results_table(title, results):
    """결과를 비교 테이블로 출력한다."""
    if not results:
        print(f"\n  {title}: 결과 없음\n")
        return

    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)
    header = (
        f"{'전략':22s} {'수익률':>10s} {'CAGR':>8s} "
        f"{'Sharpe':>8s} {'MDD':>8s} {'거래':>8s} {'시간':>6s}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        res = r["results"]
        total_ret = res.get("total_return", 0)
        cagr = res.get("cagr", 0)
        sharpe = res.get("sharpe_ratio", 0)
        mdd = res.get("mdd", 0)
        trades = res.get("total_trades", res.get("num_windows", "-"))
        elapsed = r.get("elapsed", 0)

        print(
            f"  {r['name']:20s} "
            f"{total_ret:9.2f}% "
            f"{cagr:7.2f}% "
            f"{sharpe:8.2f} "
            f"{mdd:7.2f}% "
            f"{str(trades):>8s} "
            f"{elapsed:5.0f}s"
        )

    print("-" * 100)

    # Best 전략
    best_sharpe = max(results, key=lambda x: x["results"].get("sharpe_ratio", -999))
    best_return = max(results, key=lambda x: x["results"].get("cagr", -999))
    best_mdd = max(results, key=lambda x: x["results"].get("mdd", -999))

    print(
        f"  Best Sharpe:  {best_sharpe['name']} "
        f"(Sharpe={best_sharpe['results'].get('sharpe_ratio', 0):.2f})"
    )
    print(
        f"  Best CAGR:    {best_return['name']} "
        f"(CAGR={best_return['results'].get('cagr', 0):.2f}%)"
    )
    print(
        f"  Lowest MDD:   {best_mdd['name']} "
        f"(MDD={best_mdd['results'].get('mdd', 0):.2f}%)"
    )
    print()


def print_delta_table(title, results):
    """Baseline 대비 Delta를 출력한다."""
    if len(results) < 2:
        return

    baseline = results[0]["results"]
    bl_sharpe = baseline.get("sharpe_ratio", 0)
    bl_cagr = baseline.get("cagr", 0)
    bl_mdd = baseline.get("mdd", 0)

    print(f"  {title} — Baseline 대비 Delta")
    print("-" * 80)
    header = f"{'전략':22s} {'dSharpe':>10s} {'dCAGR':>10s} {'dMDD':>10s}"
    print(header)
    print("-" * 80)

    for r in results[1:]:
        res = r["results"]
        ds = res.get("sharpe_ratio", 0) - bl_sharpe
        dc = res.get("cagr", 0) - bl_cagr
        dm = res.get("mdd", 0) - bl_mdd

        sign_s = "+" if ds >= 0 else ""
        sign_c = "+" if dc >= 0 else ""
        sign_m = "+" if dm >= 0 else ""

        print(
            f"  {r['name']:20s} "
            f"{sign_s}{ds:9.2f} "
            f"{sign_c}{dc:8.2f}pp "
            f"{sign_m}{dm:8.2f}pp"
        )

    print("-" * 80)
    print()


# ── 메인 ──────────────────────────────────────────────

def main():
    print()
    print("*" * 100)
    print("  Phase 7 Sharpe 개선 백테스트 — 8개 전략 비교 (2016~2025)")
    print("*" * 100)
    print()

    strategies = build_strategies()

    # 1) In-Sample 백테스트
    is_results = run_insample(strategies)
    print_results_table("In-Sample 결과 비교", is_results)
    print_delta_table("In-Sample", is_results)

    # 2) Walk-Forward OOS 백테스트
    wf_configs = build_wf_configs()
    wf_results = run_walkforward(wf_configs)
    print_results_table("Walk-Forward OOS 결과 비교", wf_results)
    print_delta_table("Walk-Forward OOS", wf_results)

    # 종합 요약
    print("*" * 100)
    print("  Phase 7 Sharpe 개선 백테스트 완료")
    print("*" * 100)


if __name__ == "__main__":
    main()
