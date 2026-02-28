"""10년 백테스트 스크립트 — Phase 6 모듈 검증.

6개 전략을 점진적으로 기능 추가하며 비교한다.
In-Sample (전체 기간) + Walk-Forward OOS 두 가지 모드를 실행한다.

전략 목록:
  1. Baseline 3F          — V33/M33/Q34, 오버레이 없음
  2. 3F + MT              — + MarketTimingOverlay(MA200, gradual)
  3. 4F + MT              — + LowVol 15%
  4. 4F + Triple Overlay  — + Drawdown + VolTargeting
  5. 4F + Full            — + 섹터 중립 + Turnover buffer
  6. 4F + Full + Regime   — + RuleBasedRegimeModel

사용법:
  python scripts/run_sharpe_backtest.py
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
from src.strategy.market_timing import MarketTimingOverlay
from src.strategy.drawdown_overlay import DrawdownOverlay
from src.strategy.vol_targeting import VolTargetingOverlay
from src.ml.regime_model import RuleBasedRegimeModel

# ── 설정 ──────────────────────────────────────────────
START_DATE = "20160101"
END_DATE = "20251231"
INITIAL_CAPITAL = 100_000_000  # 1억원
REBALANCE_FREQ = "monthly"
NUM_STOCKS = 20

# Walk-Forward 설정
WF_TRAIN_YEARS = 5
WF_TEST_YEARS = 1
WF_STEP_MONTHS = 12


# ── 전략 빌더 ─────────────────────────────────────────

def build_strategies():
    """6개 전략과 관련 오버레이를 생성하여 반환한다.

    Returns:
        list of (name, strategy, overlay_kwargs) 튜플
    """
    strategies = []

    # 공통 마켓 타이밍 오버레이
    mt_overlay = MarketTimingOverlay(
        ma_period=200,
        ma_type="SMA",
        switch_mode="gradual",
        reference_index="KOSPI",
    )

    # 드로다운 오버레이
    dd_overlay = DrawdownOverlay(
        thresholds=[(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)],
        recovery_buffer=0.02,
    )

    # 변동성 타겟팅 오버레이
    vt_overlay = VolTargetingOverlay(
        target_vol=0.15,
        lookback_days=20,
        use_downside_only=True,
    )

    # 레짐 모델
    regime_model = RuleBasedRegimeModel(
        factor_names=["value", "momentum", "quality", "low_vol"],
    )

    # ── 1. Baseline 3F ──
    strategies.append((
        "1) Baseline 3F",
        ThreeFactorStrategy(
            num_stocks=NUM_STOCKS,
            value_weight=0.33,
            momentum_weight=0.33,
            quality_weight=0.34,
        ),
        {},  # no overlays
    ))

    # ── 2. 3F + MT ──
    strategies.append((
        "2) 3F + MT",
        ThreeFactorStrategy(
            num_stocks=NUM_STOCKS,
            value_weight=0.33,
            momentum_weight=0.33,
            quality_weight=0.34,
            market_timing=mt_overlay,
        ),
        {"overlay": MarketTimingOverlay(
            ma_period=200, ma_type="SMA",
            switch_mode="gradual", reference_index="KOSPI",
        )},
    ))

    # ── 3. 4F + MT ──
    strategies.append((
        "3) 4F + MT",
        ThreeFactorStrategy(
            num_stocks=NUM_STOCKS,
            value_weight=0.28,
            momentum_weight=0.28,
            quality_weight=0.29,
            low_vol_weight=0.15,
            market_timing=mt_overlay,
        ),
        {"overlay": MarketTimingOverlay(
            ma_period=200, ma_type="SMA",
            switch_mode="gradual", reference_index="KOSPI",
        )},
    ))

    # ── 4. 4F + Triple Overlay ──
    strategies.append((
        "4) 4F + TripleOvl",
        ThreeFactorStrategy(
            num_stocks=NUM_STOCKS,
            value_weight=0.28,
            momentum_weight=0.28,
            quality_weight=0.29,
            low_vol_weight=0.15,
            market_timing=mt_overlay,
        ),
        {
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
        },
    ))

    # ── 5. 4F + Full ──
    strategies.append((
        "5) 4F + Full",
        ThreeFactorStrategy(
            num_stocks=NUM_STOCKS,
            value_weight=0.28,
            momentum_weight=0.28,
            quality_weight=0.29,
            low_vol_weight=0.15,
            market_timing=mt_overlay,
            sector_neutral=True,
            max_sector_pct=0.25,
            turnover_buffer=5,
            holding_bonus=0.5,
        ),
        {
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
        },
    ))

    # ── 6. 4F + Full + Regime ──
    strategies.append((
        "6) 4F+Full+Regime",
        ThreeFactorStrategy(
            num_stocks=NUM_STOCKS,
            value_weight=0.28,
            momentum_weight=0.28,
            quality_weight=0.29,
            low_vol_weight=0.15,
            market_timing=mt_overlay,
            sector_neutral=True,
            max_sector_pct=0.25,
            turnover_buffer=5,
            holding_bonus=0.5,
            regime_model=regime_model,
        ),
        {
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
        },
    ))

    return strategies


def make_strategy_factory(strategy_cls_kwargs):
    """Walk-Forward용 strategy_factory를 생성한다.

    학습 구간 시작/종료일을 받지만, ThreeFactorStrategy는
    학습 구간에 의존하지 않으므로 동일 파라미터로 전략을 생성한다.
    """
    def factory(train_start: str, train_end: str):
        return ThreeFactorStrategy(**strategy_cls_kwargs)
    return factory


# ── In-Sample 백테스트 실행 ────────────────────────────

def run_insample(strategies):
    """전체 기간 In-Sample 백테스트를 실행한다."""
    print("=" * 90)
    print(f"  [In-Sample] 백테스트: {START_DATE} ~ {END_DATE}")
    print(f"  초기 자본: {INITIAL_CAPITAL:,}원 | 리밸런싱: {REBALANCE_FREQ}")
    print("=" * 90)
    print()

    results = []

    for i, (name, strategy, overlay_kwargs) in enumerate(strategies):
        print(f"  [{i+1}/{len(strategies)}] {name} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            bt = Backtest(
                strategy=strategy,
                start_date=START_DATE,
                end_date=END_DATE,
                initial_capital=INITIAL_CAPITAL,
                rebalance_freq=REBALANCE_FREQ,
                **overlay_kwargs,
            )
            bt.run()
            res = bt.get_results()
            elapsed = time.time() - t0

            results.append({"name": name, "results": res, "elapsed": elapsed})
            print(
                f"OK ({elapsed:.0f}s) | "
                f"CAGR={res['cagr']:.2f}% | "
                f"Sharpe={res['sharpe_ratio']:.2f} | "
                f"MDD={res['mdd']:.2f}%"
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAIL ({elapsed:.0f}s): {e}")
            import traceback
            traceback.print_exc()

    return results


# ── Walk-Forward OOS 백테스트 실행 ─────────────────────

def run_walkforward(strategies):
    """Walk-Forward OOS 백테스트를 실행한다."""
    print()
    print("=" * 90)
    print(
        f"  [Walk-Forward OOS] 백테스트: {START_DATE} ~ {END_DATE}"
        f"  (학습={WF_TRAIN_YEARS}년, 검증={WF_TEST_YEARS}년, 스텝={WF_STEP_MONTHS}개월)"
    )
    print("=" * 90)
    print()

    results = []

    # 전략별 ThreeFactorStrategy kwargs 추출
    strategy_configs = [
        # (name, strategy_kwargs, overlay_kwargs)
        ("1) Baseline 3F", {
            "num_stocks": NUM_STOCKS,
            "value_weight": 0.33,
            "momentum_weight": 0.33,
            "quality_weight": 0.34,
        }, {}),
        ("2) 3F + MT", {
            "num_stocks": NUM_STOCKS,
            "value_weight": 0.33,
            "momentum_weight": 0.33,
            "quality_weight": 0.34,
        }, {"overlay": MarketTimingOverlay(
            ma_period=200, ma_type="SMA",
            switch_mode="gradual", reference_index="KOSPI",
        )}),
        ("3) 4F + MT", {
            "num_stocks": NUM_STOCKS,
            "value_weight": 0.28,
            "momentum_weight": 0.28,
            "quality_weight": 0.29,
            "low_vol_weight": 0.15,
        }, {"overlay": MarketTimingOverlay(
            ma_period=200, ma_type="SMA",
            switch_mode="gradual", reference_index="KOSPI",
        )}),
        ("4) 4F + TripleOvl", {
            "num_stocks": NUM_STOCKS,
            "value_weight": 0.28,
            "momentum_weight": 0.28,
            "quality_weight": 0.29,
            "low_vol_weight": 0.15,
        }, {
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
        }),
        ("5) 4F + Full", {
            "num_stocks": NUM_STOCKS,
            "value_weight": 0.28,
            "momentum_weight": 0.28,
            "quality_weight": 0.29,
            "low_vol_weight": 0.15,
            "sector_neutral": True,
            "max_sector_pct": 0.25,
            "turnover_buffer": 5,
            "holding_bonus": 0.5,
        }, {
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
        }),
        ("6) 4F+Full+Regime", {
            "num_stocks": NUM_STOCKS,
            "value_weight": 0.28,
            "momentum_weight": 0.28,
            "quality_weight": 0.29,
            "low_vol_weight": 0.15,
            "sector_neutral": True,
            "max_sector_pct": 0.25,
            "turnover_buffer": 5,
            "holding_bonus": 0.5,
            "regime_model": RuleBasedRegimeModel(
                factor_names=["value", "momentum", "quality", "low_vol"],
            ),
        }, {
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
        }),
    ]

    for i, (name, strat_kwargs, overlay_kwargs) in enumerate(strategy_configs):
        print(f"  [{i+1}/{len(strategy_configs)}] {name} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            wf = WalkForwardBacktest(
                strategy_factory=make_strategy_factory(strat_kwargs),
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
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    header = (
        f"{'전략':22s} {'수익률':>10s} {'CAGR':>8s} "
        f"{'Sharpe':>8s} {'MDD':>8s} {'시간':>6s}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        res = r["results"]
        total_ret = res.get("total_return", 0)
        cagr = res.get("cagr", 0)
        sharpe = res.get("sharpe_ratio", 0)
        mdd = res.get("mdd", 0)
        elapsed = r.get("elapsed", 0)

        print(
            f"  {r['name']:20s} "
            f"{total_ret:9.2f}% "
            f"{cagr:7.2f}% "
            f"{sharpe:8.2f} "
            f"{mdd:7.2f}% "
            f"{elapsed:5.0f}s"
        )

    print("-" * 90)

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


# ── Delta 비교 ────────────────────────────────────────

def print_delta_table(title, results):
    """Baseline 대비 Delta를 출력한다."""
    if len(results) < 2:
        return

    baseline = results[0]["results"]
    bl_sharpe = baseline.get("sharpe_ratio", 0)
    bl_cagr = baseline.get("cagr", 0)
    bl_mdd = baseline.get("mdd", 0)

    print(f"  {title} — Baseline 대비 Delta")
    print("-" * 70)
    header = f"{'전략':22s} {'dSharpe':>10s} {'dCAGR':>10s} {'dMDD':>10s}"
    print(header)
    print("-" * 70)

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

    print("-" * 70)
    print()


# ── 메인 ──────────────────────────────────────────────

def main():
    print()
    print("*" * 90)
    print("  10년 백테스트 — Phase 6 모듈 검증 (2016~2025)")
    print("*" * 90)
    print()

    strategies = build_strategies()

    # 1) In-Sample 백테스트
    is_results = run_insample(strategies)
    print_results_table("In-Sample 결과 비교", is_results)
    print_delta_table("In-Sample", is_results)

    # 2) Walk-Forward OOS 백테스트
    wf_results = run_walkforward(strategies)
    print_results_table("Walk-Forward OOS 결과 비교", wf_results)
    print_delta_table("Walk-Forward OOS", wf_results)

    # 종합 요약
    print("*" * 90)
    print("  10년 백테스트 완료")
    print("*" * 90)


if __name__ == "__main__":
    main()
