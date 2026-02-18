"""실데이터 백테스트 스크립트.

여러 전략을 실제 한국 주식 데이터로 백테스트하고 결과를 비교한다.
"""
import sys
import os
import time

sys.path.insert(0, "/mnt/data/quant")

from src.backtest.engine import Backtest
from src.strategy.value import ValueStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.multi_factor import MultiFactorStrategy
from src.strategy.market_timing import MarketTimingOverlay

# 백테스트 설정
START_DATE = "20230101"
END_DATE = "20251231"
INITIAL_CAPITAL = 100_000_000  # 1억원
REBALANCE_FREQ = "monthly"

strategies = []

# 1. 밸류 전략 (저PBR 상위 20종목)
strategies.append(("Value(PBR)", ValueStrategy(factor="pbr", num_stocks=20)))

# 2. 밸류 전략 (저PER 상위 20종목)
strategies.append(("Value(PER)", ValueStrategy(factor="per", num_stocks=20)))

# 3. 모멘텀 전략 (12개월 모멘텀)
strategies.append(("Momentum(12M)", MomentumStrategy(
    lookback_months=[12], skip_month=True, num_stocks=20
)))

# 4. 멀티팩터 (밸류+모멘텀)
strategies.append(("MultiFactor(V+M)", MultiFactorStrategy(
    factors=["value", "momentum"],
    weights=[0.5, 0.5],
    combine_method="zscore",
    num_stocks=20,
)))

# 5. 멀티팩터 + 마켓타이밍
overlay = MarketTimingOverlay(
    ma_period=200,
    ma_type="SMA",
    switch_mode="gradual",
    reference_index="KOSPI",
)
strategies.append(("MultiFactor+MT", MultiFactorStrategy(
    factors=["value", "momentum"],
    weights=[0.5, 0.5],
    combine_method="zscore",
    num_stocks=20,
)))

print("=" * 80)
print(f"  실데이터 백테스트: {START_DATE} ~ {END_DATE}")
print(f"  초기 자본: {INITIAL_CAPITAL:,}원 | 리밸런싱: {REBALANCE_FREQ}")
print("=" * 80)
print()

all_results = []

for i, (name, strategy) in enumerate(strategies):
    print(f"[{i+1}/{len(strategies)}] {name} 백테스트 시작...")
    t0 = time.time()

    try:
        if name == "MultiFactor+MT":
            bt = Backtest(
                strategy=strategy,
                start_date=START_DATE,
                end_date=END_DATE,
                initial_capital=INITIAL_CAPITAL,
                rebalance_freq=REBALANCE_FREQ,
                overlay=overlay,
            )
        else:
            bt = Backtest(
                strategy=strategy,
                start_date=START_DATE,
                end_date=END_DATE,
                initial_capital=INITIAL_CAPITAL,
                rebalance_freq=REBALANCE_FREQ,
            )

        bt.run()
        results = bt.get_results()
        elapsed = time.time() - t0

        all_results.append({
            "name": name,
            "results": results,
            "history": bt.get_portfolio_history(),
            "elapsed": elapsed,
        })

        print(f"  완료 ({elapsed:.1f}초) | 수익률: {results['total_return']:.2f}% | "
              f"CAGR: {results['cagr']:.2f}% | Sharpe: {results['sharpe_ratio']:.2f} | "
              f"MDD: {results['mdd']:.2f}%")
    except Exception as e:
        print(f"  실패: {e}")
        import traceback
        traceback.print_exc()

    print()

# 결과 비교 테이블
print()
print("=" * 80)
print("  전략 비교 결과")
print("=" * 80)
header = f"{'전략':22s} {'수익률':>10s} {'CAGR':>8s} {'Sharpe':>8s} {'MDD':>8s} {'승률':>8s} {'최종자산':>16s}"
print(header)
print("-" * 80)

for r in all_results:
    res = r["results"]
    print(
        f"{r['name']:22s} "
        f"{res['total_return']:9.2f}% "
        f"{res['cagr']:7.2f}% "
        f"{res['sharpe_ratio']:8.2f} "
        f"{res['mdd']:7.2f}% "
        f"{res['win_rate']:7.1f}% "
        f"{res['final_value']:>15,.0f}"
    )

print("-" * 80)

# 베스트 전략
if all_results:
    best = max(all_results, key=lambda x: x["results"]["sharpe_ratio"])
    print(f"\n  Best Sharpe: {best['name']} (Sharpe={best['results']['sharpe_ratio']:.2f})")
    best_return = max(all_results, key=lambda x: x["results"]["total_return"])
    print(f"  Best Return: {best_return['name']} ({best_return['results']['total_return']:.2f}%)")
    lowest_mdd = max(all_results, key=lambda x: x["results"]["mdd"])
    print(f"  Lowest MDD: {lowest_mdd['name']} ({lowest_mdd['results']['mdd']:.2f}%)")

print()
