#!/bin/bash
# JAE-4 Backtest Runner
# Usage: bash run-jae4-backtest.sh

echo "JAE-4 Backtest Suite"
echo "===================="
echo "Testing: ASY, RMOM, VDS"
echo "Period: 2022-03-12 ~ 2026-03-12 (4 years)"
echo "Rebalance: Monthly"
echo ""

cd /mnt/data/quant-dev

python3 << 'PYTHON'
import sys
from datetime import datetime
from scripts.batch_backtest import STRATEGY_REGISTRY, create_strategy, _run_long_term_backtest

strategies = ['residual_momentum', 'advanced_shareholder_yield', 'value_up_disclosure']
start_date = "2022-03-12"
end_date = "2026-03-12"
initial_capital = 1_500_000

print(f"Started: {datetime.now().isoformat()}\n")

results = {}
for strategy_name in strategies:
    print(f"{'='*60}")
    print(f"{strategy_name}")
    print(f"{'='*60}")
    
    try:
        reg = STRATEGY_REGISTRY.get(strategy_name)
        config = reg['configs'][0]
        strategy = create_strategy(strategy_name, config)
        
        if strategy:
            result = _run_long_term_backtest(strategy, strategy_name, config, start_date, end_date, initial_capital)
            if result and 'results' in result:
                r = result['results']
                results[strategy_name] = r
                print(f"CAGR: {r.get('cagr', 0):.2%} | Sharpe: {r.get('sharpe_ratio', 0):.2f} | MDD: {r.get('mdd', 0):.2%}")
                print(f"Final: {r.get('final_value', 0):,.0f}원\n")
    except Exception as e:
        print(f"ERROR: {e}\n")

print(f"{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, r in sorted(results.items(), key=lambda x: -x[1].get('cagr', 0)):
    print(f"{name:<40} {r.get('cagr', 0):>7.2%} | {r.get('sharpe_ratio', 0):>6.2f} | {r.get('mdd', 0):>7.2%}")

print(f"\nCompleted: {datetime.now().isoformat()}")
PYTHON
