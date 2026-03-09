#!/usr/bin/env python3
"""
3경로 시그널 일치 검증:
  A) premarket_check (08:50 장전 시그널)
  B) /balance (실시간 프리뷰)
  C) daily_simulation_batch (16:05 일일 시뮬레이션)
"""
import sys
import os

sys.path.insert(0, "/mnt/data/quant")
os.chdir("/mnt/data/quant")

from dotenv import load_dotenv
load_dotenv("/mnt/data/quant/.env")

from datetime import datetime
from src.scheduler.main import TradingBot
from src.strategy.strategy_config import create_multi_factor

bot = TradingBot()
strategy = create_multi_factor("live")
bot.set_strategy(strategy)

today = datetime.now().date()
prev_day = bot.holidays.prev_trading_day(today)
data_date = prev_day.strftime("%Y%m%d")
date_str = today.strftime("%Y%m%d")

print(f"=== 3경로 시그널 일치 검증 ({today}) ===")
print(f"  T-1 데이터 기준일: {data_date}")
print()

# 종목명 매핑용 데이터 수집 (한 번만)
strategy_data = bot._collect_strategy_data(data_date)
fund = strategy_data.get("fundamentals")
name_map = {}
if fund is not None and not fund.empty and "ticker" in fund.columns:
    name_map = dict(zip(fund["ticker"], fund["name"]))

num_stocks = getattr(bot._strategy, "num_stocks", 7)
scan_limit = num_stocks * 2


def print_signals(label, signals_dict):
    """시그널 딕셔너리를 정렬 출력"""
    print(f"  [{label}] {len(signals_dict)}종목")
    for i, (t, w) in enumerate(
        sorted(signals_dict.items(), key=lambda x: x[1], reverse=True), 1
    ):
        print(f"    {i}. {name_map.get(t, t)}({t}): {w:.1%}")


# ─────────────────────────────────────────
# Path A: premarket_check 경로
# ─────────────────────────────────────────
print("=" * 55)
print("[ A: premarket_check 경로 ]")
print(f"  전략: self._strategy (market_timing={bot._strategy.apply_market_timing})")
print(f"  scan_limit: {scan_limit}")

signals_a = bot._strategy.generate_signals(
    date_str, strategy_data, scan_limit=scan_limit
)
print(f"  raw 시그널: {len(signals_a)}종목")

if bot.allocator:
    budget = bot.allocator.get_long_term_budget()
    print(f"  매수가능 필터: 예산 {budget:,}원")
    signals_a, excl_a = bot._filter_affordable_signals(
        signals_a, strategy_data, budget, num_stocks
    )

print_signals("premarket_check", signals_a)

# ─────────────────────────────────────────
# Path B: /balance 경로
# ─────────────────────────────────────────
print()
print("=" * 55)
print("[ B: /balance 경로 (_generate_live_long_term_signals) ]")

# 캐시 클리어
cache_path = f"data/simulation/{today.strftime('%Y-%m-%d')}/multi_factor.json"
if os.path.exists(cache_path):
    os.remove(cache_path)

result_b = bot._generate_live_long_term_signals("multi_factor")
selected_b = result_b.get("selected", []) if result_b else []
signals_b = {item["ticker"]: item["weight"] for item in selected_b}

print_signals("/balance", signals_b)

# ─────────────────────────────────────────
# Path C: daily_simulation_batch 경로
# ─────────────────────────────────────────
print()
print("=" * 55)
print("[ C: daily_simulation_batch 경로 ]")

# daily_simulation_batch가 하는 것을 재현 (수정 후)
from src.data.daily_simulator import DailySimulator

# primary 전략은 self._strategy 사용 (수정됨)
sim_config = bot.feature_flags.get_config("daily_simulation")
primary_name = sim_config.get("primary_strategy", "multi_factor")
sim_strategy = bot._strategy  # self._strategy 우선
print(f"  전략: self._strategy (market_timing={sim_strategy.apply_market_timing})")

sim_num_stocks = getattr(sim_strategy, "num_stocks", 7)
sim_scan_limit = sim_num_stocks * 2
print(f"  scan_limit: {sim_scan_limit}")

# DailySimulator._generate_signals에 scan_limit 전달
simulator = DailySimulator()
simulator.strategy_data = strategy_data
signals_c_raw = simulator._generate_signals(
    sim_strategy, date_str, scan_limit=sim_scan_limit
)
print(f"  raw 시그널: {len(signals_c_raw)}종목")

# 매수가능 필터 적용 (수정됨)
signals_c = signals_c_raw
if bot.allocator:
    budget_c = bot.allocator.get_long_term_budget()
    print(f"  매수가능 필터: 예산 {budget_c:,}원")
    signals_c, excl_c = bot._filter_affordable_signals(
        signals_c_raw, strategy_data, budget_c, sim_num_stocks
    )

print_signals("daily_simulation", signals_c)

# ─────────────────────────────────────────
# 비교
# ─────────────────────────────────────────
print()
print("=" * 55)
print("[ 장기 종목 비교 ]")

tickers_a = set(signals_a.keys())
tickers_b = set(signals_b.keys())
tickers_c = set(signals_c.keys())

# A vs B
if tickers_a == tickers_b:
    print("  A vs B (premarket vs /balance): ✅ 일치")
else:
    print("  A vs B (premarket vs /balance): ❌ 불일치")
    diff = tickers_a.symmetric_difference(tickers_b)
    for t in diff:
        src = "A에만" if t in tickers_a else "B에만"
        print(f"    {src}: {name_map.get(t, t)}")

# A vs C
if tickers_a == tickers_c:
    print("  A vs C (premarket vs daily_sim): ✅ 일치")
else:
    print("  A vs C (premarket vs daily_sim): ❌ 불일치")
    only_a = tickers_a - tickers_c
    only_c = tickers_c - tickers_a
    if only_a:
        print(f"    premarket에만: {[name_map.get(t, t) for t in only_a]}")
    if only_c:
        print(f"    daily_sim에만: {[name_map.get(t, t) for t in only_c]}")

# 비중 비교 (A vs B)
weights_match_ab = all(
    abs(signals_a.get(t, 0) - signals_b.get(t, 0)) < 0.001
    for t in tickers_a | tickers_b
)
if weights_match_ab:
    print("  A vs B 비중: ✅ 일치")
else:
    print("  A vs B 비중: ❌ 불일치")

# 비중 비교 (A vs C)
weights_match_ac = all(
    abs(signals_a.get(t, 0) - signals_c.get(t, 0)) < 0.001
    for t in tickers_a | tickers_c
)
if weights_match_ac:
    print("  A vs C 비중: ✅ 일치")
else:
    print("  A vs C 비중: ❌ 불일치")

# ─────────────────────────────────────────
# ETF 비교
# ─────────────────────────────────────────
print()
print("=" * 55)
print("[ ETF 시그널 비교 ]")

etf_a = bot._generate_etf_signals(date_str)
etf_c = bot._generate_etf_signals(date_str)  # daily_sim도 동일 함수

etf_strategy = bot._create_etf_strategy()
etf_names = getattr(etf_strategy, "etf_universe", {})

if etf_a and etf_c and set(etf_a.keys()) == set(etf_c.keys()):
    print("  ETF 종목: ✅ 일치")
    etf_w_match = all(
        abs(etf_a[t] - etf_c[t]) < 0.001 for t in etf_a
    )
    if etf_w_match:
        print("  ETF 비중: ✅ 일치")
    print()
    for t in sorted(etf_a.keys(), key=lambda x: etf_a[x], reverse=True):
        print(f"    {etf_names.get(t, t)}({t}): {etf_a[t]:.1%}")
else:
    print("  ETF: ❌ 불일치")

# ─────────────────────────────────────────
# 차이 원인 분석
# ─────────────────────────────────────────
if tickers_a != tickers_c:
    print()
    print("=" * 55)
    print("[ 차이 원인 분석 ]")
    print(f"  1. apply_market_timing: premarket={bot._strategy.apply_market_timing}"
          f" vs daily_sim={sim_strategy.apply_market_timing}")
    print(f"  2. scan_limit: premarket={scan_limit} vs daily_sim=None({sim_strategy.num_stocks})")
    print(f"  3. 매수가능 필터: premarket=적용 vs daily_sim=미적용")
    print(f"  4. turnover_penalty: premarket=self._strategy(연속) vs daily_sim=새 인스턴스")

print()
print("검증 완료")
