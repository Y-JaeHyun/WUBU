"""3/2 리밸런싱 시뮬레이션 스크립트.

3월 2일(월) 09:05 통합 리밸런싱을 시뮬레이션한다.
T-1 = 2/27(금) 종가 기준 데이터 사용.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, date

# 데이터 수집
from src.data.collector import get_all_fundamentals, get_price_data
from src.data.index_collector import get_index_data
from src.strategy.multi_factor import MultiFactorStrategy
from src.strategy.etf_rotation import ETFRotationStrategy
from src.execution.portfolio_allocator import PortfolioAllocator
from src.execution.position_manager import PositionManager
from src.execution.risk_guard import RiskGuard
from src.execution.kis_client import KISClient
# HolidayManager not needed for direct simulation

print("=" * 60)
print("  3/2(월) 통합 리밸런싱 시뮬레이션")
print("  데이터 기준: T-1 = 2/27(금) 종가")
print("=" * 60)

# 1. 데이터 수집 (T-1 = 20260227)
data_date = "20260227"
print(f"\n[1] 데이터 수집 (기준일: {data_date})...")

try:
    fundamentals = get_all_fundamentals(data_date)
    print(f"  펀더멘탈: {len(fundamentals)}종목")
except Exception as e:
    print(f"  펀더멘탈 수집 실패: {e}")
    sys.exit(1)

# 시총 상위 200 종목 가격 데이터
from datetime import timedelta
start_dt = datetime.strptime(data_date, "%Y%m%d") - timedelta(days=400)
start_date = start_dt.strftime("%Y%m%d")

top_tickers = fundamentals.nlargest(200, "시가총액").index.tolist() if "시가총액" in fundamentals.columns else fundamentals.head(200).index.tolist()
prices = {}
for ticker in top_tickers:
    try:
        df = get_price_data(ticker, start_date, data_date)
        if df is not None and not df.empty:
            prices[ticker] = df
    except Exception:
        pass

print(f"  가격 데이터: {len(prices)}종목")

# KOSPI 지수
try:
    index_df = get_index_data("KOSPI", start_date, data_date)
    print(f"  KOSPI 지수: {len(index_df)}일")
except Exception:
    index_df = None
    print("  KOSPI 지수: 수집 실패")

strategy_data = {
    "fundamentals": fundamentals,
    "prices": prices,
    "index_prices": index_df["종가"] if index_df is not None and not index_df.empty else None,
}

# 2. 장기 시그널 생성
print(f"\n[2] 장기 전략 시그널 생성 (MultiFactor)...")
strategy = MultiFactorStrategy(
    value_weight=0.4,
    momentum_weight=0.6,
    num_stocks=10,
    turnover_penalty=0.1,
    apply_market_timing=True,
)

today_str = "20260302"
long_signals = strategy.generate_signals(today_str, strategy_data)
print(f"  장기 종목 수: {len(long_signals)}개")
if long_signals:
    for ticker, weight in sorted(long_signals.items(), key=lambda x: x[1], reverse=True):
        name = fundamentals.loc[ticker, "종목명"] if ticker in fundamentals.index and "종목명" in fundamentals.columns else ticker
        print(f"    {name}({ticker}): {weight:.1%}")

# 3. ETF 시그널 생성
print(f"\n[3] ETF 로테이션 시그널 생성...")
etf_universe = {
    "069500": "KODEX 200",
    "360750": "TIGER 미국S&P500",
    "133690": "TIGER 미국나스닥100",
    "091160": "KODEX 반도체",
    "091170": "KODEX 은행",
    "117700": "KODEX 건설",
    "132030": "KODEX 골드선물(H)",
    "471510": "TIGER AI로보틱스",
    "470950": "ACE AI반도체포커스",
    "153130": "KODEX 단기채권",
}

etf_prices = {}
for ticker in etf_universe:
    try:
        df = get_price_data(ticker, start_date, data_date)
        if df is not None and not df.empty:
            etf_prices[ticker] = df
    except Exception:
        pass

print(f"  ETF 가격 데이터: {len(etf_prices)}종목")

etf_strategy = ETFRotationStrategy(
    etf_universe=etf_universe,
    lookback_months=12,
    n_select=3,
    safe_asset="153130",
)

etf_signals = etf_strategy.generate_signals(today_str, {"etf_prices": etf_prices})
print(f"  ETF 시그널 수: {len(etf_signals)}개")
if etf_signals:
    for ticker, weight in sorted(etf_signals.items(), key=lambda x: x[1], reverse=True):
        name = etf_universe.get(ticker, ticker)
        print(f"    {name}({ticker}): {weight:.1%}")

# 4. 통합 병합
print(f"\n[4] 풀별 시그널 병합 (장기 70% + ETF 30%)...")
allocator = PortfolioAllocator()

pool_signals = {}
if long_signals:
    pool_signals["long_term"] = long_signals
if etf_signals:
    pool_signals["etf_rotation"] = etf_signals

merged = allocator.merge_pool_targets(pool_signals)
print(f"  병합 종목 수: {len(merged)}개")
print(f"  비중 합계: {sum(merged.values()):.1%}")
for ticker, weight in sorted(merged.items(), key=lambda x: x[1], reverse=True):
    # 종목명 찾기
    if ticker in fundamentals.index and "종목명" in fundamentals.columns:
        name = fundamentals.loc[ticker, "종목명"]
    elif ticker in etf_universe:
        name = etf_universe[ticker]
    else:
        name = ticker
    pool = "ETF" if ticker in etf_universe else "장기"
    print(f"    [{pool}] {name}({ticker}): {weight:.1%}")

# 5. 리스크 체크
print(f"\n[5] 리스크 체크...")
risk_guard = RiskGuard(is_live=True)
passed, warnings = risk_guard.check_rebalance(merged)
print(f"  비중 검증: {'PASS' if passed else 'FAIL'}")
if warnings:
    for w in warnings:
        print(f"    ⚠ {w}")

# 6. 현재 포지션 vs 타겟 diff
print(f"\n[6] 현재 보유 → 타겟 Diff 계산...")
kis = KISClient()
try:
    balance = kis.get_balance()
    portfolio_value = balance.get("total_value", 0)
    holdings = balance.get("holdings", [])
    print(f"  포트폴리오 총액: {portfolio_value:,}원")
    print(f"  현재 보유: {len(holdings)}종목")
    for h in holdings:
        ticker = h.get("ticker", "")
        name = h.get("name", ticker)
        qty = h.get("qty", 0)
        value = h.get("eval_amount", 0)
        pct = value / portfolio_value * 100 if portfolio_value > 0 else 0
        print(f"    {name}({ticker}): {qty}주, {value:,}원 ({pct:.1f}%)")
except Exception as e:
    print(f"  잔고 조회 실패: {e}")
    portfolio_value = 0
    holdings = []

# 7. 주문 계산
print(f"\n[7] 매도/매수 주문 계산...")
pm = PositionManager(kis)
try:
    orders = pm.calculate_rebalance_orders(merged, integrated=True)
    sell_orders = orders.get("sell", [])
    buy_orders = orders.get("buy", [])

    if sell_orders:
        print(f"\n  📉 매도 예상: {len(sell_orders)}건")
        total_sell = 0
        for o in sell_orders:
            name = o.get("name", o.get("ticker", ""))
            amount = o.get("amount", 0)
            total_sell += amount
            print(f"    {name}: {o.get('qty', 0)}주, {amount:,}원")
        print(f"    총 매도: {total_sell:,}원")

    if buy_orders:
        print(f"\n  📈 매수 예상: {len(buy_orders)}건")
        total_buy = 0
        for o in buy_orders:
            name = o.get("name", o.get("ticker", ""))
            amount = o.get("amount", 0)
            total_buy += amount
            print(f"    {name}: {o.get('qty', 0)}주, {amount:,}원")
        print(f"    총 매수: {total_buy:,}원")

    # 8. Turnover 체크
    if sell_orders or buy_orders:
        total_sell = sum(o.get("amount", 0) for o in sell_orders)
        total_buy = sum(o.get("amount", 0) for o in buy_orders)
        turnover = (total_sell + total_buy) / portfolio_value if portfolio_value > 0 else 0
        passed_t, reason_t = risk_guard.check_turnover(sell_orders, buy_orders, portfolio_value)
        print(f"\n[8] Turnover 체크")
        print(f"  회전율: {turnover:.1%} (한도: {risk_guard.max_daily_turnover:.0%})")
        print(f"  결과: {'PASS ✅' if passed_t else 'FAIL ❌'}")
        if reason_t:
            print(f"    {reason_t}")

except Exception as e:
    print(f"  주문 계산 실패: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'=' * 60}")
print("  시뮬레이션 완료")
print("=" * 60)
