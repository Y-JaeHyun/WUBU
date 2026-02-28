"""다음 거래일 리밸런싱 시뮬레이션 스크립트.

오늘(2/27 금) 종가 기준 → 다음 거래일(3/2 월) 09:05 통합 리밸런싱 시뮬레이션.
3-Pool 아키텍처: 장기 70% (MultiFactor) + ETF 30% (ETFRotation) + 단기 0%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from src.data.collector import get_all_fundamentals, get_price_data
from src.data.index_collector import get_index_data
from src.strategy.multi_factor import MultiFactorStrategy
from src.strategy.etf_rotation import ETFRotationStrategy
from src.execution.portfolio_allocator import PortfolioAllocator
from src.execution.position_manager import PositionManager
from src.execution.risk_guard import RiskGuard
from src.execution.kis_client import KISClient

# ── 설정 ──
DATA_DATE = "20260227"       # T-1 = 오늘(금) 종가
NEXT_TRADING_DATE = "20260302"  # 다음 거래일 (월)
LONG_TERM_PCT = 0.70
ETF_ROTATION_PCT = 0.30

ETF_UNIVERSE = {
    "069500": "KODEX 200",
    "371460": "TIGER 미국S&P500",
    "133690": "TIGER 미국나스닥100",
    "091160": "KODEX 반도체",
    "091170": "KODEX 은행",
    "117700": "KODEX 건설",
    "132030": "KODEX 골드선물(H)",
    "464310": "TIGER 글로벌AI&로보틱스INDXX",
    "469150": "ACE AI반도체포커스",
    "439870": "KODEX 단기채권",
}

print("=" * 60)
print(f"  다음 거래일({NEXT_TRADING_DATE}) 통합 리밸런싱 시뮬레이션")
print(f"  데이터 기준: T-1 = {DATA_DATE} 종가")
print(f"  풀 배분: 장기 {LONG_TERM_PCT:.0%} + ETF {ETF_ROTATION_PCT:.0%}")
print("=" * 60)

# ── 1. 데이터 수집 ──
print(f"\n[1] 데이터 수집 (기준일: {DATA_DATE})...")

try:
    fundamentals = get_all_fundamentals(DATA_DATE)
    print(f"  펀더멘탈: {len(fundamentals)}종목")
except Exception as e:
    print(f"  펀더멘탈 수집 실패: {e}")
    sys.exit(1)

# 시총 상위 200 종목 가격 데이터 (400일)
start_dt = datetime.strptime(DATA_DATE, "%Y%m%d") - timedelta(days=400)
start_date = start_dt.strftime("%Y%m%d")

# 시가총액 컬럼명 탐색
market_cap_col = None
for col in ["market_cap", "시가총액"]:
    if col in fundamentals.columns:
        market_cap_col = col
        break

if market_cap_col and "ticker" in fundamentals.columns:
    top_tickers = fundamentals.nlargest(200, market_cap_col)["ticker"].tolist()
elif market_cap_col:
    top_tickers = fundamentals.nlargest(200, market_cap_col).index.tolist()
else:
    top_tickers = fundamentals.head(200).index.tolist()

print(f"  시총 상위 200 종목 가격 수집 중...")
prices = {}
for ticker in top_tickers:
    try:
        df = get_price_data(ticker, start_date, DATA_DATE)
        if df is not None and not df.empty:
            prices[ticker] = df
    except Exception:
        pass

print(f"  가격 데이터: {len(prices)}종목")

# KOSPI 지수
try:
    index_df = get_index_data("KOSPI", start_date, DATA_DATE)
    print(f"  KOSPI 지수: {len(index_df)}일")
except Exception:
    index_df = None
    print("  KOSPI 지수: 수집 실패")

strategy_data = {
    "fundamentals": fundamentals,
    "prices": prices,
    "index_prices": index_df["종가"] if index_df is not None and not index_df.empty else None,
}

# ── 2. 장기 전략 시그널 (MultiFactor: V0.4 + M0.6, top10, MT, turnover_penalty=0.1) ──
print(f"\n[2] 장기 전략 시그널 생성 (MultiFactor V0.4+M0.6, top10)...")
long_strategy = MultiFactorStrategy(
    value_weight=0.4,
    momentum_weight=0.6,
    num_stocks=10,
    turnover_penalty=0.1,
    apply_market_timing=True,
)

long_signals = long_strategy.generate_signals(NEXT_TRADING_DATE, strategy_data)
print(f"  장기 종목 수: {len(long_signals)}개")

# 종목명 매핑
name_map = {}
if "ticker" in fundamentals.columns and "name" in fundamentals.columns:
    name_map = dict(zip(fundamentals["ticker"], fundamentals["name"]))
elif "종목명" in fundamentals.columns:
    name_map = dict(zip(fundamentals.index, fundamentals["종목명"]))

if long_signals:
    print(f"\n  {'순위':>4} {'종목명':>20} {'코드':>8} {'비중':>8}")
    print(f"  {'-'*4} {'-'*20} {'-'*8} {'-'*8}")
    for i, (ticker, weight) in enumerate(
        sorted(long_signals.items(), key=lambda x: x[1], reverse=True), 1
    ):
        name = name_map.get(ticker, ticker)
        print(f"  {i:>4} {name:>20} {ticker:>8} {weight:>7.1%}")
    print(f"  {'합계':>35} {sum(long_signals.values()):>7.1%}")

# ── 3. ETF 로테이션 시그널 ──
print(f"\n[3] ETF 로테이션 시그널 생성 (12M lookback, top3)...")

etf_prices = {}
for ticker in ETF_UNIVERSE:
    try:
        df = get_price_data(ticker, start_date, DATA_DATE)
        if df is not None and not df.empty:
            etf_prices[ticker] = df
    except Exception:
        pass

print(f"  ETF 가격 데이터: {len(etf_prices)}/{len(ETF_UNIVERSE)}종목")

etf_strategy = ETFRotationStrategy(
    etf_universe=ETF_UNIVERSE,
    lookback=252,
    num_etfs=3,
    safe_asset="439870",
)

etf_signals = etf_strategy.generate_signals(NEXT_TRADING_DATE, {"etf_prices": etf_prices})
print(f"  ETF 시그널 수: {len(etf_signals)}개")

# ETF 모멘텀 순위표
diag = etf_strategy.last_diagnostics
per_ticker = diag.get("per_ticker", {})
if per_ticker:
    print(f"\n  {'순위':>4} {'ETF명':>30} {'코드':>8} {'모멘텀':>10} {'상태':>8}")
    print(f"  {'-'*4} {'-'*30} {'-'*8} {'-'*10} {'-'*8}")
    mom_list = [
        (t, info.get("momentum", float("-inf")), info.get("status", "?"))
        for t, info in per_ticker.items()
    ]
    mom_list.sort(key=lambda x: x[1], reverse=True)
    for i, (ticker, mom, status) in enumerate(mom_list, 1):
        name = ETF_UNIVERSE.get(ticker, ticker)
        selected = "*" if ticker in etf_signals else " "
        mom_str = f"{mom:+.1%}" if mom != float("-inf") else "N/A"
        print(f"  {i:>3}{selected} {name:>30} {ticker:>8} {mom_str:>10} {status:>8}")

if etf_signals:
    print(f"\n  선정 ETF:")
    for ticker, weight in sorted(etf_signals.items(), key=lambda x: x[1], reverse=True):
        name = ETF_UNIVERSE.get(ticker, ticker)
        print(f"    {name}({ticker}): {weight:.1%}")

# ── 4. 풀별 시그널 병합 (장기 70% + ETF 30%) ──
print(f"\n[4] 풀별 시그널 병합 (장기 {LONG_TERM_PCT:.0%} + ETF {ETF_ROTATION_PCT:.0%})...")

# PortfolioAllocator (KIS 미사용 - 비중 계산만)
class MockKIS:
    def get_balance(self):
        return {"total_eval": 0, "cash": 0, "holdings": []}
    def get_current_price(self, ticker):
        return {"price": 0}

mock_allocator = PortfolioAllocator(
    kis_client=MockKIS(),
    long_term_pct=LONG_TERM_PCT,
    short_term_pct=0.0,
    etf_rotation_pct=ETF_ROTATION_PCT,
    allocation_path="/tmp/sim_allocation.json",
)

pool_signals = {}
if long_signals:
    pool_signals["long_term"] = long_signals
if etf_signals:
    pool_signals["etf_rotation"] = etf_signals

merged = mock_allocator.merge_pool_targets(pool_signals)
print(f"  병합 종목 수: {len(merged)}개")
print(f"  비중 합계: {sum(merged.values()):.1%}")

print(f"\n  {'순위':>4} {'풀':>6} {'종목명':>30} {'코드':>8} {'비중':>8}")
print(f"  {'-'*4} {'-'*6} {'-'*30} {'-'*8} {'-'*8}")
for i, (ticker, weight) in enumerate(
    sorted(merged.items(), key=lambda x: x[1], reverse=True), 1
):
    pool = "ETF" if ticker in ETF_UNIVERSE else "장기"
    name = ETF_UNIVERSE.get(ticker, name_map.get(ticker, ticker))
    print(f"  {i:>4} {pool:>6} {name:>30} {ticker:>8} {weight:>7.1%}")

# ── 5. 리스크 체크 ──
print(f"\n[5] 리스크 체크 (실전 모드)...")
risk_guard = RiskGuard(is_live=True)
passed, warnings = risk_guard.check_rebalance(merged)
print(f"  비중 검증: {'PASS' if passed else 'FAIL'}")
if warnings:
    for w in warnings:
        print(f"    - {w}")

# ── 6. 현재 보유 포지션 vs 타겟 Diff ──
print(f"\n[6] 현재 보유 포지션 조회...")
kis = KISClient()
portfolio_value = 0
holdings = []

try:
    balance = kis.get_balance()
    portfolio_value = balance.get("total_eval", 0)
    cash = balance.get("cash", 0)
    holdings = balance.get("holdings", [])

    print(f"  총 평가금액: {portfolio_value:,}원")
    print(f"  현금: {cash:,}원")
    print(f"  보유종목: {len(holdings)}개")

    for h in holdings:
        ticker = h.get("ticker", "")
        name = h.get("name", name_map.get(ticker, ticker))
        qty = h.get("qty", 0)
        value = h.get("eval_amount", 0)
        pnl_pct = h.get("pnl_pct", 0.0)
        pct = value / portfolio_value * 100 if portfolio_value > 0 else 0
        print(f"    {name}({ticker}): {qty}주, {value:,}원 ({pct:.1f}%), 수익률 {pnl_pct:+.1f}%")
except Exception as e:
    print(f"  잔고 조회 실패: {e}")

# ── 7. 리밸런싱 주문 계산 (Dry Run) ──
if portfolio_value > 0:
    print(f"\n[7] 리밸런싱 주문 계산 (Dry Run)...")
    pm = PositionManager(kis)

    try:
        sell_orders, buy_orders = pm.calculate_rebalance_orders(
            merged, allocator=None, pool=None, integrated=True
        )

        total_sell = sum(o.get("amount", 0) for o in sell_orders)
        total_buy = sum(o.get("amount", 0) for o in buy_orders)

        if sell_orders:
            print(f"\n  매도 예상: {len(sell_orders)}건 (총 {total_sell:,}원)")
            for o in sell_orders:
                ticker = o.get("ticker", "")
                name = ETF_UNIVERSE.get(ticker, name_map.get(ticker, ticker))
                qty = o.get("qty", 0)
                amount = o.get("amount", 0)
                print(f"    매도 {name}({ticker}): {qty}주, {amount:,}원")

        if buy_orders:
            print(f"\n  매수 예상: {len(buy_orders)}건 (총 {total_buy:,}원)")
            for o in buy_orders:
                ticker = o.get("ticker", "")
                name = ETF_UNIVERSE.get(ticker, name_map.get(ticker, ticker))
                qty = o.get("qty", 0)
                amount = o.get("amount", 0)
                print(f"    매수 {name}({ticker}): {qty}주, {amount:,}원")

        if not sell_orders and not buy_orders:
            print("  변경 없음 (현재 = 목표)")

        # ── 8. Turnover 체크 ──
        if sell_orders or buy_orders:
            turnover = (total_sell + total_buy) / portfolio_value if portfolio_value > 0 else 0
            passed_t, reason_t = risk_guard.check_turnover(sell_orders, buy_orders, portfolio_value)
            print(f"\n[8] Turnover 체크")
            print(f"  회전율: {turnover:.1%} (한도: {risk_guard.max_daily_turnover:.0%})")
            print(f"  결과: {'PASS' if passed_t else 'FAIL'}")
            if reason_t:
                print(f"    {reason_t}")

            # 순 현금 흐름
            net_flow = total_sell - total_buy
            print(f"\n  순 현금 흐름: {net_flow:+,}원")
            if net_flow < 0:
                print(f"  추가 현금 필요: {abs(net_flow):,}원")
                if cash >= abs(net_flow):
                    print(f"  현금 잔고 충분 ({cash:,}원 >= {abs(net_flow):,}원)")
                else:
                    print(f"  현금 부족! ({cash:,}원 < {abs(net_flow):,}원)")

    except Exception as e:
        print(f"  주문 계산 실패: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n[7] 포트폴리오 가치 0원 - 주문 계산 스킵")

# ── 요약 ──
print(f"\n{'=' * 60}")
print(f"  시뮬레이션 요약")
print(f"  - 데이터 기준: {DATA_DATE} 종가")
print(f"  - 다음 거래일: {NEXT_TRADING_DATE}")
print(f"  - 장기 {len(long_signals)}종목 ({LONG_TERM_PCT:.0%})")
print(f"  - ETF {len(etf_signals)}종목 ({ETF_ROTATION_PCT:.0%})")
print(f"  - 통합 {len(merged)}종목 (비중합 {sum(merged.values()):.1%})")
print(f"  - 리스크: {'PASS' if passed else 'FAIL'}")
if portfolio_value > 0:
    print(f"  - 포트폴리오: {portfolio_value:,}원")
print("=" * 60)
