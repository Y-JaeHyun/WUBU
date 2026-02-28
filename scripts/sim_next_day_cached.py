"""다음 거래일 리밸런싱 시뮬레이션 (캐시 기반).

캐시된 펀더멘탈(2/27 종가) + pykrx OHLCV로 시뮬레이션.
3-Pool: 장기 70% (MultiFactor) + ETF 30% (ETFRotation)
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from pykrx import stock as pykrx_stock

from src.strategy.multi_factor import MultiFactorStrategy
from src.strategy.etf_rotation import ETFRotationStrategy
from src.execution.portfolio_allocator import PortfolioAllocator
from src.execution.position_manager import PositionManager
from src.execution.risk_guard import RiskGuard
from src.execution.kis_client import KISClient

# ── 설정 ──
DATA_DATE = "20260227"
NEXT_TRADING_DATE = "20260302"
LONG_TERM_PCT = 0.70
ETF_ROTATION_PCT = 0.30
CACHE_FUND_PATH = "/mnt/data/quant/data/cache/4ea16834afc34c5f93fdbe2758445269.parquet"

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
print(f"  데이터 기준: T-1 = {DATA_DATE} 종가 (캐시)")
print(f"  풀 배분: 장기 {LONG_TERM_PCT:.0%} + ETF {ETF_ROTATION_PCT:.0%}")
print("=" * 60)

# ── 1. 펀더멘탈 데이터 (캐시) ──
print(f"\n[1] 펀더멘탈 데이터 (캐시에서 로드)...")
fundamentals = pd.read_parquet(CACHE_FUND_PATH)
print(f"  전체: {len(fundamentals)}종목")
print(f"  컬럼: {list(fundamentals.columns)}")

# 종목명 매핑
name_map = dict(zip(fundamentals["ticker"], fundamentals["name"]))

# ── 2. 가격 데이터 수집 (시총 상위 200, 400일) ──
print(f"\n[2] 가격 데이터 수집 (시총 상위 200, OHLCV)...")
start_dt = datetime.strptime(DATA_DATE, "%Y%m%d") - timedelta(days=400)
start_date = start_dt.strftime("%Y%m%d")

top200 = fundamentals.nlargest(200, "market_cap")
top_tickers = top200["ticker"].tolist()
print(f"  시총 1위: {top200.iloc[0]['name']} ({top200.iloc[0]['market_cap']:,.0f}원)")
print(f"  시총 200위: {top200.iloc[-1]['name']} ({top200.iloc[-1]['market_cap']:,.0f}원)")

prices = {}
errors = 0
for i, ticker in enumerate(top_tickers):
    try:
        df = pykrx_stock.get_market_ohlcv_by_date(start_date, DATA_DATE, ticker)
        if not df.empty:
            df = df.rename(columns={
                "시가": "open", "고가": "high", "저가": "low",
                "종가": "close", "거래량": "volume",
            })
            prices[ticker] = df
    except Exception:
        errors += 1
    if (i + 1) % 50 == 0:
        print(f"  진행: {i+1}/200 ({len(prices)}종목 수집완료)")
        time.sleep(1)

print(f"  가격 데이터: {len(prices)}/200 종목 (에러: {errors})")

# ── 3. KOSPI 지수 (KODEX200 대용 - pykrx index API 장애) ──
print(f"\n[3] KOSPI 지수 수집 (KODEX200 대용)...")
try:
    kospi_df = pykrx_stock.get_market_ohlcv_by_date(start_date, DATA_DATE, "069500")
    if not kospi_df.empty:
        index_prices = kospi_df["종가"]
        print(f"  KODEX200 → KOSPI 대용: {len(index_prices)}일")
        print(f"  최근 종가: {index_prices.iloc[-1]:,} (2/27)")
    else:
        index_prices = None
        print("  KODEX200 데이터 없음")
except Exception as e:
    index_prices = None
    print(f"  KOSPI 지수 수집 실패 ({e})")

strategy_data = {
    "fundamentals": fundamentals,
    "prices": prices,
    "index_prices": index_prices,
}

# ── 4. 장기 전략 시그널 (MultiFactor) ──
print(f"\n[4] 장기 전략 시그널 생성 (MultiFactor V0.4+M0.6, top10, MT)...")
long_strategy = MultiFactorStrategy(
    factors=["value", "momentum"],
    weights=[0.4, 0.6],
    num_stocks=10,
    turnover_penalty=0.1,
    apply_market_timing=True,
)

long_signals = long_strategy.generate_signals(NEXT_TRADING_DATE, strategy_data)
print(f"  장기 종목 수: {len(long_signals)}개")

if long_signals:
    print(f"\n  {'순위':>4} {'종목명':>20} {'코드':>8} {'비중':>8}")
    print(f"  {'-'*4} {'-'*20} {'-'*8} {'-'*8}")
    for i, (ticker, weight) in enumerate(
        sorted(long_signals.items(), key=lambda x: x[1], reverse=True), 1
    ):
        name = name_map.get(ticker, ticker)
        print(f"  {i:>4} {name:>20} {ticker:>8} {weight:>7.1%}")
    print(f"  {'합계':>35} {sum(long_signals.values()):>7.1%}")

# ── 5. ETF 로테이션 시그널 ──
print(f"\n[5] ETF 로테이션 시그널 생성 (12M lookback, top3)...")

etf_prices = {}
for ticker, etf_name in ETF_UNIVERSE.items():
    try:
        df = pykrx_stock.get_market_ohlcv_by_date(start_date, DATA_DATE, ticker)
        if not df.empty:
            df = df.rename(columns={
                "시가": "open", "고가": "high", "저가": "low",
                "종가": "close", "거래량": "volume",
            })
            etf_prices[ticker] = df
    except Exception:
        pass
    time.sleep(0.3)

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
    print(f"\n  {'순위':>4} {'ETF명':>35} {'코드':>8} {'모멘텀':>10} {'상태':>8}")
    print(f"  {'-'*4} {'-'*35} {'-'*8} {'-'*10} {'-'*8}")
    mom_list = [
        (t, info.get("momentum", float("-inf")), info.get("status", "?"))
        for t, info in per_ticker.items()
    ]
    mom_list.sort(key=lambda x: x[1], reverse=True)
    for i, (ticker, mom, status) in enumerate(mom_list, 1):
        name = ETF_UNIVERSE.get(ticker, ticker)
        selected = "*" if ticker in etf_signals else " "
        mom_str = f"{mom:+.1%}" if mom != float("-inf") else "N/A"
        print(f"  {i:>3}{selected} {name:>35} {ticker:>8} {mom_str:>10} {status:>8}")

if etf_signals:
    print(f"\n  선정 ETF:")
    for ticker, weight in sorted(etf_signals.items(), key=lambda x: x[1], reverse=True):
        name = ETF_UNIVERSE.get(ticker, ticker)
        print(f"    {name}({ticker}): {weight:.1%}")

# ── 6. 풀별 시그널 병합 ──
print(f"\n[6] 풀별 시그널 병합 (장기 {LONG_TERM_PCT:.0%} + ETF {ETF_ROTATION_PCT:.0%})...")

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

print(f"\n  {'순위':>4} {'풀':>6} {'종목명':>35} {'코드':>8} {'비중':>8}")
print(f"  {'-'*4} {'-'*6} {'-'*35} {'-'*8} {'-'*8}")
for i, (ticker, weight) in enumerate(
    sorted(merged.items(), key=lambda x: x[1], reverse=True), 1
):
    pool = "ETF" if ticker in ETF_UNIVERSE else "장기"
    name = ETF_UNIVERSE.get(ticker, name_map.get(ticker, ticker))
    print(f"  {i:>4} {pool:>6} {name:>35} {ticker:>8} {weight:>7.1%}")

# ── 7. 리스크 체크 ──
print(f"\n[7] 리스크 체크 (실전 모드)...")
risk_guard = RiskGuard(is_live=True)
passed, warnings = risk_guard.check_rebalance(merged)
print(f"  비중 검증: {'PASS' if passed else 'FAIL'}")
if warnings:
    for w in warnings:
        print(f"    - {w}")

# ── 8. 현재 보유 포지션 조회 + Dry Run ──
print(f"\n[8] 현재 보유 포지션 조회 (KIS API)...")
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
        avg_price = h.get("avg_price", 0)
        pct = value / portfolio_value * 100 if portfolio_value > 0 else 0
        print(f"    {name}({ticker}): {qty}주 x 평단{avg_price:,}원, "
              f"평가 {value:,}원 ({pct:.1f}%), 수익률 {pnl_pct:+.2f}%")
except Exception as e:
    print(f"  잔고 조회 실패: {e}")

# ── 9. 매도/매수 주문 계산 ──
if portfolio_value > 0:
    print(f"\n[9] 리밸런싱 주문 계산 (Dry Run)...")
    pm = PositionManager(kis)

    try:
        sell_orders, buy_orders = pm.calculate_rebalance_orders(
            merged, allocator=None, pool=None, integrated=True
        )

        total_sell = sum(o.get("amount", 0) for o in sell_orders)
        total_buy = sum(o.get("amount", 0) for o in buy_orders)

        if sell_orders:
            print(f"\n  [매도 예상] {len(sell_orders)}건 (총 {total_sell:,}원)")
            for o in sell_orders:
                ticker = o.get("ticker", "")
                name = ETF_UNIVERSE.get(ticker, name_map.get(ticker, ticker))
                qty = o.get("qty", 0)
                amount = o.get("amount", 0)
                print(f"    매도 {name}({ticker}): {qty}주, {amount:,}원")

        if buy_orders:
            print(f"\n  [매수 예상] {len(buy_orders)}건 (총 {total_buy:,}원)")
            for o in buy_orders:
                ticker = o.get("ticker", "")
                name = ETF_UNIVERSE.get(ticker, name_map.get(ticker, ticker))
                qty = o.get("qty", 0)
                amount = o.get("amount", 0)
                print(f"    매수 {name}({ticker}): {qty}주, {amount:,}원")

        if not sell_orders and not buy_orders:
            print("  변경 없음 (현재 = 목표)")

        # Turnover 체크
        if sell_orders or buy_orders:
            turnover = (total_sell + total_buy) / portfolio_value if portfolio_value > 0 else 0
            passed_t, reason_t = risk_guard.check_turnover(sell_orders, buy_orders, portfolio_value)
            print(f"\n  [Turnover 체크]")
            print(f"  회전율: {turnover:.1%} (한도: {risk_guard.max_daily_turnover:.0%})")
            print(f"  결과: {'PASS' if passed_t else 'FAIL'}")
            if reason_t:
                print(f"    {reason_t}")

            net_flow = total_sell - total_buy
            print(f"\n  순 현금 흐름: {net_flow:+,}원")
            if net_flow < 0:
                if cash >= abs(net_flow):
                    print(f"  현금 충분 ({cash:,}원 >= {abs(net_flow):,}원)")
                else:
                    print(f"  현금 부족! ({cash:,}원 < {abs(net_flow):,}원)")
    except Exception as e:
        print(f"  주문 계산 실패: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n[9] 포트폴리오 가치 0원 - 주문 계산 스킵")

# ── 요약 ──
print(f"\n{'=' * 60}")
print(f"  시뮬레이션 요약")
print(f"  - 데이터 기준: {DATA_DATE} 종가 (캐시)")
print(f"  - 다음 거래일: {NEXT_TRADING_DATE}")
print(f"  - 장기 {len(long_signals)}종목 ({LONG_TERM_PCT:.0%})")
print(f"  - ETF {len(etf_signals)}종목 ({ETF_ROTATION_PCT:.0%})")
print(f"  - 통합 {len(merged)}종목 (비중합 {sum(merged.values()):.1%})")
print(f"  - 리스크: {'PASS' if passed else 'FAIL'}")
if portfolio_value > 0:
    print(f"  - 포트폴리오: {portfolio_value:,}원")
print("=" * 60)
