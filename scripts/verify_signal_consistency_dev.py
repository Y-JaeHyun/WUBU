#!/usr/bin/env python3
"""
4경로 시그널 일치 검증 (dev 환경 전용):
  A) premarket_check 경로 (T-1 데이터, live 프로필)
  B) /balance 경로 (T-1 데이터, live 프로필)
  C) daily_simulation 경로 (T 데이터, live 프로필)
  D) backtest 엔진 경로 (live 프로필, 동일 데이터)

운영환경(/mnt/data/quant)을 참조하지 않고,
dev 환경(/mnt/data/quant-dev)에서 독립 실행한다.
"""
import sys
import os

# dev 환경 설정
DEV_DIR = "/mnt/data/quant-dev"
sys.path.insert(0, DEV_DIR)
os.chdir(DEV_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(DEV_DIR, ".env"))

from datetime import datetime
from src.strategy.strategy_config import create_multi_factor
from src.data.collector import get_all_fundamentals, get_price_data
from src.data.daily_simulator import DailySimulator
from src.backtest.engine import Backtest


def collect_strategy_data(data_date: str) -> dict:
    """전략 데이터를 수집한다 (scheduler._collect_strategy_data 간소 버전)."""
    fundamentals = get_all_fundamentals(data_date)
    prices: dict = {}
    index_prices = None

    if fundamentals is not None and not fundamentals.empty:
        tickers = fundamentals["ticker"].tolist()[:200]
        for ticker in tickers:
            try:
                df = get_price_data(ticker, period=400)
                if df is not None and not df.empty:
                    prices[ticker] = df
            except Exception:
                pass

        try:
            from pykrx import stock as pykrx_stock
            import pandas as pd
            end = data_date
            start = (pd.Timestamp(data_date) - pd.tseries.offsets.BDay(400)).strftime("%Y%m%d")
            kospi = pykrx_stock.get_index_ohlcv(start, end, "1001")
            if not kospi.empty:
                index_prices = kospi["종가"]
        except Exception:
            pass

    import pandas as pd
    return {
        "fundamentals": fundamentals if fundamentals is not None else pd.DataFrame(),
        "prices": prices,
        "index_prices": index_prices if index_prices is not None else pd.Series(dtype=float),
    }


def print_signals(label: str, signals: dict, name_map: dict) -> None:
    """시그널을 정렬 출력한다."""
    print(f"  [{label}] {len(signals)}종목")
    for i, (t, w) in enumerate(
        sorted(signals.items(), key=lambda x: x[1], reverse=True), 1
    ):
        print(f"    {i}. {name_map.get(t, t)}({t}): {w:.1%}")


def compare_signals(label: str, sig_a: dict, sig_b: dict, name_map: dict) -> bool:
    """두 시그널의 종목 일치와 비중 일치를 비교한다."""
    tickers_a = set(sig_a.keys())
    tickers_b = set(sig_b.keys())

    tickers_match = tickers_a == tickers_b
    if tickers_match:
        print(f"  {label} 종목: OK 일치")
    else:
        print(f"  {label} 종목: MISMATCH 불일치")
        only_a = tickers_a - tickers_b
        only_b = tickers_b - tickers_a
        if only_a:
            print(f"    좌측에만: {[name_map.get(t, t) for t in only_a]}")
        if only_b:
            print(f"    우측에만: {[name_map.get(t, t) for t in only_b]}")

    weights_match = all(
        abs(sig_a.get(t, 0) - sig_b.get(t, 0)) < 0.001
        for t in tickers_a | tickers_b
    )
    if weights_match:
        print(f"  {label} 비중: OK 일치")
    else:
        print(f"  {label} 비중: MISMATCH 불일치")
        for t in tickers_a | tickers_b:
            wa = sig_a.get(t, 0)
            wb = sig_b.get(t, 0)
            if abs(wa - wb) >= 0.001:
                print(f"    {name_map.get(t, t)}: {wa:.3%} vs {wb:.3%}")

    return tickers_match and weights_match


def verify_4path(target_date: str) -> bool:
    """4경로 시그널 정합성을 검증한다.

    Args:
        target_date: 검증할 날짜 (YYYYMMDD).

    Returns:
        4경로 모두 일치하면 True.
    """
    import pandas as pd
    from src.scheduler.holidays import KoreanHolidays

    holidays = KoreanHolidays()
    target = datetime.strptime(target_date, "%Y%m%d").date()

    # T-1 데이터 (Path A, B)
    prev_day = holidays.prev_trading_day(target)
    data_date_t1 = prev_day.strftime("%Y%m%d")

    print(f"=== 4경로 시그널 정합성 검증 ({target_date}) ===")
    print(f"  T-1 데이터: {data_date_t1}")
    print()

    # 데이터 수집
    print("데이터 수집 중...")
    strategy_data_t1 = collect_strategy_data(data_date_t1)
    strategy_data_t = collect_strategy_data(target_date)

    fund = strategy_data_t1.get("fundamentals")
    name_map = {}
    if fund is not None and not fund.empty and "ticker" in fund.columns:
        name_map = dict(zip(fund["ticker"], fund["name"]))

    # 전략 생성 (live 프로필, 동일 인스턴스)
    strategy = create_multi_factor("live")
    num_stocks = getattr(strategy, "num_stocks", 7)
    scan_limit = num_stocks * 2

    # ─── Path A: premarket_check 경로 ───
    print("=" * 55)
    print("[ A: premarket_check 경로 ]")
    signals_a = strategy.generate_signals(
        target_date, strategy_data_t1, scan_limit=scan_limit
    )
    print_signals("premarket", signals_a, name_map)

    # ─── Path B: /balance 경로 ───
    # 동일 전략 인스턴스, 동일 데이터 → A와 동일해야 함
    print()
    print("=" * 55)
    print("[ B: /balance 경로 ]")
    signals_b = strategy.generate_signals(
        target_date, strategy_data_t1, scan_limit=scan_limit
    )
    print_signals("/balance", signals_b, name_map)

    # ─── Path C: daily_simulation 경로 ───
    print()
    print("=" * 55)
    print("[ C: daily_simulation 경로 ]")
    simulator = DailySimulator()
    simulator.strategy_data = strategy_data_t1  # T-1 데이터 사용 (일관성 위해)
    signals_c = simulator._generate_signals(
        strategy, target_date, scan_limit=scan_limit
    )
    print_signals("daily_sim", signals_c, name_map)

    # ─── Path D: backtest 엔진 경로 ───
    print()
    print("=" * 55)
    print("[ D: backtest 엔진 경로 ]")
    bt_strategy = create_multi_factor("live")
    # 이전 보유 상태를 동기화 (turnover penalty 일관성)
    if hasattr(strategy, "_prev_holdings") and hasattr(bt_strategy, "_prev_holdings"):
        bt_strategy._prev_holdings = strategy._prev_holdings.copy()
    bt = Backtest(
        strategy=bt_strategy,
        start_date=data_date_t1,
        end_date=target_date,
    )
    signals_d = bt.extract_signals_for_date(target_date)
    print_signals("backtest", signals_d, name_map)

    # ─── 비교 ───
    print()
    print("=" * 55)
    print("[ 4경로 비교 결과 ]")

    ok_ab = compare_signals("A vs B", signals_a, signals_b, name_map)
    ok_ac = compare_signals("A vs C", signals_a, signals_c, name_map)
    ok_ad = compare_signals("A vs D", signals_a, signals_d, name_map)

    all_ok = ok_ab and ok_ac and ok_ad
    print()
    if all_ok:
        print("결과: 4경로 시그널 100% 일치")
    else:
        print("결과: 시그널 불일치 감지")
        if not ok_ab:
            print("  - A vs B 불일치: premarket과 /balance 경로 차이")
        if not ok_ac:
            print("  - A vs C 불일치: premarket과 daily_simulation 경로 차이")
        if not ok_ad:
            print("  - A vs D 불일치: premarket과 backtest 엔진 경로 차이")

    return all_ok


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        date = _sys.argv[1].replace("-", "")
    else:
        date = datetime.now().strftime("%Y%m%d")

    ok = verify_4path(date)
    _sys.exit(0 if ok else 1)
