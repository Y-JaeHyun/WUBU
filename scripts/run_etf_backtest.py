#!/usr/bin/env python3
"""ETF 로테이션 전략 백테스트 + 파라미터 그리드 서치.

ETFRotationStrategy의 lookback_months, num_etfs 조합을 백테스트하고
최적 파라미터를 탐색한다.

사용법:
    python scripts/run_etf_backtest.py
    python scripts/run_etf_backtest.py --start 20200101 --end 20251231 --capital 1450000
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.etf_collector import get_etf_price
from src.strategy.etf_rotation import ETFRotationStrategy, DEFAULT_ETF_UNIVERSE
from src.utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "etf_backtest_cache"
RESULT_PATH = PROJECT_ROOT / "data" / "etf_backtest_results.json"

# 그리드 서치 파라미터
LOOKBACK_MONTHS_GRID = [3, 6, 9, 12]
NUM_ETFS_GRID = [1, 2, 3]
WEIGHTING_GRID = ["equal"]


def load_etf_prices(
    etf_universe: dict[str, str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """ETF 유니버스의 가격 데이터를 로드한다 (parquet 캐시 활용)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    cache_key = f"{start_date}_{end_date}"

    for ticker, name in etf_universe.items():
        cache_file = CACHE_DIR / f"etf_{ticker}_{cache_key}.parquet"

        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    result[ticker] = df
                    logger.debug("캐시 HIT: %s (%s)", ticker, name)
                    continue
            except Exception:
                pass

        try:
            logger.info("ETF 가격 수집: %s (%s)", ticker, name)
            df = get_etf_price(ticker, start_date, end_date)
            if not df.empty:
                df.to_parquet(cache_file)
                result[ticker] = df
        except Exception as e:
            logger.warning("ETF %s (%s) 로드 실패: %s", ticker, name, e)

        time.sleep(0.5)

    logger.info("ETF 데이터 로드: %d/%d", len(result), len(etf_universe))
    return result


def run_etf_backtest(
    strategy: ETFRotationStrategy,
    etf_prices: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    initial_capital: int,
    buy_cost: float = 0.00015,
    sell_cost: float = 0.00015,  # ETF: 거래세 없음 (수수료만)
) -> dict:
    """ETF 로테이션 전략 백테스트를 실행한다."""
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    if dates.empty:
        return {"error": "유효한 영업일 없음"}

    # 월별 첫 거래일 = 리밸런싱일
    rebal_series = dates.to_series().groupby(dates.to_period("M")).first()
    rebalance_dates = set(d.strftime("%Y%m%d") for d in rebal_series)

    cash = float(initial_capital)
    holdings: dict[str, int] = {}  # ticker -> qty
    portfolio_values = []
    trade_count = 0

    def get_price_on_date(ticker: str, date_str: str):
        if ticker not in etf_prices:
            return None
        df = etf_prices[ticker]
        if df.empty or "close" not in df.columns:
            return None
        target = pd.Timestamp(date_str)
        if target in df.index:
            return float(df.loc[target, "close"])
        mask = df.index <= target
        if mask.any():
            return float(df.loc[mask].iloc[-1]["close"])
        return None

    for date in dates:
        date_str = date.strftime("%Y%m%d")

        if date_str in rebalance_dates:
            # 리밸런싱일까지의 가격만 사용 (look-ahead bias 방지)
            sliced_prices = {}
            for ticker, df in etf_prices.items():
                sliced = df[df.index <= date]
                if not sliced.empty:
                    sliced_prices[ticker] = sliced

            signals = strategy.generate_signals(
                date_str, {"etf_prices": sliced_prices}
            )

            if signals:
                # 포트폴리오 가치 계산
                pv = cash
                for t, q in holdings.items():
                    p = get_price_on_date(t, date_str)
                    if p:
                        pv += q * p

                # 전량 매도
                for t, q in list(holdings.items()):
                    if q > 0:
                        p = get_price_on_date(t, date_str)
                        if p:
                            cash += q * p * (1 - sell_cost)
                            trade_count += 1
                holdings.clear()

                # 시그널대로 매수
                for ticker, weight in signals.items():
                    if weight <= 0:
                        continue
                    p = get_price_on_date(ticker, date_str)
                    if p and p > 0:
                        target_amount = pv * weight
                        buy_qty = int(target_amount / (p * (1 + buy_cost)))
                        if buy_qty > 0:
                            cost = buy_qty * p * (1 + buy_cost)
                            if cost <= cash:
                                cash -= cost
                                holdings[ticker] = holdings.get(ticker, 0) + buy_qty
                                trade_count += 1

        # 일일 포트폴리오 가치
        pv = cash
        for t, q in holdings.items():
            p = get_price_on_date(t, date_str)
            if p:
                pv += q * p
        portfolio_values.append({"date": date_str, "value": pv})

    if not portfolio_values:
        return {"error": "이력 없음"}

    # 성과 지표 계산
    values = pd.Series(
        [h["value"] for h in portfolio_values],
        index=pd.to_datetime([h["date"] for h in portfolio_values]),
    )

    final = float(values.iloc[-1])
    total_return = (final / initial_capital - 1) * 100
    days = (values.index[-1] - values.index[0]).days
    years = days / 365.25 if days > 0 else 1
    cagr = ((final / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    daily_returns = values.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        risk_free_daily = 0.03 / 252
        sharpe = float(
            (daily_returns.mean() - risk_free_daily) / daily_returns.std()
            * np.sqrt(252)
        )
    else:
        sharpe = 0.0

    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    mdd = float(drawdown.min()) * 100

    return {
        "total_return": round(total_return, 2),
        "cagr": round(cagr, 2),
        "sharpe_ratio": round(sharpe, 2),
        "mdd": round(mdd, 2),
        "final_value": round(final, 0),
        "trades": trade_count,
    }


def run_grid_search(
    start_date: str,
    end_date: str,
    initial_capital: int,
    etf_prices: dict[str, pd.DataFrame],
    etf_universe: dict[str, str],
) -> list[dict]:
    """그리드 서치를 실행한다."""
    results = []

    combos = list(product(LOOKBACK_MONTHS_GRID, NUM_ETFS_GRID, WEIGHTING_GRID))
    print(f"\n  그리드 서치: {len(combos)}개 조합\n")

    for i, (lookback_m, num_etfs, weighting) in enumerate(combos):
        lookback_days = lookback_m * 21
        label = f"ETF(lb={lookback_m}m, n={num_etfs}, w={weighting})"
        print(f"  [{i + 1}/{len(combos)}] {label}...", end=" ", flush=True)

        t0 = time.time()

        strategy = ETFRotationStrategy(
            lookback=lookback_days,
            num_etfs=num_etfs,
            etf_universe=etf_universe,
            weighting=weighting,
        )

        res = run_etf_backtest(
            strategy, etf_prices,
            start_date, end_date, initial_capital,
        )

        elapsed = time.time() - t0

        entry = {
            "lookback_months": lookback_m,
            "num_etfs": num_etfs,
            "weighting": weighting,
            **res,
            "elapsed_sec": round(elapsed, 1),
        }
        results.append(entry)

        if "error" not in res:
            print(
                f"({elapsed:.0f}s) | "
                f"수익={res['total_return']:+.1f}% | "
                f"CAGR={res['cagr']:+.1f}% | "
                f"Sharpe={res['sharpe_ratio']:.2f} | "
                f"MDD={res['mdd']:.1f}% | "
                f"거래={res['trades']}"
            )
        else:
            print(f"실패: {res['error']}")

    return results


def print_grid_results(results: list[dict]) -> dict | None:
    """그리드 서치 결과를 출력한다."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("  유효한 결과 없음")
        return None

    sorted_results = sorted(valid, key=lambda x: x["sharpe_ratio"], reverse=True)

    print(f"\n{'=' * 95}")
    print("  ETF 로테이션 그리드 서치 결과 (Sharpe 내림차순)")
    print(f"{'=' * 95}")
    header = (
        f"{'Lookback':>10s} {'N_ETFs':>8s} {'Weight':>10s} "
        f"{'수익률':>10s} {'CAGR':>8s} {'Sharpe':>8s} {'MDD':>8s} {'거래':>6s}"
    )
    print(header)
    print("-" * 95)

    for r in sorted_results:
        print(
            f"{r['lookback_months']:>8d}M "
            f"{r['num_etfs']:>8d} "
            f"{r['weighting']:>10s} "
            f"{r['total_return']:>+9.1f}% "
            f"{r['cagr']:>+7.1f}% "
            f"{r['sharpe_ratio']:>7.2f} "
            f"{r['mdd']:>7.1f}% "
            f"{r['trades']:>6d}"
        )

    print("-" * 95)
    best = sorted_results[0]
    print(
        f"\n  ** 최적: lookback={best['lookback_months']}M, "
        f"num_etfs={best['num_etfs']}, "
        f"Sharpe={best['sharpe_ratio']:.2f}, "
        f"CAGR={best['cagr']:+.1f}%, MDD={best['mdd']:.1f}% **"
    )
    return best


def main():
    parser = argparse.ArgumentParser(
        description="ETF 로테이션 백테스트 그리드 서치"
    )
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--capital", type=int, default=1_450_000)
    args = parser.parse_args()

    print(f"\n{'#' * 80}")
    print("  ETF 로테이션 백테스트 그리드 서치")
    print(
        f"  기간: {args.start} ~ {args.end}, 자본금: {args.capital:,}원"
    )
    print(f"{'#' * 80}")

    # ETF 데이터 로드
    print("\n  [데이터] ETF 가격 로드 중...")
    etf_prices = load_etf_prices(DEFAULT_ETF_UNIVERSE, args.start, args.end)
    loaded = len(etf_prices)
    total = len(DEFAULT_ETF_UNIVERSE)
    print(f"  [데이터] {loaded}/{total}개 ETF 로드 완료\n")

    if loaded == 0:
        print("  ETF 데이터를 로드하지 못했습니다.")
        sys.exit(1)

    # 그리드 서치
    results = run_grid_search(
        args.start, args.end, args.capital,
        etf_prices, DEFAULT_ETF_UNIVERSE,
    )

    best = print_grid_results(results)

    # 결과 저장
    output = {
        "config": {
            "start_date": args.start,
            "end_date": args.end,
            "capital": args.capital,
            "grid": {
                "lookback_months": LOOKBACK_MONTHS_GRID,
                "num_etfs": NUM_ETFS_GRID,
                "weighting": WEIGHTING_GRID,
            },
            "etf_universe": DEFAULT_ETF_UNIVERSE,
        },
        "results": results,
        "best": best,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
