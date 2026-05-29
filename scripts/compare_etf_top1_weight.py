"""JAE-64: ETF 로테이션 top-1 가중치 비교 시뮬레이션.

equal weight vs top1_weight=0.50 vs top1_weight=0.55
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.etf_rotation import ETFRotationStrategy, ETF_SECTOR_MAP


SAFE_ASSET = "439870"

# 섹터별로 다른 3개 ETF (max_same_sector=1 통과 보장)
#   069500: 국내지수, 360750: 미국지수, 091160: 반도체
TEST_UNIVERSE = {
    "069500": "KODEX 200",
    "360750": "TIGER 미국S&P500",
    "091160": "KODEX 반도체",
    SAFE_ASSET: "KODEX 단기채권",
}


def make_prices(annual_returns: dict, n_days: int = 400) -> dict:
    """ETF 합성 가격 데이터를 생성한다."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    out = {}
    for ticker, ann_ret in annual_returns.items():
        daily = (1 + ann_ret) ** (1 / 252) - 1
        noise = np.random.normal(0, 0.012, n_days)
        prices = 10000 * np.cumprod(1 + daily + noise)
        out[ticker] = pd.DataFrame(
            {"close": prices, "volume": [1_000_000] * n_days},
            index=dates,
        )
    # 안전자산: 거의 변동 없음
    out[SAFE_ASSET] = pd.DataFrame(
        {"close": np.linspace(10000, 10200, n_days), "volume": [1_000_000] * n_days},
        index=dates,
    )
    return out


def simulate(top1_weight: float, prices: dict, n_periods: int = 20) -> dict:
    """월별 리밸런싱 포트폴리오를 시뮬레이션한다."""
    strategy = ETFRotationStrategy(
        num_etfs=3,
        lookback=63,
        weighting="equal",
        top1_weight=top1_weight,
        safe_asset=SAFE_ASSET,
        etf_universe=TEST_UNIVERSE,
        abs_momentum=False,
        max_same_sector=1,
    )

    risky_tickers = [t for t in TEST_UNIVERSE if t != SAFE_ASSET]
    all_dates = list(prices[risky_tickers[0]].index)
    rebal_dates = all_dates[63::21][:n_periods]

    portfolio_value = 100.0
    returns = []

    for i, date in enumerate(rebal_dates[:-1]):
        etf_at_date = {t: df[df.index <= date] for t, df in prices.items()}
        signals = strategy.generate_signals(
            date.strftime("%Y%m%d"), {"etf_prices": etf_at_date},
        )
        next_date = rebal_dates[i + 1]

        period_return = 0.0
        for ticker, weight in signals.items():
            if ticker in prices:
                p_now  = float(prices[ticker].loc[prices[ticker].index <= date, "close"].iloc[-1])
                p_next = float(prices[ticker].loc[prices[ticker].index <= next_date, "close"].iloc[-1])
                period_return += weight * (p_next / p_now - 1)

        portfolio_value *= (1 + period_return)
        returns.append(period_return)

    arr = np.array(returns)
    n_years = len(arr) / 12
    cagr = (portfolio_value / 100) ** (1 / max(n_years, 0.01)) - 1
    sharpe = (arr.mean() / arr.std() * np.sqrt(12)) if arr.std() > 0 else 0
    cum = np.cumprod(1 + arr) * 100
    mdd = float(((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min()) * 100

    return {"cagr": cagr * 100, "sharpe_ratio": sharpe, "mdd": mdd}


def verify_weight_split():
    """비중 분배 수식 검증 (3 ETF 선택 시 50/25/25)."""
    print("\n[비중 분배 검증 — 3섹터 ETF, no abs_momentum]")
    prices = make_prices({"069500": 0.20, "360750": 0.10, "091160": 0.05})

    for tw in [0.0, 0.50, 0.55]:
        strategy = ETFRotationStrategy(
            num_etfs=3, lookback=63, weighting="equal",
            top1_weight=tw, safe_asset=SAFE_ASSET,
            etf_universe=TEST_UNIVERSE, abs_momentum=False, max_same_sector=1,
        )
        etf_data = {t: df.head(100) for t, df in {**prices,
            SAFE_ASSET: pd.DataFrame({"close": [10000]*100, "volume": [0]*100},
                index=pd.date_range("2022-01-03", periods=100, freq="B"))}.items()}
        signals = strategy.generate_signals("20230101", {"etf_prices": etf_data})
        total = sum(signals.values())
        sorted_w = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        print(f"  top1_weight={tw:.2f}: " +
              " | ".join(f"{t}={w:.1%}" for t, w in sorted_w) +
              f"  (합계={total:.3f})")


def run_comparison():
    # 시나리오: top-1이 명확히 우월한 경우 (top-1이 다른 두 ETF보다 2배 이상 수익)
    prices_bull = make_prices({"069500": 0.30, "360750": 0.10, "091160": 0.08})

    configs = [
        ("Equal(33/33/33)", 0.0),
        ("Top1-50(50/25/25)", 0.50),
        ("Top1-55(55/22.5/22.5)", 0.55),
    ]

    print(f"\n" + "=" * 68)
    print("JAE-64: ETF 로테이션 top-1 가중치 비교 시뮬레이션")
    print("합성 데이터 — 상승장: top-1 +30%/yr, top-2 +10%, top-3 +8%")
    print("=" * 68)
    print(f"{'전략':<32} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8}")
    print("-" * 68)

    results = []
    for label, tw in configs:
        r = simulate(top1_weight=tw, prices=prices_bull)
        results.append((label, tw, r))
        print(f"{label:<32} {r['cagr']:>7.1f}% {r['sharpe_ratio']:>8.2f} {r['mdd']:>7.1f}%")

    print("=" * 68)
    baseline = results[0][2]
    print("\n[Equal 대비 개선폭]")
    for label, tw, r in results[1:]:
        dc = r['cagr'] - baseline['cagr']
        ds = r['sharpe_ratio'] - baseline['sharpe_ratio']
        dm = r['mdd'] - baseline['mdd']
        print(f"  {label}: CAGR {'+' if dc>=0 else ''}{dc:.2f}pp, "
              f"Sharpe {'+' if ds>=0 else ''}{ds:.3f}, "
              f"MDD {'+' if dm>=0 else ''}{dm:.2f}pp")

    verify_weight_split()


if __name__ == "__main__":
    run_comparison()
