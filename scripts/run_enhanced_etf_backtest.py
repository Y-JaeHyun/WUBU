#!/usr/bin/env python3
"""Enhanced ETF 전략 백테스트 스크립트.

신규 전략들의 백테스트를 실행하고 기존 전략과 비교한다.
캐시된 ETF 가격 데이터를 활용하여 pykrx API 호출 없이 실행 가능하다.

사용법:
    python scripts/run_enhanced_etf_backtest.py
    python scripts/run_enhanced_etf_backtest.py --start 20200101 --end 20251231
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

from src.utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "etf_backtest_cache"
RESULT_PATH = PROJECT_ROOT / "data" / "enhanced_etf_backtest_results.json"

# 기본 ETF 유니버스
DEFAULT_ETF_UNIVERSE = {
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


def load_etf_prices(
    etf_universe: dict[str, str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """ETF 가격 데이터를 캐시에서 로드한다."""
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
                    continue
            except Exception:
                pass
        logger.warning(f"캐시 데이터 없음: {ticker} ({name})")

    logger.info(f"ETF 데이터 로드: {len(result)}/{len(etf_universe)}")
    return result


def run_etf_backtest(
    strategy,
    etf_prices: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    initial_capital: int,
    buy_cost: float = 0.00015,
    sell_cost: float = 0.00015,
) -> dict:
    """ETF 전략 백테스트를 실행한다."""
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    if dates.empty:
        return {"error": "유효한 영업일 없음"}

    rebal_series = dates.to_series().groupby(dates.to_period("M")).first()
    rebalance_dates = set(d.strftime("%Y%m%d") for d in rebal_series)

    cash = float(initial_capital)
    holdings: dict[str, int] = {}
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
            # look-ahead bias 방지: 현재까지의 가격만 사용
            sliced_prices = {}
            for ticker, df in etf_prices.items():
                sliced = df[df.index <= date]
                if not sliced.empty:
                    sliced_prices[ticker] = sliced

            signals = strategy.generate_signals(
                date_str, {"etf_prices": sliced_prices}
            )

            if signals:
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

    # 연도별 수익률
    yearly_returns = {}
    for year in range(values.index[0].year, values.index[-1].year + 1):
        year_data = values[values.index.year == year]
        if len(year_data) > 1:
            yr = (year_data.iloc[-1] / year_data.iloc[0] - 1) * 100
            yearly_returns[str(year)] = round(yr, 2)

    return {
        "total_return": round(total_return, 2),
        "cagr": round(cagr, 2),
        "sharpe_ratio": round(sharpe, 2),
        "mdd": round(mdd, 2),
        "final_value": round(final, 0),
        "trades": trade_count,
        "yearly_returns": yearly_returns,
    }


def build_strategies() -> list[tuple[str, object]]:
    """백테스트할 전략 리스트를 생성한다."""
    from src.strategy.etf_rotation import ETFRotationStrategy
    from src.strategy.enhanced_etf_rotation import EnhancedETFRotationStrategy
    from src.strategy.cross_asset_momentum import CrossAssetMomentumStrategy

    strategies = []

    # ── 기존 전략 (베이스라인) ──
    # 1. 기존 ETF Rotation 3M/3 (현재 최고: Sharpe 1.02)
    strategies.append((
        "Baseline: ETF(3M,top3)",
        ETFRotationStrategy(lookback=63, num_etfs=3),
    ))

    # 2. 기존 ETF Rotation 12M/3
    strategies.append((
        "Baseline: ETF(12M,top3)",
        ETFRotationStrategy(lookback=252, num_etfs=3),
    ))

    # 3. 기존 ETF + inverse_vol
    strategies.append((
        "Baseline: ETF(3M,top3,invol)",
        ETFRotationStrategy(lookback=63, num_etfs=3, weighting="inverse_vol"),
    ))

    # ── Enhanced ETF Rotation ──
    # 4. 기본 Enhanced (모든 필터 ON)
    strategies.append((
        "Enhanced: all_filters",
        EnhancedETFRotationStrategy(
            num_etfs=3,
            use_vol_weight=True,
            use_market_filter=True,
            use_trend_filter=True,
        ),
    ))

    # 5. Enhanced: vol_weight만
    strategies.append((
        "Enhanced: vol_only",
        EnhancedETFRotationStrategy(
            num_etfs=3,
            use_vol_weight=True,
            use_market_filter=False,
            use_trend_filter=False,
        ),
    ))

    # 6. Enhanced: market_filter만
    strategies.append((
        "Enhanced: mkt_only",
        EnhancedETFRotationStrategy(
            num_etfs=3,
            use_vol_weight=False,
            use_market_filter=True,
            use_trend_filter=False,
        ),
    ))

    # 7. Enhanced: vol + market
    strategies.append((
        "Enhanced: vol+mkt",
        EnhancedETFRotationStrategy(
            num_etfs=3,
            use_vol_weight=True,
            use_market_filter=True,
            use_trend_filter=False,
        ),
    ))

    # 8. Enhanced: vol + trend
    strategies.append((
        "Enhanced: vol+trend",
        EnhancedETFRotationStrategy(
            num_etfs=3,
            use_vol_weight=True,
            use_market_filter=False,
            use_trend_filter=True,
        ),
    ))

    # 9. Enhanced: top2 (더 집중)
    strategies.append((
        "Enhanced: top2_all",
        EnhancedETFRotationStrategy(
            num_etfs=2,
            use_vol_weight=True,
            use_market_filter=True,
            use_trend_filter=True,
        ),
    ))

    # 10. Enhanced: 커스텀 모멘텀 가중치 (단기 중시)
    strategies.append((
        "Enhanced: short_mom",
        EnhancedETFRotationStrategy(
            num_etfs=3,
            use_vol_weight=True,
            use_market_filter=True,
            use_trend_filter=True,
            momentum_weights={21: 0.35, 63: 0.35, 126: 0.20, 252: 0.10},
        ),
    ))

    # 11. Enhanced: 커스텀 모멘텀 가중치 (장기 중시)
    strategies.append((
        "Enhanced: long_mom",
        EnhancedETFRotationStrategy(
            num_etfs=3,
            use_vol_weight=True,
            use_market_filter=True,
            use_trend_filter=True,
            momentum_weights={21: 0.10, 63: 0.20, 126: 0.35, 252: 0.35},
        ),
    ))

    # 12. Enhanced: 낮은 RISK_OFF 비율 (30%)
    strategies.append((
        "Enhanced: mild_riskoff",
        EnhancedETFRotationStrategy(
            num_etfs=3,
            use_vol_weight=True,
            use_market_filter=True,
            use_trend_filter=True,
            cash_ratio_risk_off=0.3,
        ),
    ))

    # 13. Enhanced: 높은 RISK_OFF 비율 (70%)
    strategies.append((
        "Enhanced: strong_riskoff",
        EnhancedETFRotationStrategy(
            num_etfs=3,
            use_vol_weight=True,
            use_market_filter=True,
            use_trend_filter=True,
            cash_ratio_risk_off=0.7,
        ),
    ))

    # ── Cross-Asset Momentum ──
    # 14. 기본 Cross-Asset Momentum
    strategies.append((
        "CrossAsset: base",
        CrossAssetMomentumStrategy(
            num_assets=3,
            use_trend_filter=True,
            use_correlation_filter=True,
        ),
    ))

    # 15. Cross-Asset: 트렌드 필터 OFF
    strategies.append((
        "CrossAsset: no_trend",
        CrossAssetMomentumStrategy(
            num_assets=3,
            use_trend_filter=False,
            use_correlation_filter=True,
        ),
    ))

    # 16. Cross-Asset: 상관관계 필터 OFF
    strategies.append((
        "CrossAsset: no_corr",
        CrossAssetMomentumStrategy(
            num_assets=3,
            use_trend_filter=True,
            use_correlation_filter=False,
        ),
    ))

    # 17. Cross-Asset: 단기 모멘텀 중시 (80%)
    strategies.append((
        "CrossAsset: short_bias",
        CrossAssetMomentumStrategy(
            num_assets=3,
            lookback_short=63,
            lookback_long=252,
            short_weight=0.8,
            use_trend_filter=True,
            use_correlation_filter=True,
        ),
    ))

    # 18. Cross-Asset: top2
    strategies.append((
        "CrossAsset: top2",
        CrossAssetMomentumStrategy(
            num_assets=2,
            use_trend_filter=True,
            use_correlation_filter=True,
        ),
    ))

    # 19. Cross-Asset: 자산군당 2개 허용
    strategies.append((
        "CrossAsset: max2_class",
        CrossAssetMomentumStrategy(
            num_assets=3,
            max_per_asset_class=2,
            use_trend_filter=True,
            use_correlation_filter=True,
        ),
    ))

    # 20. Cross-Asset: 높은 상관관계 허용
    strategies.append((
        "CrossAsset: high_corr",
        CrossAssetMomentumStrategy(
            num_assets=3,
            max_correlation=0.85,
            use_trend_filter=True,
            use_correlation_filter=True,
        ),
    ))

    return strategies


def print_results_table(results: list[dict]) -> None:
    """결과 테이블을 출력한다."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("  유효한 결과 없음")
        return

    sorted_results = sorted(valid, key=lambda x: x["sharpe_ratio"], reverse=True)

    print(f"\n{'=' * 110}")
    print("  ETF 전략 백테스트 비교 결과 (Sharpe 내림차순)")
    print(f"{'=' * 110}")
    header = (
        f"{'전략':30s} {'수익률':>10s} {'CAGR':>8s} "
        f"{'Sharpe':>8s} {'MDD':>8s} {'거래':>6s} {'최종자산':>14s}"
    )
    print(header)
    print("-" * 110)

    for r in sorted_results:
        name = r["strategy_name"]
        if len(name) > 30:
            name = name[:27] + "..."
        print(
            f"{name:30s} "
            f"{r['total_return']:>+9.1f}% "
            f"{r['cagr']:>+7.1f}% "
            f"{r['sharpe_ratio']:>7.2f} "
            f"{r['mdd']:>7.1f}% "
            f"{r['trades']:>6d} "
            f"{r['final_value']:>13,.0f}"
        )

    print("-" * 110)

    # 상위 3개
    top3 = sorted_results[:3]
    print("\n  ** TOP 3 전략 (Sharpe 기준) **")
    for i, r in enumerate(top3, 1):
        print(
            f"    {i}. {r['strategy_name']}: "
            f"Sharpe={r['sharpe_ratio']:.2f}, "
            f"CAGR={r['cagr']:+.1f}%, "
            f"MDD={r['mdd']:.1f}%"
        )

    # MDD가 가장 작은 전략
    best_mdd = min(sorted_results, key=lambda x: abs(x["mdd"]))
    print(
        f"\n  ** 가장 안정적 (최소 MDD): {best_mdd['strategy_name']} "
        f"(MDD={best_mdd['mdd']:.1f}%, Sharpe={best_mdd['sharpe_ratio']:.2f})"
    )

    # Sharpe / MDD 비율이 가장 좋은 전략
    for r in sorted_results:
        r["risk_adjusted"] = (
            r["sharpe_ratio"] / abs(r["mdd"]) * 100 if r["mdd"] != 0 else 0
        )
    best_ra = max(sorted_results, key=lambda x: x["risk_adjusted"])
    print(
        f"  ** 최적 위험조정 (Sharpe/|MDD|): {best_ra['strategy_name']} "
        f"(ratio={best_ra['risk_adjusted']:.4f})"
    )


def main():
    parser = argparse.ArgumentParser(description="Enhanced ETF 백테스트")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--capital", type=int, default=1_450_000)
    args = parser.parse_args()

    print(f"\n{'#' * 80}")
    print("  Enhanced ETF 전략 백테스트")
    print(f"  기간: {args.start} ~ {args.end}, 자본금: {args.capital:,}원")
    print(f"{'#' * 80}")

    # 데이터 로드
    print("\n  [데이터] ETF 가격 로드 중...")
    etf_prices = load_etf_prices(DEFAULT_ETF_UNIVERSE, args.start, args.end)
    print(f"  [데이터] {len(etf_prices)}/{len(DEFAULT_ETF_UNIVERSE)}개 ETF 로드 완료\n")

    if len(etf_prices) == 0:
        print("  ETF 데이터 없음. 종료.")
        sys.exit(1)

    # 전략 생성
    strategies = build_strategies()
    results = []

    print(f"  전략 수: {len(strategies)}개\n")

    for i, (name, strategy) in enumerate(strategies):
        print(f"  [{i+1}/{len(strategies)}] {name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            res = run_etf_backtest(
                strategy, etf_prices,
                args.start, args.end, args.capital,
            )
            elapsed = time.time() - t0

            entry = {
                "strategy_name": name,
                **res,
                "elapsed_sec": round(elapsed, 1),
            }
            results.append(entry)

            if "error" not in res:
                print(
                    f"({elapsed:.1f}s) | "
                    f"수익={res['total_return']:+.1f}% | "
                    f"CAGR={res['cagr']:+.1f}% | "
                    f"Sharpe={res['sharpe_ratio']:.2f} | "
                    f"MDD={res['mdd']:.1f}%"
                )
            else:
                print(f"실패: {res['error']}")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"에러: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "strategy_name": name,
                "total_return": 0, "cagr": 0, "sharpe_ratio": -999,
                "mdd": 0, "trades": 0, "final_value": 0,
                "error": str(e), "elapsed_sec": round(elapsed, 1),
            })

    # 결과 출력
    print_results_table(results)

    # 결과 저장
    output = {
        "config": {
            "start_date": args.start,
            "end_date": args.end,
            "capital": args.capital,
            "etf_universe": DEFAULT_ETF_UNIVERSE,
        },
        "results": results,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
