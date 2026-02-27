#!/usr/bin/env python3
"""전략 전체 백테스트 비교 스크립트.

장기 전략 12종 + 단기 전략 4종을 한국 시장 실데이터로 백테스트하고 비교한다.
결과는 data/all_backtest_results.json에 저장된다.

사용법:
    # 전체 (장기 + 단기)
    python scripts/run_all_backtest.py

    # 장기만
    python scripts/run_all_backtest.py --mode long

    # 단기만
    python scripts/run_all_backtest.py --mode short

    # 기간/자본 지정
    python scripts/run_all_backtest.py --start 20200101 --end 20251231 --capital 1450000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "backtest_cache"
RESULT_PATH = PROJECT_ROOT / "data" / "all_backtest_results.json"


# ──────────────────────────────────────────────────────────
# 장기 전략 정의
# ──────────────────────────────────────────────────────────

def build_long_term_strategies() -> list[tuple[str, object, dict]]:
    """장기 전략 인스턴스 리스트를 생성한다.

    Returns:
        (이름, 전략 객체, 옵션) 튜플 리스트.
        옵션: {"overlay": MarketTimingOverlay} 등.
    """
    from src.strategy.value import ValueStrategy
    from src.strategy.momentum import MomentumStrategy
    from src.strategy.quality import QualityStrategy
    from src.strategy.multi_factor import MultiFactorStrategy
    from src.strategy.three_factor import ThreeFactorStrategy
    from src.strategy.risk_parity import RiskParityStrategy
    from src.strategy.shareholder_yield import ShareholderYieldStrategy
    from src.strategy.pead import PEADStrategy
    from src.strategy.low_vol_quality import LowVolQualityStrategy
    from src.strategy.accrual import AccrualStrategy
    from src.strategy.market_timing import MarketTimingOverlay

    strategies = []

    # 1. Value (저PBR)
    strategies.append((
        "Value(PBR)",
        ValueStrategy(factor="pbr", num_stocks=10, min_market_cap=0),
        {},
    ))

    # 2. Value (Composite: PBR+PER+PSR+PCR)
    strategies.append((
        "Value(Composite)",
        ValueStrategy(factor="composite", num_stocks=10, min_market_cap=0),
        {},
    ))

    # 3. Momentum (12M, 잔차 모멘텀)
    strategies.append((
        "Momentum(Residual)",
        MomentumStrategy(
            lookback_months=[12], skip_month=True, num_stocks=10,
            residual=True, min_market_cap=0,
        ),
        {},
    ))

    # 4. Quality (ROE+GP/A+부채비율+발생액)
    strategies.append((
        "Quality",
        QualityStrategy(num_stocks=10, min_market_cap=0, strict_accrual=True),
        {},
    ))

    # 5. MultiFactor (V+M, 회전율 페널티)
    strategies.append((
        "MultiFactor(V+M)",
        MultiFactorStrategy(
            factors=["value", "momentum"],
            weights=[0.5, 0.5],
            combine_method="zscore",
            num_stocks=10,
            turnover_penalty=0.1,
        ),
        {},
    ))

    # 6. MultiFactor + 마켓타이밍
    overlay = MarketTimingOverlay(
        ma_period=200, ma_type="SMA", switch_mode="gradual",
        reference_index="KOSPI",
    )
    strategies.append((
        "MultiFactor+MT",
        MultiFactorStrategy(
            factors=["value", "momentum"],
            weights=[0.5, 0.5],
            combine_method="zscore",
            num_stocks=10,
        ),
        {"overlay": overlay},
    ))

    # 7. ThreeFactor (V+M+Q)
    strategies.append((
        "ThreeFactor(V+M+Q)",
        ThreeFactorStrategy(
            num_stocks=10, min_market_cap=0,
            value_weight=0.33, momentum_weight=0.33, quality_weight=0.34,
        ),
        {},
    ))

    # 8. ShareholderYield (배당+자사주)
    strategies.append((
        "ShareholderYield",
        ShareholderYieldStrategy(num_stocks=10, min_market_cap=0),
        {},
    ))

    # 9. PEAD (실적 서프라이즈)
    strategies.append((
        "PEAD",
        PEADStrategy(num_stocks=10, min_market_cap=0),
        {},
    ))

    # 10. LowVolQuality (저변동성+품질)
    strategies.append((
        "LowVolQuality",
        LowVolQualityStrategy(num_stocks=10, min_market_cap=0),
        {},
    ))

    # 11. Accrual (발생액 전략)
    strategies.append((
        "Accrual",
        AccrualStrategy(num_stocks=10, min_market_cap=0),
        {},
    ))

    # 12. RiskParity(MultiFactor)
    base_strategy = MultiFactorStrategy(
        factors=["value", "momentum"],
        weights=[0.5, 0.5],
        num_stocks=10,
    )
    strategies.append((
        "RiskParity(MF)",
        RiskParityStrategy(stock_selector=base_strategy, max_weight=0.15),
        {},
    ))

    return strategies


# ──────────────────────────────────────────────────────────
# 단기 전략 정의
# ──────────────────────────────────────────────────────────

def build_short_term_strategies() -> list[tuple[str, object]]:
    """단기 전략 인스턴스 리스트를 생성한다."""
    from src.strategy.bb_squeeze import BBSqueezeStrategy
    from src.strategy.high_breakout import HighBreakoutStrategy
    from src.strategy.swing_reversion import SwingReversionStrategy

    strategies = [
        ("bb_squeeze", BBSqueezeStrategy(params={"min_market_cap": 0})),
        ("bb_squeeze_noATR", BBSqueezeStrategy(params={
            "min_market_cap": 0, "use_atr_stops": False,
        })),
        ("high_breakout", HighBreakoutStrategy(params={"min_market_cap": 0})),
        ("swing_reversion", SwingReversionStrategy(params={"min_market_cap": 0})),
        ("swing_rev+regime", SwingReversionStrategy(params={
            "min_market_cap": 0, "regime_filter": True,
        })),
    ]
    return strategies


# ──────────────────────────────────────────────────────────
# 데이터 로드 (단기 전략용)
# ──────────────────────────────────────────────────────────

def collect_universe(date: str, top_n: int = 100) -> list[str]:
    """시가총액 상위 종목 코드를 수집한다."""
    from pykrx import stock as pykrx_stock

    tickers = []
    for market in ["KOSPI", "KOSDAQ"]:
        try:
            df = pykrx_stock.get_market_cap(date, market=market)
            if df is not None and not df.empty:
                df = df.sort_values("시가총액", ascending=False)
                tickers.extend(df.head(top_n).index.tolist())
        except Exception as e:
            logger.warning("유니버스 수집 실패 (%s): %s", market, e)

    unique = list(dict.fromkeys(tickers))[:top_n]
    logger.info("유니버스 수집: %d종목", len(unique))
    return unique


def load_price_data(
    tickers: list[str], start_date: str, end_date: str, cache_dir: Path,
) -> dict[str, pd.DataFrame]:
    """종목별 일봉 데이터를 로드한다 (캐시 활용)."""
    from pykrx import stock as pykrx_stock

    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    cache_key = f"{start_date}_{end_date}"

    for i, ticker in enumerate(tickers):
        cache_file = cache_dir / f"{ticker}_{cache_key}.parquet"

        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    result[ticker] = df
                    continue
            except Exception:
                pass

        try:
            df = pykrx_stock.get_market_ohlcv(start_date, end_date, ticker)
            if df is not None and not df.empty:
                col_map = {
                    "시가": "open", "고가": "high", "저가": "low",
                    "종가": "close", "거래량": "volume",
                }
                df = df.rename(columns=col_map)

                try:
                    mcap = pykrx_stock.get_market_cap(end_date, end_date, ticker)
                    if mcap is not None and not mcap.empty:
                        df["시가총액"] = mcap["시가총액"].iloc[0]
                    else:
                        df["시가총액"] = 0
                except Exception:
                    df["시가총액"] = 0

                df.to_parquet(cache_file)
                result[ticker] = df

            if (i + 1) % 20 == 0:
                logger.info("데이터 로드: %d/%d", i + 1, len(tickers))
                time.sleep(1)

        except Exception as e:
            logger.debug("종목 %s 데이터 실패: %s", ticker, e)

    logger.info("데이터 로드 완료: %d/%d 종목", len(result), len(tickers))
    return result


# ──────────────────────────────────────────────────────────
# 장기 전략 백테스트 실행
# ──────────────────────────────────────────────────────────

def run_long_term_backtests(
    start_date: str, end_date: str, initial_capital: int,
) -> list[dict]:
    """장기 전략 백테스트를 실행한다."""
    from src.backtest.engine import Backtest

    strategies = build_long_term_strategies()
    results = []

    print(f"\n{'='*80}")
    print(f"  장기 전략 백테스트: {start_date} ~ {end_date}")
    print(f"  초기 자본: {initial_capital:,}원 | 리밸런싱: monthly")
    print(f"  전략 수: {len(strategies)}")
    print(f"{'='*80}\n")

    for i, (name, strategy, opts) in enumerate(strategies):
        print(f"  [{i+1}/{len(strategies)}] {name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            bt = Backtest(
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                rebalance_freq="monthly",
                overlay=opts.get("overlay"),
            )
            bt.run()
            res = bt.get_results()
            elapsed = time.time() - t0

            result_entry = {
                "category": "long_term",
                "strategy_name": name,
                "total_return": res["total_return"],
                "cagr": res["cagr"],
                "sharpe_ratio": res["sharpe_ratio"],
                "mdd": res["mdd"],
                "win_rate": res.get("win_rate", 0),
                "total_trades": res.get("total_trades", 0),
                "final_value": res.get("final_value", 0),
                "elapsed_sec": round(elapsed, 1),
            }
            results.append(result_entry)

            print(
                f"완료 ({elapsed:.0f}s) | "
                f"수익={res['total_return']:.1f}% | "
                f"Sharpe={res['sharpe_ratio']:.2f} | "
                f"MDD={res['mdd']:.1f}%"
            )

        except Exception as e:
            print(f"실패: {e}")
            results.append({
                "category": "long_term",
                "strategy_name": name,
                "total_return": 0, "cagr": 0, "sharpe_ratio": -999,
                "mdd": 0, "win_rate": 0, "total_trades": 0,
                "final_value": 0, "elapsed_sec": 0, "error": str(e),
            })

    return results


# ──────────────────────────────────────────────────────────
# 단기 전략 백테스트 실행
# ──────────────────────────────────────────────────────────

def run_short_term_backtests(
    start_date: str, end_date: str, initial_capital: int, top_n: int = 100,
) -> list[dict]:
    """단기 전략 백테스트를 실행한다."""
    from src.backtest.short_term_backtest import ShortTermBacktest

    print(f"\n{'='*80}")
    print(f"  단기 전략 백테스트: {start_date} ~ {end_date}")
    print(f"  초기 자본: {initial_capital:,}원 | 유니버스: 시총 상위 {top_n}")
    print(f"{'='*80}\n")

    # 데이터 로드
    print("  [데이터] 유니버스 수집 중...")
    tickers = collect_universe(end_date, top_n=top_n)
    if not tickers:
        print("  유니버스 수집 실패!")
        return []

    print(f"  [데이터] {len(tickers)}종목 가격 데이터 로드 중...")
    price_data = load_price_data(tickers, start_date, end_date, CACHE_DIR)
    if not price_data:
        print("  데이터 로드 실패!")
        return []
    print(f"  [데이터] {len(price_data)}종목 로드 완료\n")

    strategies = build_short_term_strategies()
    results = []

    for i, (name, strategy) in enumerate(strategies):
        print(f"  [{i+1}/{len(strategies)}] {name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            bt = ShortTermBacktest(
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
            )
            bt.run(preloaded_data=price_data)
            res = bt.get_results()
            elapsed = time.time() - t0

            result_entry = {
                "category": "short_term",
                "strategy_name": name,
                "total_return": round(res["total_return"] * 100, 2),
                "cagr": round(res["cagr"] * 100, 2),
                "sharpe_ratio": round(res["sharpe_ratio"], 2),
                "mdd": round(res["max_drawdown"] * 100, 2),
                "win_rate": round(res["win_rate"] * 100, 1),
                "total_trades": res["total_trades"],
                "profit_factor": round(res.get("profit_factor", 0), 2),
                "avg_holding_days": round(res.get("avg_holding_days", 0), 1),
                "commission_total": round(res.get("commission_total", 0), 0),
                "final_value": round(
                    initial_capital * (1 + res["total_return"]), 0
                ),
                "elapsed_sec": round(elapsed, 1),
            }
            results.append(result_entry)

            print(
                f"완료 ({elapsed:.0f}s) | "
                f"수익={result_entry['total_return']:.1f}% | "
                f"Sharpe={result_entry['sharpe_ratio']:.2f} | "
                f"MDD={result_entry['mdd']:.1f}% | "
                f"거래={res['total_trades']}건"
            )

        except Exception as e:
            print(f"실패: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "category": "short_term",
                "strategy_name": name,
                "total_return": 0, "cagr": 0, "sharpe_ratio": -999,
                "mdd": 0, "win_rate": 0, "total_trades": 0,
                "final_value": 0, "elapsed_sec": 0, "error": str(e),
            })

    return results


# ──────────────────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────────────────

def print_comparison_table(results: list[dict], category: str) -> None:
    """결과 비교 테이블을 출력한다."""
    filtered = [r for r in results if r["category"] == category and "error" not in r]
    if not filtered:
        print(f"  {category}: 유효한 결과 없음")
        return

    sorted_results = sorted(filtered, key=lambda x: x["sharpe_ratio"], reverse=True)

    title = "장기 전략" if category == "long_term" else "단기 전략"
    print(f"\n{'='*90}")
    print(f"  {title} 비교 결과 (Sharpe 내림차순)")
    print(f"{'='*90}")

    if category == "long_term":
        header = (
            f"{'전략':22s} {'수익률':>10s} {'CAGR':>8s} "
            f"{'Sharpe':>8s} {'MDD':>8s} {'승률':>7s} {'최종자산':>16s}"
        )
    else:
        header = (
            f"{'전략':22s} {'수익률':>10s} {'CAGR':>8s} "
            f"{'Sharpe':>8s} {'MDD':>8s} {'승률':>7s} {'PF':>7s} {'거래':>6s}"
        )
    print(header)
    print("-" * 90)

    for r in sorted_results:
        if category == "long_term":
            print(
                f"{r['strategy_name']:22s} "
                f"{r['total_return']:>9.2f}% "
                f"{r['cagr']:>7.2f}% "
                f"{r['sharpe_ratio']:>7.2f} "
                f"{r['mdd']:>7.2f}% "
                f"{r['win_rate']:>6.1f}% "
                f"{r['final_value']:>15,.0f}"
            )
        else:
            print(
                f"{r['strategy_name']:22s} "
                f"{r['total_return']:>9.2f}% "
                f"{r['cagr']:>7.2f}% "
                f"{r['sharpe_ratio']:>7.2f} "
                f"{r['mdd']:>7.2f}% "
                f"{r['win_rate']:>6.1f}% "
                f"{r.get('profit_factor', 0):>6.2f} "
                f"{r['total_trades']:>6d}"
            )

    print("-" * 90)

    # 최고 전략
    best = sorted_results[0]
    print(f"\n  ** 최고 전략 (Sharpe): {best['strategy_name']} "
          f"(Sharpe={best['sharpe_ratio']:.2f}, "
          f"CAGR={best['cagr']:.2f}%, MDD={best['mdd']:.2f}%) **")


def print_recommendation(results: list[dict]) -> dict:
    """최적 전략 조합을 추천한다."""
    long_results = [r for r in results if r["category"] == "long_term" and "error" not in r]
    short_results = [r for r in results if r["category"] == "short_term" and "error" not in r]

    recommendation = {}

    print(f"\n{'='*90}")
    print("  전략 추천 (Sharpe 기준)")
    print(f"{'='*90}")

    if long_results:
        best_long = max(long_results, key=lambda x: x["sharpe_ratio"])
        recommendation["long_term_best"] = best_long["strategy_name"]
        recommendation["long_term_sharpe"] = best_long["sharpe_ratio"]
        print(f"\n  [장기] 추천: {best_long['strategy_name']}")
        print(f"    Sharpe={best_long['sharpe_ratio']:.2f}, "
              f"CAGR={best_long['cagr']:.2f}%, MDD={best_long['mdd']:.2f}%")

        # Sharpe > 0 이고 MDD 절대값이 작은 전략도 별도 추천
        stable = [r for r in long_results if r["sharpe_ratio"] > 0]
        if stable:
            least_mdd = min(stable, key=lambda x: abs(x["mdd"]))
            if least_mdd["strategy_name"] != best_long["strategy_name"]:
                recommendation["long_term_stable"] = least_mdd["strategy_name"]
                print(f"  [장기] 안정적 추천: {least_mdd['strategy_name']}")
                print(f"    Sharpe={least_mdd['sharpe_ratio']:.2f}, "
                      f"MDD={least_mdd['mdd']:.2f}%")

    if short_results:
        best_short = max(short_results, key=lambda x: x["sharpe_ratio"])
        recommendation["short_term_best"] = best_short["strategy_name"]
        recommendation["short_term_sharpe"] = best_short["sharpe_ratio"]
        print(f"\n  [단기] 추천: {best_short['strategy_name']}")
        print(f"    Sharpe={best_short['sharpe_ratio']:.2f}, "
              f"CAGR={best_short['cagr']:.2f}%, MDD={best_short['mdd']:.2f}%")

    print()
    return recommendation


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="전략 전체 백테스트 비교")
    parser.add_argument("--start", default="20200101", help="시작일 (YYYYMMDD)")
    parser.add_argument("--end", default="20251231", help="종료일 (YYYYMMDD)")
    parser.add_argument("--capital", type=int, default=1_450_000, help="초기 자본금")
    parser.add_argument("--mode", choices=["all", "long", "short"], default="all",
                        help="백테스트 모드 (all/long/short)")
    parser.add_argument("--top-n", type=int, default=100,
                        help="단기 전략 유니버스 종목수")
    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"  전략 전체 백테스트")
    print(f"  기간: {args.start} ~ {args.end}")
    print(f"  자본금: {args.capital:,}원")
    print(f"  모드: {args.mode}")
    print(f"{'#'*80}")

    all_results = []
    total_start = time.time()

    # 장기 전략 백테스트
    if args.mode in ("all", "long"):
        long_results = run_long_term_backtests(
            args.start, args.end, args.capital,
        )
        all_results.extend(long_results)
        print_comparison_table(all_results, "long_term")

    # 단기 전략 백테스트
    if args.mode in ("all", "short"):
        short_results = run_short_term_backtests(
            args.start, args.end, args.capital, top_n=args.top_n,
        )
        all_results.extend(short_results)
        print_comparison_table(all_results, "short_term")

    # 추천
    recommendation = print_recommendation(all_results)

    # 결과 저장
    total_elapsed = time.time() - total_start
    output = {
        "backtest_config": {
            "start_date": args.start,
            "end_date": args.end,
            "initial_capital": args.capital,
            "mode": args.mode,
        },
        "results": all_results,
        "recommendation": recommendation,
        "total_elapsed_sec": round(total_elapsed, 1),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {RESULT_PATH}")
    print(f"총 소요시간: {total_elapsed:.0f}초")


if __name__ == "__main__":
    main()
