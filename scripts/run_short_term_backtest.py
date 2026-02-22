#!/usr/bin/env python3
"""단기 전략 백테스트 비교 스크립트.

4개 전략을 2020~2025 한국 시장 데이터로 백테스트하고 비교한다.
결과는 data/short_term_backtest_results.csv에 저장된다.

사용법:
    python scripts/run_short_term_backtest.py [--start 20200101] [--end 20251231]
        [--capital 145000] [--top-n 100] [--cache-dir data/backtest_cache]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.short_term_backtest import ShortTermBacktest
from src.strategy.swing_reversion import SwingReversionStrategy
from src.strategy.high_breakout import HighBreakoutStrategy
from src.strategy.bb_squeeze import BBSqueezeStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "backtest_cache"
RESULT_PATH = PROJECT_ROOT / "data" / "short_term_backtest_results.csv"


def collect_universe(date: str, top_n: int = 100) -> list[str]:
    """시가총액 상위 종목 코드를 수집한다.

    Args:
        date: 기준일 (YYYYMMDD).
        top_n: 상위 N개.

    Returns:
        종목 코드 리스트.
    """
    from pykrx import stock as pykrx_stock

    # 코스피 + 코스닥 시총 상위
    tickers = []
    for market in ["KOSPI", "KOSDAQ"]:
        try:
            df = pykrx_stock.get_market_cap(date, market=market)
            if df is not None and not df.empty:
                df = df.sort_values("시가총액", ascending=False)
                tickers.extend(df.head(top_n).index.tolist())
        except Exception as e:
            logger.warning("유니버스 수집 실패 (%s): %s", market, e)

    # 중복 제거 후 시총 기준 상위 top_n
    unique = list(dict.fromkeys(tickers))[:top_n]
    logger.info("유니버스 수집: %d종목 (상위 %d)", len(unique), top_n)
    return unique


def load_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: Path,
) -> dict[str, pd.DataFrame]:
    """종목별 일봉 데이터를 로드한다 (캐시 활용).

    Args:
        tickers: 종목 코드 리스트.
        start_date: 시작일 (YYYYMMDD).
        end_date: 종료일 (YYYYMMDD).
        cache_dir: 캐시 디렉토리.

    Returns:
        {ticker: DataFrame} 딕셔너리.
    """
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

        # pykrx에서 다운로드
        try:
            df = pykrx_stock.get_market_ohlcv(start_date, end_date, ticker)
            if df is not None and not df.empty:
                # 컬럼 매핑: 시가/고가/저가/종가/거래량 → open/high/low/close/volume
                col_map = {
                    "시가": "open",
                    "고가": "high",
                    "저가": "low",
                    "종가": "close",
                    "거래량": "volume",
                }
                df = df.rename(columns=col_map)

                # 시가총액 추가
                try:
                    mcap = pykrx_stock.get_market_cap(
                        end_date, end_date, ticker
                    )
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
                time.sleep(1)  # API rate limit 보호

        except Exception as e:
            logger.debug("종목 %s 데이터 실패: %s", ticker, e)

    logger.info("데이터 로드 완료: %d/%d 종목", len(result), len(tickers))
    return result


def run_backtest(
    strategy,
    price_data: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    initial_capital: int,
) -> dict:
    """단일 전략 백테스트를 실행한다."""
    bt = ShortTermBacktest(
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )
    bt.run(preloaded_data=price_data)
    return bt.get_results()


def format_results_table(results: list[dict]) -> str:
    """결과를 보기 좋은 테이블로 포맷한다."""
    lines = [
        "=" * 90,
        "단기 전략 백테스트 비교 결과",
        "=" * 90,
        "",
        f"{'전략':20s} {'총수익률':>10s} {'CAGR':>8s} {'Sharpe':>8s} "
        f"{'MDD':>8s} {'승률':>7s} {'PF':>7s} {'거래수':>7s} {'보유일':>7s}",
        "-" * 90,
    ]

    # Sharpe 기준 정렬
    sorted_results = sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True)

    for r in sorted_results:
        lines.append(
            f"{r['strategy_name']:20s} "
            f"{r['total_return']:>9.2%} "
            f"{r['cagr']:>7.2%} "
            f"{r['sharpe_ratio']:>7.2f} "
            f"{r['max_drawdown']:>7.2%} "
            f"{r['win_rate']:>6.1%} "
            f"{r['profit_factor']:>6.2f} "
            f"{r['total_trades']:>7d} "
            f"{r['avg_holding_days']:>6.1f}"
        )

    lines.append("-" * 90)

    best = sorted_results[0]
    lines.append("")
    lines.append(f"** 최고 전략 (Sharpe 기준): {best['strategy_name']} **")
    lines.append(f"   Sharpe={best['sharpe_ratio']:.2f}, "
                 f"CAGR={best['cagr']:.2%}, MDD={best['max_drawdown']:.2%}")
    lines.append("")

    # 수수료 정보
    lines.append("[수수료 상세]")
    for r in sorted_results:
        lines.append(
            f"  {r['strategy_name']:20s}: "
            f"총 {r['commission_total']:,.0f}원 "
            f"(거래당 {r['commission_total'] / r['total_trades']:.0f}원)"
            if r["total_trades"] > 0
            else f"  {r['strategy_name']:20s}: 거래 없음"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="단기 전략 백테스트 비교")
    parser.add_argument("--start", default="20200101", help="시작일 (YYYYMMDD)")
    parser.add_argument("--end", default="20251231", help="종료일 (YYYYMMDD)")
    parser.add_argument("--capital", type=int, default=145_000, help="초기 자본금")
    parser.add_argument("--top-n", type=int, default=100, help="유니버스 종목수")
    parser.add_argument(
        "--cache-dir",
        default=str(CACHE_DIR),
        help="캐시 디렉토리",
    )
    args = parser.parse_args()

    print(f"\n단기 전략 백테스트")
    print(f"  기간: {args.start} ~ {args.end}")
    print(f"  자본금: {args.capital:,}원")
    print(f"  유니버스: 시총 상위 {args.top_n}종목")
    print()

    # 1. 유니버스 수집
    print("[1/4] 유니버스 수집 중...")
    tickers = collect_universe(args.end, top_n=args.top_n)
    if not tickers:
        print("유니버스 수집 실패. 종료합니다.")
        return

    # 2. 데이터 로드
    print(f"[2/4] {len(tickers)}종목 데이터 로드 중 (캐시: {args.cache_dir})...")
    price_data = load_price_data(
        tickers, args.start, args.end, Path(args.cache_dir)
    )
    if not price_data:
        print("데이터 로드 실패. 종료합니다.")
        return
    print(f"  {len(price_data)}종목 로드 완료")

    # 3. 전략 정의
    # 유니버스가 이미 시총 상위 N이므로 시총 필터 비활성화 (데이터에 시총 없을 수 있음)
    strategies = [
        ("swing_reversion", SwingReversionStrategy(params={"min_market_cap": 0})),
        ("swing_reversion_obv", SwingReversionStrategy(
            params={"use_obv_filter": True, "min_market_cap": 0}
        )),
        ("high_breakout", HighBreakoutStrategy(params={"min_market_cap": 0})),
        ("bb_squeeze", BBSqueezeStrategy(params={"min_market_cap": 0})),
    ]

    # 4. 백테스트 실행
    print(f"[3/4] {len(strategies)}개 전략 백테스트 실행 중...")
    all_results = []

    for name, strategy in strategies:
        print(f"  실행 중: {name}...", end=" ", flush=True)
        start_time = time.time()

        result = run_backtest(
            strategy=strategy,
            price_data=price_data,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
        )
        elapsed = time.time() - start_time
        print(f"완료 ({elapsed:.1f}s, {result['total_trades']}건)")
        all_results.append(result)

    # 5. 결과 출력 + 저장
    print()
    print("[4/4] 결과 비교")
    print()
    print(format_results_table(all_results))

    # CSV 저장
    df = pd.DataFrame(all_results)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {RESULT_PATH}")

    # 최고 전략 식별 (Sharpe 기준)
    best = max(all_results, key=lambda x: x["sharpe_ratio"])
    print(f"\n추천 기본 전략: {best['strategy_name']}")

    # JSON으로도 저장 (프로그래밍 접근용)
    json_path = RESULT_PATH.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"results": all_results, "best_strategy": best["strategy_name"]},
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
