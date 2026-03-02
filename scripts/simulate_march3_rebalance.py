#!/usr/bin/env python3
"""3/3 리밸런싱 시뮬레이션 스크립트.

실제 실행 로직(executor.py 09:05)과 동일하게 T-1 데이터 기준으로
3월 3일(월) 리밸런싱을 시뮬레이션한다.

현재 설정:
- 장기 70% (MultiFactor V+M) + ETF 30% (EnhancedETFRotation) + 단기 0%
- enhanced_etf_rotation ON, drawdown_overlay ON
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_simulation(
    capital: int = 2_000_000,
    output_path: str = "docs/simulation_march3_rebalance.md",
):
    """3/3 리밸런싱 시뮬레이션을 실행한다."""
    from src.data.collector import get_all_fundamentals, get_price_data
    from src.data.index_collector import get_index_data
    from src.data.etf_collector import get_etf_price
    from src.strategy.multi_factor import MultiFactorStrategy
    from src.strategy.enhanced_etf_rotation import EnhancedETFRotationStrategy
    from src.strategy.drawdown_overlay import DrawdownOverlay
    from src.strategy.etf_rotation import DEFAULT_ETF_UNIVERSE, DEFAULT_SAFE_ASSET

    print(f"\n{'='*70}")
    print(f"  3/3 리밸런싱 시뮬레이션 (T-1 기준: 20260228)")
    print(f"  자본금: {capital:,}원")
    print(f"  배분: 장기 70% / ETF 30% / 단기 0%")
    print(f"  전략: MultiFactor(V+M) + EnhancedETFRotation")
    print(f"  오버레이: drawdown_overlay ON")
    print(f"{'='*70}\n")

    t0 = time.time()

    # T-1 = 2026-02-27 (금요일, 3/3 직전 거래일)
    # 2/28(토), 3/1(일), 3/2(월-삼일절 대체공휴일) → T-1 = 2/27
    data_date = "20260227"
    sim_date = "20260303"

    # ── 1. 데이터 수집 (실제 실행과 동일한 흐름) ──
    print("[1/6] 펀더멘탈 데이터 수집 중...")
    try:
        fundamentals = get_all_fundamentals(data_date)
        print(f"  -> {len(fundamentals)}종목 수집 완료")
    except Exception as e:
        print(f"  -> 수집 실패: {e}")
        fundamentals = pd.DataFrame()

    print("[2/6] 가격 데이터 수집 중...")
    prices = {}
    if not fundamentals.empty and "ticker" in fundamentals.columns:
        top_tickers = (
            fundamentals[fundamentals["market_cap"] > 0]
            .sort_values("market_cap", ascending=False)
            .head(200)["ticker"]
            .tolist()
        )
        start_dt = datetime.strptime(data_date, "%Y%m%d") - pd.Timedelta(days=400)
        start_date = start_dt.strftime("%Y%m%d")

        import signal as _signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("pykrx request timeout")

        loaded = 0
        skipped_tickers = []
        for ticker in top_tickers:
            try:
                _signal.signal(_signal.SIGALRM, _timeout_handler)
                _signal.alarm(15)  # 15초 timeout
                df = get_price_data(ticker, start_date, data_date)
                _signal.alarm(0)
                if not df.empty:
                    prices[ticker] = df
                    loaded += 1
            except TimeoutError:
                _signal.alarm(0)
                skipped_tickers.append(ticker)
                print(f"  !! timeout: {ticker}", flush=True)
            except Exception:
                _signal.alarm(0)
                pass
        print(f"  -> {loaded}/{len(top_tickers)}종목 가격 로드 (timeout: {len(skipped_tickers)}개)")

    print("[3/6] KOSPI 지수 수집 중...")
    index_prices = pd.Series(dtype=float)
    try:
        start_dt = datetime.strptime(data_date, "%Y%m%d") - pd.Timedelta(days=400)
        index_df = get_index_data("KOSPI", start_dt.strftime("%Y%m%d"), data_date)
        if not index_df.empty and "close" in index_df.columns:
            index_prices = index_df["close"]
        print(f"  -> {len(index_prices)}일 지수 데이터")
    except Exception as e:
        print(f"  -> 실패: {e}")

    print("[4/6] ETF 가격 수집 중...")
    etf_universe = dict(DEFAULT_ETF_UNIVERSE)
    etf_prices = {}
    for ticker, name in etf_universe.items():
        try:
            start_dt = datetime.strptime(data_date, "%Y%m%d") - pd.Timedelta(days=400)
            df = get_etf_price(ticker, start_dt.strftime("%Y%m%d"), data_date)
            if not df.empty:
                etf_prices[ticker] = df
        except Exception:
            pass
    etf_prices[DEFAULT_SAFE_ASSET] = etf_prices.get(DEFAULT_SAFE_ASSET, pd.DataFrame())
    print(f"  -> {len(etf_prices)}/{len(etf_universe)} ETF 로드")

    # ── 2. 전략 데이터 구성 ──
    strategy_data = {
        "fundamentals": fundamentals,
        "prices": prices,
        "index_prices": index_prices,
        "etf_prices": etf_prices,
    }

    # ── 3. 장기 전략 시그널 (MultiFactor V+M) ──
    print("[5/6] 장기 전략 시그널 생성 중...")
    long_strategy = MultiFactorStrategy(
        factors=["value", "momentum"],
        weights=[0.5, 0.5],
        combine_method="zscore",
        num_stocks=10,
        turnover_penalty=0.1,
    )
    try:
        long_signals = long_strategy.generate_signals(data_date, strategy_data)
        print(f"  -> MultiFactor: {len(long_signals)}종목 선정")
    except Exception as e:
        print(f"  -> MultiFactor 실패: {e}")
        long_signals = {}

    # ── 4. ETF 전략 시그널 (EnhancedETFRotation) ──
    print("[6/6] ETF 전략 시그널 생성 중...")
    etf_strategy = EnhancedETFRotationStrategy(
        num_etfs=3,
        use_vol_weight=True,
        use_market_filter=True,
        use_trend_filter=True,
        max_drawdown_filter=0.15,
        vol_lookback=60,
        cash_ratio_risk_off=0.7,
    )
    try:
        etf_data = dict(strategy_data)
        etf_data["etf_prices"] = etf_prices
        etf_signals = etf_strategy.generate_signals(data_date, etf_data)
        print(f"  -> EnhancedETF: {len(etf_signals)}종목 선정")
    except Exception as e:
        print(f"  -> EnhancedETF 실패: {e}")
        etf_signals = {}

    # ── 5. 70/30 비중 병합 ──
    long_pct = 0.70
    etf_pct = 0.30

    merged_signals = {}
    for ticker, weight in long_signals.items():
        scaled = round(weight * long_pct, 6)
        merged_signals[ticker] = merged_signals.get(ticker, 0.0) + scaled

    for ticker, weight in etf_signals.items():
        scaled = round(weight * etf_pct, 6)
        merged_signals[ticker] = merged_signals.get(ticker, 0.0) + scaled

    # ── 6. Drawdown Overlay 적용 ──
    drawdown_overlay = DrawdownOverlay(
        thresholds=[(-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25)],
        recovery_buffer=0.02,
    )
    # 시뮬레이션이므로 현재 포트폴리오 가치 = 초기 자본금
    merged_after_overlay = drawdown_overlay.apply_overlay(merged_signals, capital)

    # ── 7. 주문 시뮬레이션 ──
    print(f"\n{'='*70}")
    print("  리밸런싱 시뮬레이션 결과")
    print(f"{'='*70}\n")

    # 종목명 매핑
    ticker_names = {}
    if not fundamentals.empty and "ticker" in fundamentals.columns and "name" in fundamentals.columns:
        ticker_names = dict(zip(fundamentals["ticker"], fundamentals["name"]))
    for tk, nm in etf_universe.items():
        ticker_names[tk] = nm
    ticker_names[DEFAULT_SAFE_ASSET] = "KODEX 단기채권"

    # 장기 시그널 상세
    long_detail = []
    for ticker, weight in sorted(long_signals.items(), key=lambda x: x[1], reverse=True):
        name = ticker_names.get(ticker, ticker)
        scaled_w = round(weight * long_pct, 4)
        target_amount = int(capital * scaled_w)
        long_detail.append({
            "ticker": ticker,
            "name": name,
            "raw_weight": round(weight, 4),
            "scaled_weight": scaled_w,
            "target_amount": target_amount,
        })

    # ETF 시그널 상세
    etf_detail = []
    for ticker, weight in sorted(etf_signals.items(), key=lambda x: x[1], reverse=True):
        name = ticker_names.get(ticker, ticker)
        scaled_w = round(weight * etf_pct, 4)
        target_amount = int(capital * scaled_w)
        etf_detail.append({
            "ticker": ticker,
            "name": name,
            "raw_weight": round(weight, 4),
            "scaled_weight": scaled_w,
            "target_amount": target_amount,
        })

    # 병합 포트폴리오 상세
    merged_detail = []
    for ticker, weight in sorted(merged_after_overlay.items(), key=lambda x: x[1], reverse=True):
        name = ticker_names.get(ticker, ticker)
        target_amount = int(capital * weight)
        pool = "ETF" if ticker in etf_universe or ticker == DEFAULT_SAFE_ASSET else "장기"
        merged_detail.append({
            "ticker": ticker,
            "name": name,
            "weight": round(weight, 4),
            "target_amount": target_amount,
            "pool": pool,
        })

    # 소규모 자본 필터 (최소 주문 70,000원)
    MIN_ORDER = 70000
    executable = [m for m in merged_detail if m["target_amount"] >= MIN_ORDER]
    skipped = [m for m in merged_detail if m["target_amount"] < MIN_ORDER]

    # ── 결과 출력 ──
    total_weight = sum(m["weight"] for m in merged_detail)
    cash_weight = 1.0 - total_weight

    print(f"총 목표 종목: {len(merged_detail)}개")
    print(f"실행 가능: {len(executable)}개 (>= {MIN_ORDER:,}원)")
    print(f"스킵: {len(skipped)}개 (< {MIN_ORDER:,}원)")
    print(f"투자 비중: {total_weight:.1%} | 현금 비중: {cash_weight:.1%}")
    print()

    for m in merged_detail:
        flag = "  " if m["target_amount"] >= MIN_ORDER else "!!"
        print(f"  {flag} [{m['pool']:3s}] {m['name']}({m['ticker']}) "
              f"{m['weight']:.1%} = {m['target_amount']:>10,}원")

    elapsed = time.time() - t0
    print(f"\n소요시간: {elapsed:.0f}초")

    # ── MD 파일 생성 ──
    output = Path(PROJECT_ROOT) / output_path
    output.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# 3/3 리밸런싱 시뮬레이션 결과",
        "",
        f"**실행일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**시뮬레이션 대상일**: 2026-03-03 (월)",
        f"**데이터 기준일**: 2026-02-27 (T-1, 직전 거래일 - 금)",
        f"**초기 자본금**: {capital:,}원",
        "",
        "## 설정",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        f"| 장기 전략 | MultiFactor(V+M, zscore, 10종목) |",
        f"| ETF 전략 | EnhancedETFRotation(복합모멘텀+레짐+추세+하락필터) |",
        f"| 배분 비율 | 장기 70% / ETF 30% / 단기 0% |",
        f"| drawdown_overlay | ON (10%/15%/20% 단계적 디레버리징) |",
        f"| 리밸런싱 주기 | monthly |",
        "",
        "## 시뮬레이션 vs 실제 실행 로직 비교",
        "",
        "| 항목 | DailySimulator (16:05) | 실제 실행 (09:05) | 본 시뮬레이션 |",
        "|------|------------------------|-------------------|--------------|",
        "| 데이터 기준 | T (당일) | T-1 (전일) | **T-1** (실제 기준) |",
        "| 실행 빈도 | 매일 | 월 1회 리밸런싱일 | 3/3 1회 |",
        "| 주문 실행 | dry_run | 실주문 | 시뮬레이션 |",
        "| 비중 병합 | merge_pool_targets | merge_pool_targets | 동일 로직 |",
        "",
        "**핵심 차이**: 기존 DailySimulator는 T(당일) 데이터를 사용하여 사후 분석 용도. ",
        "실제 09:05 리밸런싱은 T-1 데이터로 시그널 생성. 본 시뮬레이션은 **실제 실행과 동일하게 T-1 기준**으로 수행.",
        "",
        "## 장기 전략 시그널 (MultiFactor V+M, 70%)",
        "",
        "| 순위 | 종목 | 코드 | 원시비중 | 스케일비중(70%) | 목표금액 |",
        "|------|------|------|----------|----------------|----------|",
    ]

    for i, d in enumerate(long_detail, 1):
        md_lines.append(
            f"| {i} | {d['name']} | {d['ticker']} | "
            f"{d['raw_weight']:.2%} | {d['scaled_weight']:.2%} | "
            f"{d['target_amount']:,}원 |"
        )

    md_lines.extend([
        "",
        "## ETF 전략 시그널 (EnhancedETFRotation, 30%)",
        "",
        "| 순위 | ETF | 코드 | 원시비중 | 스케일비중(30%) | 목표금액 |",
        "|------|-----|------|----------|----------------|----------|",
    ])

    for i, d in enumerate(etf_detail, 1):
        md_lines.append(
            f"| {i} | {d['name']} | {d['ticker']} | "
            f"{d['raw_weight']:.2%} | {d['scaled_weight']:.2%} | "
            f"{d['target_amount']:,}원 |"
        )

    md_lines.extend([
        "",
        "## 통합 포트폴리오 (Drawdown Overlay 적용 후)",
        "",
        f"총 투자 비중: {total_weight:.1%} | 현금: {cash_weight:.1%}",
        "",
        "| 순위 | 종목 | 코드 | 풀 | 비중 | 목표금액 | 실행가능 |",
        "|------|------|------|----|------|----------|----------|",
    ])

    for i, m in enumerate(merged_detail, 1):
        exec_flag = "O" if m["target_amount"] >= MIN_ORDER else "X (< 7만원)"
        md_lines.append(
            f"| {i} | {m['name']} | {m['ticker']} | {m['pool']} | "
            f"{m['weight']:.2%} | {m['target_amount']:,}원 | {exec_flag} |"
        )

    md_lines.extend([
        "",
        "## 소규모 자본 분석",
        "",
        f"- 총 자본금: {capital:,}원",
        f"- 최소 주문 금액: {MIN_ORDER:,}원",
        f"- 실행 가능 종목: {len(executable)}개",
        f"- 스킵 종목: {len(skipped)}개",
    ])

    if skipped:
        md_lines.append("")
        md_lines.append("### 스킵 종목 (최소 주문 미달)")
        for s in skipped:
            md_lines.append(f"- {s['name']}({s['ticker']}): {s['target_amount']:,}원 ({s['weight']:.2%})")

    md_lines.extend([
        "",
        "## 주요 참고사항",
        "",
        "1. **T-1 데이터 사용**: 실제 실행(09:05)과 동일하게 2/27(금) 데이터 기준",
        "2. **Drawdown Overlay**: 초기 자본금 기준이므로 첫 실행 시 영향 없음 (MDD=0%)",
        "3. **소액 제약**: 200만원으로 10+3종목 분산은 종목당 할당액이 작아 추적오차 발생 가능",
        "4. **ETF 거래세 면제**: ETF는 매도 시 0.015% (주식 0.245%), 소액 투자에 유리",
        f"5. **시뮬레이션 소요시간**: {elapsed:.0f}초",
        "",
        f"---",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    output.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n결과 저장: {output}")

    # JSON 결과도 저장 (Task 3 비교용)
    json_result = {
        "sim_date": sim_date,
        "data_date": data_date,
        "capital": capital,
        "allocation": {"long_term": 0.70, "etf_rotation": 0.30, "short_term": 0.0},
        "long_strategy": "MultiFactor(V+M)",
        "etf_strategy": "EnhancedETFRotation",
        "overlays": ["drawdown_overlay"],
        "long_signals": {d["ticker"]: d["raw_weight"] for d in long_detail},
        "etf_signals": {d["ticker"]: d["raw_weight"] for d in etf_detail},
        "merged_portfolio": {m["ticker"]: m["weight"] for m in merged_detail},
        "executable_count": len(executable),
        "skipped_count": len(skipped),
        "total_weight": round(total_weight, 4),
        "elapsed_sec": round(elapsed, 1),
    }

    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps(json_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON 저장: {json_path}")

    return json_result


if __name__ == "__main__":
    run_simulation()
