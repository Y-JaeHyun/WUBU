#!/usr/bin/env python3
"""펀더멘탈 캐시 재수집 스크립트 (JAE-24).

pykrx get_market_fundamental() 장애로 인해 2026-03 이후 캐시가 비어있거나
PBR 0값 비율이 높은 경우, pykrx 또는 DART 기반으로 재수집한다.

사용법:
    python scripts/update_fundamentals_cache.py [--from YYYYMMDD] [--force] [--dart-only]

Options:
    --from YYYYMMDD  재수집 시작일 (기본: 20260201 — 마지막 유효 캐시 이후)
    --force          기존 캐시가 있어도 덮어쓰기
    --dart-only      pykrx 시도 없이 DART fallback만 사용
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.data.price_store import PriceStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_rebalancing_dates(from_date: str) -> list[str]:
    """재수집 대상 날짜 목록을 반환한다 (월말 영업일 기준).

    매월 마지막 영업일만 대상으로 한다. 현재 전략의 리밸런싱이 월단위이기 때문이다.
    """
    today = pd.Timestamp.now("Asia/Seoul").strftime("%Y%m%d")
    dates = pd.date_range(
        start=pd.Timestamp(from_date),
        end=pd.Timestamp(today),
        freq="BME",  # Business Month End
    )
    return [d.strftime("%Y%m%d") for d in dates if d.strftime("%Y%m%d") <= today]


def _collect_with_pykrx(date: str) -> pd.DataFrame:
    """pykrx로 펀더멘탈을 수집한다."""
    from src.data.collector import get_all_fundamentals
    return get_all_fundamentals(date)


def _collect_with_dart(date: str) -> pd.DataFrame:
    """DART 기반으로 펀더멘탈을 수집한다."""
    from src.data.dart_collector import get_all_fundamentals_dart
    return get_all_fundamentals_dart(date)


def main():
    parser = argparse.ArgumentParser(description="펀더멘탈 캐시 재수집")
    parser.add_argument(
        "--from", dest="from_date", default="20260201",
        help="재수집 시작일 YYYYMMDD (기본: 20260201)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="기존 캐시 덮어쓰기"
    )
    parser.add_argument(
        "--dart-only", action="store_true",
        help="DART fallback만 사용 (pykrx 시도 안 함)"
    )
    args = parser.parse_args()

    store = PriceStore()
    dates = _get_rebalancing_dates(args.from_date)

    logger.info("재수집 대상: %d개 날짜 (%s ~ %s)", len(dates), dates[0] if dates else "N/A", dates[-1] if dates else "N/A")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for date in dates:
        parquet_path = store._fundamentals_dir / f"{date}.parquet"

        # 기존 캐시 품질 확인
        if not args.force and parquet_path.exists():
            is_valid, zero_ratio = store.check_fundamentals_quality(date)
            if is_valid:
                logger.info("[SKIP] %s — 캐시 유효 (PBR 0값 %.1f%%)", date, zero_ratio * 100)
                skip_count += 1
                continue
            else:
                logger.warning("[재수집] %s — 캐시 불량 (PBR 0값 %.1f%%)", date, zero_ratio * 100)
        elif args.force and parquet_path.exists():
            logger.info("[강제] %s — 기존 캐시 덮어쓰기", date)
        else:
            logger.info("[신규] %s — 캐시 없음, 수집 시작", date)

        # 수집 실행
        df = pd.DataFrame()
        if not args.dart_only:
            try:
                df = _collect_with_pykrx(date)
                if not df.empty:
                    is_valid, zero_ratio = (
                        (True, (df["pbr"] == 0).sum() / len(df))
                        if "pbr" in df.columns else (False, 1.0)
                    )
                    if not is_valid or zero_ratio > 0.5:
                        logger.warning(
                            "%s pykrx 데이터 불량 (PBR 0값 %.1f%%) — DART fallback",
                            date, zero_ratio * 100
                        )
                        df = pd.DataFrame()
            except Exception as e:
                logger.warning("%s pykrx 수집 실패: %s — DART fallback", date, e)
                df = pd.DataFrame()

        if df.empty:
            try:
                df = _collect_with_dart(date)
            except Exception as e:
                logger.error("%s DART 수집 실패: %s", date, e)

        if df.empty:
            logger.error("[실패] %s — 모든 소스 실패", date)
            fail_count += 1
            continue

        # 캐시 저장
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path)
        store._set_meta(date, "fundamentals", date, date, len(df))

        valid_pbr = (df["pbr"] > 0).sum() if "pbr" in df.columns else 0
        logger.info(
            "[완료] %s — %d개 종목, 유효 PBR %d개 (%.1f%%)",
            date, len(df), valid_pbr, valid_pbr / len(df) * 100
        )
        success_count += 1

    logger.info(
        "캐시 재수집 완료: 성공 %d, 건너뜀 %d, 실패 %d",
        success_count, skip_count, fail_count
    )

    # 품질 요약 출력
    print("\n=== 캐시 품질 요약 ===")
    for date in store.list_fundamentals_dates():
        is_valid, zero_ratio = store.check_fundamentals_quality(date)
        status = "OK" if is_valid else "불량"
        print(f"  {date}: {status} (PBR 0값 {zero_ratio:.1%})")


if __name__ == "__main__":
    main()
