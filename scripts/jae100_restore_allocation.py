"""JAE-100 복구: portfolio_allocation.json 재태깅.

ETF 유니버스에 포함된 보유 종목은 etf_rotation, 그 외는 long_term으로
backfill한다. backfill_untagged_positions()와 동일 로직을 사용해
이미 태깅된 포지션은 건드리지 않는다.

사용법:
    # dry-run (변경 미적용)
    python scripts/jae100_restore_allocation.py --dry-run

    # 실제 적용
    python scripts/jae100_restore_allocation.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="JAE-100 allocation.json 복구")
    parser.add_argument(
        "--apply", action="store_true",
        help="실제 파일에 적용. 미지정 시 dry-run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="명시적 dry-run (기본 동작).",
    )
    parser.add_argument(
        "--allocation-path",
        default="/mnt/data/quant/data/portfolio_allocation.json",
        help="복구 대상 allocation.json 경로",
    )
    args = parser.parse_args()

    if args.apply and args.dry_run:
        print("ERROR: --apply 와 --dry-run 동시 지정 불가")
        return 2

    apply_mode = args.apply

    sys.path.insert(0, "/mnt/data/quant-dev")
    from src.execution.kis_client import KISClient
    from src.execution.portfolio_allocator import PortfolioAllocator
    from src.scheduler.main import TradingBot

    alloc_path = Path(args.allocation_path)
    if not alloc_path.exists():
        print(f"ERROR: allocation 파일 없음: {alloc_path}")
        return 1

    print(f"[1/5] 현재 allocation.json 백업 준비")
    backup_path = alloc_path.with_suffix(
        f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    print(f"[2/5] 현재 allocation 내용:")
    current = json.loads(alloc_path.read_text(encoding="utf-8"))
    print(json.dumps(current, ensure_ascii=False, indent=2))

    print(f"[3/5] KIS 잔고 조회 및 ETF 유니버스 수집")
    kis = KISClient()
    bot = TradingBot(kis_client=kis)
    etf_universe = bot._get_etf_universe_tickers()
    print(f"  ETF 유니버스 종목 수: {len(etf_universe)}")
    print(f"  ETF 유니버스: {sorted(etf_universe)}")

    balance = kis.get_balance()
    holdings = balance.get("holdings", [])
    print(f"  KIS 보유 종목 수: {len(holdings)}")
    for h in holdings:
        print(
            f"    {h.get('ticker')} qty={h.get('qty')} "
            f"eval={h.get('eval_amount')}"
        )

    print(f"[4/5] 복구 시뮬레이션 (long_term_pct={bot.allocator._long_term_pct})")
    # 복구 후 예상 상태 계산
    existing_tags = current.get("positions", {})
    plan: dict[str, dict] = dict(existing_tags)
    new_tags = 0
    for h in holdings:
        ticker = h.get("ticker", "")
        qty = h.get("qty", 0)
        if not ticker or qty <= 0:
            continue
        if ticker in plan:
            continue  # 이미 태깅
        if ticker in etf_universe:
            plan[ticker] = {
                "pool": "etf_rotation",
                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                "backfilled": True,
                "jae100_restore": True,
            }
        else:
            plan[ticker] = {
                "pool": "long_term",
                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                "backfilled": True,
                "jae100_restore": True,
            }
        new_tags += 1
        print(f"    + 신규 태깅: {ticker} → {plan[ticker]['pool']}")

    print(f"  신규 태깅 종목 수: {new_tags}")

    print(f"[5/5] 적용 단계")
    if not apply_mode:
        print("  --apply 미지정 → dry-run 종료. 적용하려면 --apply 사용.")
        print(f"  예상 결과 미리보기:")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if new_tags == 0:
        print("  변경 없음 — 파일 그대로 유지.")
        return 0

    print(f"  백업: {backup_path}")
    shutil.copy2(alloc_path, backup_path)

    new_doc = dict(current)
    new_doc["positions"] = plan
    new_doc["updated_at"] = datetime.now().isoformat(timespec="seconds")
    new_doc["jae100_restore_note"] = (
        f"JAE-100 복구: {new_tags}개 미태깅 포지션을 재태깅. "
        f"백업={backup_path.name}"
    )
    alloc_path.write_text(
        json.dumps(new_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  적용 완료: {alloc_path}")
    print(
        "  주의: 운영 중인 quant-bot 프로세스는 메모리 상태를 캐시한다. "
        "변경 사항이 즉시 반영되려면 systemd 재시작이 필요하다 (CEO 승인 필요)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
