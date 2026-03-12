#!/usr/bin/env python3
"""전략 배치 백테스트 러너.

모든 전략 & 옵션 조합을 순차 실행하고 결과를 JSON으로 저장한다.
체크포인트 기반으로 중단/재개가 가능하다.

사용법:
    python scripts/batch_backtest.py                    # 전체 실행
    python scripts/batch_backtest.py --group A          # 그룹 A만 실행
    python scripts/batch_backtest.py --strategy multi_factor  # 특정 전략만
    python scripts/batch_backtest.py --resume            # 중단된 지점부터 재개
"""
import sys
import os
import json
import hashlib
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, "/mnt/data/quant-dev")
os.chdir("/mnt/data/quant-dev")

from dotenv import load_dotenv
load_dotenv("/mnt/data/quant-dev/.env")

import pandas as pd

from src.backtest.engine import Backtest, Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── 설정 ──────────────────────────────────────────
RESULTS_DIR = Path("data/backtest_results")
PROGRESS_FILE = RESULTS_DIR / "progress.json"
DEFAULT_START = "20220312"
DEFAULT_END = "20260312"
DEFAULT_CAPITAL = 1_500_000


# ─── 전략 레지스트리 ───────────────────────────────

# 그룹 분류:
#   A: 기본 팩터 전략 (value, momentum, quality, three_factor, multi_factor)
#   B: ETF/자산배분 전략 (etf_rotation, enhanced_etf_rotation, risk_parity, dual_momentum)
#   C: 고급 팩터 전략 (low_volatility, low_vol_quality, shareholder_yield, accrual, pead,
#                      cross_asset_momentum, hybrid_strategy, ml_factor)
#   D: 단기 전략 (bb_squeeze, high_breakout, swing_reversion, orb_daytrading)

STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    # ─── Group A: 기본 팩터 ───
    "value": {
        "group": "A",
        "module": "src.strategy.value",
        "class": "ValueStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
            {"label": "top10", "kwargs": {"num_stocks": 10}},
        ],
    },
    "momentum": {
        "group": "A",
        "module": "src.strategy.momentum",
        "class": "MomentumStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
            {"label": "top10", "kwargs": {"num_stocks": 10}},
        ],
    },
    "quality": {
        "group": "A",
        "module": "src.strategy.quality",
        "class": "QualityStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "three_factor": {
        "group": "A",
        "module": "src.strategy.three_factor",
        "class": "ThreeFactorStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
            {"label": "top10", "kwargs": {"num_stocks": 10}},
        ],
    },
    "multi_factor": {
        "group": "A",
        "module": "src.strategy.multi_factor",
        "class": "MultiFactorStrategy",
        "base": "Strategy",
        "configs": [
            {
                "label": "live",
                "kwargs": {
                    "factors": ["value", "momentum"],
                    "weights": [0.5, 0.5],
                    "combine_method": "zscore",
                    "num_stocks": 7,
                    "apply_market_timing": True,
                    "turnover_penalty": 0.1,
                    "max_group_weight": 0.25,
                    "max_stocks_per_conglomerate": 2,
                    "spike_filter": True,
                },
                "is_live": True,
            },
            {
                "label": "backtest",
                "kwargs": {
                    "factors": ["value", "momentum"],
                    "weights": [0.5, 0.5],
                    "combine_method": "zscore",
                    "num_stocks": 10,
                    "apply_market_timing": False,
                    "turnover_penalty": 0.1,
                    "max_group_weight": 0.25,
                    "max_stocks_per_conglomerate": 2,
                    "spike_filter": True,
                },
            },
        ],
    },
    # ─── Group B: ETF/자산배분 ───
    "etf_rotation": {
        "group": "B",
        "module": "src.strategy.etf_rotation",
        "class": "ETFRotationStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "enhanced_etf_rotation": {
        "group": "B",
        "module": "src.strategy.enhanced_etf_rotation",
        "class": "EnhancedETFRotationStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "risk_parity": {
        "group": "B",
        "module": "src.strategy.risk_parity",
        "class": "RiskParityStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "dual_momentum": {
        "group": "B",
        "module": "src.strategy.dual_momentum",
        "class": "DualMomentumStrategy",
        "base": "independent",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    # ─── Group C: 고급 팩터 ───
    "low_volatility": {
        "group": "C",
        "module": "src.strategy.low_volatility",
        "class": "LowVolatilityStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "low_vol_quality": {
        "group": "C",
        "module": "src.strategy.low_vol_quality",
        "class": "LowVolQualityStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "shareholder_yield": {
        "group": "C",
        "module": "src.strategy.shareholder_yield",
        "class": "ShareholderYieldStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "accrual": {
        "group": "C",
        "module": "src.strategy.accrual",
        "class": "AccrualStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "pead": {
        "group": "C",
        "module": "src.strategy.pead",
        "class": "PEADStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "cross_asset_momentum": {
        "group": "C",
        "module": "src.strategy.cross_asset_momentum",
        "class": "CrossAssetMomentumStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "hybrid_strategy": {
        "group": "C",
        "module": "src.strategy.hybrid_strategy",
        "class": "HybridStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "ml_factor": {
        "group": "C",
        "module": "src.strategy.ml_factor",
        "class": "MLFactorStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    # ─── R&D: 신규 전략 ───
    "size_value": {
        "group": "C",
        "module": "src.strategy.size_value",
        "class": "SizeValueStrategy",
        "base": "Strategy",
        "configs": [
            {"label": "default", "kwargs": {}},
            {"label": "aggressive", "kwargs": {
                "size_pct": 0.20,
                "value_pct": 0.15,
                "max_stocks": 15,
            }},
        ],
    },
    # ─── Group D: 단기 전략 ───
    "bb_squeeze": {
        "group": "D",
        "module": "src.strategy.bb_squeeze",
        "class": "BBSqueezeStrategy",
        "base": "ShortTermStrategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "high_breakout": {
        "group": "D",
        "module": "src.strategy.high_breakout",
        "class": "HighBreakoutStrategy",
        "base": "ShortTermStrategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "swing_reversion": {
        "group": "D",
        "module": "src.strategy.swing_reversion",
        "class": "SwingReversionStrategy",
        "base": "ShortTermStrategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
    "orb_daytrading": {
        "group": "D",
        "module": "src.strategy.orb_daytrading",
        "class": "ORBDaytradingStrategy",
        "base": "ShortTermStrategy",
        "configs": [
            {"label": "default", "kwargs": {}},
        ],
    },
}


def _config_hash(strategy_name: str, label: str) -> str:
    """전략+설정 조합의 고유 해시를 생성한다."""
    key = f"{strategy_name}_{label}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _task_id(strategy_name: str, label: str) -> str:
    """체크포인트용 태스크 ID를 생성한다."""
    return f"{strategy_name}_{label}"


# ─── 체크포인트 관리 ──────────────────────────────

def load_progress() -> dict:
    """체크포인트 파일을 로드한다."""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"done": [], "pending": [], "failed": [], "updated_at": ""}


def save_progress(progress: dict) -> None:
    """체크포인트 파일을 저장한다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = datetime.now().isoformat(timespec="seconds")
    PROGRESS_FILE.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ─── 전략 인스턴스 생성 ──────────────────────────

def create_strategy(name: str, config: dict) -> Optional[Strategy]:
    """전략 인스턴스를 동적으로 생성한다."""
    import importlib

    reg = STRATEGY_REGISTRY.get(name)
    if not reg:
        logger.error(f"알 수 없는 전략: {name}")
        return None

    try:
        mod = importlib.import_module(reg["module"])
        cls = getattr(mod, reg["class"])
        kwargs = config.get("kwargs", {})
        return cls(**kwargs)
    except Exception as e:
        logger.error(f"전략 생성 실패 ({name}): {e}")
        return None


# ─── 단일 백테스트 실행 ──────────────────────────

def run_single_backtest(
    strategy_name: str,
    config: dict,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    initial_capital: float = DEFAULT_CAPITAL,
) -> Optional[dict]:
    """단일 전략의 백테스트를 실행하고 결과를 반환한다."""
    reg = STRATEGY_REGISTRY[strategy_name]
    label = config.get("label", "default")
    base_type = reg.get("base", "Strategy")

    logger.info(f"백테스트 시작: {strategy_name} ({label})")

    strategy = create_strategy(strategy_name, config)
    if strategy is None:
        return None

    try:
        if base_type == "ShortTermStrategy":
            # 단기 전략은 별도 백테스트 엔진 사용
            result = _run_short_term_backtest(
                strategy, strategy_name, config,
                start_date, end_date, initial_capital,
            )
        else:
            result = _run_long_term_backtest(
                strategy, strategy_name, config,
                start_date, end_date, initial_capital,
            )

        if result:
            # 운영환경 비교 정보 추가
            result["live_comparison"] = _compare_with_live(strategy_name, config)

        return result
    except Exception as e:
        logger.error(f"백테스트 실패 ({strategy_name}/{label}): {e}")
        logger.debug(traceback.format_exc())
        return None


def _run_long_term_backtest(
    strategy: Strategy,
    strategy_name: str,
    config: dict,
    start_date: str,
    end_date: str,
    initial_capital: float,
) -> Optional[dict]:
    """장기 전략 백테스트를 실행한다."""
    label = config.get("label", "default")

    bt = Backtest(
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        rebalance_freq="monthly",
    )
    bt.run()

    results = bt.get_results()
    trades_df = bt.get_trades()
    history_df = bt.get_portfolio_history()

    # 리밸런싱 로그 구축
    rebalancing_log = _build_rebalancing_log(trades_df, history_df)

    return {
        "strategy": strategy_name,
        "label": label,
        "options": config.get("kwargs", {}),
        "period": {"start": start_date, "end": end_date},
        "initial_capital": initial_capital,
        "rebalancing_log": rebalancing_log,
        "summary": {
            "total_return": results.get("total_return", 0),
            "cagr": results.get("cagr", 0),
            "max_drawdown": results.get("mdd", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "win_rate": results.get("win_rate", 0),
            "num_rebalancing": results.get("rebalance_count", 0),
            "total_trades": results.get("total_trades", 0),
            "final_value": results.get("final_value", 0),
        },
        "is_live": config.get("is_live", False),
    }


def _run_short_term_backtest(
    strategy,
    strategy_name: str,
    config: dict,
    start_date: str,
    end_date: str,
    initial_capital: float,
) -> Optional[dict]:
    """단기 전략 백테스트를 실행한다."""
    label = config.get("label", "default")

    try:
        from src.backtest.short_term_backtest import ShortTermBacktest
        bt = ShortTermBacktest(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )
        bt.run()
        results = bt.get_results()

        return {
            "strategy": strategy_name,
            "label": label,
            "options": config.get("kwargs", {}),
            "period": {"start": start_date, "end": end_date},
            "initial_capital": initial_capital,
            "rebalancing_log": [],
            "summary": {
                "total_return": results.get("total_return", 0),
                "cagr": results.get("cagr", 0),
                "max_drawdown": results.get("mdd", 0),
                "sharpe_ratio": results.get("sharpe_ratio", 0),
                "win_rate": results.get("win_rate", 0),
                "num_rebalancing": 0,
                "total_trades": results.get("total_trades", 0),
                "final_value": results.get("final_value", 0),
            },
            "is_live": config.get("is_live", False),
        }
    except Exception as e:
        logger.warning(f"단기 백테스트 엔진 실행 실패 ({strategy_name}): {e}")
        return None


def _build_rebalancing_log(
    trades_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> list[dict]:
    """거래 내역과 포트폴리오 이력에서 리밸런싱 로그를 생성한다."""
    if trades_df.empty:
        return []

    log: list[dict] = []
    for date, group in trades_df.groupby("date"):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        buys = []
        sells = []

        for _, trade in group.iterrows():
            entry = {
                "ticker": trade["ticker"],
                "quantity": int(trade.get("quantity", 0)),
                "price": float(trade.get("price", 0)),
            }
            if trade["action"] == "buy":
                buys.append(entry)
            else:
                sells.append(entry)

        pv = None
        cash = None
        ts = pd.Timestamp(date)
        if ts in history_df.index:
            pv = float(history_df.loc[ts, "portfolio_value"])
            cash = float(history_df.loc[ts, "cash"])

        log.append({
            "date": date_str,
            "buys": buys,
            "sells": sells,
            "portfolio_value": pv,
            "cash": cash,
        })

    return log


def _compare_with_live(strategy_name: str, config: dict) -> dict:
    """운영환경 설정과 비교한다."""
    comparison: dict[str, Any] = {"is_live_config": config.get("is_live", False)}

    if strategy_name == "multi_factor":
        try:
            from src.strategy.strategy_config import (
                MULTI_FACTOR_BASE,
                MULTI_FACTOR_PROFILES,
            )
            live_config = {**MULTI_FACTOR_BASE, **MULTI_FACTOR_PROFILES.get("live", {})}
            bt_kwargs = config.get("kwargs", {})

            diffs = {}
            for key in set(live_config.keys()) | set(bt_kwargs.keys()):
                live_val = live_config.get(key)
                bt_val = bt_kwargs.get(key)
                if live_val != bt_val and bt_val is not None:
                    diffs[key] = {"live": live_val, "backtest": bt_val}

            comparison["config_diffs"] = diffs
        except Exception:
            pass

    # 운영환경 파일 확인
    prod_config_path = "/mnt/data/quant/src/strategy/strategy_config.py"
    comparison["prod_config_exists"] = os.path.exists(prod_config_path)

    return comparison


# ─── 결과 저장 ────────────────────────────────────

def save_result(strategy_name: str, label: str, result: dict) -> Path:
    """백테스트 결과를 JSON 파일로 저장한다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    h = _config_hash(strategy_name, label)
    filename = f"{strategy_name}_{label}_{h}.json"
    filepath = RESULTS_DIR / filename

    result["saved_at"] = datetime.now().isoformat(timespec="seconds")
    filepath.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"결과 저장: {filepath}")
    return filepath


# ─── 배치 실행 ────────────────────────────────────

def get_tasks(
    group: Optional[str] = None,
    strategy: Optional[str] = None,
) -> list[tuple[str, dict]]:
    """실행할 태스크 목록을 반환한다."""
    tasks = []
    for name, reg in STRATEGY_REGISTRY.items():
        if group and reg["group"] != group:
            continue
        if strategy and name != strategy:
            continue
        for config in reg["configs"]:
            tasks.append((name, config))
    return tasks


def run_batch(
    group: Optional[str] = None,
    strategy: Optional[str] = None,
    resume: bool = True,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    initial_capital: float = DEFAULT_CAPITAL,
) -> dict:
    """배치 백테스트를 실행한다."""
    tasks = get_tasks(group=group, strategy=strategy)
    progress = load_progress() if resume else {
        "done": [], "pending": [], "failed": [], "updated_at": "",
    }

    # pending 초기화
    all_ids = [_task_id(name, cfg["label"]) for name, cfg in tasks]
    done_set = set(progress.get("done", []))
    progress["pending"] = [tid for tid in all_ids if tid not in done_set]
    save_progress(progress)

    total = len(tasks)
    completed = 0
    failed = 0

    print(f"=== 배치 백테스트 시작 ===")
    print(f"  전략 수: {total}")
    print(f"  기간: {start_date} ~ {end_date}")
    print(f"  초기자본: {initial_capital:,.0f}원")
    if group:
        print(f"  그룹: {group}")
    print()

    for name, config in tasks:
        tid = _task_id(name, config["label"])

        if resume and tid in done_set:
            print(f"  [SKIP] {tid} (이미 완료)")
            completed += 1
            continue

        print(f"  [{completed + failed + 1}/{total}] {tid} 실행 중...", flush=True)

        result = run_single_backtest(
            name, config, start_date, end_date, initial_capital
        )

        if result:
            save_result(name, config["label"], result)
            progress["done"].append(tid)
            if tid in progress["pending"]:
                progress["pending"].remove(tid)
            completed += 1

            summary = result.get("summary", {})
            tag = " [LIVE]" if config.get("is_live") else ""
            print(
                f"    -> CAGR={summary.get('cagr', 0):.1f}%, "
                f"MDD={summary.get('max_drawdown', 0):.1f}%, "
                f"Sharpe={summary.get('sharpe_ratio', 0):.2f}"
                f"{tag}"
            )
        else:
            progress["failed"].append(tid)
            if tid in progress["pending"]:
                progress["pending"].remove(tid)
            failed += 1
            print(f"    -> FAILED")

        save_progress(progress)

    print()
    print(f"=== 배치 백테스트 완료 ===")
    print(f"  완료: {completed}, 실패: {failed}, 잔여: {len(progress['pending'])}")

    return progress


# ─── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="전략 배치 백테스트 러너")
    parser.add_argument("--group", choices=["A", "B", "C", "D"], help="실행할 그룹")
    parser.add_argument("--strategy", help="특정 전략만 실행")
    parser.add_argument("--resume", action="store_true", default=True, help="중단 지점부터 재개")
    parser.add_argument("--fresh", action="store_true", help="처음부터 실행 (체크포인트 무시)")
    parser.add_argument("--start", default=DEFAULT_START, help="시작일 (YYYYMMDD)")
    parser.add_argument("--end", default=DEFAULT_END, help="종료일 (YYYYMMDD)")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL, help="초기자본금")

    args = parser.parse_args()

    run_batch(
        group=args.group,
        strategy=args.strategy,
        resume=not args.fresh,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )


if __name__ == "__main__":
    main()
