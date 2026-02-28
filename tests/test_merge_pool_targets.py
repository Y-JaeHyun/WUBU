"""merge_pool_targets / get_pool_pct / auto_tag / backfill 테스트.

PortfolioAllocator의 통합 리밸런싱 메서드들을 검증한다.
KIS 클라이언트는 Mock으로 대체하며, tmp_path로 격리된 환경에서 테스트한다.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.execution.portfolio_allocator import PortfolioAllocator


# ===================================================================
# 헬퍼
# ===================================================================


def _alloc_path(tmp_path: Path) -> str:
    """테스트용 JSON 파일 경로를 반환한다."""
    return str(tmp_path / "test_allocation.json")


def _mock_kis(
    total_eval: int = 1_000_000,
    cash: int = 500_000,
    holdings: list | None = None,
):
    """KIS 클라이언트 Mock을 생성한다."""
    kis = MagicMock()
    kis.get_balance.return_value = {
        "total_eval": total_eval,
        "cash": cash,
        "holdings": holdings or [],
    }
    kis.get_current_price.return_value = {"price": 70000}
    return kis


def _make_allocator(
    tmp_path: Path,
    long_term_pct: float = 0.70,
    short_term_pct: float = 0.0,
    etf_rotation_pct: float = 0.30,
    total_eval: int = 1_000_000,
    holdings: list | None = None,
) -> PortfolioAllocator:
    """테스트용 PortfolioAllocator를 간편하게 생성한다."""
    kis = _mock_kis(total_eval=total_eval, holdings=holdings)
    return PortfolioAllocator(
        kis,
        long_term_pct=long_term_pct,
        short_term_pct=short_term_pct,
        etf_rotation_pct=etf_rotation_pct,
        allocation_path=_alloc_path(tmp_path),
    )


# ===================================================================
# 1. merge_pool_targets 테스트
# ===================================================================


class TestMergePoolTargets:
    """merge_pool_targets 메서드를 검증한다."""

    def test_single_pool_long_term_only(self, tmp_path):
        """장기 풀만 있으면 0.7로 스케일링된다."""
        alloc = _make_allocator(tmp_path, long_term_pct=0.70, etf_rotation_pct=0.30)

        pool_signals = {
            "long_term": {"005930": 0.5, "000660": 0.5},
        }
        merged = alloc.merge_pool_targets(pool_signals)

        assert merged["005930"] == pytest.approx(0.35)
        assert merged["000660"] == pytest.approx(0.35)
        total = sum(merged.values())
        assert total == pytest.approx(0.70)

    def test_two_pools_proper_scaling(self, tmp_path):
        """장기 70% + ETF 30% 비중으로 올바르게 스케일링된다."""
        alloc = _make_allocator(tmp_path, long_term_pct=0.70, etf_rotation_pct=0.30)

        pool_signals = {
            "long_term": {"005930": 0.5, "000660": 0.5},
            "etf_rotation": {"069500": 0.5, "371460": 0.5},
        }
        merged = alloc.merge_pool_targets(pool_signals)

        # long_term: 0.5*0.7=0.35 each
        assert merged["005930"] == pytest.approx(0.35)
        assert merged["000660"] == pytest.approx(0.35)
        # etf_rotation: 0.5*0.3=0.15 each
        assert merged["069500"] == pytest.approx(0.15)
        assert merged["371460"] == pytest.approx(0.15)
        total = sum(merged.values())
        assert total == pytest.approx(1.0)

    def test_overlapping_tickers_weights_sum(self, tmp_path):
        """같은 종목이 여러 풀에 포함되면 비중을 합산한다."""
        alloc = _make_allocator(tmp_path, long_term_pct=0.70, etf_rotation_pct=0.30)

        pool_signals = {
            "long_term": {"069500": 0.2, "005930": 0.8},
            "etf_rotation": {"069500": 1.0},
        }
        merged = alloc.merge_pool_targets(pool_signals)

        # 069500: 0.2*0.7 + 1.0*0.3 = 0.14 + 0.30 = 0.44
        assert merged["069500"] == pytest.approx(0.44)
        # 005930: 0.8*0.7 = 0.56
        assert merged["005930"] == pytest.approx(0.56)

    def test_empty_signals_for_one_pool(self, tmp_path):
        """한 풀의 시그널이 비어 있으면 다른 풀만 반영된다."""
        alloc = _make_allocator(tmp_path, long_term_pct=0.70, etf_rotation_pct=0.30)

        pool_signals = {
            "long_term": {"005930": 0.5, "000660": 0.5},
            "etf_rotation": {},
        }
        merged = alloc.merge_pool_targets(pool_signals)

        assert len(merged) == 2
        total = sum(merged.values())
        assert total == pytest.approx(0.70)

    def test_all_empty_signals(self, tmp_path):
        """모든 풀의 시그널이 비어 있으면 빈 딕셔너리를 반환한다."""
        alloc = _make_allocator(tmp_path)

        pool_signals = {
            "long_term": {},
            "etf_rotation": {},
        }
        merged = alloc.merge_pool_targets(pool_signals)

        assert merged == {}

    def test_pool_with_zero_allocation_excluded(self, tmp_path):
        """비율이 0인 풀의 시그널은 무시된다."""
        alloc = _make_allocator(
            tmp_path, long_term_pct=1.0, short_term_pct=0.0, etf_rotation_pct=0.0,
        )

        pool_signals = {
            "long_term": {"005930": 1.0},
            "etf_rotation": {"069500": 1.0},  # pct=0.0 -> 무시
        }
        merged = alloc.merge_pool_targets(pool_signals)

        assert "069500" not in merged
        assert merged["005930"] == pytest.approx(1.0)

    def test_sum_of_merged_weights_lte_one(self, tmp_path):
        """병합 비중 합은 1.0 이하이다."""
        alloc = _make_allocator(tmp_path, long_term_pct=0.70, etf_rotation_pct=0.30)

        pool_signals = {
            "long_term": {f"STOCK_{i:02d}": 0.1 for i in range(10)},
            "etf_rotation": {"ETF_A": 0.5, "ETF_B": 0.5},
        }
        merged = alloc.merge_pool_targets(pool_signals)

        total = sum(merged.values())
        assert total <= 1.0 + 1e-9

    def test_rounding_precision_no_artifacts(self, tmp_path):
        """부동소수점 아티팩트 없이 소수점 6자리로 반올림된다."""
        alloc = _make_allocator(
            tmp_path, long_term_pct=0.70, etf_rotation_pct=0.30,
        )

        pool_signals = {
            "long_term": {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
            "etf_rotation": {"D": 1 / 3, "E": 1 / 3, "F": 1 / 3},
        }
        merged = alloc.merge_pool_targets(pool_signals)

        for ticker, weight in merged.items():
            # 소수점 6자리 이하인지 확인
            s = f"{weight:.10f}"
            # round(x, 6) 결과이므로 7자리 이후는 0
            decimal_part = s.split(".")[1]
            assert decimal_part[6:] == "0000", (
                f"{ticker}={weight} has more than 6 decimal places"
            )


# ===================================================================
# 2. get_pool_pct 테스트
# ===================================================================


class TestGetPoolPct:
    """get_pool_pct 메서드를 검증한다."""

    def test_returns_long_term_pct(self, tmp_path):
        """long_term 풀의 비율을 정확히 반환한다."""
        alloc = _make_allocator(tmp_path, long_term_pct=0.70)

        assert alloc.get_pool_pct("long_term") == 0.70

    def test_returns_etf_rotation_pct(self, tmp_path):
        """etf_rotation 풀의 비율을 정확히 반환한다."""
        alloc = _make_allocator(tmp_path, etf_rotation_pct=0.30)

        assert alloc.get_pool_pct("etf_rotation") == 0.30

    def test_unknown_pool_returns_zero(self, tmp_path):
        """알 수 없는 풀 이름에 대해 0.0을 반환한다."""
        alloc = _make_allocator(tmp_path)

        assert alloc.get_pool_pct("unknown_pool") == 0.0
        assert alloc.get_pool_pct("") == 0.0
        assert alloc.get_pool_pct("day_trading") == 0.0


# ===================================================================
# 3. auto_tag_from_pool_signals 테스트
# ===================================================================


class TestAutoTagFromPoolSignals:
    """auto_tag_from_pool_signals 메서드를 검증한다."""

    def test_tags_positions_based_on_pool_signals(self, tmp_path):
        """풀 시그널에 따라 포지션을 태깅한다."""
        alloc = _make_allocator(tmp_path, long_term_pct=0.70, etf_rotation_pct=0.30)

        pool_signals = {
            "long_term": {"005930": 0.5, "000660": 0.5},
            "etf_rotation": {"069500": 1.0},
        }
        alloc.auto_tag_from_pool_signals(pool_signals)

        assert alloc.get_position_pool("005930") == "long_term"
        assert alloc.get_position_pool("000660") == "long_term"
        assert alloc.get_position_pool("069500") == "etf_rotation"

    def test_overlapping_ticker_tagged_to_higher_weight_pool(self, tmp_path):
        """동일 종목이 여러 풀에 포함되면 비중이 큰 풀로 태깅한다."""
        alloc = _make_allocator(tmp_path, long_term_pct=0.70, etf_rotation_pct=0.30)

        # 069500: long_term에서 0.2*0.7=0.14, etf_rotation에서 1.0*0.3=0.30
        # -> etf_rotation이 더 크므로 etf_rotation으로 태깅
        pool_signals = {
            "long_term": {"069500": 0.2},
            "etf_rotation": {"069500": 1.0},
        }
        alloc.auto_tag_from_pool_signals(pool_signals)

        assert alloc.get_position_pool("069500") == "etf_rotation"

    def test_overlapping_ticker_tagged_to_long_term_when_higher(self, tmp_path):
        """long_term의 스케일 비중이 더 크면 long_term으로 태깅한다."""
        alloc = _make_allocator(tmp_path, long_term_pct=0.70, etf_rotation_pct=0.30)

        # 069500: long_term에서 1.0*0.7=0.70, etf_rotation에서 0.1*0.3=0.03
        # -> long_term이 더 크므로 long_term으로 태깅
        pool_signals = {
            "long_term": {"069500": 1.0},
            "etf_rotation": {"069500": 0.1},
        }
        alloc.auto_tag_from_pool_signals(pool_signals)

        assert alloc.get_position_pool("069500") == "long_term"

    def test_stale_positions_cleaned(self, tmp_path):
        """시그널에 더 이상 없는 기존 장기/ETF 태그는 제거된다."""
        alloc = _make_allocator(tmp_path, long_term_pct=0.70, etf_rotation_pct=0.30)

        # 1차: 005930 태깅
        alloc.tag_position("005930", "long_term")
        assert alloc.get_position_pool("005930") == "long_term"

        # 2차: 005930이 시그널에 없음 -> 태그 제거됨
        pool_signals = {
            "long_term": {"000660": 1.0},
            "etf_rotation": {},
        }
        alloc.auto_tag_from_pool_signals(pool_signals)

        assert alloc.get_position_pool("005930") is None
        assert alloc.get_position_pool("000660") == "long_term"

    def test_stale_short_term_positions_preserved(self, tmp_path):
        """short_term 풀의 기존 태그는 시그널에 없어도 보존된다."""
        alloc = _make_allocator(
            tmp_path, long_term_pct=0.70, short_term_pct=0.0, etf_rotation_pct=0.30,
        )

        # short_term으로 태깅된 종목
        alloc.tag_position("035420", "short_term")
        assert alloc.get_position_pool("035420") == "short_term"

        # 시그널에 035420 없지만 short_term이므로 보존
        pool_signals = {
            "long_term": {"005930": 1.0},
        }
        alloc.auto_tag_from_pool_signals(pool_signals)

        assert alloc.get_position_pool("035420") == "short_term"
        assert alloc.get_position_pool("005930") == "long_term"

    def test_empty_signals_no_changes_except_cleanup(self, tmp_path):
        """모든 시그널이 비어 있으면 기존 장기/ETF 태그가 정리된다."""
        alloc = _make_allocator(tmp_path)

        alloc.tag_position("005930", "long_term")
        alloc.tag_position("069500", "etf_rotation")

        pool_signals = {
            "long_term": {},
            "etf_rotation": {},
        }
        alloc.auto_tag_from_pool_signals(pool_signals)

        # 시그널에 없으므로 태그 제거됨
        assert alloc.get_position_pool("005930") is None
        assert alloc.get_position_pool("069500") is None

    def test_multiple_pools_multiple_tickers(self, tmp_path):
        """여러 풀 x 여러 종목을 올바르게 태깅한다."""
        alloc = _make_allocator(
            tmp_path, long_term_pct=0.60, short_term_pct=0.10, etf_rotation_pct=0.30,
        )

        pool_signals = {
            "long_term": {"005930": 0.3, "000660": 0.3, "035420": 0.4},
            "etf_rotation": {"069500": 0.5, "371460": 0.5},
            "short_term": {"003490": 1.0},
        }
        alloc.auto_tag_from_pool_signals(pool_signals)

        assert alloc.get_position_pool("005930") == "long_term"
        assert alloc.get_position_pool("000660") == "long_term"
        assert alloc.get_position_pool("035420") == "long_term"
        assert alloc.get_position_pool("069500") == "etf_rotation"
        assert alloc.get_position_pool("371460") == "etf_rotation"
        assert alloc.get_position_pool("003490") == "short_term"


# ===================================================================
# 4. backfill_untagged_positions 테스트
# ===================================================================


class TestBackfillUntaggedPositions:
    """backfill_untagged_positions 메서드를 검증한다."""

    def test_etf_tickers_tagged_as_etf_rotation(self, tmp_path):
        """ETF 유니버스에 포함된 미태깅 종목은 etf_rotation으로 태깅된다."""
        holdings = [
            {"ticker": "069500", "eval_amount": 300_000, "current_price": 30000,
             "qty": 10, "pnl": 0, "pnl_pct": 0.0},
            {"ticker": "371460", "eval_amount": 200_000, "current_price": 20000,
             "qty": 10, "pnl": 0, "pnl_pct": 0.0},
        ]
        alloc = _make_allocator(tmp_path, holdings=holdings)

        etf_universe = {"069500", "371460", "252670"}
        count = alloc.backfill_untagged_positions(etf_tickers=etf_universe)

        assert count == 2
        assert alloc.get_position_pool("069500") == "etf_rotation"
        assert alloc.get_position_pool("371460") == "etf_rotation"

    def test_non_etf_tickers_tagged_as_default_pool(self, tmp_path):
        """ETF가 아닌 미태깅 종목은 기본 풀(long_term)로 태깅된다."""
        holdings = [
            {"ticker": "005930", "eval_amount": 500_000, "current_price": 70000,
             "qty": 7, "pnl": 0, "pnl_pct": 0.0},
            {"ticker": "000660", "eval_amount": 300_000, "current_price": 130000,
             "qty": 2, "pnl": 0, "pnl_pct": 0.0},
        ]
        alloc = _make_allocator(tmp_path, holdings=holdings)

        etf_universe = {"069500", "371460"}
        count = alloc.backfill_untagged_positions(etf_tickers=etf_universe)

        assert count == 2
        assert alloc.get_position_pool("005930") == "long_term"
        assert alloc.get_position_pool("000660") == "long_term"

    def test_mixed_etf_and_non_etf(self, tmp_path):
        """ETF와 주식이 섞여 있으면 각각 적절한 풀로 태깅된다."""
        holdings = [
            {"ticker": "005930", "eval_amount": 500_000, "current_price": 70000,
             "qty": 7, "pnl": 0, "pnl_pct": 0.0},
            {"ticker": "069500", "eval_amount": 300_000, "current_price": 30000,
             "qty": 10, "pnl": 0, "pnl_pct": 0.0},
        ]
        alloc = _make_allocator(tmp_path, holdings=holdings)

        etf_universe = {"069500"}
        count = alloc.backfill_untagged_positions(etf_tickers=etf_universe)

        assert count == 2
        assert alloc.get_position_pool("005930") == "long_term"
        assert alloc.get_position_pool("069500") == "etf_rotation"

    def test_already_tagged_positions_not_retagged(self, tmp_path):
        """이미 태깅된 종목은 건너뛴다."""
        holdings = [
            {"ticker": "005930", "eval_amount": 500_000, "current_price": 70000,
             "qty": 7, "pnl": 0, "pnl_pct": 0.0},
            {"ticker": "069500", "eval_amount": 300_000, "current_price": 30000,
             "qty": 10, "pnl": 0, "pnl_pct": 0.0},
        ]
        alloc = _make_allocator(tmp_path, holdings=holdings)

        # 미리 태깅
        alloc.tag_position("005930", "short_term")

        etf_universe = {"069500"}
        count = alloc.backfill_untagged_positions(etf_tickers=etf_universe)

        # 005930은 이미 태깅 -> 건너뜀, 069500만 태깅
        assert count == 1
        assert alloc.get_position_pool("005930") == "short_term"  # 변경 안 됨
        assert alloc.get_position_pool("069500") == "etf_rotation"

    def test_empty_holdings_no_crash(self, tmp_path):
        """보유 종목이 없어도 에러 없이 0을 반환한다."""
        alloc = _make_allocator(tmp_path, holdings=[])

        count = alloc.backfill_untagged_positions(etf_tickers={"069500"})

        assert count == 0

    def test_no_etf_tickers_param_all_go_to_default(self, tmp_path):
        """etf_tickers가 None이면 모든 미태깅 종목이 기본 풀로 태깅된다."""
        holdings = [
            {"ticker": "069500", "eval_amount": 300_000, "current_price": 30000,
             "qty": 10, "pnl": 0, "pnl_pct": 0.0},
        ]
        alloc = _make_allocator(tmp_path, holdings=holdings)

        count = alloc.backfill_untagged_positions(etf_tickers=None)

        assert count == 1
        # etf_tickers가 None이므로 ETF라도 long_term으로 태깅
        assert alloc.get_position_pool("069500") == "long_term"

    def test_kis_failure_returns_zero(self, tmp_path):
        """KIS 잔고 조회 실패 시 0을 반환하고 크래시하지 않는다."""
        kis = _mock_kis()
        kis.get_balance.side_effect = Exception("KIS 점검 중")
        alloc = PortfolioAllocator(
            kis,
            long_term_pct=0.70,
            etf_rotation_pct=0.30,
            allocation_path=_alloc_path(tmp_path),
        )

        count = alloc.backfill_untagged_positions(etf_tickers={"069500"})

        assert count == 0
