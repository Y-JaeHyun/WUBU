"""PortfolioAllocator 테스트.

JSON 기반 포지션 태깅과 장기/단기 풀 분리를 검증한다.
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


def _read_json(path: str) -> dict:
    """JSON 파일을 읽어 딕셔너리를 반환한다."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mock_kis(total_eval: int = 1_000_000, cash: int = 500_000, holdings: list | None = None):
    """KIS 클라이언트 Mock을 생성한다."""
    kis = MagicMock()
    kis.get_balance.return_value = {
        "total_eval": total_eval,
        "cash": cash,
        "holdings": holdings or [],
    }
    kis.get_current_price.return_value = {"price": 70000}
    return kis


# ===================================================================
# 1. 초기화 테스트
# ===================================================================


class TestInitialization:
    """PortfolioAllocator 초기화를 검증한다."""

    def test_init_default_values(self, tmp_path):
        """기본 비율로 초기화된다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        assert alloc._long_term_pct == 0.90
        assert alloc._short_term_pct == 0.10

    def test_init_custom_values(self, tmp_path):
        """사용자 지정 비율로 초기화된다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(
            kis,
            long_term_pct=0.80,
            short_term_pct=0.20,
            allocation_path=path,
        )

        assert alloc._long_term_pct == 0.80
        assert alloc._short_term_pct == 0.20

    def test_init_creates_json_file(self, tmp_path):
        """초기화 시 JSON 파일이 생성된다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        assert Path(path).exists()
        data = _read_json(path)
        assert data["version"] == 1
        assert "config" in data
        assert "positions" in data

    def test_init_loads_existing_file(self, tmp_path):
        """기존 JSON 파일이 있으면 해당 파일을 로드한다."""
        path = _alloc_path(tmp_path)
        existing = {
            "version": 1,
            "updated_at": "2026-01-01T00:00:00",
            "config": {
                "long_term_pct": 0.70,
                "short_term_pct": 0.30,
                "soft_cap_mode": True,
            },
            "positions": {
                "005930": {
                    "pool": "long_term",
                    "entry_date": "2026-01-01",
                }
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        # 파일에서 비율 복원
        assert alloc._long_term_pct == 0.70
        assert alloc._short_term_pct == 0.30
        assert alloc.get_position_pool("005930") == "long_term"


# ===================================================================
# 2. 예산 조회 테스트
# ===================================================================


class TestBudget:
    """예산 관련 메서드를 검증한다."""

    def test_get_total_portfolio_value(self, tmp_path):
        """KIS 잔고에서 총 포트폴리오 가치를 반환한다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis(total_eval=2_000_000)
        alloc = PortfolioAllocator(kis, allocation_path=path)

        assert alloc.get_total_portfolio_value() == 2_000_000
        kis.get_balance.assert_called()

    def test_get_total_portfolio_value_uses_cache_on_failure(self, tmp_path):
        """KIS 조회 실패 시 캐시 값을 사용한다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis(total_eval=1_500_000)
        alloc = PortfolioAllocator(kis, allocation_path=path)

        # 첫 호출 성공 -> 캐시 저장
        alloc.get_total_portfolio_value()

        # 이후 실패
        kis.get_balance.side_effect = Exception("KIS 점검 중")
        result = alloc.get_total_portfolio_value()
        assert result == 1_500_000

    def test_get_long_term_budget(self, tmp_path):
        """장기 풀 예산을 반환한다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis(total_eval=1_000_000)
        alloc = PortfolioAllocator(
            kis, long_term_pct=0.90, short_term_pct=0.10, allocation_path=path
        )

        assert alloc.get_long_term_budget() == 900_000

    def test_get_short_term_budget(self, tmp_path):
        """단기 풀 예산을 반환한다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis(total_eval=1_000_000)
        alloc = PortfolioAllocator(
            kis, long_term_pct=0.90, short_term_pct=0.10, allocation_path=path
        )

        assert alloc.get_short_term_budget() == 100_000

    def test_get_short_term_cash(self, tmp_path):
        """단기 풀의 사용 가능 현금을 반환한다."""
        path = _alloc_path(tmp_path)
        holdings = [
            {"ticker": "005930", "eval_amount": 30_000, "current_price": 70000, "qty": 1, "pnl": 0, "pnl_pct": 0.0},
        ]
        kis = _mock_kis(total_eval=1_000_000, holdings=holdings)
        alloc = PortfolioAllocator(kis, allocation_path=path)

        # 005930을 단기로 태깅
        alloc.tag_position("005930", "short_term")

        # 단기 예산: 100,000 - 30,000 = 70,000
        cash = alloc.get_short_term_cash()
        assert cash == 70_000

    def test_get_long_term_cash(self, tmp_path):
        """장기 풀의 사용 가능 현금을 반환한다."""
        path = _alloc_path(tmp_path)
        holdings = [
            {"ticker": "005930", "eval_amount": 500_000, "current_price": 70000, "qty": 7, "pnl": 0, "pnl_pct": 0.0},
        ]
        kis = _mock_kis(total_eval=1_000_000, holdings=holdings)
        alloc = PortfolioAllocator(kis, allocation_path=path)

        alloc.tag_position("005930", "long_term")

        # 장기 예산: 900,000 - 500,000 = 400,000
        cash = alloc.get_long_term_cash()
        assert cash == 400_000

    def test_get_short_term_cash_no_negative(self, tmp_path):
        """단기 현금이 음수가 되지 않는다."""
        path = _alloc_path(tmp_path)
        holdings = [
            {"ticker": "005930", "eval_amount": 200_000, "current_price": 70000, "qty": 3, "pnl": 0, "pnl_pct": 0.0},
        ]
        kis = _mock_kis(total_eval=1_000_000, holdings=holdings)
        alloc = PortfolioAllocator(kis, allocation_path=path)

        alloc.tag_position("005930", "short_term")

        # 단기 예산: 100,000, 사용: 200,000 -> max(100000 - 200000, 0) = 0
        assert alloc.get_short_term_cash() == 0


# ===================================================================
# 3. 포지션 태깅 테스트
# ===================================================================


class TestPositionTagging:
    """포지션 태깅 관련 메서드를 검증한다."""

    def test_tag_position_long_term(self, tmp_path):
        """종목을 장기 풀에 태깅한다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        alloc.tag_position("005930", "long_term", {"entry_price": 70000, "strategy": "multi_factor"})

        assert alloc.get_position_pool("005930") == "long_term"

    def test_tag_position_short_term(self, tmp_path):
        """종목을 단기 풀에 태깅한다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        alloc.tag_position("000660", "short_term")

        assert alloc.get_position_pool("000660") == "short_term"

    def test_tag_position_invalid_pool(self, tmp_path):
        """유효하지 않은 풀 이름에 ValueError를 발생시킨다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        with pytest.raises(ValueError, match="long_term.*short_term"):
            alloc.tag_position("005930", "invalid_pool")

    def test_untag_position(self, tmp_path):
        """종목의 태그를 제거한다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        alloc.tag_position("005930", "long_term")
        assert alloc.get_position_pool("005930") == "long_term"

        alloc.untag_position("005930")
        assert alloc.get_position_pool("005930") is None

    def test_untag_nonexistent_position(self, tmp_path):
        """존재하지 않는 종목의 untag는 에러 없이 무시된다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        # 예외 없이 통과
        alloc.untag_position("999999")

    def test_get_positions_by_pool(self, tmp_path):
        """특정 풀에 속한 포지션 목록을 반환한다."""
        path = _alloc_path(tmp_path)
        holdings = [
            {"ticker": "005930", "eval_amount": 500_000, "current_price": 70000, "qty": 7, "pnl": 5000, "pnl_pct": 1.0},
            {"ticker": "000660", "eval_amount": 200_000, "current_price": 130000, "qty": 2, "pnl": -1000, "pnl_pct": -0.5},
            {"ticker": "035420", "eval_amount": 50_000, "current_price": 250000, "qty": 1, "pnl": 0, "pnl_pct": 0.0},
        ]
        kis = _mock_kis(total_eval=750_000, holdings=holdings)
        alloc = PortfolioAllocator(kis, allocation_path=path)

        alloc.tag_position("005930", "long_term")
        alloc.tag_position("000660", "long_term")
        alloc.tag_position("035420", "short_term")

        long_positions = alloc.get_positions_by_pool("long_term")
        short_positions = alloc.get_positions_by_pool("short_term")

        assert len(long_positions) == 2
        assert len(short_positions) == 1
        tickers_long = {p["ticker"] for p in long_positions}
        assert tickers_long == {"005930", "000660"}
        assert short_positions[0]["ticker"] == "035420"
        assert short_positions[0]["eval_amount"] == 50_000

    def test_get_position_pool_untagged(self, tmp_path):
        """태깅되지 않은 종목은 None을 반환한다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        assert alloc.get_position_pool("999999") is None

    def test_tag_position_overwrite(self, tmp_path):
        """같은 종목을 다른 풀로 재태깅하면 덮어쓴다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        alloc.tag_position("005930", "long_term")
        assert alloc.get_position_pool("005930") == "long_term"

        alloc.tag_position("005930", "short_term")
        assert alloc.get_position_pool("005930") == "short_term"


# ===================================================================
# 4. 리밸런싱 통합 테스트
# ===================================================================


class TestRebalanceIntegration:
    """리밸런싱 관련 메서드를 검증한다."""

    def test_filter_long_term_weights(self, tmp_path):
        """장기 전략 weights를 장기 비율로 스케일링한다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(
            kis, long_term_pct=0.90, short_term_pct=0.10, allocation_path=path
        )

        target = {"A": 0.10, "B": 0.10, "C": 0.10}
        scaled = alloc.filter_long_term_weights(target)

        assert scaled["A"] == pytest.approx(0.09)
        assert scaled["B"] == pytest.approx(0.09)
        assert scaled["C"] == pytest.approx(0.09)

    def test_filter_long_term_weights_scales_correctly(self, tmp_path):
        """10개 종목 각 10%를 장기 90%로 스케일링하면 각 9%가 된다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(
            kis, long_term_pct=0.90, short_term_pct=0.10, allocation_path=path
        )

        target = {f"STOCK_{i:02d}": 0.10 for i in range(10)}
        scaled = alloc.filter_long_term_weights(target)

        total_weight = sum(scaled.values())
        assert total_weight == pytest.approx(0.90, abs=0.001)
        for w in scaled.values():
            assert w == pytest.approx(0.09, abs=0.0001)

    def test_filter_long_term_weights_empty(self, tmp_path):
        """빈 weights는 빈 딕셔너리를 반환한다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        assert alloc.filter_long_term_weights({}) == {}

    def test_rebalance_allocation(self, tmp_path):
        """풀 비율 재조정 결과를 반환한다."""
        path = _alloc_path(tmp_path)
        holdings = [
            {"ticker": "005930", "eval_amount": 800_000, "current_price": 70000, "qty": 11, "pnl": 0, "pnl_pct": 0.0},
            {"ticker": "000660", "eval_amount": 50_000, "current_price": 130000, "qty": 1, "pnl": 0, "pnl_pct": 0.0},
        ]
        kis = _mock_kis(total_eval=1_000_000, holdings=holdings)
        alloc = PortfolioAllocator(kis, allocation_path=path)

        alloc.tag_position("005930", "long_term")
        alloc.tag_position("000660", "short_term")

        result = alloc.rebalance_allocation()

        assert result["total_eval"] == 1_000_000
        assert result["long_term_eval"] == 800_000
        assert result["short_term_eval"] == 50_000
        assert result["long_term_target"] == 0.90
        assert result["short_term_target"] == 0.10
        assert result["long_term_actual"] == 0.80
        assert result["short_term_actual"] == 0.05
        # drift = |0.80 - 0.90| = 0.10 -> 10%, rebalance_needed
        assert result["rebalance_needed"] is True
        assert result["drift_pct"] == 10.0

    def test_rebalance_allocation_zero_total(self, tmp_path):
        """총 가치가 0이면 리밸런싱 불필요."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis(total_eval=0)
        alloc = PortfolioAllocator(kis, allocation_path=path)

        result = alloc.rebalance_allocation()
        assert result["rebalance_needed"] is False
        assert result["total_eval"] == 0


# ===================================================================
# 5. 영속화 테스트
# ===================================================================


class TestPersistence:
    """JSON 저장/로드를 검증한다."""

    def test_save_and_load_persistence(self, tmp_path):
        """태깅 후 새 인스턴스에서 데이터가 유지된다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()

        # 인스턴스 1: 태깅
        alloc1 = PortfolioAllocator(kis, allocation_path=path)
        alloc1.tag_position("005930", "long_term", {"entry_price": 70000})
        alloc1.tag_position("000660", "short_term")

        # 인스턴스 2: 같은 파일로 새로 로드
        alloc2 = PortfolioAllocator(kis, allocation_path=path)

        assert alloc2.get_position_pool("005930") == "long_term"
        assert alloc2.get_position_pool("000660") == "short_term"

    def test_json_structure_valid(self, tmp_path):
        """저장된 JSON 파일 구조가 올바르다."""
        path = _alloc_path(tmp_path)
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)
        alloc.tag_position("005930", "long_term", {"entry_price": 70000, "strategy": "multi_factor"})

        data = _read_json(path)

        assert data["version"] == 1
        assert "updated_at" in data
        assert "config" in data
        assert data["config"]["long_term_pct"] == 0.90
        assert data["config"]["short_term_pct"] == 0.10
        assert data["config"]["soft_cap_mode"] is True
        assert "positions" in data
        assert "005930" in data["positions"]
        assert data["positions"]["005930"]["pool"] == "long_term"
        assert data["positions"]["005930"]["entry_price"] == 70000

    def test_corrupted_json_falls_back_to_defaults(self, tmp_path):
        """손상된 JSON 파일이면 기본값으로 폴백한다."""
        path = _alloc_path(tmp_path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("{{{INVALID", encoding="utf-8")

        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        # 기본값으로 동작
        assert alloc._long_term_pct == 0.90
        assert alloc.get_position_pool("005930") is None

    def test_nested_directory_creation(self, tmp_path):
        """중첩 디렉토리가 자동 생성된다."""
        path = str(tmp_path / "deep" / "nested" / "allocation.json")
        kis = _mock_kis()
        alloc = PortfolioAllocator(kis, allocation_path=path)

        assert Path(path).exists()


# ===================================================================
# 6. Feature Flag 통합 테스트
# ===================================================================


class TestFeatureFlagIntegration:
    """short_term_trading Feature Flag을 검증한다."""

    def test_short_term_trading_flag_exists(self):
        """DEFAULT_FLAGS에 short_term_trading이 존재한다."""
        from src.utils.feature_flags import FeatureFlags

        assert "short_term_trading" in FeatureFlags.DEFAULT_FLAGS

    def test_short_term_trading_default_disabled(self, tmp_path):
        """short_term_trading은 기본 비활성화 상태이다."""
        from src.utils.feature_flags import FeatureFlags

        path = str(tmp_path / "flags.json")
        ff = FeatureFlags(flags_path=path)
        assert ff.is_enabled("short_term_trading") is False

    def test_short_term_trading_config_values(self, tmp_path):
        """short_term_trading의 기본 config 값이 올바르다."""
        from src.utils.feature_flags import FeatureFlags

        path = str(tmp_path / "flags.json")
        ff = FeatureFlags(flags_path=path)
        config = ff.get_config("short_term_trading")

        assert config["long_term_pct"] == 0.90
        assert config["short_term_pct"] == 0.10
        assert config["stop_loss_pct"] == -0.05
        assert config["take_profit_pct"] == 0.10
        assert config["max_concurrent_positions"] == 3
        assert config["max_daily_loss_pct"] == -0.03
        assert config["confirm_timeout_minutes"] == 30
        assert config["mode"] == "swing"

    def test_short_term_trading_toggle(self, tmp_path):
        """short_term_trading을 토글할 수 있다."""
        from src.utils.feature_flags import FeatureFlags

        path = str(tmp_path / "flags.json")
        ff = FeatureFlags(flags_path=path)

        assert ff.is_enabled("short_term_trading") is False
        ff.toggle("short_term_trading", enabled=True)
        assert ff.is_enabled("short_term_trading") is True
