"""섹터 중립화 모듈 테스트."""

import pandas as pd
import pytest

from src.strategy.sector_neutral import sector_neutral_rank, select_sector_neutral


# ---------------------------------------------------------------------------
# sector_neutral_rank 테스트
# ---------------------------------------------------------------------------

class TestSectorNeutralRank:

    def test_basic_ranking(self):
        """기본 섹터 내 순위 변환."""
        scores = pd.Series({
            "A": 10, "B": 20, "C": 30,   # 섹터1
            "D": 100, "E": 200, "F": 300, # 섹터2
        })
        sector_map = {
            "A": "IT", "B": "IT", "C": "IT",
            "D": "금융", "E": "금융", "F": "금융",
        }
        result = sector_neutral_rank(scores, sector_map)

        assert len(result) == 6
        # 섹터 내 순위이므로, 각 섹터에서 1위는 같은 pct_rank
        # C가 IT에서 1위, F가 금융에서 1위 -> 둘 다 같은 값
        assert result["C"] == result["F"]  # 각 섹터 내 1위

    def test_empty_scores(self):
        """빈 스코어 -> 빈 결과."""
        result = sector_neutral_rank(pd.Series(dtype=float), {})
        assert result.empty

    def test_single_sector(self):
        """단일 섹터일 때도 정상 동작."""
        scores = pd.Series({"A": 1, "B": 2, "C": 3})
        sector_map = {"A": "IT", "B": "IT", "C": "IT"}
        result = sector_neutral_rank(scores, sector_map)
        assert len(result) == 3

    def test_unclassified_ticker(self):
        """미분류 종목은 '기타' 섹터로 처리."""
        scores = pd.Series({"A": 10, "B": 20, "C": 30})
        sector_map = {"A": "IT"}  # B, C는 미분류
        result = sector_neutral_rank(scores, sector_map)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# select_sector_neutral 테스트
# ---------------------------------------------------------------------------

class TestSelectSectorNeutral:

    def _make_scores_and_sectors(self, n_sectors=4, n_per_sector=25):
        """테스트용 스코어 및 섹터맵 생성."""
        sector_names = [f"섹터{i}" for i in range(n_sectors)]
        scores_dict = {}
        sector_map = {}
        score_val = 1.0

        for s_idx, sector in enumerate(sector_names):
            for j in range(n_per_sector):
                ticker = f"S{s_idx}T{j:02d}"
                scores_dict[ticker] = score_val
                sector_map[ticker] = sector
                score_val += 1.0

        return pd.Series(scores_dict), sector_map

    def test_basic_selection(self):
        """기본 섹터 균등 선정."""
        scores, sector_map = self._make_scores_and_sectors(4, 25)
        selected = select_sector_neutral(scores, sector_map, num_stocks=20, max_sector_pct=0.25)

        assert len(selected) == 20

    def test_max_sector_pct_enforced(self):
        """max_sector_pct 제한 준수."""
        scores, sector_map = self._make_scores_and_sectors(4, 25)
        selected = select_sector_neutral(scores, sector_map, num_stocks=20, max_sector_pct=0.25)

        # 각 섹터별 선정 종목 수 확인
        sector_counts = {}
        for ticker in selected:
            sector = sector_map[ticker]
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        max_allowed = max(1, int(20 * 0.25))  # = 5
        for sector, count in sector_counts.items():
            assert count <= max_allowed, f"{sector}: {count} > {max_allowed}"

    def test_sector_distribution_balanced(self):
        """4섹터 x 25종목 -> 섹터당 5종목 선정."""
        scores, sector_map = self._make_scores_and_sectors(4, 25)
        selected = select_sector_neutral(scores, sector_map, num_stocks=20, max_sector_pct=0.25)

        sector_counts = {}
        for ticker in selected:
            sector = sector_map[ticker]
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # 4개 섹터에서 균등하게 5개씩
        assert len(sector_counts) == 4
        for count in sector_counts.values():
            assert count == 5

    def test_empty_scores(self):
        """빈 스코어 -> 빈 리스트."""
        result = select_sector_neutral(pd.Series(dtype=float), {}, num_stocks=20)
        assert result == []

    def test_unclassified_tickers(self):
        """미분류 종목 -> '기타' 섹터 할당."""
        scores = pd.Series({"A": 10, "B": 20, "C": 30, "D": 40, "E": 50})
        sector_map = {"A": "IT", "B": "IT"}  # C, D, E는 미분류

        selected = select_sector_neutral(scores, sector_map, num_stocks=4, max_sector_pct=0.50)
        assert len(selected) == 4

    def test_fewer_stocks_than_requested(self):
        """종목 수 부족 시 가능한 만큼만 선정."""
        scores = pd.Series({"A": 10, "B": 20, "C": 30})
        sector_map = {"A": "IT", "B": "금융", "C": "IT"}

        selected = select_sector_neutral(scores, sector_map, num_stocks=10, max_sector_pct=0.50)
        assert len(selected) <= 3

    def test_single_sector_respects_cap(self):
        """단일 섹터만 있으면 max_per_sector까지만 선정."""
        scores = pd.Series({f"T{i}": float(i) for i in range(20)})
        sector_map = {f"T{i}": "IT" for i in range(20)}

        selected = select_sector_neutral(scores, sector_map, num_stocks=10, max_sector_pct=0.50)
        # max_per_sector = int(10 * 0.50) = 5
        assert len(selected) == 5
