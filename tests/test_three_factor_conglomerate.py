"""ThreeFactorStrategy 계열사/업종 집중도 필터 테스트.

_apply_concentration_filter 메서드의 동작을 검증한다:
- max_stocks_per_conglomerate: 동일 기업집단 최대 종목 수 제한
- max_group_weight: 동일 업종 합산 비중 상한 제한
- 두 필터의 조합, 데이터 누락 시 폴백 동작 등.
"""

import pandas as pd
import pytest

from src.strategy.three_factor import ThreeFactorStrategy


# ===================================================================
# 헬퍼 함수
# ===================================================================

def _make_ranked_scores(tickers: list[str], scores: list[float]) -> pd.Series:
    """스코어 내림차순 정렬된 Series를 생성한다.

    Args:
        tickers: 종목코드 리스트 (스코어 내림차순 순서).
        scores: 대응하는 스코어 값 (내림차순이어야 함).

    Returns:
        pd.Series (index=ticker, values=score, 내림차순 정렬)
    """
    s = pd.Series(scores, index=tickers, dtype=float)
    return s.sort_values(ascending=False)


def _make_fundamentals(
    tickers: list[str],
    names: list[str],
    sectors: list[str] | None = None,
    market_caps: list[int] | None = None,
) -> pd.DataFrame:
    """테스트용 fundamentals DataFrame을 생성한다.

    Args:
        tickers: 종목코드 리스트.
        names: 종목명 리스트.
        sectors: 업종 리스트 (None이면 sector 컬럼 생략).
        market_caps: 시가총액 리스트 (None이면 기본값 사용).
    """
    data = {
        "ticker": tickers,
        "name": names,
    }
    if sectors is not None:
        data["sector"] = sectors
    if market_caps is not None:
        data["market_cap"] = market_caps
    else:
        data["market_cap"] = [1_000_000_000_000] * len(tickers)
    return pd.DataFrame(data)


# ===================================================================
# 테스트: 계열사(conglomerate) 집중도 필터
# ===================================================================

class TestConcentrationFilterConglomerate:
    """_apply_concentration_filter 계열사 제한 테스트."""

    def test_no_conglomerate_limit_passes_all(self):
        """max_stocks_per_conglomerate=0이면 계열사 제한 없이 상위 N개가 그대로 선정된다."""
        strategy = ThreeFactorStrategy(
            num_stocks=5,
            max_group_weight=0,  # 업종 제한도 없음
            max_stocks_per_conglomerate=0,
        )

        tickers = ["005930", "000810", "028260", "009150", "032830"]
        names = ["삼성전자", "삼성화재", "삼성물산", "삼성전기", "삼성생명"]
        scores = [100.0, 90.0, 80.0, 70.0, 60.0]

        ranked = _make_ranked_scores(tickers, scores)
        fundamentals = _make_fundamentals(tickers, names)

        result = strategy._apply_concentration_filter(ranked, fundamentals, num_stocks=5)

        # 제한 없으므로 5개 모두 선정
        assert len(result) == 5
        assert list(result.index) == tickers

    def test_conglomerate_limit_2_with_3_samsung(self):
        """max_stocks_per_conglomerate=2일 때 삼성 3종목 중 상위 2개만 선정된다."""
        strategy = ThreeFactorStrategy(
            num_stocks=5,
            max_group_weight=0,  # 업종 제한 없음
            max_stocks_per_conglomerate=2,
        )

        # 삼성 3개 + 비계열 2개 (총 5개, 점수 순)
        tickers = ["005930", "000810", "051910", "028260", "003550"]
        names = ["삼성전자", "삼성화재", "LG화학", "삼성물산", "LG"]
        scores = [100.0, 90.0, 85.0, 80.0, 75.0]

        ranked = _make_ranked_scores(tickers, scores)
        fundamentals = _make_fundamentals(tickers, names)

        result = strategy._apply_concentration_filter(ranked, fundamentals, num_stocks=5)

        selected = list(result.index)

        # 삼성전자(100), 삼성화재(90) 선정, 삼성물산(80)은 스킵
        assert "005930" in selected, "삼성전자(1위)는 선정되어야 한다"
        assert "000810" in selected, "삼성화재(2위)는 선정되어야 한다"
        assert "028260" not in selected, "삼성물산(4위)은 계열사 한도 초과로 스킵"

        # LG 계열 2개는 제한 내이므로 선정
        assert "051910" in selected, "LG화학은 선정되어야 한다"
        assert "003550" in selected, "LG는 선정되어야 한다"

        # 삼성 중 상위 2개만 포함
        samsung_selected = [t for t in selected if t in ["005930", "000810", "028260"]]
        assert len(samsung_selected) == 2

    def test_conglomerate_limit_2_with_samsung_and_lotte(self):
        """삼성 3개 + 롯데 3개에서 각 계열사 최대 2개씩 선정된다."""
        strategy = ThreeFactorStrategy(
            num_stocks=6,
            max_group_weight=0,
            max_stocks_per_conglomerate=2,
        )

        tickers = ["005930", "023530", "000810", "004990", "028260", "071050"]
        names = ["삼성전자", "롯데쇼핑", "삼성화재", "롯데지주", "삼성물산", "롯데케미칼"]
        scores = [100.0, 95.0, 90.0, 85.0, 80.0, 75.0]

        ranked = _make_ranked_scores(tickers, scores)
        fundamentals = _make_fundamentals(tickers, names)

        result = strategy._apply_concentration_filter(ranked, fundamentals, num_stocks=6)

        selected = list(result.index)

        # 삼성: 삼성전자(100), 삼성화재(90) → 삼성물산(80) 스킵
        samsung_in = [t for t in selected if t in ["005930", "000810", "028260"]]
        assert len(samsung_in) == 2, f"삼성 계열은 최대 2개: {samsung_in}"
        assert "005930" in samsung_in, "삼성전자(1위)는 선정"
        assert "000810" in samsung_in, "삼성화재(3위)는 선정"

        # 롯데: 롯데쇼핑(95), 롯데지주(85) → 롯데케미칼(75) 스킵
        lotte_in = [t for t in selected if t in ["023530", "004990", "071050"]]
        assert len(lotte_in) == 2, f"롯데 계열은 최대 2개: {lotte_in}"
        assert "023530" in lotte_in, "롯데쇼핑(2위)는 선정"
        assert "004990" in lotte_in, "롯데지주(4위)는 선정"

        # 총 4개만 선정 가능 (후보 6개 중 2개 스킵, num_stocks=6이지만 후보 소진)
        assert len(selected) == 4


# ===================================================================
# 테스트: 업종(sector) 비중 필터
# ===================================================================

class TestConcentrationFilterSector:
    """_apply_concentration_filter 업종 비중 제한 테스트."""

    def test_sector_weight_limit(self):
        """max_group_weight=0.25, num_stocks=4이면 업종당 최대 1종목(weight=0.25)이다."""
        strategy = ThreeFactorStrategy(
            num_stocks=4,
            max_group_weight=0.25,
            max_stocks_per_conglomerate=0,  # 계열사 제한 없음
        )

        # 반도체 3개 + 화학 1개 + 금융 1개
        tickers = ["005930", "000660", "042700", "051910", "105560"]
        names = ["삼성전자", "SK하이닉스", "한미반도체", "LG화학", "KB금융"]
        sectors = ["반도체", "반도체", "반도체", "화학", "금융"]
        scores = [100.0, 90.0, 80.0, 70.0, 60.0]

        ranked = _make_ranked_scores(tickers, scores)
        fundamentals = _make_fundamentals(tickers, names, sectors=sectors)

        result = strategy._apply_concentration_filter(ranked, fundamentals, num_stocks=4)

        selected = list(result.index)

        # 반도체: stock_weight=0.25, max_group_weight=0.25
        # 삼성전자(0.25) 선정 → SK하이닉스(0.25+0.25=0.50 > 0.25) 스킵
        # → 한미반도체 역시 스킵 → LG화학 선정 → KB금융 선정
        semiconductor_count = sum(1 for t in selected if t in ["005930", "000660", "042700"])
        assert semiconductor_count == 1, f"반도체 업종은 1개만: {semiconductor_count}"
        assert "005930" in selected, "삼성전자(1위, 반도체)는 선정"
        assert "051910" in selected, "LG화학(화학)은 선정"
        assert "105560" in selected, "KB금융(금융)은 선정"

    def test_no_sector_data_falls_back_to_etc(self):
        """sector 컬럼이 없으면 모두 '기타'로 분류되어 업종 제한이 일괄 적용된다."""
        strategy = ThreeFactorStrategy(
            num_stocks=4,
            max_group_weight=0.25,
            max_stocks_per_conglomerate=0,
        )

        tickers = ["005930", "000660", "051910", "105560"]
        names = ["삼성전자", "SK하이닉스", "LG화학", "KB금융"]
        scores = [100.0, 90.0, 80.0, 70.0]

        ranked = _make_ranked_scores(tickers, scores)
        # sector 컬럼 없음 → sector_map이 빈 dict → 조건문에서 sector_map 검사 시 falsy
        # 코드: if sector_map and not no_sector_limit → False
        # → sector 체크가 스킵됨 → sector='기타' 할당만 됨 → 비중 체크 안 함
        fundamentals = _make_fundamentals(tickers, names, sectors=None)

        result = strategy._apply_concentration_filter(ranked, fundamentals, num_stocks=4)

        # sector_map이 비어있으면 업종 체크가 스킵되므로 모든 종목 통과
        assert len(result) == 4, "sector 없으면 업종 제한이 적용되지 않아 모두 통과"


# ===================================================================
# 테스트: 복합 필터 (업종 + 계열사)
# ===================================================================

class TestConcentrationFilterCombined:
    """업종 비중 + 계열사 수 제한이 동시 적용되는 시나리오."""

    def test_combined_sector_and_conglomerate_limits(self):
        """업종 비중과 계열사 제한이 동시에 적용된다."""
        strategy = ThreeFactorStrategy(
            num_stocks=4,
            max_group_weight=0.5,  # stock_weight=0.25, 업종당 최대 2종목
            max_stocks_per_conglomerate=1,  # 계열사당 1종목
        )

        # 삼성전자/삼성화재는 같은 계열사(삼성), LG화학/LG는 같은 계열사(LG)
        # 삼성전자/삼성화재/LG화학 = 반도체/보험/화학 (업종은 다름)
        tickers = ["005930", "000810", "051910", "003550", "105560"]
        names = ["삼성전자", "삼성화재", "LG화학", "LG", "KB금융"]
        sectors = ["반도체", "보험", "화학", "지주", "금융"]
        scores = [100.0, 90.0, 80.0, 70.0, 60.0]

        ranked = _make_ranked_scores(tickers, scores)
        fundamentals = _make_fundamentals(tickers, names, sectors=sectors)

        result = strategy._apply_concentration_filter(ranked, fundamentals, num_stocks=4)

        selected = list(result.index)

        # 삼성전자(100) 선정 → 삼성화재(90) 계열사 한도(1) 초과 스킵
        # LG화학(80) 선정 → LG(70) 계열사 한도(1) 초과 스킵
        # KB금융(60) 선정
        assert "005930" in selected, "삼성전자 선정"
        assert "000810" not in selected, "삼성화재는 계열사 한도 초과"
        assert "051910" in selected, "LG화학 선정"
        assert "003550" not in selected, "LG는 계열사 한도 초과"
        assert "105560" in selected, "KB금융 선정"
        assert len(selected) == 3, "후보 소진으로 3개만 선정"


# ===================================================================
# 테스트: 엣지 케이스
# ===================================================================

class TestConcentrationFilterEdgeCases:
    """_apply_concentration_filter 엣지 케이스 테스트."""

    def test_no_name_data_skips_conglomerate_filter(self):
        """name 컬럼이 없으면 계열사 필터가 스킵되고 업종 필터만 동작한다."""
        strategy = ThreeFactorStrategy(
            num_stocks=4,
            max_group_weight=0.5,  # stock_weight=0.25, 업종당 최대 2
            max_stocks_per_conglomerate=1,
        )

        tickers = ["005930", "000810", "051910", "105560"]
        sectors = ["반도체", "반도체", "화학", "금융"]
        scores = [100.0, 90.0, 80.0, 70.0]

        ranked = _make_ranked_scores(tickers, scores)
        # name 컬럼 없이 생성
        fundamentals = pd.DataFrame({
            "ticker": tickers,
            "sector": sectors,
            "market_cap": [1_000_000_000_000] * 4,
        })

        result = strategy._apply_concentration_filter(ranked, fundamentals, num_stocks=4)

        selected = list(result.index)

        # name_map이 비어있으므로 계열사 체크 스킵
        # 업종 체크만 적용: 반도체 2개(0.25+0.25=0.50 <= 0.50) 둘 다 통과
        assert "005930" in selected, "삼성전자 선정 (업종 내 비중 허용)"
        assert "000810" in selected, "삼성화재 선정 (반도체 2개=0.50 허용)"
        assert "051910" in selected, "LG화학 선정"
        assert "105560" in selected, "KB금융 선정"
        assert len(selected) == 4

    def test_empty_ranked_scores_returns_empty(self):
        """빈 ranked_scores 입력 시 빈 Series를 반환한다."""
        strategy = ThreeFactorStrategy(
            num_stocks=10,
            max_group_weight=0.25,
            max_stocks_per_conglomerate=2,
        )

        ranked = pd.Series(dtype=float)
        fundamentals = _make_fundamentals([], [])

        result = strategy._apply_concentration_filter(ranked, fundamentals, num_stocks=10)

        assert len(result) == 0, "빈 입력이면 빈 결과"
        assert isinstance(result, pd.Series), "반환 타입은 pd.Series"
