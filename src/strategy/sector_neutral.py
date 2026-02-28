"""섹터 중립 스코어링 및 종목 선정 모듈.

팩터 스코어를 섹터 내 순위로 변환하고,
섹터별 균등 배분(라운드로빈) 방식으로 종목을 선정하여
특정 섹터 쏠림을 방지한다.
"""

from typing import Dict, List, Optional
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def sector_neutral_rank(
    scores: pd.Series,
    sector_map: Dict[str, str],
) -> pd.Series:
    """섹터 내 백분위 순위로 변환한다.

    각 종목의 스코어를 해당 섹터 내에서의 백분위 순위(0~1)로 변환.
    섹터 간 스코어 수준 차이를 제거하여 공정한 비교를 가능하게 한다.

    Args:
        scores: 원본 스코어 (index=ticker)
        sector_map: {ticker: sector_name} 딕셔너리

    Returns:
        섹터 내 백분위 순위 (pd.Series, index=ticker, values=0~1)
    """
    if scores.empty:
        return pd.Series(dtype=float)

    # 섹터 정보 매핑
    sectors = scores.index.map(lambda t: sector_map.get(t, "기타"))
    df = pd.DataFrame({"score": scores, "sector": sectors})

    # 섹터 내 백분위 순위
    df["sector_rank"] = df.groupby("sector")["score"].rank(pct=True)

    result = df["sector_rank"]
    result.index = scores.index

    logger.info(
        f"섹터 중립 랭킹 완료: {len(result)}개 종목, "
        f"{df['sector'].nunique()}개 섹터"
    )

    return result


def select_sector_neutral(
    scores: pd.Series,
    sector_map: Dict[str, str],
    num_stocks: int = 20,
    max_sector_pct: float = 0.25,
) -> List[str]:
    """섹터 균등 배분 방식으로 종목을 선정한다.

    라운드로빈 방식으로 각 섹터에서 순위가 높은 종목을 번갈아 선정.
    특정 섹터가 전체의 max_sector_pct를 초과하지 않도록 제한.

    Args:
        scores: 원본 스코어 (index=ticker, 높을수록 좋음)
        sector_map: {ticker: sector_name} 딕셔너리
        num_stocks: 목표 종목 수 (기본 20)
        max_sector_pct: 단일 섹터 최대 비중 (기본 0.25 = 25%)

    Returns:
        선정된 종목 코드 리스트
    """
    if scores.empty:
        return []

    max_per_sector = max(1, int(num_stocks * max_sector_pct))

    # 섹터별 스코어 내림차순 정렬
    sectors = scores.index.map(lambda t: sector_map.get(t, "기타"))
    df = pd.DataFrame({"score": scores, "sector": sectors})
    df = df.sort_values("score", ascending=False)

    # 섹터별 그룹화, 각 섹터 내 순위순
    sector_groups: Dict[str, List[str]] = {}
    for ticker, row in df.iterrows():
        sector = row["sector"]
        if sector not in sector_groups:
            sector_groups[sector] = []
        sector_groups[sector].append(str(ticker))

    # 라운드로빈 선정
    selected: List[str] = []
    sector_counts: Dict[str, int] = {s: 0 for s in sector_groups}
    sector_pointers: Dict[str, int] = {s: 0 for s in sector_groups}

    # 섹터를 종목 수 기준 내림차순 정렬 (큰 섹터부터)
    sorted_sectors = sorted(
        sector_groups.keys(),
        key=lambda s: len(sector_groups[s]),
        reverse=True,
    )

    while len(selected) < num_stocks:
        added_this_round = False
        for sector in sorted_sectors:
            if len(selected) >= num_stocks:
                break

            ptr = sector_pointers[sector]
            candidates = sector_groups[sector]

            # 해당 섹터에서 아직 선정 가능한 종목이 있고, 섹터 상한 미달
            if ptr < len(candidates) and sector_counts[sector] < max_per_sector:
                selected.append(candidates[ptr])
                sector_counts[sector] += 1
                sector_pointers[sector] += 1
                added_this_round = True

        if not added_this_round:
            # 더 이상 선정 가능한 종목 없음
            break

    logger.info(
        f"섹터 중립 선정 완료: {len(selected)}/{num_stocks}개 종목, "
        f"섹터 분포={dict(sector_counts)}"
    )

    return selected
