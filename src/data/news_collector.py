"""DART 공시 자동 수집 + 뉴스레터 생성 모듈.

DART OpenAPI를 사용하여 최근 공시를 수집하고,
중요 공시 필터링, 포트폴리오 영향도 스코어링, 뉴스레터 생성 기능을 제공한다.
DART_API_KEY가 없으면 graceful하게 빈 결과를 반환한다.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Optional

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUEST_TIMEOUT = 10

# 중요 공시 카테고리별 키워드 매핑
IMPORTANT_KEYWORDS: dict[str, list[str]] = {
    "earnings": ["영업(잠정)실적", "매출액또는손익구조", "분기보고서", "사업보고서", "반기보고서"],
    "rights_issue": ["유상증자", "무상증자", "신주인수권"],
    "buyback": ["자기주식", "자사주"],
    "major_shareholder": ["최대주주", "대주주", "임원ㆍ주요주주"],
    "merger": ["합병", "분할", "영업양수도"],
    "dividend": ["배당", "현금ㆍ현물배당"],
    "delisting": ["상장폐지", "관리종목"],
    "investment": ["타법인주식", "투자판단"],
}

# 카테고리별 기본 영향도 점수
CATEGORY_SCORES: dict[str, int] = {
    "earnings": 8,
    "rights_issue": 7,
    "buyback": 6,
    "major_shareholder": 7,
    "merger": 9,
    "dividend": 5,
    "delisting": 10,
    "investment": 4,
}


class NewsCollector:
    """DART 공시 수집 및 뉴스레터 생성.

    Args:
        dart_api_key: DART API 키. None이면 환경변수에서 로드.
    """

    BASE_URL = "https://opendart.fss.or.kr/api"

    # 공시 카테고리별 투자 교육 정보
    _DISCLOSURE_EDUCATION: dict[str, dict[str, str]] = {
        "earnings": {
            "impact": "실적 공시는 주가에 직접적 영향. 컨센서스 대비 서프라이즈/쇼크 확인 필요",
            "action": "서프라이즈 → 추가매수 검토, 쇼크 → 손절/비중축소 검토",
            "risk": "medium",
        },
        "merger": {
            "impact": "합병/인수는 기업가치 재평가. 피인수기업 프리미엄, 인수기업 희석 가능",
            "action": "합병비율, 시너지 가능성 분석 후 판단",
            "risk": "high",
        },
        "delisting": {
            "impact": "상장폐지는 투자금 전액 손실 위험. 즉시 매도 검토",
            "action": "즉시 매도 강력 권고",
            "risk": "critical",
        },
        "rights_issue": {
            "impact": "유상증자는 주식 가치 희석. 기존 주주 가치 하락 가능",
            "action": "증자 목적(시설/운영/차환) 확인, 투기적 증자는 부정적",
            "risk": "high",
        },
        "buyback": {
            "impact": "자사주 매입은 주주환원 + 주가 하방 지지. 긍정적 시그널",
            "action": "매입 규모와 소각 여부 확인. 소각 시 더 긍정적",
            "risk": "low",
        },
        "dividend": {
            "impact": "배당 공시는 주주환원 의지 표명. 배당수익률 변화 확인",
            "action": "배당 증가 → 긍정, 배당 삭감 → 부정",
            "risk": "low",
        },
        "major_shareholder": {
            "impact": "대주주 변경은 경영권 이슈. 지분율 변화 방향 확인",
            "action": "매수 → 경영참여/인수 가능, 매도 → 엑시트 신호",
            "risk": "medium",
        },
        "investment": {
            "impact": "투자판단 관련 공시. 구체적 내용에 따라 영향 다름",
            "action": "공시 전문 확인 후 판단",
            "risk": "medium",
        },
    }

    def __init__(self, dart_api_key: Optional[str] = None) -> None:
        self.dart_api_key = dart_api_key or os.getenv("DART_API_KEY", "")
        if not self.dart_api_key:
            logger.warning(
                "DART_API_KEY가 설정되지 않았습니다. "
                "공시 수집 기능이 비활성화됩니다."
            )

    def is_configured(self) -> bool:
        """API 키가 설정되어 있는지 확인한다."""
        return bool(self.dart_api_key)

    def fetch_recent_disclosures(
        self,
        corp_codes: Optional[list[str]] = None,
        days: int = 1,
    ) -> list[dict[str, Any]]:
        """최근 N일간 공시를 조회한다.

        Args:
            corp_codes: 조회할 기업 고유번호 리스트. None이면 전체.
            days: 조회할 일수 (기본 1일).

        Returns:
            공시 리스트. API 미설정이나 오류 시 빈 리스트.
        """
        if not self.is_configured():
            logger.info("DART API 미설정, 빈 결과 반환")
            return []

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params: dict[str, Any] = {
            "crtfc_key": self.dart_api_key,
            "bgn_de": start_date.strftime("%Y%m%d"),
            "end_de": end_date.strftime("%Y%m%d"),
            "page_count": 100,
        }
        if corp_codes and len(corp_codes) == 1:
            params["corp_code"] = corp_codes[0]

        try:
            response = requests.get(
                f"{self.BASE_URL}/list.json",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "000":
                msg = data.get("message", "알 수 없는 오류")
                logger.warning("DART API 오류: %s", msg)
                return []

            disclosures = data.get("list", [])

            # 특정 기업 필터링 (corp_codes가 여러 개인 경우)
            if corp_codes and len(corp_codes) > 1:
                corp_set = set(corp_codes)
                disclosures = [
                    d for d in disclosures
                    if d.get("corp_code") in corp_set
                ]

            logger.info("공시 %d건 조회 완료 (%d일간)", len(disclosures), days)
            return disclosures

        except requests.exceptions.Timeout:
            logger.error("DART API 타임아웃")
            return []
        except requests.exceptions.ConnectionError:
            logger.error("DART API 연결 실패")
            return []
        except requests.exceptions.RequestException as e:
            logger.error("DART API 요청 실패: %s", e)
            return []
        except Exception as e:
            logger.error("공시 조회 중 예외: %s", e)
            return []

    def filter_important(
        self,
        disclosures: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """중요 공시만 필터링한다.

        Args:
            disclosures: 전체 공시 리스트.

        Returns:
            중요 공시 리스트 (카테고리 정보 추가).
        """
        important: list[dict[str, Any]] = []

        for disc in disclosures:
            report_nm = disc.get("report_nm", "")
            category = self._categorize(report_nm)

            if category is not None:
                disc_copy = dict(disc)
                disc_copy["category"] = category
                important.append(disc_copy)

        logger.info(
            "중요 공시 필터링: %d/%d건",
            len(important),
            len(disclosures),
        )
        return important

    def score_impact(
        self,
        disclosure: dict[str, Any],
        holdings: Optional[list[str]] = None,
    ) -> int:
        """공시의 포트폴리오 영향도를 스코어링한다.

        Args:
            disclosure: 공시 딕셔너리.
            holdings: 보유 종목코드 리스트.

        Returns:
            영향도 점수 (0~10).
        """
        category = disclosure.get("category")
        if category is None:
            report_nm = disclosure.get("report_nm", "")
            category = self._categorize(report_nm)

        base_score = CATEGORY_SCORES.get(category, 3) if category else 3

        # 보유종목 관련이면 가산
        if holdings:
            corp_name = disclosure.get("corp_name", "")
            stock_code = disclosure.get("stock_code", "")
            if stock_code in holdings:
                base_score = min(base_score + 3, 10)
            elif any(h in corp_name for h in holdings):
                base_score = min(base_score + 2, 10)

        return base_score

    def generate_newsletter(
        self,
        date: Optional[str] = None,
        holdings: Optional[list[str]] = None,
        days: int = 1,
    ) -> str:
        """일일 뉴스레터를 생성한다.

        Args:
            date: 날짜 (표시용). None이면 오늘.
            holdings: 보유 종목코드 리스트.
            days: 조회할 일수.

        Returns:
            포매팅된 뉴스레터 문자열.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        if not self.is_configured():
            return f"[공시 뉴스레터] {date}\nDART API 미설정"

        disclosures = self.fetch_recent_disclosures(days=days)
        important = self.filter_important(disclosures)

        if not important:
            return f"[공시 뉴스레터] {date}\n중요 공시 없음"

        # 스코어링 및 정렬
        scored: list[tuple[dict, int]] = []
        for disc in important:
            score = self.score_impact(disc, holdings)
            scored.append((disc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 포매팅
        lines = [
            f"[공시 뉴스레터] {date}",
            "=" * 40,
            f"중요 공시 {len(scored)}건",
            "",
        ]

        for disc, score in scored[:15]:
            corp_name = disc.get("corp_name", "?")
            report_nm = disc.get("report_nm", "?")
            rcept_dt = disc.get("rcept_dt", "?")
            category = disc.get("category", "기타")

            # 보유종목 표시
            is_holding = ""
            if holdings:
                stock_code = disc.get("stock_code", "")
                if stock_code in holdings:
                    is_holding = " [보유]"

            lines.append(
                f"  [{score}/10] {corp_name}{is_holding}"
            )
            lines.append(f"    {report_nm}")
            lines.append(f"    ({category} | {rcept_dt})")
            lines.append("")

        lines.append("=" * 40)
        return "\n".join(lines)

    def format_morning_checklist(
        self,
        disclosures: Optional[list[dict[str, Any]]] = None,
        macro_data: Optional[dict[str, Any]] = None,
    ) -> str:
        """08:00 오전 체크리스트를 생성한다.

        Args:
            disclosures: 공시 리스트. None이면 자동 수집.
            macro_data: 매크로 데이터 딕셔너리.

        Returns:
            포매팅된 오전 체크리스트.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        lines = [
            f"[오전 체크리스트] {today}",
            "=" * 35,
        ]

        # 공시 섹션
        if disclosures is None and self.is_configured():
            disclosures = self.fetch_recent_disclosures(days=1)
            disclosures = self.filter_important(disclosures)

        if disclosures:
            lines.append(f"\n[주요 공시] {len(disclosures)}건")
            for disc in disclosures[:5]:
                corp_name = disc.get("corp_name", "?")
                report_nm = disc.get("report_nm", "?")
                lines.append(f"  - {corp_name}: {report_nm}")
        else:
            lines.append("\n[주요 공시] 없음")

        # 매크로 섹션
        if macro_data:
            lines.append("\n[매크로 환경]")
            for key, value in macro_data.items():
                lines.append(f"  - {key}: {value}")

        lines.append("")
        lines.append("=" * 35)
        return "\n".join(lines)

    def format_eod_news(
        self,
        disclosures: Optional[list[dict[str, Any]]] = None,
        holdings: Optional[list[str]] = None,
    ) -> str:
        """15:40 장 마감 뉴스를 생성한다.

        Args:
            disclosures: 공시 리스트. None이면 자동 수집.
            holdings: 보유 종목코드 리스트.

        Returns:
            포매팅된 장 마감 뉴스.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        lines = [
            f"[장마감 공시] {today}",
            "-" * 30,
        ]

        if disclosures is None and self.is_configured():
            disclosures = self.fetch_recent_disclosures(days=1)
            disclosures = self.filter_important(disclosures)

        if not disclosures:
            lines.append("오늘의 중요 공시 없음")
            return "\n".join(lines)

        # 보유종목 관련 우선
        holding_news: list[dict] = []
        other_news: list[dict] = []

        for disc in disclosures:
            stock_code = disc.get("stock_code", "")
            if holdings and stock_code in holdings:
                holding_news.append(disc)
            else:
                other_news.append(disc)

        if holding_news:
            lines.append(f"\n[보유종목 관련] {len(holding_news)}건")
            for disc in holding_news:
                corp_name = disc.get("corp_name", "?")
                report_nm = disc.get("report_nm", "?")
                lines.append(f"  * {corp_name}: {report_nm}")

        if other_news:
            lines.append(f"\n[기타 주요 공시] {len(other_news)}건")
            for disc in other_news[:10]:
                corp_name = disc.get("corp_name", "?")
                report_nm = disc.get("report_nm", "?")
                lines.append(f"  - {corp_name}: {report_nm}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # EOD 공시 모니터링 + 투자 교육
    # ------------------------------------------------------------------

    def analyze_disclosure_impact(
        self,
        disclosures: list[dict[str, Any]],
        holdings: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """각 공시의 투자 영향 분석 + 학습 제안을 추가한다.

        Args:
            disclosures: 공시 리스트 (filter_important() 결과 등).
            holdings: 보유 종목코드 리스트.

        Returns:
            교육 정보가 추가된 공시 리스트. priority 기준 내림차순 정렬.
        """
        if not disclosures:
            return []

        _PRIORITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        enriched: list[dict[str, Any]] = []
        for disc in disclosures:
            disc_copy = dict(disc)

            # 카테고리 판별
            category = disc_copy.get("category")
            if category is None:
                report_nm = disc_copy.get("report_nm", "")
                category = self._categorize(report_nm)
                if category is not None:
                    disc_copy["category"] = category

            # 교육 정보 추가
            education = self._DISCLOSURE_EDUCATION.get(category, {
                "impact": "분류되지 않은 공시. 공시 전문을 직접 확인 필요",
                "action": "공시 원문 확인 후 판단",
                "risk": "low",
            }) if category else {
                "impact": "분류되지 않은 공시. 공시 전문을 직접 확인 필요",
                "action": "공시 원문 확인 후 판단",
                "risk": "low",
            }
            disc_copy["education"] = dict(education)

            # 보유 여부 판별
            is_holding = False
            if holdings:
                stock_code = disc_copy.get("stock_code", "")
                if stock_code in holdings:
                    is_holding = True
            disc_copy["is_holding"] = is_holding

            # 우선순위 결정: 교육 정보의 risk를 기본으로, 보유종목이면 +3
            base_priority = _PRIORITY_ORDER.get(education["risk"], 1)
            if is_holding:
                base_priority += 3

            # 숫자를 다시 문자열 priority로 매핑
            if base_priority >= 4:
                priority = "critical"
            elif base_priority >= 3:
                priority = "high"
            elif base_priority >= 2:
                priority = "medium"
            else:
                priority = "low"

            disc_copy["priority"] = priority
            disc_copy["_priority_score"] = base_priority
            enriched.append(disc_copy)

        # priority score 내림차순 정렬
        enriched.sort(key=lambda x: x["_priority_score"], reverse=True)
        return enriched

    def format_eod_with_education(
        self,
        disclosures: Optional[list[dict[str, Any]]] = None,
        holdings: Optional[list[str]] = None,
    ) -> str:
        """EOD 공시 리포트 + 투자 학습 제안 포맷.

        Args:
            disclosures: 공시 리스트. None이면 자동 수집 + 필터링.
            holdings: 보유 종목코드 리스트.

        Returns:
            포매팅된 EOD 공시 모니터링 리포트 문자열.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        lines = [
            f"[EOD 공시 모니터링] {today}",
            "=" * 32,
        ]

        # 공시 데이터 준비
        if disclosures is None and self.is_configured():
            disclosures = self.fetch_recent_disclosures(days=1)
            disclosures = self.filter_important(disclosures)

        if not disclosures:
            lines.append("오늘의 중요 공시 없음")
            return "\n".join(lines)

        # 영향 분석 + 교육 정보 추가
        analyzed = self.analyze_disclosure_impact(disclosures, holdings)

        # 보유종목 / 기타 분리
        holding_disclosures = [d for d in analyzed if d.get("is_holding")]
        other_disclosures = [d for d in analyzed if not d.get("is_holding")]

        idx = 1

        if holding_disclosures:
            lines.append("[보유종목 공시] (우선 체크)")
            for disc in holding_disclosures:
                corp_name = disc.get("corp_name", "?")
                report_nm = disc.get("report_nm", "?")
                category = disc.get("category", "기타")
                edu = disc.get("education", {})

                lines.append(f"  {idx}. [보유] {corp_name} - {report_nm} ({category})")
                if edu.get("impact"):
                    lines.append(f"     영향: {edu['impact']}")
                if edu.get("action"):
                    lines.append(f"     조치: {edu['action']}")
                if edu.get("risk"):
                    lines.append(f"     리스크: {edu['risk']}")
                lines.append("")
                idx += 1

        if other_disclosures:
            lines.append("[기타 주요 공시]")
            for disc in other_disclosures[:10]:
                corp_name = disc.get("corp_name", "?")
                report_nm = disc.get("report_nm", "?")
                category = disc.get("category", "기타")
                edu = disc.get("education", {})

                lines.append(f"  {idx}. {corp_name} - {report_nm} ({category})")
                if edu.get("impact"):
                    lines.append(f"     영향: {edu['impact']}")
                if edu.get("action"):
                    lines.append(f"     조치: {edu['action']}")
                if edu.get("risk"):
                    lines.append(f"     리스크: {edu['risk']}")
                lines.append("")
                idx += 1

        # 투자 학습 팁
        all_categories = [d.get("category") for d in analyzed if d.get("category")]
        tip = self._generate_learning_tip(all_categories)
        if tip:
            lines.append("[투자 학습 팁]")
            lines.append(f"  {tip}")

        return "\n".join(lines)

    @staticmethod
    def _generate_learning_tip(categories: list[str]) -> str:
        """당일 공시 카테고리를 기반으로 학습 팁을 생성한다."""
        tips: dict[str, str] = {
            "rights_issue": (
                "오늘의 핵심: 유상증자 공시가 나온 경우, "
                "증자 목적(시설투자/운영자금/차환)을 반드시 확인하세요. "
                "투기적 증자(운영자금 목적)는 주가에 부정적인 경우가 많습니다."
            ),
            "delisting": (
                "오늘의 핵심: 상장폐지 관련 공시가 있습니다. "
                "관리종목 지정/상장폐지 사유를 확인하고, "
                "보유 시 즉시 매도를 검토하세요."
            ),
            "merger": (
                "오늘의 핵심: 합병/인수 공시가 있습니다. "
                "합병비율과 시너지 효과를 분석하세요. "
                "피인수기업은 프리미엄이, 인수기업은 희석 가능성이 있습니다."
            ),
            "earnings": (
                "오늘의 핵심: 실적 공시를 확인할 때는 "
                "시장 컨센서스 대비 서프라이즈/쇼크 여부가 핵심입니다. "
                "매출과 영업이익 모두 확인하세요."
            ),
            "buyback": (
                "오늘의 핵심: 자사주 매입은 긍정적 신호이지만, "
                "소각 여부도 확인하세요. 소각 없는 자사주 매입은 "
                "추후 재매각 가능성이 있습니다."
            ),
        }

        # 가장 중요한(risk 높은) 카테고리 우선
        priority_order = ["delisting", "merger", "rights_issue", "earnings", "buyback"]
        for cat in priority_order:
            if cat in categories:
                return tips[cat]

        if categories:
            return (
                "오늘의 핵심: 공시 내용을 꼼꼼히 읽고, "
                "투자 판단에 미치는 영향을 분석해보세요."
            )
        return ""

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _categorize(report_nm: str) -> Optional[str]:
        """공시 제목으로 카테고리를 판별한다."""
        for category, keywords in IMPORTANT_KEYWORDS.items():
            for kw in keywords:
                if kw in report_nm:
                    return category
        return None
