"""EOD 공시 모니터링 + 투자 교육 기능 테스트.

NewsCollector의 analyze_disclosure_impact(), format_eod_with_education() 기능과
_DISCLOSURE_EDUCATION 데이터, 우선순위 스코어링을 검증한다.
"""

import pytest

from src.data.news_collector import NewsCollector


# ===================================================================
# 공통 픽스처
# ===================================================================


@pytest.fixture
def collector():
    """API 키가 설정된 NewsCollector."""
    return NewsCollector(dart_api_key="test_api_key")


@pytest.fixture
def sample_disclosures():
    """카테고리가 이미 지정된 샘플 공시 리스트."""
    return [
        {
            "corp_code": "00126380",
            "corp_name": "삼성전자",
            "stock_code": "005930",
            "report_nm": "영업(잠정)실적(공정공시)",
            "rcept_dt": "20260227",
            "category": "earnings",
        },
        {
            "corp_code": "00164779",
            "corp_name": "SK하이닉스",
            "stock_code": "000660",
            "report_nm": "유상증자결정",
            "rcept_dt": "20260227",
            "category": "rights_issue",
        },
        {
            "corp_code": "00200000",
            "corp_name": "네이버",
            "stock_code": "035420",
            "report_nm": "자기주식취득결정",
            "rcept_dt": "20260227",
            "category": "buyback",
        },
        {
            "corp_code": "00300000",
            "corp_name": "위험기업",
            "stock_code": "999999",
            "report_nm": "상장폐지결정",
            "rcept_dt": "20260227",
            "category": "delisting",
        },
    ]


# ===================================================================
# TestDisclosureEducation: 교육 정보 매핑 테스트
# ===================================================================


class TestDisclosureEducation:
    """_DISCLOSURE_EDUCATION 데이터 및 analyze_disclosure_impact() 교육 매핑 테스트."""

    def test_earnings_education(self, collector):
        """earnings 카테고리에 올바른 교육 정보가 매핑된다."""
        disclosures = [
            {
                "corp_name": "삼성전자",
                "stock_code": "005930",
                "report_nm": "분기보고서",
                "category": "earnings",
            }
        ]
        result = collector.analyze_disclosure_impact(disclosures)
        assert len(result) == 1
        edu = result[0]["education"]
        assert "실적 공시" in edu["impact"]
        assert "서프라이즈" in edu["action"]
        assert edu["risk"] == "medium"

    def test_delisting_education(self, collector):
        """delisting 카테고리는 critical 리스크로 매핑된다."""
        disclosures = [
            {
                "corp_name": "위험기업",
                "stock_code": "999999",
                "report_nm": "상장폐지결정",
                "category": "delisting",
            }
        ]
        result = collector.analyze_disclosure_impact(disclosures)
        assert len(result) == 1
        edu = result[0]["education"]
        assert "전액 손실" in edu["impact"]
        assert "즉시 매도" in edu["action"]
        assert edu["risk"] == "critical"
        assert result[0]["priority"] == "critical"

    def test_buyback_education(self, collector):
        """buyback 카테고리는 low 리스크 + 긍정적 시그널로 매핑된다."""
        disclosures = [
            {
                "corp_name": "네이버",
                "stock_code": "035420",
                "report_nm": "자기주식취득결정",
                "category": "buyback",
            }
        ]
        result = collector.analyze_disclosure_impact(disclosures)
        assert len(result) == 1
        edu = result[0]["education"]
        assert "주주환원" in edu["impact"]
        assert "소각" in edu["action"]
        assert edu["risk"] == "low"

    def test_unknown_category_gets_default(self, collector):
        """알 수 없는 카테고리는 기본 교육 정보를 받는다."""
        disclosures = [
            {
                "corp_name": "테스트",
                "stock_code": "111111",
                "report_nm": "일반 보고서",
                # category가 없고, report_nm으로도 분류 불가
            }
        ]
        result = collector.analyze_disclosure_impact(disclosures)
        assert len(result) == 1
        edu = result[0]["education"]
        assert "분류되지 않은" in edu["impact"]
        assert edu["risk"] == "low"
        assert result[0]["priority"] == "low"


# ===================================================================
# TestEodFormat: format_eod_with_education() 포맷 테스트
# ===================================================================


class TestEodFormat:
    """format_eod_with_education() 출력 포맷 테스트."""

    def test_with_holdings(self, collector, sample_disclosures):
        """보유종목이 있으면 [보유종목 공시] 섹션이 생성된다."""
        result = collector.format_eod_with_education(
            disclosures=sample_disclosures,
            holdings=["005930"],
        )
        assert "[EOD 공시 모니터링]" in result
        assert "[보유종목 공시] (우선 체크)" in result
        assert "[보유] 삼성전자" in result
        assert "영향:" in result
        assert "조치:" in result
        assert "리스크:" in result
        assert "[기타 주요 공시]" in result
        assert "[투자 학습 팁]" in result

    def test_without_holdings(self, collector, sample_disclosures):
        """보유종목이 없으면 [보유종목 공시] 섹션이 생기지 않는다."""
        result = collector.format_eod_with_education(
            disclosures=sample_disclosures,
            holdings=[],
        )
        assert "[EOD 공시 모니터링]" in result
        assert "[보유종목 공시]" not in result
        assert "[기타 주요 공시]" in result
        # 교육 정보는 여전히 포함
        assert "영향:" in result
        assert "조치:" in result


# ===================================================================
# TestAnalyzeImpact: analyze_disclosure_impact() 동작 테스트
# ===================================================================


class TestAnalyzeImpact:
    """analyze_disclosure_impact() 우선순위 및 정렬 테스트."""

    def test_holding_prioritization(self, collector, sample_disclosures):
        """보유종목 공시는 priority가 +3 부스트되어 상위로 정렬된다."""
        # 삼성전자(005930)를 보유종목으로 지정 (earnings, medium risk)
        result = collector.analyze_disclosure_impact(
            sample_disclosures,
            holdings=["005930"],
        )

        # 삼성전자는 보유종목이므로 is_holding=True
        samsung = [d for d in result if d["corp_name"] == "삼성전자"][0]
        assert samsung["is_holding"] is True
        # medium(2) + 3 = 5 → critical
        assert samsung["priority"] == "critical"

        # 비보유 종목은 is_holding=False
        sk = [d for d in result if d["corp_name"] == "SK하이닉스"][0]
        assert sk["is_holding"] is False

        # delisting(critical=4)과 보유 earnings(medium+3=5) 중
        # 보유 삼성전자가 최상위에 있어야 함
        assert result[0]["corp_name"] == "삼성전자"

    def test_empty_disclosures(self, collector):
        """빈 공시 리스트를 입력하면 빈 리스트를 반환한다."""
        result = collector.analyze_disclosure_impact([])
        assert result == []
