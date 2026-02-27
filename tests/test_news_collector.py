"""뉴스/공시 수집기(src/data/news_collector.py) 테스트.

NewsCollector 클래스의 공시 수집, 필터링, 스코어링,
뉴스레터/체크리스트/장마감 뉴스 포맷 기능을 검증한다.
DART API 호출은 mock 처리한다.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.data.news_collector import (
    CATEGORY_SCORES,
    IMPORTANT_KEYWORDS,
    NewsCollector,
)


# ===================================================================
# 공통 픽스처
# ===================================================================


@pytest.fixture
def collector():
    """API 키가 설정된 NewsCollector."""
    return NewsCollector(dart_api_key="test_api_key")


@pytest.fixture
def unconfigured_collector():
    """API 키가 없는 NewsCollector."""
    with patch.dict("os.environ", {"DART_API_KEY": ""}, clear=False):
        return NewsCollector(dart_api_key="")


@pytest.fixture
def sample_disclosures():
    """샘플 공시 리스트."""
    return [
        {
            "corp_code": "00126380",
            "corp_name": "삼성전자",
            "stock_code": "005930",
            "report_nm": "영업(잠정)실적(공정공시)",
            "rcept_dt": "20260226",
        },
        {
            "corp_code": "00164779",
            "corp_name": "SK하이닉스",
            "stock_code": "000660",
            "report_nm": "자기주식취득결정",
            "rcept_dt": "20260226",
        },
        {
            "corp_code": "00999999",
            "corp_name": "테스트기업",
            "stock_code": "999999",
            "report_nm": "기타 일반 보고서",
            "rcept_dt": "20260226",
        },
        {
            "corp_code": "00164779",
            "corp_name": "네이버",
            "stock_code": "035420",
            "report_nm": "합병등종료보고서",
            "rcept_dt": "20260226",
        },
    ]


# ===================================================================
# 초기화 테스트
# ===================================================================


class TestInit:
    """NewsCollector 초기화 테스트."""

    def test_configured_with_key(self, collector):
        """API 키를 전달하면 configured 상태이다."""
        assert collector.is_configured()

    def test_unconfigured_without_key(self, unconfigured_collector):
        """API 키가 없으면 unconfigured 상태이다."""
        assert not unconfigured_collector.is_configured()

    @patch.dict("os.environ", {"DART_API_KEY": "env_key"})
    def test_loads_from_env(self):
        """환경변수에서 API 키를 로드한다."""
        nc = NewsCollector()
        assert nc.dart_api_key == "env_key"


# ===================================================================
# fetch_recent_disclosures() 테스트
# ===================================================================


class TestFetchRecentDisclosures:
    """fetch_recent_disclosures() 테스트."""

    def test_unconfigured_returns_empty(self, unconfigured_collector):
        """API 미설정 시 빈 리스트를 반환한다."""
        result = unconfigured_collector.fetch_recent_disclosures()
        assert result == []

    @patch("src.data.news_collector.requests.get")
    def test_successful_fetch(self, mock_get, collector):
        """성공적인 API 호출 시 공시 리스트를 반환한다."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "000",
            "list": [
                {"corp_name": "삼성전자", "report_nm": "사업보고서"},
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = collector.fetch_recent_disclosures(days=1)
        assert len(result) == 1
        assert result[0]["corp_name"] == "삼성전자"

    @patch("src.data.news_collector.requests.get")
    def test_api_error_returns_empty(self, mock_get, collector):
        """API 오류 시 빈 리스트를 반환한다."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "013",
            "message": "조회된 데이터가 없습니다.",
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = collector.fetch_recent_disclosures()
        assert result == []

    @patch("src.data.news_collector.requests.get")
    def test_timeout_returns_empty(self, mock_get, collector):
        """타임아웃 시 빈 리스트를 반환한다."""
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()

        result = collector.fetch_recent_disclosures()
        assert result == []

    @patch("src.data.news_collector.requests.get")
    def test_connection_error_returns_empty(self, mock_get, collector):
        """연결 오류 시 빈 리스트를 반환한다."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = collector.fetch_recent_disclosures()
        assert result == []

    @patch("src.data.news_collector.requests.get")
    def test_corp_codes_filter(self, mock_get, collector):
        """corp_codes 파라미터로 기업을 필터링한다."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "000",
            "list": [
                {"corp_code": "001", "corp_name": "A"},
                {"corp_code": "002", "corp_name": "B"},
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = collector.fetch_recent_disclosures(
            corp_codes=["001", "002"], days=1
        )
        assert len(result) == 2


# ===================================================================
# filter_important() 테스트
# ===================================================================


class TestFilterImportant:
    """filter_important() 중요 공시 필터링 테스트."""

    def test_filters_earnings(self, collector, sample_disclosures):
        """실적 공시가 필터링된다."""
        result = collector.filter_important(sample_disclosures)
        categories = [d["category"] for d in result]
        assert "earnings" in categories

    def test_filters_buyback(self, collector, sample_disclosures):
        """자사주 공시가 필터링된다."""
        result = collector.filter_important(sample_disclosures)
        categories = [d["category"] for d in result]
        assert "buyback" in categories

    def test_filters_merger(self, collector, sample_disclosures):
        """합병 공시가 필터링된다."""
        result = collector.filter_important(sample_disclosures)
        categories = [d["category"] for d in result]
        assert "merger" in categories

    def test_excludes_unimportant(self, collector, sample_disclosures):
        """중요하지 않은 공시는 제외된다."""
        result = collector.filter_important(sample_disclosures)
        # "기타 일반 보고서"는 제외
        corp_names = [d["corp_name"] for d in result]
        assert "테스트기업" not in corp_names

    def test_empty_list_returns_empty(self, collector):
        """빈 리스트 입력 시 빈 리스트를 반환한다."""
        result = collector.filter_important([])
        assert result == []

    def test_all_categories_covered(self):
        """모든 카테고리에 대한 키워드가 정의되어 있다."""
        for category in CATEGORY_SCORES:
            assert category in IMPORTANT_KEYWORDS


# ===================================================================
# score_impact() 테스트
# ===================================================================


class TestScoreImpact:
    """score_impact() 영향도 스코어링 테스트."""

    def test_base_score_by_category(self, collector):
        """카테고리별 기본 점수가 올바르다."""
        disc = {"report_nm": "영업(잠정)실적", "category": "earnings"}
        score = collector.score_impact(disc)
        assert score == CATEGORY_SCORES["earnings"]

    def test_holding_bonus(self, collector):
        """보유종목 관련 공시는 가산점을 받는다."""
        disc = {
            "report_nm": "영업(잠정)실적",
            "category": "earnings",
            "stock_code": "005930",
            "corp_name": "삼성전자",
        }
        base = collector.score_impact(disc, holdings=None)
        boosted = collector.score_impact(disc, holdings=["005930"])
        assert boosted > base

    def test_max_score_is_10(self, collector):
        """최대 점수는 10이다."""
        disc = {
            "report_nm": "합병",
            "category": "merger",
            "stock_code": "005930",
            "corp_name": "삼성전자",
        }
        score = collector.score_impact(disc, holdings=["005930"])
        assert score <= 10

    def test_no_category_gets_base_score(self, collector):
        """카테고리가 없는 공시는 기본 점수를 받는다."""
        disc = {"report_nm": "알 수 없는 보고서"}
        score = collector.score_impact(disc)
        assert score == 3


# ===================================================================
# generate_newsletter() 테스트
# ===================================================================


class TestGenerateNewsletter:
    """generate_newsletter() 뉴스레터 생성 테스트."""

    def test_unconfigured_returns_message(self, unconfigured_collector):
        """API 미설정 시 안내 메시지를 반환한다."""
        result = unconfigured_collector.generate_newsletter()
        assert "DART API 미설정" in result

    @patch.object(NewsCollector, "fetch_recent_disclosures")
    def test_no_important_disclosures(self, mock_fetch, collector):
        """중요 공시가 없으면 안내 메시지를 반환한다."""
        mock_fetch.return_value = [
            {"report_nm": "기타 일반 보고서", "corp_name": "A"},
        ]
        result = collector.generate_newsletter()
        assert "중요 공시 없음" in result

    @patch.object(NewsCollector, "fetch_recent_disclosures")
    def test_with_disclosures(self, mock_fetch, collector, sample_disclosures):
        """중요 공시가 있으면 뉴스레터를 생성한다."""
        mock_fetch.return_value = sample_disclosures
        result = collector.generate_newsletter()

        assert "[공시 뉴스레터]" in result
        assert "삼성전자" in result
        assert "중요 공시" in result

    @patch.object(NewsCollector, "fetch_recent_disclosures")
    def test_holdings_marked(self, mock_fetch, collector, sample_disclosures):
        """보유종목 관련 공시가 표시된다."""
        mock_fetch.return_value = sample_disclosures
        result = collector.generate_newsletter(holdings=["005930"])
        assert "[보유]" in result


# ===================================================================
# format_morning_checklist() 테스트
# ===================================================================


class TestFormatMorningChecklist:
    """format_morning_checklist() 오전 체크리스트 테스트."""

    def test_basic_format(self, collector):
        """기본 포맷이 올바르다."""
        result = collector.format_morning_checklist(disclosures=[])
        assert "[오전 체크리스트]" in result

    def test_with_disclosures(self, collector, sample_disclosures):
        """공시 정보가 포함된다."""
        important = collector.filter_important(sample_disclosures)
        result = collector.format_morning_checklist(disclosures=important)
        assert "[주요 공시]" in result
        assert "삼성전자" in result

    def test_with_macro_data(self, collector):
        """매크로 데이터가 포함된다."""
        macro = {"기준금리": "3.50%", "VIX": "18.5"}
        result = collector.format_morning_checklist(
            disclosures=[], macro_data=macro
        )
        assert "[매크로 환경]" in result
        assert "기준금리" in result

    def test_no_disclosures_shows_none(self, collector):
        """공시가 없으면 '없음'이 표시된다."""
        result = collector.format_morning_checklist(disclosures=[])
        assert "없음" in result


# ===================================================================
# format_eod_news() 테스트
# ===================================================================


class TestFormatEodNews:
    """format_eod_news() 장마감 뉴스 테스트."""

    def test_no_disclosures_message(self, collector):
        """공시가 없으면 안내 메시지를 반환한다."""
        result = collector.format_eod_news(disclosures=[])
        assert "중요 공시 없음" in result

    def test_holding_news_prioritized(self, collector, sample_disclosures):
        """보유종목 관련 공시가 우선 표시된다."""
        important = collector.filter_important(sample_disclosures)
        result = collector.format_eod_news(
            disclosures=important, holdings=["005930"]
        )
        assert "[보유종목 관련]" in result

    def test_other_news_shown(self, collector, sample_disclosures):
        """비보유 종목 공시도 표시된다."""
        important = collector.filter_important(sample_disclosures)
        result = collector.format_eod_news(disclosures=important, holdings=[])
        assert "[기타 주요 공시]" in result


# ===================================================================
# _categorize() 테스트
# ===================================================================


class TestCategorize:
    """_categorize() 카테고리 판별 테스트."""

    def test_earnings(self):
        """실적 공시를 earnings로 분류한다."""
        assert NewsCollector._categorize("영업(잠정)실적(공정공시)") == "earnings"

    def test_buyback(self):
        """자사주 공시를 buyback으로 분류한다."""
        assert NewsCollector._categorize("자기주식취득결정") == "buyback"

    def test_merger(self):
        """합병 공시를 merger로 분류한다."""
        assert NewsCollector._categorize("합병등종료보고서") == "merger"

    def test_dividend(self):
        """배당 공시를 dividend로 분류한다."""
        assert NewsCollector._categorize("현금ㆍ현물배당결정") == "dividend"

    def test_unknown_returns_none(self):
        """분류할 수 없는 공시는 None을 반환한다."""
        assert NewsCollector._categorize("일반 보고서") is None

    def test_delisting(self):
        """상장폐지 공시를 delisting으로 분류한다."""
        assert NewsCollector._categorize("상장폐지결정") == "delisting"

    def test_major_shareholder(self):
        """대주주 변동 공시를 major_shareholder로 분류한다."""
        assert (
            NewsCollector._categorize("임원ㆍ주요주주특정증권등소유상황보고서")
            == "major_shareholder"
        )
