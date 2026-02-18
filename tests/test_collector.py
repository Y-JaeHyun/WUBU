"""데이터 수집 모듈(src/data/collector.py) 테스트.

pykrx API를 직접 호출하는 테스트는 @pytest.mark.slow 로 마킹한다.
빠른 단위 테스트는 mock/샘플 데이터를 사용한다.
"""

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.collector import (
    _format_date,
    get_stock_list,
    get_price_data,
    get_market_cap,
    get_fundamental,
    get_all_fundamentals,
)


# ===================================================================
# _format_date 헬퍼 테스트
# ===================================================================

class TestFormatDate:
    """날짜 포맷 변환 함수 테스트."""

    def test_yyyymmdd_string(self):
        """'YYYYMMDD' 문자열이 그대로 반환된다."""
        assert _format_date("20240101") == "20240101"

    def test_yyyy_mm_dd_string(self):
        """'YYYY-MM-DD' 문자열이 'YYYYMMDD'로 변환된다."""
        assert _format_date("2024-01-01") == "20240101"

    def test_datetime_date(self):
        """datetime.date 객체가 'YYYYMMDD'로 변환된다."""
        d = datetime.date(2024, 1, 15)
        assert _format_date(d) == "20240115"

    def test_datetime_datetime(self):
        """datetime.datetime 객체가 'YYYYMMDD'로 변환된다."""
        dt = datetime.datetime(2024, 3, 20, 10, 30)
        assert _format_date(dt) == "20240320"

    def test_pd_timestamp(self):
        """pandas Timestamp가 'YYYYMMDD'로 변환된다."""
        ts = pd.Timestamp("2024-06-15")
        assert _format_date(ts) == "20240615"


# ===================================================================
# get_stock_list 테스트
# ===================================================================

class TestGetStockList:
    """종목 리스트 조회 테스트."""

    @patch("src.data.collector.pykrx_stock")
    def test_kospi_only(self, mock_pykrx):
        """KOSPI 마켓만 조회할 때 KOSPI 종목만 반환된다."""
        mock_pykrx.get_market_ticker_list.return_value = ["005930", "000660"]
        mock_pykrx.get_market_ticker_name.side_effect = lambda t: {
            "005930": "삼성전자",
            "000660": "SK하이닉스",
        }[t]

        df = get_stock_list("KOSPI")

        assert len(df) == 2
        assert list(df.columns) == ["ticker", "name", "market"]
        assert all(df["market"] == "KOSPI")
        mock_pykrx.get_market_ticker_list.assert_called_once()

    @patch("src.data.collector.pykrx_stock")
    def test_kosdaq_only(self, mock_pykrx):
        """KOSDAQ 마켓만 조회할 때 KOSDAQ 종목만 반환된다."""
        mock_pykrx.get_market_ticker_list.return_value = ["247540"]
        mock_pykrx.get_market_ticker_name.return_value = "에코프로비엠"

        df = get_stock_list("KOSDAQ")

        assert len(df) == 1
        assert df.iloc[0]["market"] == "KOSDAQ"

    @patch("src.data.collector.pykrx_stock")
    def test_all_markets(self, mock_pykrx):
        """ALL 조회 시 KOSPI + KOSDAQ 모두 가져온다."""
        # KOSPI 호출 시 2개, KOSDAQ 호출 시 1개 반환
        mock_pykrx.get_market_ticker_list.side_effect = [
            ["005930", "000660"],  # KOSPI
            ["247540"],            # KOSDAQ
        ]
        mock_pykrx.get_market_ticker_name.side_effect = lambda t: {
            "005930": "삼성전자",
            "000660": "SK하이닉스",
            "247540": "에코프로비엠",
        }[t]

        df = get_stock_list("ALL")

        assert len(df) == 3
        assert set(df["market"].unique()) == {"KOSPI", "KOSDAQ"}

    @patch("src.data.collector.pykrx_stock")
    def test_case_insensitive_market(self, mock_pykrx):
        """마켓 이름이 소문자/대소문자 혼용이어도 정상 동작한다."""
        mock_pykrx.get_market_ticker_list.return_value = ["005930"]
        mock_pykrx.get_market_ticker_name.return_value = "삼성전자"

        df = get_stock_list("kospi")

        assert len(df) == 1
        assert df.iloc[0]["market"] == "KOSPI"

    @patch("src.data.collector.pykrx_stock")
    def test_empty_result(self, mock_pykrx):
        """종목이 없는 경우 빈 DataFrame을 반환한다."""
        mock_pykrx.get_market_ticker_list.return_value = []

        df = get_stock_list("KOSPI")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("src.data.collector.pykrx_stock")
    def test_columns(self, mock_pykrx):
        """반환되는 DataFrame의 컬럼이 ticker, name, market이다."""
        mock_pykrx.get_market_ticker_list.return_value = ["005930"]
        mock_pykrx.get_market_ticker_name.return_value = "삼성전자"

        df = get_stock_list("KOSPI")

        assert "ticker" in df.columns
        assert "name" in df.columns
        assert "market" in df.columns


# ===================================================================
# get_price_data 테스트
# ===================================================================

class TestGetPriceData:
    """가격 데이터 반환 형식 검증."""

    @patch("src.data.collector.pykrx_stock")
    def test_return_columns(self, mock_pykrx):
        """반환 DataFrame에 open, high, low, close, volume 컬럼이 있다."""
        dates = pd.bdate_range("2024-01-02", periods=5)
        mock_df = pd.DataFrame(
            {
                "시가": [70000] * 5,
                "고가": [71000] * 5,
                "저가": [69000] * 5,
                "종가": [70500] * 5,
                "거래량": [1000000] * 5,
                "등락률": [0.5] * 5,
            },
            index=dates,
        )
        mock_pykrx.get_market_ohlcv_by_date.return_value = mock_df

        df = get_price_data("005930", "2024-01-02", "2024-01-08")

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns, f"'{col}' 컬럼이 없습니다."

    @patch("src.data.collector.pykrx_stock")
    def test_index_name_is_date(self, mock_pykrx):
        """인덱스 이름이 'date' 이다."""
        dates = pd.bdate_range("2024-01-02", periods=3)
        mock_df = pd.DataFrame(
            {
                "시가": [70000] * 3,
                "고가": [71000] * 3,
                "저가": [69000] * 3,
                "종가": [70500] * 3,
                "거래량": [1000000] * 3,
                "등락률": [0.5] * 3,
            },
            index=dates,
        )
        mock_pykrx.get_market_ohlcv_by_date.return_value = mock_df

        df = get_price_data("005930", "20240102", "20240104")

        assert df.index.name == "date"

    @patch("src.data.collector.pykrx_stock")
    def test_empty_data(self, mock_pykrx):
        """데이터가 없으면 빈 DataFrame을 반환한다."""
        mock_pykrx.get_market_ohlcv_by_date.return_value = pd.DataFrame()

        df = get_price_data("999999", "20240101", "20240110")

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("src.data.collector.pykrx_stock")
    def test_date_format_yyyymmdd(self, mock_pykrx):
        """'YYYYMMDD' 날짜 포맷이 지원된다."""
        mock_pykrx.get_market_ohlcv_by_date.return_value = pd.DataFrame()

        get_price_data("005930", "20240102", "20240108")

        mock_pykrx.get_market_ohlcv_by_date.assert_called_once_with(
            "20240102", "20240108", "005930"
        )

    @patch("src.data.collector.pykrx_stock")
    def test_date_format_with_dash(self, mock_pykrx):
        """'YYYY-MM-DD' 날짜 포맷이 지원된다 (하이픈 제거 후 호출)."""
        mock_pykrx.get_market_ohlcv_by_date.return_value = pd.DataFrame()

        get_price_data("005930", "2024-01-02", "2024-01-08")

        mock_pykrx.get_market_ohlcv_by_date.assert_called_once_with(
            "20240102", "20240108", "005930"
        )

    @patch("src.data.collector.pykrx_stock")
    def test_exception_propagated(self, mock_pykrx):
        """pykrx 예외가 상위로 전파된다."""
        mock_pykrx.get_market_ohlcv_by_date.side_effect = Exception("API 에러")

        with pytest.raises(Exception, match="API 에러"):
            get_price_data("005930", "20240101", "20240110")


# ===================================================================
# get_market_cap 테스트
# ===================================================================

class TestGetMarketCap:
    """시가총액 데이터 반환 형식 검증."""

    @patch("src.data.collector.pykrx_stock")
    def test_return_columns(self, mock_pykrx):
        """반환 DataFrame에 market_cap, volume 등의 컬럼이 있다."""
        dates = pd.bdate_range("2024-01-02", periods=3)
        mock_df = pd.DataFrame(
            {
                "시가총액": [400_000_000_000_000] * 3,
                "거래량": [10_000_000] * 3,
                "거래대금": [700_000_000_000] * 3,
                "상장주식수": [5_969_782_550] * 3,
            },
            index=dates,
        )
        mock_pykrx.get_market_cap_by_date.return_value = mock_df

        df = get_market_cap("005930", "20240102", "20240104")

        for col in ["market_cap", "volume", "trade_value", "listed_shares"]:
            assert col in df.columns

    @patch("src.data.collector.pykrx_stock")
    def test_empty_data(self, mock_pykrx):
        """데이터가 없으면 빈 DataFrame을 반환한다."""
        mock_pykrx.get_market_cap_by_date.return_value = pd.DataFrame()

        df = get_market_cap("999999", "20240101", "20240110")

        assert df.empty


# ===================================================================
# get_fundamental 테스트
# ===================================================================

class TestGetFundamental:
    """펀더멘탈 데이터 반환 형식 검증."""

    @patch("src.data.collector.pykrx_stock")
    def test_return_columns(self, mock_pykrx):
        """반환 DataFrame에 bps, per, pbr, eps, div_yield 컬럼이 있다."""
        dates = pd.bdate_range("2024-01-02", periods=3)
        mock_df = pd.DataFrame(
            {
                "BPS": [45000] * 3,
                "PER": [12.5] * 3,
                "PBR": [1.5] * 3,
                "EPS": [5600] * 3,
                "DIV": [2.1] * 3,
                "DPS": [1444] * 3,
            },
            index=dates,
        )
        mock_pykrx.get_market_fundamental_by_date.return_value = mock_df

        df = get_fundamental("005930", "20240102", "20240104")

        for col in ["bps", "per", "pbr", "eps", "div_yield"]:
            assert col in df.columns

    @patch("src.data.collector.pykrx_stock")
    def test_index_name(self, mock_pykrx):
        """인덱스 이름이 'date' 이다."""
        dates = pd.bdate_range("2024-01-02", periods=3)
        mock_df = pd.DataFrame(
            {
                "BPS": [45000] * 3,
                "PER": [12.5] * 3,
                "PBR": [1.5] * 3,
                "EPS": [5600] * 3,
                "DIV": [2.1] * 3,
                "DPS": [1444] * 3,
            },
            index=dates,
        )
        mock_pykrx.get_market_fundamental_by_date.return_value = mock_df

        df = get_fundamental("005930", "20240102", "20240104")

        assert df.index.name == "date"

    @patch("src.data.collector.pykrx_stock")
    def test_empty_data(self, mock_pykrx):
        """데이터가 없으면 빈 DataFrame을 반환한다."""
        mock_pykrx.get_market_fundamental_by_date.return_value = pd.DataFrame()

        df = get_fundamental("999999", "20240101", "20240110")

        assert df.empty

    @patch("src.data.collector.pykrx_stock")
    def test_exception_propagated(self, mock_pykrx):
        """pykrx 예외가 상위로 전파된다."""
        mock_pykrx.get_market_fundamental_by_date.side_effect = RuntimeError(
            "네트워크 오류"
        )

        with pytest.raises(RuntimeError, match="네트워크 오류"):
            get_fundamental("005930", "20240101", "20240110")


# ===================================================================
# get_all_fundamentals 테스트
# ===================================================================

class TestGetAllFundamentals:
    """전종목 펀더멘탈 데이터 조회 테스트."""

    @patch("src.data.collector.pykrx_stock")
    def test_all_market(self, mock_pykrx):
        """ALL 마켓 조회 시 KOSPI, KOSDAQ 모두 호출된다."""
        # 빈 DataFrame 반환하여 빠르게 테스트
        mock_pykrx.get_market_fundamental.return_value = pd.DataFrame()
        mock_pykrx.get_market_cap.return_value = pd.DataFrame()

        get_all_fundamentals("20240102", market="ALL")

        # KOSPI, KOSDAQ 각각 호출 확인
        assert mock_pykrx.get_market_fundamental.call_count == 2
        assert mock_pykrx.get_market_cap.call_count == 2

    @patch("src.data.collector.pykrx_stock")
    def test_single_market(self, mock_pykrx):
        """단일 마켓 조회 시 해당 마켓만 호출된다."""
        mock_pykrx.get_market_fundamental.return_value = pd.DataFrame()
        mock_pykrx.get_market_cap.return_value = pd.DataFrame()

        get_all_fundamentals("20240102", market="KOSPI")

        assert mock_pykrx.get_market_fundamental.call_count == 1

    @patch("src.data.collector.pykrx_stock")
    def test_return_columns_on_data(self, mock_pykrx):
        """데이터가 있으면 필수 컬럼이 포함된 DataFrame을 반환한다."""
        fund_df = pd.DataFrame(
            {
                "BPS": [45000, 30000],
                "PER": [12.5, 8.0],
                "PBR": [1.5, 0.8],
                "EPS": [5600, 3800],
                "DIV": [2.1, 3.5],
                "DPS": [1444, 1000],
            },
            index=pd.Index(["005930", "000660"], name="티커"),
        )
        cap_df = pd.DataFrame(
            {
                "종가": [70000, 130000],
                "시가총액": [400_000_000_000_000, 90_000_000_000_000],
                "거래량": [10_000_000, 3_000_000],
                "거래대금": [700_000_000_000, 390_000_000_000],
                "상장주식수": [5_969_782_550, 728_002_365],
            },
            index=pd.Index(["005930", "000660"], name="티커"),
        )
        mock_pykrx.get_market_fundamental.return_value = fund_df
        mock_pykrx.get_market_cap.return_value = cap_df
        mock_pykrx.get_market_ticker_name.side_effect = lambda t: {
            "005930": "삼성전자",
            "000660": "SK하이닉스",
        }.get(t, "UNKNOWN")

        df = get_all_fundamentals("20240102", market="KOSPI")

        assert not df.empty
        for col in ["ticker", "name", "market", "per", "pbr", "close", "market_cap"]:
            assert col in df.columns, f"'{col}' 컬럼이 없습니다."

    @patch("src.data.collector.pykrx_stock")
    def test_date_format_with_dash(self, mock_pykrx):
        """'YYYY-MM-DD' 날짜 포맷이 'YYYYMMDD'로 변환된다."""
        mock_pykrx.get_market_fundamental.return_value = pd.DataFrame()
        mock_pykrx.get_market_cap.return_value = pd.DataFrame()

        get_all_fundamentals("2024-01-02", market="KOSPI")

        # get_market_fundamental 이 "20240102" 로 호출됐는지 확인
        call_args = mock_pykrx.get_market_fundamental.call_args
        assert call_args[0][0] == "20240102"

    @patch("src.data.collector.pykrx_stock")
    def test_empty_when_no_data(self, mock_pykrx):
        """데이터가 없으면 빈 DataFrame을 반환한다."""
        mock_pykrx.get_market_fundamental.return_value = pd.DataFrame()
        mock_pykrx.get_market_cap.return_value = pd.DataFrame()

        df = get_all_fundamentals("20240102", market="KOSPI")

        assert isinstance(df, pd.DataFrame)
        assert df.empty
