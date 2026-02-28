"""src/utils/market_utils.py 테스트."""

from src.utils.market_utils import is_etf, _KNOWN_ETF_TICKERS


class TestIsETF:
    """is_etf() 함수 테스트."""

    def test_known_etf_tickers_return_true(self):
        """ETF 유니버스에 정의된 티커는 True를 반환한다."""
        etf_tickers = [
            "069500",   # KODEX 200
            "371460",   # TIGER 미국S&P500
            "133690",   # TIGER 미국나스닥100
            "132030",   # KODEX 골드선물(H)
            "439870",   # KODEX 단기채권
            "091160",   # KODEX 반도체
        ]
        for ticker in etf_tickers:
            assert is_etf(ticker), f"{ticker}은 ETF인데 False 반환"

    def test_stock_tickers_return_false(self):
        """일반 주식 티커는 False를 반환한다."""
        stock_tickers = [
            "005930",   # 삼성전자
            "000660",   # SK하이닉스
            "035420",   # NAVER
        ]
        for ticker in stock_tickers:
            assert not is_etf(ticker), f"{ticker}은 주식인데 True 반환"

    def test_etf_collector_universe_included(self):
        """etf_collector.ETF_UNIVERSE의 티커도 포함된다."""
        # ETF_UNIVERSE에만 있는 티커들
        assert is_etf("360750")   # TIGER 미국S&P500 (etf_collector 전용)
        assert is_etf("214980")   # KODEX 단기채권PLUS
        assert is_etf("114820")   # KODEX 국고채3년

    def test_cross_asset_universe_included(self):
        """cross_asset_momentum 유니버스도 포함된다."""
        # cross_asset에도 있는 티커 (이미 etf_rotation에도 있음)
        assert is_etf("069500")
