"""Phase 5-A 모듈 테스트.

GlobalCollector, StockReviewer, AutoBacktester, NightResearcher를 검증한다.
외부 의존성(yfinance, get_price_data, Backtest)은 mock 처리한다.
"""

import sys
from datetime import datetime
from types import ModuleType
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest


# ===================================================================
# GlobalCollector 테스트
# ===================================================================


class TestIsYfinanceAvailable:
    """_is_yfinance_available() 함수 검증."""

    def test_yfinance_not_installed_returns_false(self):
        """yfinance가 설치되어 있지 않으면 False를 반환한다."""
        with patch.dict(sys.modules, {"yfinance": None}):
            # 모듈 캐시를 우회하기 위해 함수를 새로 import
            from src.data.global_collector import _is_yfinance_available

            # yfinance import 시 ImportError가 발생하도록 설정
            with patch("builtins.__import__", side_effect=_make_import_raiser("yfinance")):
                result = _is_yfinance_available()

            assert result is False, (
                "yfinance가 설치되어 있지 않으면 _is_yfinance_available()은 False를 반환해야 한다."
            )

    def test_yfinance_installed_returns_true(self):
        """yfinance가 설치되어 있으면 True를 반환한다."""
        fake_yfinance = ModuleType("yfinance")
        with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
            from src.data.global_collector import _is_yfinance_available

            result = _is_yfinance_available()

        assert result is True, (
            "yfinance가 설치되어 있으면 _is_yfinance_available()은 True를 반환해야 한다."
        )


class TestGetGlobalSnapshot:
    """get_global_snapshot() 함수 검증."""

    def test_returns_empty_dataframe_when_yfinance_unavailable(self):
        """yfinance가 없으면 빈 DataFrame을 반환한다."""
        with patch(
            "src.data.global_collector._is_yfinance_available", return_value=False
        ):
            from src.data.global_collector import get_global_snapshot

            result = get_global_snapshot()

        assert isinstance(result, pd.DataFrame), "반환값은 DataFrame이어야 한다."
        assert result.empty, "yfinance 미설치 시 빈 DataFrame이어야 한다."

    def test_with_mock_yfinance_returns_correct_dataframe(self):
        """mock yfinance로 정상 데이터를 반환한다."""
        # mock fast_info 객체 생성
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 5000.0
        mock_fast_info.previous_close = 4950.0

        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info

        fake_yf = MagicMock()
        fake_yf.Ticker.return_value = mock_ticker

        with patch(
            "src.data.global_collector._is_yfinance_available", return_value=True
        ):
            with patch.dict(sys.modules, {"yfinance": fake_yf}):
                from src.data.global_collector import get_global_snapshot

                result = get_global_snapshot(symbols=["^GSPC"])

        assert not result.empty, "mock yfinance가 있으면 비어 있지 않아야 한다."
        assert len(result) == 1, "심볼 1개 조회 시 행이 1개여야 한다."
        assert result.iloc[0]["symbol"] == "^GSPC"
        assert result.iloc[0]["price"] == 5000.0
        # change_pct = (5000/4950 - 1) * 100 = 약 1.01%
        expected_change = round(((5000.0 / 4950.0) - 1) * 100, 2)
        assert result.iloc[0]["change_pct"] == expected_change

    def test_with_mock_yfinance_multiple_symbols(self):
        """여러 심볼을 조회하면 해당 수만큼 행을 반환한다."""
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 100.0
        mock_fast_info.previous_close = 100.0

        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info

        fake_yf = MagicMock()
        fake_yf.Ticker.return_value = mock_ticker

        with patch(
            "src.data.global_collector._is_yfinance_available", return_value=True
        ):
            with patch.dict(sys.modules, {"yfinance": fake_yf}):
                from src.data.global_collector import get_global_snapshot

                result = get_global_snapshot(symbols=["^GSPC", "^VIX"])

        assert len(result) == 2, "2개 심볼 조회 시 행이 2개여야 한다."

    def test_ticker_exception_skips_symbol(self):
        """특정 심볼 조회 중 예외가 발생하면 해당 심볼만 건너뛴다."""
        fake_yf = MagicMock()
        fake_yf.Ticker.side_effect = Exception("Network error")

        with patch(
            "src.data.global_collector._is_yfinance_available", return_value=True
        ):
            with patch.dict(sys.modules, {"yfinance": fake_yf}):
                from src.data.global_collector import get_global_snapshot

                result = get_global_snapshot(symbols=["^GSPC"])

        assert result.empty, "예외 발생 시 해당 심볼 스킵 후 빈 DataFrame 가능."


class TestFormatGlobalSnapshot:
    """format_global_snapshot() 함수 검증."""

    def test_empty_dataframe_returns_no_data_message(self):
        """빈 DataFrame이면 '데이터 없음' 메시지를 반환한다."""
        from src.data.global_collector import format_global_snapshot

        result = format_global_snapshot(pd.DataFrame())

        assert "데이터 없음" in result, (
            "빈 DataFrame에서는 '데이터 없음'이 포함되어야 한다."
        )

    def test_with_sample_data_returns_formatted_text(self):
        """샘플 데이터가 있으면 포매팅된 텍스트를 반환한다."""
        from src.data.global_collector import format_global_snapshot

        sample = pd.DataFrame(
            [
                {
                    "symbol": "^GSPC",
                    "name": "S&P 500",
                    "price": 5050.25,
                    "change_pct": 1.23,
                    "prev_close": 4988.78,
                },
                {
                    "symbol": "^VIX",
                    "name": "VIX",
                    "price": 18.50,
                    "change_pct": -2.10,
                    "prev_close": 18.90,
                },
            ]
        )

        result = format_global_snapshot(sample)

        assert "[글로벌 시장 현황]" in result, "헤더가 포함되어야 한다."
        assert "S&P 500" in result, "S&P 500 이름이 포함되어야 한다."
        assert "VIX" in result, "VIX 이름이 포함되어야 한다."
        assert "+1.23%" in result, "양수 변동률에 + 부호가 있어야 한다."
        assert "-2.10%" in result, "음수 변동률에 - 부호가 있어야 한다."

    def test_zero_change_has_plus_sign(self):
        """변동률이 0이면 + 부호를 붙인다."""
        from src.data.global_collector import format_global_snapshot

        sample = pd.DataFrame(
            [
                {
                    "symbol": "^GSPC",
                    "name": "S&P 500",
                    "price": 5000.0,
                    "change_pct": 0.0,
                    "prev_close": 5000.0,
                }
            ]
        )

        result = format_global_snapshot(sample)
        assert "+0.00%" in result, "변동률 0일 때 +0.00%로 표시되어야 한다."


# ===================================================================
# StockReviewer 테스트
# ===================================================================


class TestStockReviewer:
    """StockReviewer 클래스 검증."""

    def test_review_holdings_empty(self):
        """보유 종목이 비어있으면 '보유 종목 없음' 메시지를 반환한다."""
        from src.report.stock_reviewer import StockReviewer

        reviewer = StockReviewer()
        result = reviewer.review_holdings([])

        assert "보유 종목 없음" in result, (
            "빈 holdings에서는 '보유 종목 없음'이 포함되어야 한다."
        )

    @patch("src.report.stock_reviewer.get_price_data")
    def test_review_holdings_with_sample_data(self, mock_get_price):
        """샘플 보유 종목으로 리뷰 텍스트를 생성한다."""
        # 52주 가격 데이터 mock
        dates = pd.date_range("2025-01-01", periods=252, freq="B")
        mock_df = pd.DataFrame(
            {
                "close": [50000 + i * 100 for i in range(252)],
                "volume": [100000 + i * 10 for i in range(252)],
            },
            index=dates,
        )
        mock_get_price.return_value = mock_df

        from src.report.stock_reviewer import StockReviewer

        reviewer = StockReviewer()
        holdings = [
            {
                "ticker": "005930",
                "name": "삼성전자",
                "pnl_pct": 5.0,
                "current_price": 70000,
            },
        ]

        result = reviewer.review_holdings(holdings)

        assert "삼성전자" in result, "종목명이 포함되어야 한다."
        assert "005930" in result, "종목코드가 포함되어야 한다."
        assert "+5.00%" in result, "수익률이 포함되어야 한다."
        assert "52주 고가" in result, "52주 고가 정보가 포함되어야 한다."
        assert "52주 저가" in result, "52주 저가 정보가 포함되어야 한다."
        assert "HOLD" in result, "5% 수익률은 HOLD 시그널이어야 한다."

    @patch("src.report.stock_reviewer.get_price_data")
    def test_signal_profit_taking(self, mock_get_price):
        """pnl_pct > 20이면 '익절 검토' 시그널을 반환한다."""
        mock_get_price.return_value = pd.DataFrame()

        from src.report.stock_reviewer import StockReviewer

        reviewer = StockReviewer()
        holdings = [
            {
                "ticker": "005930",
                "name": "삼성전자",
                "pnl_pct": 25.0,
                "current_price": 80000,
            },
        ]

        result = reviewer.review_holdings(holdings)

        assert "익절 검토" in result, (
            "pnl_pct > 20이면 '익절 검토' 시그널이 포함되어야 한다."
        )

    @patch("src.report.stock_reviewer.get_price_data")
    def test_signal_stop_loss(self, mock_get_price):
        """pnl_pct < -15이면 '손절 검토' 시그널을 반환한다."""
        mock_get_price.return_value = pd.DataFrame()

        from src.report.stock_reviewer import StockReviewer

        reviewer = StockReviewer()
        holdings = [
            {
                "ticker": "005930",
                "name": "삼성전자",
                "pnl_pct": -20.0,
                "current_price": 50000,
            },
        ]

        result = reviewer.review_holdings(holdings)

        assert "손절 검토" in result, (
            "pnl_pct < -15이면 '손절 검토' 시그널이 포함되어야 한다."
        )

    @patch("src.report.stock_reviewer.get_price_data")
    def test_signal_hold(self, mock_get_price):
        """pnl_pct가 -15 ~ 20 사이이면 'HOLD' 시그널을 반환한다."""
        mock_get_price.return_value = pd.DataFrame()

        from src.report.stock_reviewer import StockReviewer

        reviewer = StockReviewer()
        holdings = [
            {
                "ticker": "005930",
                "name": "삼성전자",
                "pnl_pct": 10.0,
                "current_price": 65000,
            },
        ]

        result = reviewer.review_holdings(holdings)

        assert "HOLD" in result, (
            "pnl_pct가 -15 ~ 20 범위이면 'HOLD' 시그널이어야 한다."
        )

    @patch("src.report.stock_reviewer.get_price_data")
    def test_signal_boundary_20_is_hold(self, mock_get_price):
        """pnl_pct == 20이면 '익절 검토'가 아닌 'HOLD'이다 (> 20 조건)."""
        mock_get_price.return_value = pd.DataFrame()

        from src.report.stock_reviewer import StockReviewer

        reviewer = StockReviewer()
        holdings = [
            {
                "ticker": "005930",
                "name": "삼성전자",
                "pnl_pct": 20.0,
                "current_price": 72000,
            },
        ]

        result = reviewer.review_holdings(holdings)

        assert "익절 검토" not in result, (
            "pnl_pct == 20 (경계값)은 '익절 검토'가 아니라 HOLD이어야 한다."
        )
        assert "HOLD" in result

    @patch("src.report.stock_reviewer.get_price_data")
    def test_signal_boundary_minus15_is_hold(self, mock_get_price):
        """pnl_pct == -15이면 '손절 검토'가 아닌 'HOLD'이다 (< -15 조건)."""
        mock_get_price.return_value = pd.DataFrame()

        from src.report.stock_reviewer import StockReviewer

        reviewer = StockReviewer()
        holdings = [
            {
                "ticker": "005930",
                "name": "삼성전자",
                "pnl_pct": -15.0,
                "current_price": 51000,
            },
        ]

        result = reviewer.review_holdings(holdings)

        assert "손절 검토" not in result, (
            "pnl_pct == -15 (경계값)은 '손절 검토'가 아니라 HOLD이어야 한다."
        )
        assert "HOLD" in result

    @patch("src.report.stock_reviewer.get_price_data")
    def test_max_stocks_limit(self, mock_get_price):
        """max_stocks를 초과하는 종목은 리뷰하지 않는다."""
        mock_get_price.return_value = pd.DataFrame()

        from src.report.stock_reviewer import StockReviewer

        reviewer = StockReviewer(max_stocks=2)
        holdings = [
            {"ticker": f"00000{i}", "name": f"종목{i}", "pnl_pct": 0, "current_price": 10000}
            for i in range(5)
        ]

        result = reviewer.review_holdings(holdings)

        # 종목0, 종목1만 리뷰, 종목2~4는 빠져야 함
        assert "종목0" in result
        assert "종목1" in result
        assert "종목2" not in result

    @patch("src.report.stock_reviewer.get_price_data")
    def test_volume_spike_detected(self, mock_get_price):
        """20일 평균 대비 2배 이상 거래량이면 거래량 정보가 표시된다."""
        dates = pd.date_range("2025-01-01", periods=30, freq="B")
        volumes = [100000] * 29 + [300000]  # 마지막 날 3배
        mock_df = pd.DataFrame(
            {
                "close": [50000] * 30,
                "volume": volumes,
            },
            index=dates,
        )
        mock_get_price.return_value = mock_df

        from src.report.stock_reviewer import StockReviewer

        reviewer = StockReviewer()
        holdings = [
            {
                "ticker": "005930",
                "name": "삼성전자",
                "pnl_pct": 5.0,
                "current_price": 50000,
            },
        ]

        result = reviewer.review_holdings(holdings)

        assert "거래량" in result, "거래량 급등 시 거래량 정보가 표시되어야 한다."
        assert "배" in result, "거래량 배수 정보가 포함되어야 한다."


# ===================================================================
# AutoBacktester 테스트
# ===================================================================


class TestAutoBacktester:
    """AutoBacktester 클래스 검증."""

    def test_default_init_strategies(self):
        """기본 초기화 시 올바른 전략 리스트를 가진다."""
        from src.backtest.auto_runner import AutoBacktester

        ab = AutoBacktester()

        assert ab.strategies == ["value", "momentum", "multi_factor"], (
            "기본 전략 리스트는 ['value', 'momentum', 'multi_factor']이어야 한다."
        )
        assert ab.lookback_months == 6, "기본 lookback_months는 6이어야 한다."

    def test_custom_init(self):
        """사용자 지정 인자로 초기화할 수 있다."""
        from src.backtest.auto_runner import AutoBacktester

        ab = AutoBacktester(lookback_months=12, strategies=["value"])

        assert ab.lookback_months == 12
        assert ab.strategies == ["value"]

    def test_create_strategy_value(self):
        """'value' 전략 이름으로 ValueStrategy를 생성한다."""
        mock_strategy = MagicMock()
        mock_strategy.name = "value"

        with patch(
            "src.strategy.value.ValueStrategy", return_value=mock_strategy
        ):
            from src.backtest.auto_runner import AutoBacktester

            result = AutoBacktester._create_strategy("value")

        assert result is not None, "'value' 전략 생성은 None이 아니어야 한다."

    def test_create_strategy_momentum(self):
        """'momentum' 전략 이름으로 MomentumStrategy를 생성한다."""
        mock_strategy = MagicMock()
        mock_strategy.name = "momentum"

        with patch(
            "src.strategy.momentum.MomentumStrategy", return_value=mock_strategy
        ):
            from src.backtest.auto_runner import AutoBacktester

            result = AutoBacktester._create_strategy("momentum")

        assert result is not None, "'momentum' 전략 생성은 None이 아니어야 한다."

    def test_create_strategy_multi_factor(self):
        """'multi_factor' 전략 이름으로 MultiFactorStrategy를 생성한다."""
        mock_strategy = MagicMock()
        mock_strategy.name = "multi_factor"

        with patch(
            "src.strategy.multi_factor.MultiFactorStrategy",
            return_value=mock_strategy,
        ):
            from src.backtest.auto_runner import AutoBacktester

            result = AutoBacktester._create_strategy("multi_factor")

        assert result is not None, "'multi_factor' 전략 생성은 None이 아니어야 한다."

    def test_create_strategy_unknown_returns_none(self):
        """알 수 없는 전략 이름이면 None을 반환한다."""
        from src.backtest.auto_runner import AutoBacktester

        result = AutoBacktester._create_strategy("unknown_strategy")

        assert result is None, "알 수 없는 전략 이름이면 None이어야 한다."

    @patch("src.backtest.auto_runner.Backtest")
    @patch("src.backtest.auto_runner.AutoBacktester._create_strategy")
    def test_run_all_with_mocked_backtest(self, mock_create, mock_bt_cls):
        """run_all()이 모든 전략에 대해 백테스트를 실행한다."""
        from src.backtest.auto_runner import AutoBacktester

        # 전략 mock
        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_create.return_value = mock_strategy

        # Backtest 인스턴스 mock
        mock_bt_instance = MagicMock()
        mock_bt_instance.get_results.return_value = {
            "strategy_name": "test_strategy",
            "total_return": 15.5,
            "cagr": 12.3,
            "sharpe_ratio": 1.5,
            "mdd": -8.2,
            "win_rate": 60.0,
        }
        mock_bt_cls.return_value = mock_bt_instance

        ab = AutoBacktester(strategies=["value", "momentum"])
        result = ab.run_all()

        assert "[자동 백테스트]" in result, "결과에 헤더가 포함되어야 한다."
        assert "+15.50%" in result, "총수익률이 표시되어야 한다."
        assert "+12.30%" in result, "CAGR이 표시되어야 한다."
        assert "1.50" in result, "Sharpe Ratio가 표시되어야 한다."
        assert "-8.20%" in result, "MDD가 표시되어야 한다."
        assert "60.0%" in result, "승률이 표시되어야 한다."

        # 2개 전략 모두 실행되었는지 확인
        assert mock_bt_instance.run.call_count == 2, (
            "2개 전략에 대해 run()이 각각 호출되어야 한다."
        )

    @patch("src.backtest.auto_runner.Backtest")
    @patch("src.backtest.auto_runner.AutoBacktester._create_strategy")
    def test_run_all_strategy_creation_failure(self, mock_create, mock_bt_cls):
        """전략 생성 실패 시 '전략 생성 실패' 메시지가 포함된다."""
        from src.backtest.auto_runner import AutoBacktester

        mock_create.return_value = None

        ab = AutoBacktester(strategies=["bad_strategy"])
        result = ab.run_all()

        assert "전략 생성 실패" in result, (
            "전략 생성 실패 시 오류 메시지가 포함되어야 한다."
        )

    @patch("src.backtest.auto_runner.Backtest")
    @patch("src.backtest.auto_runner.AutoBacktester._create_strategy")
    def test_run_all_backtest_exception(self, mock_create, mock_bt_cls):
        """백테스트 실행 중 예외가 발생하면 오류 메시지를 포함한다."""
        from src.backtest.auto_runner import AutoBacktester

        mock_strategy = MagicMock()
        mock_create.return_value = mock_strategy

        mock_bt_instance = MagicMock()
        mock_bt_instance.run.side_effect = Exception("Data not found")
        mock_bt_cls.return_value = mock_bt_instance

        ab = AutoBacktester(strategies=["value"])
        result = ab.run_all()

        assert "오류" in result, "예외 발생 시 '오류' 메시지가 포함되어야 한다."
        assert "Data not found" in result, "예외 메시지가 포함되어야 한다."


# ===================================================================
# NightResearcher 테스트
# ===================================================================


class TestNightResearcher:
    """NightResearcher 클래스 검증."""

    def test_generate_report_no_data(self):
        """데이터 없이 리포트를 생성하면 '데이터 없음' 메시지가 포함된다."""
        from src.report.night_research import NightResearcher

        nr = NightResearcher()
        result = nr.generate_report()

        assert "[야간 리서치]" in result, "헤더가 포함되어야 한다."
        assert "데이터 없음" in result, (
            "글로벌 데이터 없으면 '데이터 없음'이 포함되어야 한다."
        )
        assert "[내일 전략 시사점]" in result, "전략 시사점 섹션이 포함되어야 한다."

    def test_generate_report_with_none_snapshot(self):
        """global_snapshot=None이면 '데이터 없음' 메시지를 포함한다."""
        from src.report.night_research import NightResearcher

        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=None)

        assert "데이터 없음" in result

    def test_generate_report_with_empty_snapshot(self):
        """빈 DataFrame을 전달하면 '데이터 없음' 메시지를 포함한다."""
        from src.report.night_research import NightResearcher

        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=pd.DataFrame())

        assert "데이터 없음" in result

    def test_generate_report_include_global_false(self):
        """include_global=False이면 글로벌 데이터가 있어도 표시하지 않는다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=1.5, vix=18.0)
        nr = NightResearcher(include_global=False)
        result = nr.generate_report(global_snapshot=snapshot)

        assert "데이터 없음" in result, (
            "include_global=False이면 글로벌 데이터를 표시하지 않아야 한다."
        )

    def test_vix_above_30_shows_fear(self):
        """VIX > 30이면 '공포' 메시지가 포함된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=0.5, vix=35.0)
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "공포" in result, "VIX > 30이면 '공포' 메시지가 포함되어야 한다."
        assert "VIX > 30" in result

    def test_vix_between_20_30_shows_caution(self):
        """VIX가 20~30 사이이면 '경계' 메시지가 포함된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=0.0, vix=25.0)
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "경계" in result, "VIX 20-30이면 '경계' 메시지가 포함되어야 한다."

    def test_vix_below_20_shows_stable(self):
        """VIX < 20이면 '안정' 메시지가 포함된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=0.0, vix=15.0)
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "안정" in result, "VIX < 20이면 '안정' 메시지가 포함되어야 한다."

    def test_sp_strong_shows_positive_direction(self):
        """S&P > 1%이면 '미국 강세 -> 한국 긍정적' 방향성이 표시된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=1.5, vix=18.0)
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "미국 강세" in result, "S&P > 1%이면 '미국 강세' 방향성이어야 한다."
        assert "한국 긍정적" in result

    def test_sp_weak_shows_caution_direction(self):
        """S&P < -1%이면 '미국 약세 -> 한국 주의' 방향성이 표시된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=-1.5, vix=22.0)
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "미국 약세" in result, "S&P < -1%이면 '미국 약세' 방향성이어야 한다."
        assert "한국 주의" in result

    def test_sp_flat_shows_neutral_direction(self):
        """S&P 변동률 -1~1% 사이이면 '미국 보합 -> 한국 중립' 방향성이다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=0.3, vix=18.0)
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "미국 보합" in result, "S&P 보합이면 '미국 보합' 방향성이어야 한다."
        assert "한국 중립" in result

    def test_risk_on_assessment(self):
        """S&P > 1% and VIX < 20이면 '리스크온' 종합 판단이 표시된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=1.5, vix=15.0)
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "리스크온" in result, (
            "S&P > 1% and VIX < 20이면 '리스크온' 종합이어야 한다."
        )
        assert "공격적 포지션" in result

    def test_risk_off_high_vix(self):
        """VIX > 25이면 '리스크오프' 종합 판단이 표시된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=0.5, vix=28.0)
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "리스크오프" in result, (
            "VIX > 25이면 '리스크오프' 종합이어야 한다."
        )
        assert "현금 비중 확대" in result

    def test_risk_off_sp_drop(self):
        """S&P < -1%이면 '리스크오프' 종합 판단이 표시된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=-1.5, vix=18.0)
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "리스크오프" in result, (
            "S&P < -1%이면 '리스크오프' 종합이어야 한다."
        )

    def test_neutral_assessment(self):
        """S&P와 VIX 모두 중립 범위이면 '중립' 종합 판단이 표시된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=0.5, vix=18.0)
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "중립" in result, "중립 조건이면 '중립' 종합이어야 한다."
        assert "기존 전략 유지" in result

    def test_portfolio_state_included(self):
        """portfolio_state가 전달되면 포트폴리오 정보가 표시된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(sp_change=0.5, vix=18.0)
        nr = NightResearcher()
        result = nr.generate_report(
            global_snapshot=snapshot,
            portfolio_state={
                "total_eval": 10_000_000,
                "cash_pct": 30.0,
                "mdd": 0.05,
            },
        )

        assert "포트폴리오" in result, "포트폴리오 정보가 포함되어야 한다."
        assert "10,000,000" in result, "평가금액이 포매팅되어야 한다."
        assert "30.0%" in result, "현금 비중이 표시되어야 한다."

    def test_portfolio_state_none_no_crash(self):
        """portfolio_state=None이어도 오류 없이 리포트를 생성한다."""
        from src.report.night_research import NightResearcher

        nr = NightResearcher()
        result = nr.generate_report(
            global_snapshot=_make_global_snapshot(sp_change=0.5, vix=18.0),
            portfolio_state=None,
        )

        assert "[내일 전략 시사점]" in result
        assert "포트폴리오" not in result

    def test_usd_krw_high_shows_weak_won(self):
        """USD/KRW > 1400이면 '원화 약세' 메시지가 표시된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(
            sp_change=0.5, vix=18.0, usd_krw=1420.0, fx_change=0.5
        )
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "원화 약세" in result, (
            "USD/KRW > 1400이면 '원화 약세' 메시지가 표시되어야 한다."
        )

    def test_usd_krw_low_shows_strong_won(self):
        """USD/KRW < 1300이면 '원화 강세' 메시지가 표시된다."""
        from src.report.night_research import NightResearcher

        snapshot = _make_global_snapshot(
            sp_change=0.5, vix=18.0, usd_krw=1280.0, fx_change=-0.3
        )
        nr = NightResearcher()
        result = nr.generate_report(global_snapshot=snapshot)

        assert "원화 강세" in result, (
            "USD/KRW < 1300이면 '원화 강세' 메시지가 표시되어야 한다."
        )


# ===================================================================
# Helper 함수
# ===================================================================


def _make_import_raiser(blocked_module: str):
    """특정 모듈만 ImportError를 발생시키는 __import__ side_effect를 생성한다."""
    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def _import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return original_import(name, *args, **kwargs)

    return _import


def _make_global_snapshot(
    sp_change: float = 0.0,
    vix: float = 18.0,
    usd_krw: float = 1350.0,
    fx_change: float = 0.0,
) -> pd.DataFrame:
    """테스트용 글로벌 스냅샷 DataFrame을 생성한다.

    Args:
        sp_change: S&P 500 변동률.
        vix: VIX 가격.
        usd_krw: USD/KRW 환율.
        fx_change: 환율 변동률.

    Returns:
        글로벌 스냅샷 DataFrame.
    """
    return pd.DataFrame(
        [
            {
                "symbol": "^GSPC",
                "name": "S&P 500",
                "price": 5000.0,
                "change_pct": sp_change,
                "prev_close": 4950.0,
            },
            {
                "symbol": "^IXIC",
                "name": "NASDAQ",
                "price": 16000.0,
                "change_pct": sp_change * 1.2,
                "prev_close": 15800.0,
            },
            {
                "symbol": "^VIX",
                "name": "VIX",
                "price": vix,
                "change_pct": 0.0,
                "prev_close": vix,
            },
            {
                "symbol": "USDKRW=X",
                "name": "USD/KRW",
                "price": usd_krw,
                "change_pct": fx_change,
                "prev_close": usd_krw,
            },
        ]
    )
