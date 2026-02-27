"""일일 시뮬레이터(src/data/daily_simulator.py) 테스트.

DailySimulator 클래스의 시뮬레이션, 저장, 히스토리 조회,
드리프트 분석, 텔레그램 리포트 기능을 검증한다.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.daily_simulator import DailySimulator


# ===================================================================
# 공통 픽스처
# ===================================================================


@pytest.fixture
def sim(tmp_path):
    """기본 DailySimulator 인스턴스 (임시 디렉토리)."""
    return DailySimulator(data_dir=str(tmp_path))


@pytest.fixture
def mock_strategy():
    """시그널을 반환하는 mock 전략."""
    strategy = MagicMock()
    strategy.generate_signals.return_value = {
        "005930": 0.15,
        "000660": 0.12,
        "035420": 0.10,
    }
    return strategy


@pytest.fixture
def sim_with_strategy(tmp_path, mock_strategy):
    """전략이 등록된 DailySimulator."""
    return DailySimulator(
        data_dir=str(tmp_path),
        strategies={"MultiFactor": mock_strategy},
    )


@pytest.fixture
def sample_result():
    """샘플 시뮬레이션 결과."""
    return {
        "date": "2026-02-26",
        "strategy": "MultiFactor",
        "universe_size": 3,
        "selected": [
            {"ticker": "005930", "name": "005930", "weight": 0.15, "rank": 1, "score": 0.15, "change": "NEW"},
            {"ticker": "000660", "name": "000660", "weight": 0.12, "rank": 2, "score": 0.12, "change": "NEW"},
            {"ticker": "035420", "name": "035420", "weight": 0.10, "rank": 3, "score": 0.10, "change": "NEW"},
        ],
        "factor_scores": {
            "005930": {"composite": 0.15},
            "000660": {"composite": 0.12},
            "035420": {"composite": 0.10},
        },
        "turnover_vs_yesterday": 1.0,
        "rebalancing_countdown": 10,
    }


# ===================================================================
# 초기화 테스트
# ===================================================================


class TestInit:
    """DailySimulator 초기화 테스트."""

    def test_default_init(self, sim):
        """기본 초기화 시 빈 전략 딕셔너리를 가진다."""
        assert sim.strategies == {}

    def test_init_with_strategies(self, sim_with_strategy):
        """전략을 전달하면 등록된다."""
        assert "MultiFactor" in sim_with_strategy.strategies

    def test_data_dir_created(self, tmp_path):
        """data_dir이 자동으로 생성된다."""
        data_dir = tmp_path / "simulation"
        sim = DailySimulator(data_dir=str(data_dir))
        assert data_dir.exists()


# ===================================================================
# save_selection / load 테스트
# ===================================================================


class TestSaveSelection:
    """save_selection() 저장 테스트."""

    def test_save_creates_json_file(self, sim, sample_result):
        """JSON 파일이 올바른 경로에 생성된다."""
        sim.save_selection("2026-02-26", "MultiFactor", sample_result)

        path = sim.data_dir / "2026-02-26" / "MultiFactor.json"
        assert path.exists()

    def test_saved_data_is_valid_json(self, sim, sample_result):
        """저장된 데이터가 유효한 JSON이다."""
        sim.save_selection("2026-02-26", "MultiFactor", sample_result)

        path = sim.data_dir / "2026-02-26" / "MultiFactor.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["strategy"] == "MultiFactor"
        assert data["date"] == "2026-02-26"
        assert len(data["selected"]) == 3

    def test_save_creates_date_directory(self, sim, sample_result):
        """날짜별 디렉토리가 생성된다."""
        sim.save_selection("2026-02-26", "Test", sample_result)
        assert (sim.data_dir / "2026-02-26").is_dir()


# ===================================================================
# get_history() 테스트
# ===================================================================


class TestGetHistory:
    """get_history() 히스토리 조회 테스트."""

    def test_empty_history(self, sim):
        """히스토리가 없으면 빈 리스트를 반환한다."""
        result = sim.get_history("MultiFactor", days=30)
        assert result == []

    def test_returns_saved_data(self, sim, sample_result):
        """저장된 데이터를 조회할 수 있다."""
        sim.save_selection("2026-02-26", "MultiFactor", sample_result)

        history = sim.get_history("MultiFactor", days=30)
        assert len(history) == 1
        assert history[0]["strategy"] == "MultiFactor"

    def test_respects_days_limit(self, sim, sample_result):
        """days 파라미터가 조회 범위를 제한한다."""
        for i in range(5):
            date = f"2026-02-{20 + i:02d}"
            data = dict(sample_result)
            data["date"] = date
            sim.save_selection(date, "MultiFactor", data)

        history = sim.get_history("MultiFactor", days=3)
        assert len(history) <= 3

    def test_sorted_reverse_chronological(self, sim, sample_result):
        """결과가 날짜 내림차순으로 정렬된다."""
        for i in range(3):
            date = f"2026-02-{24 + i:02d}"
            data = dict(sample_result)
            data["date"] = date
            sim.save_selection(date, "MultiFactor", data)

        history = sim.get_history("MultiFactor", days=10)
        if len(history) >= 2:
            assert history[0]["date"] >= history[1]["date"]


# ===================================================================
# analyze_drift() 테스트
# ===================================================================


class TestAnalyzeDrift:
    """analyze_drift() 괴리율 분석 테스트."""

    def test_no_strategy_returns_error(self, sim):
        """전략이 없으면 에러를 반환한다."""
        result = sim.analyze_drift({"005930": 0.5})
        assert result.get("error") == "전략 없음"

    def test_no_history_returns_error(self, sim_with_strategy):
        """히스토리가 없으면 에러를 반환한다."""
        result = sim_with_strategy.analyze_drift({"005930": 0.5})
        assert result.get("error") == "히스토리 없음"

    def test_zero_drift_when_same(self, sim_with_strategy, sample_result):
        """실제 보유와 가상이 동일하면 drift가 0이다."""
        sim_with_strategy.save_selection(
            "2026-02-26", "MultiFactor", sample_result
        )

        actual = {
            "005930": 0.15,
            "000660": 0.12,
            "035420": 0.10,
        }
        result = sim_with_strategy.analyze_drift(actual)
        assert result["drift_pct"] == 0.0

    def test_positive_drift_when_different(self, sim_with_strategy, sample_result):
        """실제 보유와 가상이 다르면 drift가 양수이다."""
        sim_with_strategy.save_selection(
            "2026-02-26", "MultiFactor", sample_result
        )

        actual = {"005930": 0.50, "999999": 0.50}
        result = sim_with_strategy.analyze_drift(actual)
        assert result["drift_pct"] > 0

    def test_drift_details_sorted_by_diff(self, sim_with_strategy, sample_result):
        """details가 diff 기준 내림차순으로 정렬된다."""
        sim_with_strategy.save_selection(
            "2026-02-26", "MultiFactor", sample_result
        )

        actual = {"005930": 0.50}
        result = sim_with_strategy.analyze_drift(actual)

        details = result.get("details", [])
        if len(details) >= 2:
            assert details[0]["diff"] >= details[1]["diff"]


# ===================================================================
# run_daily_simulation() 테스트
# ===================================================================


class TestRunDailySimulation:
    """run_daily_simulation() 테스트."""

    @patch("src.data.daily_simulator.DailySimulator.get_rebalancing_countdown")
    def test_returns_results_for_all_strategies(
        self, mock_countdown, sim_with_strategy
    ):
        """모든 전략에 대한 결과를 반환한다."""
        mock_countdown.return_value = 10

        results = sim_with_strategy.run_daily_simulation("2026-02-26")
        assert "MultiFactor" in results

    @patch("src.data.daily_simulator.DailySimulator.get_rebalancing_countdown")
    def test_saves_result_file(self, mock_countdown, sim_with_strategy):
        """결과가 파일로 저장된다."""
        mock_countdown.return_value = 10

        sim_with_strategy.run_daily_simulation("2026-02-26")

        path = sim_with_strategy.data_dir / "2026-02-26" / "MultiFactor.json"
        assert path.exists()

    def test_no_strategies_returns_empty(self, sim):
        """전략이 없으면 빈 딕셔너리를 반환한다."""
        results = sim.run_daily_simulation("2026-02-26")
        assert results == {}

    @patch("src.data.daily_simulator.DailySimulator.get_rebalancing_countdown")
    def test_empty_signals_skips_strategy(self, mock_countdown, tmp_path):
        """시그널이 없는 전략은 건너뛴다."""
        mock_countdown.return_value = 5
        empty_strategy = MagicMock()
        empty_strategy.generate_signals.return_value = {}

        sim = DailySimulator(
            data_dir=str(tmp_path),
            strategies={"Empty": empty_strategy},
        )
        results = sim.run_daily_simulation("2026-02-26")
        assert "Empty" not in results

    @patch("src.data.daily_simulator.DailySimulator.get_rebalancing_countdown")
    def test_strategy_exception_handled(self, mock_countdown, tmp_path):
        """전략 실행 중 예외가 발생해도 다른 전략은 계속 실행된다."""
        mock_countdown.return_value = 5

        bad_strategy = MagicMock()
        bad_strategy.generate_signals.side_effect = Exception("fail")

        good_strategy = MagicMock()
        good_strategy.generate_signals.return_value = {"005930": 0.15}

        sim = DailySimulator(
            data_dir=str(tmp_path),
            strategies={"Bad": bad_strategy, "Good": good_strategy},
        )
        results = sim.run_daily_simulation("2026-02-26")

        assert "Bad" not in results
        assert "Good" in results


# ===================================================================
# format_telegram_report() 테스트
# ===================================================================


class TestFormatTelegramReport:
    """format_telegram_report() 텔레그램 리포트 테스트."""

    def test_no_data_returns_message(self, sim):
        """데이터가 없으면 안내 메시지를 반환한다."""
        result = sim.format_telegram_report()
        assert "데이터 없음" in result

    def test_with_data_returns_formatted_text(self, sim, sample_result):
        """데이터가 있으면 포매팅된 텍스트를 반환한다."""
        sim.save_selection("2026-02-26", "MultiFactor", sample_result)

        result = sim.format_telegram_report("2026-02-26")
        assert "[일일 시뮬레이션]" in result
        assert "MultiFactor" in result
        assert "목표 종목" in result

    def test_includes_turnover(self, sim, sample_result):
        """Turnover 정보가 포함된다."""
        sim.save_selection("2026-02-26", "MultiFactor", sample_result)

        result = sim.format_telegram_report("2026-02-26")
        assert "Turnover" in result

    def test_includes_rebalancing_countdown(self, sim, sample_result):
        """리밸런싱 카운트다운이 포함된다."""
        sim.save_selection("2026-02-26", "MultiFactor", sample_result)

        result = sim.format_telegram_report("2026-02-26")
        assert "리밸런싱" in result


# ===================================================================
# 내부 헬퍼 테스트
# ===================================================================


class TestHelpers:
    """내부 헬퍼 메서드 테스트."""

    def test_build_selected_sorted_by_weight(self):
        """_build_selected가 weight 기준 내림차순으로 정렬한다."""
        signals = {"A": 0.10, "B": 0.20, "C": 0.15}
        sim = DailySimulator(data_dir="/tmp/test_build_selected")
        result = sim._build_selected(signals)

        assert result[0]["ticker"] == "B"
        assert result[0]["rank"] == 1
        assert result[1]["ticker"] == "C"
        assert result[2]["ticker"] == "A"

    def test_calc_turnover_first_day(self):
        """첫 날(전일 데이터 없음)은 turnover가 1.0이다."""
        current = [{"ticker": "A"}, {"ticker": "B"}]
        assert DailySimulator._calc_turnover(current, None) == 1.0

    def test_calc_turnover_no_change(self):
        """종목 변화가 없으면 turnover가 0이다."""
        current = [{"ticker": "A"}, {"ticker": "B"}]
        prev = {"selected": [{"ticker": "A"}, {"ticker": "B"}]}
        assert DailySimulator._calc_turnover(current, prev) == 0.0

    def test_calc_turnover_full_change(self):
        """종목이 전부 바뀌면 turnover가 1.0이다."""
        current = [{"ticker": "C"}, {"ticker": "D"}]
        prev = {"selected": [{"ticker": "A"}, {"ticker": "B"}]}
        assert DailySimulator._calc_turnover(current, prev) == 1.0

    def test_mark_changes_new(self):
        """전일 데이터 없으면 모두 NEW로 표시된다."""
        current = [{"ticker": "A", "rank": 1}]
        result = DailySimulator._mark_changes(current, None)
        assert result[0]["change"] == "NEW"

    def test_mark_changes_same_rank(self):
        """순위 변화가 없으면 = 표시이다."""
        current = [{"ticker": "A", "rank": 1}]
        prev = {"selected": [{"ticker": "A", "rank": 1}]}
        result = DailySimulator._mark_changes(current, prev)
        assert result[0]["change"] == "="

    def test_mark_changes_rank_up(self):
        """순위가 올라가면 UP 표시이다."""
        current = [{"ticker": "A", "rank": 1}]
        prev = {"selected": [{"ticker": "A", "rank": 3}]}
        result = DailySimulator._mark_changes(current, prev)
        assert result[0]["change"] == "UP"

    def test_mark_changes_rank_down(self):
        """순위가 내려가면 DOWN 표시이다."""
        current = [{"ticker": "A", "rank": 3}]
        prev = {"selected": [{"ticker": "A", "rank": 1}]}
        result = DailySimulator._mark_changes(current, prev)
        assert result[0]["change"] == "DOWN"

    def test_prev_date(self):
        """전일 날짜를 올바르게 반환한다."""
        assert DailySimulator._prev_date("2026-02-26") == "2026-02-25"
        assert DailySimulator._prev_date("2026-03-01") == "2026-02-28"

    def test_change_mark(self):
        """변동 표시 텍스트가 올바르다."""
        assert DailySimulator._change_mark("NEW") == "<- NEW"
        assert DailySimulator._change_mark("UP") == "^"
        assert DailySimulator._change_mark("DOWN") == "v"
        assert DailySimulator._change_mark("=") == "="


# ===================================================================
# 데이터 주입 테스트
# ===================================================================


class TestStrategyDataInjection:
    """strategy_data 주입이 올바르게 동작하는지 검증."""

    def test_generate_signals_passes_strategy_data(self):
        """_generate_signals가 strategy_data를 전략에 전달한다."""
        sim = DailySimulator(data_dir="/tmp/test_data_inject")
        fund_df = pd.DataFrame({
            "ticker": ["005930", "000660"],
            "name": ["삼성전자", "SK하이닉스"],
            "pbr": [1.5, 0.8],
            "market_cap": [500e12, 100e12],
        })
        sim.strategy_data = {
            "fundamentals": fund_df,
            "prices": {},
            "index_prices": pd.Series(dtype=float),
        }

        mock_strategy = MagicMock()
        mock_strategy.generate_signals.return_value = {"005930": 0.5}

        result = sim._generate_signals(mock_strategy, "20260226")

        call_args = mock_strategy.generate_signals.call_args[0]
        assert "fundamentals" in call_args[1]
        assert not call_args[1]["fundamentals"].empty
        assert result == {"005930": 0.5}

    def test_generate_signals_merges_etf_prices(self):
        """ETF 전략 시 etf_prices가 머지된다."""
        sim = DailySimulator(data_dir="/tmp/test_etf_merge")
        sim.strategy_data = {"fundamentals": pd.DataFrame()}
        sim.etf_prices = {"069500": pd.DataFrame({"close": [10000]})}

        mock_strategy = MagicMock()
        mock_strategy.etf_universe = {"069500": "KODEX200"}
        mock_strategy.generate_signals.return_value = {"069500": 1.0}

        sim._generate_signals(mock_strategy, "20260226")

        call_args = mock_strategy.generate_signals.call_args[0]
        assert "etf_prices" in call_args[1]
        assert "069500" in call_args[1]["etf_prices"]

    def test_generate_signals_empty_strategy_data(self):
        """strategy_data가 비어있어도 안전하게 동작한다."""
        sim = DailySimulator(data_dir="/tmp/test_empty_data")

        mock_strategy = MagicMock()
        mock_strategy.generate_signals.return_value = {}

        result = sim._generate_signals(mock_strategy, "20260226")

        assert result == {}

    def test_ticker_names_from_fundamentals(self):
        """run_daily_simulation이 펀더멘탈에서 종목명을 추출한다."""
        sim = DailySimulator(data_dir="/tmp/test_ticker_names")
        fund_df = pd.DataFrame({
            "ticker": ["005930", "000660"],
            "name": ["삼성전자", "SK하이닉스"],
            "market_cap": [500e12, 100e12],
        })
        sim.strategy_data = {"fundamentals": fund_df}

        mock_strategy = MagicMock()
        mock_strategy.generate_signals.return_value = {"005930": 0.5, "000660": 0.5}
        sim.strategies = {"test": mock_strategy}

        sim.run_daily_simulation()

        assert sim._ticker_names.get("005930") == "삼성전자"
        assert sim._ticker_names.get("000660") == "SK하이닉스"

    def test_build_selected_uses_ticker_names(self):
        """_build_selected가 종목명 매핑을 사용한다."""
        sim = DailySimulator(data_dir="/tmp/test_names")
        sim._ticker_names = {"005930": "삼성전자", "000660": "SK하이닉스"}

        result = sim._build_selected({"005930": 0.5, "000660": 0.5})

        names = {item["name"] for item in result}
        assert "삼성전자" in names
        assert "SK하이닉스" in names


class TestDryRunReport:
    """dry_run 결과가 텔레그램 리포트에 포함되는지 검증."""

    def test_format_includes_dry_run(self, tmp_path):
        """dry_run 결과가 매수/매도 섹션으로 표시된다."""
        sim = DailySimulator(data_dir=str(tmp_path))
        sim.save_selection("2026-02-26", "multi_factor", {
            "date": "2026-02-26",
            "strategy": "multi_factor",
            "universe_size": 2,
            "selected": [
                {"ticker": "005930", "name": "삼성전자", "weight": 0.5,
                 "rank": 1, "score": 0.5, "change": "="},
            ],
            "factor_scores": {},
            "turnover_vs_yesterday": 0.0,
            "rebalancing_countdown": 5,
        })

        sim.dry_run_results = {
            "multi_factor": {
                "sell_orders": [
                    {"ticker": "000660", "side": "sell", "qty": 3, "amount": 450000},
                ],
                "buy_orders": [
                    {"ticker": "005930", "side": "buy", "qty": 2, "amount": 140000},
                ],
                "total_sell_amount": 450000,
                "total_buy_amount": 140000,
            }
        }

        result = sim.format_telegram_report("2026-02-26")

        assert "매도 예상: 1건" in result
        assert "450,000원" in result
        assert "매수 예상: 1건" in result
        assert "140,000원" in result

    def test_format_no_dry_run(self, tmp_path):
        """dry_run 없으면 매수/매도 섹션이 없다."""
        sim = DailySimulator(data_dir=str(tmp_path))
        sim.save_selection("2026-02-26", "multi_factor", {
            "date": "2026-02-26",
            "strategy": "multi_factor",
            "universe_size": 1,
            "selected": [
                {"ticker": "005930", "name": "005930", "weight": 1.0,
                 "rank": 1, "score": 1.0, "change": "NEW"},
            ],
            "factor_scores": {},
            "turnover_vs_yesterday": 1.0,
            "rebalancing_countdown": 3,
        })

        result = sim.format_telegram_report("2026-02-26")

        assert "매도 예상" not in result
        assert "매수 예상" not in result

    def test_format_no_changes(self, tmp_path):
        """매수/매도가 없으면 '변경 없음' 표시."""
        sim = DailySimulator(data_dir=str(tmp_path))
        sim.save_selection("2026-02-26", "multi_factor", {
            "date": "2026-02-26",
            "strategy": "multi_factor",
            "universe_size": 1,
            "selected": [
                {"ticker": "005930", "name": "005930", "weight": 1.0,
                 "rank": 1, "score": 1.0, "change": "="},
            ],
            "factor_scores": {},
            "turnover_vs_yesterday": 0.0,
            "rebalancing_countdown": 5,
        })

        sim.dry_run_results = {
            "multi_factor": {
                "sell_orders": [],
                "buy_orders": [],
                "total_sell_amount": 0,
                "total_buy_amount": 0,
            }
        }

        result = sim.format_telegram_report("2026-02-26")
        assert "변경 없음" in result
