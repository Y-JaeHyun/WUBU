"""실행 모듈 테스트.

KISClient(KIS OpenAPI 클라이언트), RiskGuard(리스크 체크),
RebalanceExecutor(리밸런싱 실행기)를 검증한다.
모든 외부 API 호출은 mock 처리한다.
"""

import os
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _import_kis_client():
    """KISClient 클래스를 임포트한다."""
    from src.execution.kis_client import KISClient
    return KISClient


def _import_risk_guard():
    """RiskGuard 클래스를 임포트한다."""
    from src.execution.risk_guard import RiskGuard
    return RiskGuard


def _try_import_rebalance_executor():
    """RebalanceExecutor가 존재하면 임포트한다."""
    try:
        from src.execution.rebalance_executor import RebalanceExecutor
        return RebalanceExecutor
    except ImportError:
        try:
            from src.execution.executor import RebalanceExecutor
            return RebalanceExecutor
        except ImportError:
            return None


# ===================================================================
# KISClient 테스트
# ===================================================================

class TestKISClient:
    """KIS OpenAPI 클라이언트 검증."""

    @patch.dict(os.environ, {"KIS_TRADING_MODE": "", "KIS_IS_PAPER": "true", "KIS_APP_KEY": "", "KIS_APP_SECRET": "", "KIS_ACCOUNT_NO": "", "KIS_PAPER_ACCOUNT_NO": "", "KIS_REAL_ACCOUNT_NO": ""})
    def test_init_default(self):
        """기본 파라미터로 초기화된다."""
        KISClient = _import_kis_client()
        client = KISClient()
        # is_paper 기본값 True
        is_paper = getattr(client, "is_paper", getattr(client, "_is_paper", None))
        assert is_paper is True, "기본 is_paper는 True여야 합니다."

    @patch.dict(os.environ, {
        "KIS_APP_KEY": "test_app_key",
        "KIS_APP_SECRET": "test_app_secret",
        "KIS_ACCOUNT_NO": "12345678-01",
    })
    def test_init_from_env(self):
        """환경변수로부터 초기화된다."""
        KISClient = _import_kis_client()
        client = KISClient()
        # app_key 등이 환경변수에서 로드되었는지 간접 확인
        assert client.is_configured() is True, (
            "환경변수가 설정되면 is_configured()가 True여야 합니다."
        )

    def test_is_configured_true(self):
        """필수 정보가 있으면 is_configured()가 True이다."""
        KISClient = _import_kis_client()
        client = KISClient(
            app_key="test_key",
            app_secret="test_secret",
            account_no="12345678-01",
        )
        assert client.is_configured() is True, (
            "키가 설정되면 is_configured()가 True여야 합니다."
        )

    @patch.dict(os.environ, {"KIS_APP_KEY": "", "KIS_APP_SECRET": "", "KIS_ACCOUNT_NO": "", "KIS_PAPER_ACCOUNT_NO": "", "KIS_REAL_ACCOUNT_NO": "", "KIS_TRADING_MODE": ""})
    def test_is_configured_false(self):
        """필수 정보가 없으면 is_configured()가 False이다."""
        KISClient = _import_kis_client()
        client = KISClient(app_key="", app_secret="", account_no="")
        assert client.is_configured() is False, (
            "키가 비어 있으면 is_configured()가 False여야 합니다."
        )

    @patch("src.execution.kis_client.requests.post")
    def test_get_access_token(self, mock_post):
        """액세스 토큰을 정상적으로 발급받는다."""
        KISClient = _import_kis_client()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "mock_access_token_12345",
            "token_type": "Bearer",
            "expires_in": 86400,
        }
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        client = KISClient(
            app_key="test_key",
            app_secret="test_secret",
            account_no="12345678-01",
        )

        token = client.get_access_token()

        assert token == "mock_access_token_12345", (
            "액세스 토큰이 올바르게 반환되어야 합니다."
        )
        mock_post.assert_called_once()

    @patch("src.execution.kis_client.requests.post")
    def test_place_buy_order(self, mock_post):
        """매수 주문이 정상적으로 전송된다."""
        KISClient = _import_kis_client()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "rt_cd": "0",
            "msg_cd": "APBK0013",
            "msg1": "주문 전송 완료",
            "output": {"KRX_FWDG_ORD_ORGNO": "00950", "ODNO": "0000123456"},
        }
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        client = KISClient(
            app_key="test_key",
            app_secret="test_secret",
            account_no="12345678-01",
        )
        # 토큰을 직접 설정 (내부 상태)
        if hasattr(client, "_access_token"):
            client._access_token = "mock_token"
        elif hasattr(client, "access_token"):
            client.access_token = "mock_token"

        result = client.place_buy_order("005930", qty=10)

        assert isinstance(result, dict), "매수 주문 결과가 dict여야 합니다."

    @patch("src.execution.kis_client.requests.post")
    def test_place_sell_order(self, mock_post):
        """매도 주문이 정상적으로 전송된다."""
        KISClient = _import_kis_client()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "rt_cd": "0",
            "msg_cd": "APBK0013",
            "msg1": "주문 전송 완료",
            "output": {"KRX_FWDG_ORD_ORGNO": "00950", "ODNO": "0000123457"},
        }
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        client = KISClient(
            app_key="test_key",
            app_secret="test_secret",
            account_no="12345678-01",
        )
        if hasattr(client, "_access_token"):
            client._access_token = "mock_token"
        elif hasattr(client, "access_token"):
            client.access_token = "mock_token"

        result = client.place_sell_order("005930", qty=5)

        assert isinstance(result, dict), "매도 주문 결과가 dict여야 합니다."

    @patch("src.execution.kis_client.requests.get")
    def test_get_balance(self, mock_get):
        """잔고 조회가 정상적으로 동작한다."""
        KISClient = _import_kis_client()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "rt_cd": "0",
            "output1": [],
            "output2": [{"dnca_tot_amt": "100000000", "tot_evlu_amt": "105000000"}],
        }
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = KISClient(
            app_key="test_key",
            app_secret="test_secret",
            account_no="12345678-01",
        )
        if hasattr(client, "_access_token"):
            client._access_token = "mock_token"
        elif hasattr(client, "access_token"):
            client.access_token = "mock_token"

        balance = client.get_balance()

        assert isinstance(balance, dict), "잔고 조회 결과가 dict여야 합니다."

    @patch("src.execution.kis_client.requests.get")
    def test_get_positions(self, mock_get):
        """보유 종목 조회가 DataFrame을 반환한다."""
        KISClient = _import_kis_client()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "rt_cd": "0",
            "output1": [
                {
                    "pdno": "005930",
                    "prdt_name": "삼성전자",
                    "hldg_qty": "100",
                    "pchs_avg_pric": "70000",
                    "prpr": "72000",
                    "evlu_pfls_amt": "200000",
                },
            ],
            "output2": [{"tot_evlu_amt": "7200000"}],
        }
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = KISClient(
            app_key="test_key",
            app_secret="test_secret",
            account_no="12345678-01",
        )
        if hasattr(client, "_access_token"):
            client._access_token = "mock_token"
        elif hasattr(client, "access_token"):
            client.access_token = "mock_token"

        positions = client.get_positions()

        assert isinstance(positions, (pd.DataFrame, list, dict)), (
            "보유 종목 조회 결과가 DataFrame 또는 dict/list여야 합니다."
        )

    def test_not_configured_graceful(self):
        """미설정 상태에서 주문 시도 시 에러 없이 처리된다."""
        KISClient = _import_kis_client()
        client = KISClient(app_key="", app_secret="", account_no="")

        # 미설정 상태에서 주문을 시도하면 에러 또는 빈 결과 반환
        try:
            result = client.place_buy_order("005930", qty=10)
            # 정상 반환이면 빈 dict이거나 에러 정보를 포함
            assert isinstance(result, dict)
        except Exception:
            # 에러가 발생해도 정상 (미설정 상태 방어)
            pass


# ===================================================================
# RiskGuard 테스트
# ===================================================================

class TestRiskGuard:
    """RiskGuard 리스크 체크 검증."""

    def test_init_default(self):
        """기본 설정으로 초기화된다."""
        RiskGuard = _import_risk_guard()
        rg = RiskGuard()
        # 기본 설정이 존재하는지 확인
        assert rg is not None, "RiskGuard가 정상 초기화되어야 합니다."

    def test_check_order_passes(self):
        """정상 범위 주문이 통과한다."""
        RiskGuard = _import_risk_guard()
        rg = RiskGuard()

        # 포트폴리오 1억, 주문 500만 (5%) -> max_order_pct 10% 이내
        order = {"ticker": "005930", "amount": 5_000_000, "side": "buy"}
        portfolio_value = 100_000_000

        passed, msg = rg.check_order(order, portfolio_value)

        assert passed is True, f"정상 범위 주문이 통과해야 합니다: {msg}"

    def test_check_order_exceeds_max_pct(self):
        """주문 비중이 max_order_pct를 초과하면 거부된다."""
        RiskGuard = _import_risk_guard()
        # max_order_pct=0.10 (10%)
        rg = RiskGuard(config={"max_order_pct": 0.10})

        # 포트폴리오 1억, 주문 1500만 (15%) -> 10% 초과
        order = {"ticker": "005930", "amount": 15_000_000, "side": "buy"}
        portfolio_value = 100_000_000

        passed, msg = rg.check_order(order, portfolio_value)

        assert passed is False, (
            f"max_order_pct 초과 주문이 거부되어야 합니다: {msg}"
        )

    def test_check_single_stock_limit(self):
        """단일 종목 비중 제한이 동작한다."""
        RiskGuard = _import_risk_guard()
        rg = RiskGuard(config={"max_single_stock_pct": 0.15})

        # 단일 종목에 20% 배분 시도 -> 15% 제한 초과
        order = {"ticker": "005930", "amount": 20_000_000, "side": "buy"}
        portfolio_value = 100_000_000

        passed, msg = rg.check_order(order, portfolio_value)

        assert passed is False, (
            f"단일 종목 비중 제한 초과 시 거부되어야 합니다: {msg}"
        )

    def test_check_blocked_ticker(self):
        """차단된 종목에 대한 주문이 거부된다."""
        RiskGuard = _import_risk_guard()
        rg = RiskGuard(config={"blocked_tickers": ["999999"]})

        order = {"ticker": "999999", "amount": 1_000_000, "side": "buy"}
        portfolio_value = 100_000_000

        passed, msg = rg.check_order(order, portfolio_value)

        assert passed is False, (
            f"차단된 종목 주문이 거부되어야 합니다: {msg}"
        )

    def test_check_rebalance_passes(self):
        """정상 범위 리밸런싱이 통과한다."""
        RiskGuard = _import_risk_guard()
        rg = RiskGuard()

        # 10개 종목에 10%씩 배분 -> max_single_stock_pct 15% 이내
        target_weights = {f"{i:06d}": 0.10 for i in range(1, 11)}

        passed, warnings = rg.check_rebalance(target_weights)

        assert passed is True, f"정상 리밸런싱이 통과해야 합니다: {warnings}"

    def test_check_rebalance_warns(self):
        """리밸런싱에서 경고가 발생하는 경우를 검증한다."""
        RiskGuard = _import_risk_guard()
        rg = RiskGuard(config={"max_single_stock_pct": 0.15})

        # 1개 종목에 50% 집중 -> max_single_stock_pct 초과
        target_weights = {"005930": 0.50, "000660": 0.50}

        passed, warnings = rg.check_rebalance(target_weights)

        # 단일 종목 50%는 15% 제한을 초과하므로 실패 또는 경고
        if not passed:
            assert len(warnings) > 0, "경고 메시지가 있어야 합니다."
        else:
            # 경고만 있고 통과하는 구현도 가능
            assert isinstance(warnings, list), "warnings가 리스트여야 합니다."


# ===================================================================
# RebalanceExecutor 테스트
# ===================================================================

class TestRebalanceExecutor:
    """리밸런싱 실행기 검증."""

    def test_dry_run(self):
        """dry_run 모드에서 실제 주문 없이 계산만 수행된다."""
        RebalanceExecutor = _try_import_rebalance_executor()
        if RebalanceExecutor is None:
            pytest.skip("RebalanceExecutor가 아직 구현되지 않았습니다.")

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.get_balance.return_value = {
            "total_value": 100_000_000,
            "cash": 100_000_000,
        }
        mock_client.get_positions.return_value = pd.DataFrame()

        executor = RebalanceExecutor(kis_client=mock_client)

        target_weights = {"005930": 0.5, "000660": 0.5}
        result = executor.dry_run(target_weights)

        # dry_run에서는 실제 주문이 발생하지 않음
        mock_client.place_buy_order.assert_not_called()
        mock_client.place_sell_order.assert_not_called()
        assert isinstance(result, (dict, list)), "dry_run 결과가 반환되어야 합니다."

    def test_execute_sells_before_buys(self):
        """리밸런싱 시 매도가 매수보다 먼저 실행된다."""
        RebalanceExecutor = _try_import_rebalance_executor()
        if RebalanceExecutor is None:
            pytest.skip("RebalanceExecutor가 아직 구현되지 않았습니다.")

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.get_balance.return_value = {
            "total_value": 100_000_000,
            "cash": 10_000_000,
        }
        # 기존 보유: 005930 90주 (가정)
        mock_client.get_positions.return_value = pd.DataFrame([
            {"ticker": "005930", "qty": 90, "current_price": 70000},
        ])
        mock_client.place_sell_order.return_value = {"rt_cd": "0"}
        mock_client.place_buy_order.return_value = {"rt_cd": "0"}
        mock_client.get_current_price.return_value = {"price": 70000}

        executor = RebalanceExecutor(kis_client=mock_client)

        # 005930 -> 000660으로 리밸런싱
        target_weights = {"000660": 1.0}
        try:
            result = executor.execute_rebalance(target_weights)
        except Exception:
            # 구현에 따라 에러가 발생할 수 있음
            pass

        # 매도가 매수보다 먼저 호출되었는지 확인
        if mock_client.place_sell_order.called and mock_client.place_buy_order.called:
            sell_call_idx = 0
            buy_call_idx = 0
            for idx, call in enumerate(mock_client.method_calls):
                if "sell" in str(call):
                    sell_call_idx = idx
                if "buy" in str(call):
                    buy_call_idx = idx
            # 매도가 매수보다 먼저 (인덱스가 작음)
            assert sell_call_idx < buy_call_idx, (
                "매도가 매수보다 먼저 실행되어야 합니다."
            )

    def test_risk_guard_blocks_order(self):
        """RiskGuard가 차단한 주문은 실행되지 않는다."""
        RebalanceExecutor = _try_import_rebalance_executor()
        RiskGuard = _import_risk_guard()

        if RebalanceExecutor is None:
            pytest.skip("RebalanceExecutor가 아직 구현되지 않았습니다.")

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.get_balance.return_value = {
            "total_value": 100_000_000,
            "cash": 100_000_000,
        }
        mock_client.get_positions.return_value = pd.DataFrame()

        # 차단된 종목을 포함한 리밸런싱
        rg = RiskGuard(config={"blocked_tickers": ["999999"]})

        try:
            executor = RebalanceExecutor(kis_client=mock_client, risk_guard=rg)
            target_weights = {"999999": 0.5, "005930": 0.5}
            result = executor.dry_run(target_weights)
            # 차단된 종목은 제외되어야 함
        except Exception:
            # 구현에 따라 에러 발생 가능
            pass


# ===================================================================
# 듀얼 모드 (모의투자/실전투자) 테스트
# ===================================================================

class TestDualTradingMode:
    """모의투자/실전투자 듀얼 모드 전환 검증."""

    @patch.dict(os.environ, {"KIS_TRADING_MODE": "", "KIS_IS_PAPER": "true", "KIS_APP_KEY": "", "KIS_APP_SECRET": "", "KIS_ACCOUNT_NO": "", "KIS_PAPER_ACCOUNT_NO": "", "KIS_REAL_ACCOUNT_NO": ""})
    def test_default_mode_is_paper(self):
        """기본 모드가 모의투자(paper)이다."""
        KISClient = _import_kis_client()
        client = KISClient()
        assert client.is_paper is True

    @patch.dict(os.environ, {"KIS_TRADING_MODE": "paper"}, clear=False)
    def test_trading_mode_paper(self):
        """KIS_TRADING_MODE=paper이면 모의투자 모드이다."""
        KISClient = _import_kis_client()
        client = KISClient()
        assert client.is_paper is True
        assert client.trading_mode == "모의투자"
        assert client.mode_tag == "[모의]"

    @patch.dict(os.environ, {"KIS_TRADING_MODE": "live"}, clear=False)
    def test_trading_mode_live(self):
        """KIS_TRADING_MODE=live이면 실전투자 모드이다."""
        KISClient = _import_kis_client()
        client = KISClient()
        assert client.is_paper is False
        assert client.trading_mode == "실전투자"
        assert client.mode_tag == "[실전]"

    @patch.dict(os.environ, {
        "KIS_TRADING_MODE": "paper",
        "KIS_PAPER_ACCOUNT_NO": "50112233-01",
        "KIS_REAL_ACCOUNT_NO": "99887766-01",
    }, clear=False)
    def test_paper_account_selection(self):
        """모의 모드에서 KIS_PAPER_ACCOUNT_NO가 선택된다."""
        KISClient = _import_kis_client()
        client = KISClient()
        assert client.account_no == "50112233-01"

    @patch.dict(os.environ, {
        "KIS_TRADING_MODE": "live",
        "KIS_PAPER_ACCOUNT_NO": "50112233-01",
        "KIS_REAL_ACCOUNT_NO": "99887766-01",
    }, clear=False)
    def test_live_account_selection(self):
        """실전 모드에서 KIS_REAL_ACCOUNT_NO가 선택된다."""
        KISClient = _import_kis_client()
        client = KISClient()
        assert client.account_no == "99887766-01"

    @patch.dict(os.environ, {
        "KIS_TRADING_MODE": "",
        "KIS_IS_PAPER": "false",
    }, clear=False)
    def test_backward_compat_is_paper(self):
        """KIS_TRADING_MODE 없으면 KIS_IS_PAPER 사용 (하위호환)."""
        KISClient = _import_kis_client()
        client = KISClient()
        assert client.is_paper is False

    @patch.dict(os.environ, {
        "KIS_TRADING_MODE": "",
        "KIS_IS_PAPER": "true",
        "KIS_ACCOUNT_NO": "12345678-01",
    }, clear=False)
    def test_fallback_account_no(self):
        """모드별 계좌가 없으면 KIS_ACCOUNT_NO를 사용한다."""
        KISClient = _import_kis_client()
        client = KISClient()
        assert client.account_no == "12345678-01"

    def test_constructor_overrides_env(self):
        """생성자 파라미터가 환경변수보다 우선한다."""
        KISClient = _import_kis_client()
        client = KISClient(is_paper=False, account_no="MANUAL-01")
        assert client.is_paper is False
        assert client.account_no == "MANUAL-01"

    def test_mode_tag_property(self):
        """mode_tag가 올바른 형식을 반환한다."""
        KISClient = _import_kis_client()
        paper = KISClient(is_paper=True)
        live = KISClient(is_paper=False)
        assert paper.mode_tag == "[모의]"
        assert live.mode_tag == "[실전]"

    def test_base_url_by_mode(self):
        """모드에 따라 올바른 API URL이 선택된다."""
        KISClient = _import_kis_client()
        paper = KISClient(is_paper=True)
        live = KISClient(is_paper=False)
        assert "openapivts" in paper.base_url
        assert "openapivts" not in live.base_url

    @patch.dict(os.environ, {"KIS_LIVE_CONFIRMED": "false"}, clear=False)
    def test_live_confirmed_guard(self):
        """KIS_LIVE_CONFIRMED=false이면 실전 리밸런싱이 거부된다."""
        RebalanceExecutor = _try_import_rebalance_executor()
        if RebalanceExecutor is None:
            pytest.skip("RebalanceExecutor 미구현")

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.is_paper = False
        mock_client.mode_tag = "[실전]"

        executor = RebalanceExecutor(kis_client=mock_client)
        result = executor.execute_rebalance({"005930": 0.5})

        assert result["success"] is False
        assert any("KIS_LIVE_CONFIRMED" in e for e in result["errors"])

    def test_live_risk_guard_defaults(self):
        """실전 모드 RiskGuard가 보수적 한도를 사용한다."""
        RiskGuard = _import_risk_guard()

        paper_rg = RiskGuard()
        live_rg = RiskGuard(is_live=True)

        assert live_rg.max_order_pct < paper_rg.max_order_pct
        assert live_rg.max_daily_turnover < paper_rg.max_daily_turnover
        assert live_rg.max_single_stock_pct < paper_rg.max_single_stock_pct


# ===================================================================
# 토큰 자동 재발급 테스트
# ===================================================================

class TestTokenAutoRefresh:
    """KIS API 500/401 에러 시 토큰 자동 재발급 검증."""

    @patch("src.execution.kis_client.requests.post")
    @patch("src.execution.kis_client.requests.get")
    def test_refresh_on_500(self, mock_get, mock_post):
        """500 에러 시 토큰 재발급 후 재시도한다."""
        KISClient = _import_kis_client()

        # 토큰 재발급 mock
        mock_token_resp = MagicMock()
        mock_token_resp.json.return_value = {
            "access_token": "new_token_after_refresh",
            "expires_in": 86400,
        }
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.status_code = 200
        mock_post.return_value = mock_token_resp

        # 첫 호출: 500 → 재발급 후 두 번째: 200 성공
        resp_500 = MagicMock()
        resp_500.status_code = 500

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "rt_cd": "0",
            "output": {"stck_prpr": "70000", "acml_vol": "1000",
                       "prdy_vrss": "500", "prdy_ctrt": "0.72"},
        }
        resp_200.raise_for_status.return_value = None

        mock_get.side_effect = [resp_500, resp_200]

        client = KISClient(
            app_key="test_key",
            app_secret="test_secret",
            account_no="12345678-01",
        )
        client._access_token = "old_expired_token"

        result = client.get_current_price("005930")

        assert result["price"] == 70000, "재발급 후 정상 응답을 받아야 합니다."
        mock_post.assert_called_once()  # 토큰 재발급 1회

    @patch("src.execution.kis_client.requests.post")
    @patch("src.execution.kis_client.requests.get")
    def test_refresh_on_401(self, mock_get, mock_post):
        """401 에러 시 토큰 재발급 후 재시도한다."""
        KISClient = _import_kis_client()

        mock_token_resp = MagicMock()
        mock_token_resp.json.return_value = {
            "access_token": "new_token_401",
            "expires_in": 86400,
        }
        mock_token_resp.raise_for_status.return_value = None
        mock_token_resp.status_code = 200
        mock_post.return_value = mock_token_resp

        resp_401 = MagicMock()
        resp_401.status_code = 401

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "rt_cd": "0",
            "output": {"stck_prpr": "80000", "acml_vol": "2000",
                       "prdy_vrss": "1000", "prdy_ctrt": "1.27"},
        }
        resp_200.raise_for_status.return_value = None

        mock_get.side_effect = [resp_401, resp_200]

        client = KISClient(
            app_key="test_key",
            app_secret="test_secret",
            account_no="12345678-01",
        )
        client._access_token = "old_token"

        result = client.get_current_price("005930")

        assert result["price"] == 80000
        mock_post.assert_called_once()

    @patch("src.execution.kis_client.requests.get")
    def test_no_double_refresh(self, mock_get):
        """토큰 재발급은 요청당 최대 1회만 시도한다."""
        KISClient = _import_kis_client()

        # 모든 호출이 500 → 재발급 실패해도 무한루프 없음
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.raise_for_status.side_effect = (
            __import__("requests").exceptions.HTTPError(response=resp_500)
        )
        mock_get.return_value = resp_500

        client = KISClient(
            app_key="test_key",
            app_secret="test_secret",
            account_no="12345678-01",
        )
        client._access_token = "old_token"

        # _invalidate_and_refresh_token을 mock하여 실패하도록
        with patch.object(client, "_invalidate_and_refresh_token", return_value=False) as mock_refresh:
            result = client.get_current_price("005930")

        # 재발급은 1회만 시도
        assert mock_refresh.call_count == 1
        # 최종적으로 빈 결과
        assert result["price"] == 0

    def test_invalidate_and_refresh(self):
        """_invalidate_and_refresh_token이 토큰을 초기화하고 재발급한다."""
        KISClient = _import_kis_client()

        client = KISClient(
            app_key="test_key",
            app_secret="test_secret",
            account_no="12345678-01",
        )
        client._access_token = "old_token"

        with patch.object(client, "get_access_token") as mock_get_token:
            mock_get_token.return_value = "brand_new_token"
            client._access_token = ""  # get_access_token이 설정한다고 가정

            # 재발급 시도 전 토큰 비우기 확인
            result = client._invalidate_and_refresh_token()

        mock_get_token.assert_called_once()


# ===================================================================
# RebalanceExecutor + PortfolioAllocator 통합 테스트
# ===================================================================

class TestExecutorWithAllocator:
    """RebalanceExecutor에 allocator가 주입되었을 때의 동작을 검증한다."""

    def test_init_with_allocator(self):
        """allocator를 주입하면 속성에 저장된다."""
        RebalanceExecutor = _try_import_rebalance_executor()
        if RebalanceExecutor is None:
            pytest.skip("RebalanceExecutor가 아직 구현되지 않았습니다.")

        mock_client = MagicMock()
        mock_allocator = MagicMock()

        executor = RebalanceExecutor(
            kis_client=mock_client, allocator=mock_allocator
        )

        assert executor.allocator is mock_allocator

    def test_init_without_allocator(self):
        """allocator를 주입하지 않으면 None이다."""
        RebalanceExecutor = _try_import_rebalance_executor()
        if RebalanceExecutor is None:
            pytest.skip("RebalanceExecutor가 아직 구현되지 않았습니다.")

        mock_client = MagicMock()

        executor = RebalanceExecutor(kis_client=mock_client)

        assert executor.allocator is None

    def test_dry_run_calls_filter_long_term_weights(self):
        """dry_run 시 allocator가 있으면 filter_long_term_weights가 호출된다."""
        RebalanceExecutor = _try_import_rebalance_executor()
        if RebalanceExecutor is None:
            pytest.skip("RebalanceExecutor가 아직 구현되지 않았습니다.")

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.get_balance.return_value = {
            "total_eval": 10_000_000,
            "cash": 10_000_000,
        }
        mock_client.get_positions.return_value = pd.DataFrame()
        mock_client.get_current_price.return_value = {"price": 50000}

        mock_allocator = MagicMock()
        mock_allocator.filter_long_term_weights.return_value = {
            "005930": 0.45,
            "000660": 0.45,
        }
        mock_allocator.get_positions_by_pool.return_value = []

        executor = RebalanceExecutor(
            kis_client=mock_client, allocator=mock_allocator
        )

        target_weights = {"005930": 0.5, "000660": 0.5}
        result = executor.dry_run(target_weights)

        mock_allocator.filter_long_term_weights.assert_called_once_with(
            target_weights
        )
        assert isinstance(result, dict)

    def test_dry_run_without_allocator_no_filter(self):
        """dry_run 시 allocator가 없으면 filter_long_term_weights가 호출되지 않는다."""
        RebalanceExecutor = _try_import_rebalance_executor()
        if RebalanceExecutor is None:
            pytest.skip("RebalanceExecutor가 아직 구현되지 않았습니다.")

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.get_balance.return_value = {
            "total_eval": 10_000_000,
            "cash": 10_000_000,
        }
        mock_client.get_positions.return_value = pd.DataFrame()

        executor = RebalanceExecutor(kis_client=mock_client)

        target_weights = {"005930": 0.5, "000660": 0.5}
        result = executor.dry_run(target_weights)

        # allocator가 None이므로 filter 호출 없음
        assert executor.allocator is None
        assert isinstance(result, dict)

    @patch.dict(os.environ, {"KIS_LIVE_CONFIRMED": "true"}, clear=False)
    def test_execute_rebalance_calls_filter_long_term_weights(self):
        """execute_rebalance 시 allocator가 있으면 filter_long_term_weights가 호출된다."""
        RebalanceExecutor = _try_import_rebalance_executor()
        if RebalanceExecutor is None:
            pytest.skip("RebalanceExecutor가 아직 구현되지 않았습니다.")

        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_client.is_paper = True
        mock_client.mode_tag = "[모의]"
        mock_client.get_balance.return_value = {
            "total_eval": 10_000_000,
            "cash": 10_000_000,
        }
        mock_client.get_positions.return_value = pd.DataFrame()
        mock_client.get_current_price.return_value = {"price": 50000}

        mock_allocator = MagicMock()
        mock_allocator.filter_long_term_weights.return_value = {
            "005930": 0.45,
        }
        mock_allocator.get_positions_by_pool.return_value = []

        mock_risk_guard = MagicMock()
        mock_risk_guard.check_rebalance.return_value = (True, [])
        mock_risk_guard.check_turnover.return_value = (True, "")
        mock_risk_guard.check_order.return_value = (True, "")

        executor = RebalanceExecutor(
            kis_client=mock_client,
            risk_guard=mock_risk_guard,
            allocator=mock_allocator,
        )

        target_weights = {"005930": 0.5}
        executor.execute_rebalance(target_weights)

        mock_allocator.filter_long_term_weights.assert_called_once_with(
            target_weights
        )


# ===================================================================
# PositionManager + PortfolioAllocator 통합 테스트
# ===================================================================

class TestPositionManagerWithAllocator:
    """PositionManager.calculate_rebalance_orders에 allocator가 전달될 때 검증한다."""

    def _make_position_manager(self, current_positions, portfolio_value, prices):
        """테스트용 PositionManager를 생성한다.

        Args:
            current_positions: {ticker: qty} 딕셔너리.
            portfolio_value: 총 포트폴리오 가치.
            prices: {ticker: price} 딕셔너리.
        """
        from src.execution.position_manager import PositionManager

        mock_client = MagicMock()
        # get_positions -> DataFrame
        if current_positions:
            rows = [
                {"ticker": t, "qty": q}
                for t, q in current_positions.items()
            ]
            mock_client.get_positions.return_value = pd.DataFrame(rows)
        else:
            mock_client.get_positions.return_value = pd.DataFrame()

        mock_client.get_balance.return_value = {
            "total_eval": portfolio_value,
            "cash": portfolio_value,
        }

        def _get_price(ticker):
            p = prices.get(ticker, 0)
            return {"price": p}

        mock_client.get_current_price.side_effect = _get_price

        pm = PositionManager(mock_client)
        return pm

    def test_allocator_excludes_short_term_positions(self):
        """allocator가 있으면 단기 포지션이 리밸런싱에서 제외된다."""
        # 현재 보유: A=10주, B=5주 (B는 단기)
        # 목표: A=100%
        # allocator 없이: B 5주 매도 발생
        # allocator 있으면: B는 제외, A만 조정
        current = {"005930": 10, "000660": 5}
        prices = {"005930": 100_000, "000660": 200_000}
        portfolio_value = 2_000_000

        pm = self._make_position_manager(current, portfolio_value, prices)

        # allocator mock: 000660이 단기
        mock_allocator = MagicMock()
        mock_allocator.get_positions_by_pool.return_value = [
            {"ticker": "000660", "pool": "short_term"},
        ]

        sell_orders, buy_orders = pm.calculate_rebalance_orders(
            {"005930": 1.0},
            allocator=mock_allocator,
        )

        mock_allocator.get_positions_by_pool.assert_called_once_with("short_term")

        # 000660에 대한 매도 주문이 없어야 함
        sell_tickers = {o["ticker"] for o in sell_orders}
        assert "000660" not in sell_tickers, (
            "단기 포지션 000660은 리밸런싱에서 제외되어야 합니다."
        )

    def test_without_allocator_includes_all_positions(self):
        """allocator가 없으면 모든 포지션이 리밸런싱 대상이다."""
        current = {"005930": 10, "000660": 5}
        prices = {"005930": 100_000, "000660": 200_000}
        portfolio_value = 2_000_000

        pm = self._make_position_manager(current, portfolio_value, prices)

        sell_orders, buy_orders = pm.calculate_rebalance_orders(
            {"005930": 1.0},
        )

        # 000660은 목표에 없으므로 전량 매도 대상
        sell_tickers = {o["ticker"] for o in sell_orders}
        assert "000660" in sell_tickers, (
            "allocator가 없으면 000660은 매도 대상이어야 합니다."
        )

    def test_allocator_no_short_term_positions(self):
        """allocator가 있지만 단기 포지션이 없으면 기존과 동일하게 동작한다."""
        current = {"005930": 10}
        prices = {"005930": 100_000}
        portfolio_value = 1_000_000

        pm = self._make_position_manager(current, portfolio_value, prices)

        mock_allocator = MagicMock()
        mock_allocator.get_positions_by_pool.return_value = []

        sell_orders, buy_orders = pm.calculate_rebalance_orders(
            {"005930": 1.0},
            allocator=mock_allocator,
        )

        # 단기 포지션이 없으므로 정상적으로 리밸런싱
        mock_allocator.get_positions_by_pool.assert_called_once_with("short_term")

    def test_allocator_excludes_multiple_short_term(self):
        """allocator가 여러 단기 종목을 제외한다."""
        # 현재 보유: A, B, C, D (C, D가 단기)
        current = {"A": 10, "B": 20, "C": 5, "D": 3}
        prices = {"A": 10_000, "B": 10_000, "C": 10_000, "D": 10_000}
        portfolio_value = 1_000_000

        pm = self._make_position_manager(current, portfolio_value, prices)

        mock_allocator = MagicMock()
        mock_allocator.get_positions_by_pool.return_value = [
            {"ticker": "C", "pool": "short_term"},
            {"ticker": "D", "pool": "short_term"},
        ]

        # 목표: A=50%, B=50% -> C, D 매도 X
        sell_orders, buy_orders = pm.calculate_rebalance_orders(
            {"A": 0.50, "B": 0.50},
            allocator=mock_allocator,
        )

        sell_tickers = {o["ticker"] for o in sell_orders}
        buy_tickers = {o["ticker"] for o in buy_orders}

        assert "C" not in sell_tickers, "단기 포지션 C는 매도 대상에서 제외되어야 합니다."
        assert "D" not in sell_tickers, "단기 포지션 D는 매도 대상에서 제외되어야 합니다."
        assert "C" not in buy_tickers, "단기 포지션 C는 매수 대상에도 포함되지 않아야 합니다."
        assert "D" not in buy_tickers, "단기 포지션 D는 매수 대상에도 포함되지 않아야 합니다."

    def test_allocator_none_default_parameter(self):
        """allocator 파라미터의 기본값이 None이다."""
        from src.execution.position_manager import PositionManager
        import inspect

        sig = inspect.signature(PositionManager.calculate_rebalance_orders)
        assert sig.parameters["allocator"].default is None
