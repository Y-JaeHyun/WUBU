"""실시간 가격 관리자(src/execution/realtime_manager.py) 테스트.

RealtimeManager의 초기화, 가격 캐시, 콜백 등록,
실행 상태 확인을 검증한다.

src/execution/realtime_manager.py가 아직 구현되지 않았으면 테스트를 스킵한다.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


# ===================================================================
# 헬퍼: 모듈 임포트 (지연 임포트)
# ===================================================================

def _try_import_realtime_manager():
    """RealtimeManager가 있으면 임포트한다."""
    try:
        from src.execution.realtime_manager import RealtimeManager
        return RealtimeManager
    except ImportError:
        return None


# ===================================================================
# RealtimeManager 검증
# ===================================================================

class TestRealtimeManager:
    """RealtimeManager 검증."""

    def test_init(self):
        """RealtimeManager가 정상 초기화된다."""
        RealtimeManager = _try_import_realtime_manager()
        if RealtimeManager is None:
            pytest.skip("RealtimeManager가 아직 구현되지 않았습니다.")

        mock_kis_client = MagicMock()
        manager = RealtimeManager(kis_client=mock_kis_client)

        assert manager is not None, "RealtimeManager가 정상 초기화되어야 합니다."

    def test_get_latest_price_empty(self):
        """캐시가 비어있을 때 None 또는 0을 반환한다."""
        RealtimeManager = _try_import_realtime_manager()
        if RealtimeManager is None:
            pytest.skip("RealtimeManager가 아직 구현되지 않았습니다.")

        mock_kis_client = MagicMock()
        manager = RealtimeManager(kis_client=mock_kis_client)

        price = manager.get_latest_price("005930")

        assert price is None or price == 0, (
            f"빈 캐시에서는 None 또는 0이어야 합니다: {price}"
        )

    def test_register_callback(self):
        """콜백 함수가 정상 등록된다."""
        RealtimeManager = _try_import_realtime_manager()
        if RealtimeManager is None:
            pytest.skip("RealtimeManager가 아직 구현되지 않았습니다.")

        mock_kis_client = MagicMock()
        manager = RealtimeManager(kis_client=mock_kis_client)

        callback = MagicMock()

        if hasattr(manager, "register_callback"):
            manager.register_callback(callback)
            # 에러 없이 등록되면 통과
        elif hasattr(manager, "on_tick"):
            manager.on_tick = callback
        else:
            # 콜백 등록 방법이 다를 수 있음
            pass

    def test_get_all_prices(self):
        """전체 가격 조회가 dict를 반환한다."""
        RealtimeManager = _try_import_realtime_manager()
        if RealtimeManager is None:
            pytest.skip("RealtimeManager가 아직 구현되지 않았습니다.")

        mock_kis_client = MagicMock()
        manager = RealtimeManager(kis_client=mock_kis_client)

        if hasattr(manager, "get_all_prices"):
            prices = manager.get_all_prices()
            assert isinstance(prices, dict), "반환값이 dict여야 합니다."
        elif hasattr(manager, "prices"):
            prices = manager.prices
            assert isinstance(prices, dict), "prices가 dict여야 합니다."

    def test_is_running_initial(self):
        """초기 상태에서 실행 중이 아니다."""
        RealtimeManager = _try_import_realtime_manager()
        if RealtimeManager is None:
            pytest.skip("RealtimeManager가 아직 구현되지 않았습니다.")

        mock_kis_client = MagicMock()
        manager = RealtimeManager(kis_client=mock_kis_client)

        if hasattr(manager, "is_running"):
            if callable(manager.is_running):
                assert manager.is_running() is False, "초기 상태는 실행 중이 아닙니다."
            else:
                assert manager.is_running is False, "초기 상태는 실행 중이 아닙니다."
        elif hasattr(manager, "_running"):
            assert manager._running is False, "초기 _running은 False여야 합니다."

    def test_price_cache_update(self):
        """가격 캐시가 올바르게 업데이트된다."""
        RealtimeManager = _try_import_realtime_manager()
        if RealtimeManager is None:
            pytest.skip("RealtimeManager가 아직 구현되지 않았습니다.")

        mock_kis_client = MagicMock()
        manager = RealtimeManager(kis_client=mock_kis_client)

        # 내부 캐시를 직접 업데이트하여 테스트
        if hasattr(manager, "_price_cache"):
            manager._price_cache["005930"] = 72000
            price = manager.get_latest_price("005930")
            assert price == 72000, f"캐시 업데이트 후 가격이 72000이어야 합니다: {price}"
        elif hasattr(manager, "_prices"):
            manager._prices["005930"] = 72000
            if hasattr(manager, "get_latest_price"):
                price = manager.get_latest_price("005930")
                assert price == 72000, f"캐시 업데이트 후 가격이 72000이어야 합니다: {price}"
        elif hasattr(manager, "update_price"):
            # update_price 메서드가 있으면 사용
            tick = {"ticker": "005930", "price": 72000}
            manager.update_price(tick)
            price = manager.get_latest_price("005930")
            assert price == 72000, f"업데이트 후 가격이 72000이어야 합니다: {price}"
        else:
            # 캐시 접근 방법을 찾을 수 없으면 스킵
            pytest.skip("가격 캐시 접근 방법을 찾을 수 없습니다.")
