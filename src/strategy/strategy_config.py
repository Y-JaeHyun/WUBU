"""멀티팩터 전략 중앙 설정.

모든 운영/백테스트/시뮬레이션 코드에서 동일한 설정을 사용하도록
프로필 기반 설정 + 팩토리 함수를 제공한다.

사용법::

    from src.strategy.strategy_config import create_multi_factor

    # 운영 (num_stocks=7, market_timing=True)
    strategy = create_multi_factor("live")

    # 백테스트 (num_stocks=10, market_timing=False)
    strategy = create_multi_factor("backtest")

    # 개별 오버라이드
    strategy = create_multi_factor("backtest", num_stocks=20)

    # 설정만 조회
    from src.strategy.strategy_config import get_multi_factor_config
    config = get_multi_factor_config("live")
"""

# 모든 프로필이 공유하는 기본값
MULTI_FACTOR_BASE: dict = {
    "factors": ["value", "momentum"],
    "weights": [0.5, 0.5],
    "combine_method": "zscore",
    "turnover_penalty": 0.1,
    "max_group_weight": 0.25,
    "max_stocks_per_conglomerate": 2,
    "spike_filter": True,
    "spike_threshold_1d": 0.15,
    "spike_threshold_5d": 0.25,
    "value_trap_filter": False,
    "min_roe": 0.0,
    "min_f_score": 0,
}

# 프로필별 오버라이드 (BASE 위에 덮어씀)
MULTI_FACTOR_PROFILES: dict[str, dict] = {
    "live": {
        "num_stocks": 7,
        "apply_market_timing": True,
    },
    "backtest": {
        "num_stocks": 10,
        "apply_market_timing": False,
    },
}


def get_multi_factor_config(profile: str = "live", **overrides) -> dict:
    """프로필 기반 설정 딕셔너리를 반환한다.

    우선순위: BASE → profile 오버라이드 → 호출 시 overrides

    Args:
        profile: ``"live"`` 또는 ``"backtest"``.
        **overrides: 개별 파라미터 오버라이드.

    Returns:
        :class:`MultiFactorStrategy` 생성용 kwargs dict.
    """
    config = {**MULTI_FACTOR_BASE}
    if profile in MULTI_FACTOR_PROFILES:
        config.update(MULTI_FACTOR_PROFILES[profile])
    config.update(overrides)
    return config


def create_multi_factor(profile: str = "live", **overrides):
    """MultiFactorStrategy 인스턴스를 생성하는 팩토리 함수.

    lazy import로 순환 참조를 방지한다.

    Args:
        profile: ``"live"`` 또는 ``"backtest"``.
        **overrides: 개별 파라미터 오버라이드.

    Returns:
        :class:`MultiFactorStrategy` 인스턴스.
    """
    from src.strategy.multi_factor import MultiFactorStrategy

    config = get_multi_factor_config(profile, **overrides)
    return MultiFactorStrategy(**config)
