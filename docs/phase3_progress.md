# Phase 3 진행상황 (2026-02-18)

> ✅ Phase 3 완료

## 최종 상태: 253 passed, 0 failed, 2 skipped

### 파일 생성 완료 (23개 파일, 모두 ✅)

**Dev Team A (전략+데이터) — 완료**
| 파일 | 줄수 | 설명 |
|------|------|------|
| `src/data/dart_collector.py` | 350 | DART 재무제표 수집 (fallback: pykrx) |
| `src/data/etf_collector.py` | 155 | ETF 가격 수집 (pykrx) |
| `src/strategy/quality.py` | 338 | 퀄리티 팩터 전략 |
| `src/strategy/factor_combiner.py` | 313 | N팩터 확장 (기존 유지 + 신규 함수) |
| `src/strategy/three_factor.py` | 299 | 3팩터 통합 전략 |
| `src/strategy/dual_momentum.py` | 467 | 듀얼 모멘텀 ETF 자산배분 |
| `src/data/__init__.py` | 17 | 업데이트 |
| `src/strategy/__init__.py` | 9 | 업데이트 |

**Dev Team B (실행+스케줄러) — 완료**
| 파일 | 줄수 | 설명 |
|------|------|------|
| `src/execution/kis_client.py` | 727 | KIS OpenAPI 클라이언트 |
| `src/execution/order_manager.py` | 386 | 주문 관리 |
| `src/execution/position_manager.py` | 301 | 포지션 관리 |
| `src/execution/executor.py` | 347 | 리밸런싱 실행기 |
| `src/execution/risk_guard.py` | 285 | 리스크 가드 |
| `src/scheduler/holidays.py` | 371 | KRX 휴장일 관리 |
| `src/scheduler/main.py` | 735 | APScheduler 메인 데몬 |

**QA Team (테스트) — 완료**
| 파일 | 줄수 | 테스트수 |
|------|------|---------|
| `tests/test_quality.py` | 288 | 18 |
| `tests/test_three_factor.py` | 241 | 12 |
| `tests/test_dual_momentum.py` | 291 | 13 |
| `tests/test_n_factor_combiner.py` | 227 | 13 |
| `tests/test_execution.py` | 470 | 20 |
| `tests/test_scheduler.py` | 212 | 11 |

---

## 9개 실패 상세 분석 및 수정 방법

### 1. combine_n_factors_zscore — 3개 실패

**파일**: `tests/test_n_factor_combiner.py`
**실패 테스트**:
- `test_combine_n_factors_zscore_two_factors`
- `test_combine_n_factors_zscore_three_factors`
- `test_combine_n_factors_zscore_equal_weights`

**원인**: `src/strategy/factor_combiner.py:222`에 버그
```python
combined = combined[combined != 0.0] if len(factors) > 1 else combined
```
이 줄이 결합 결과가 0.0인 종목을 모두 제거함. 테스트에서 value=[1,2,3,4,5]와 momentum=[5,4,3,2,1]은 완벽 음의 상관관계이므로 가중합이 전부 0이 되어 모든 종목이 제거됨.

**수정**: 222줄 제거 또는 주석처리
```python
# 수정 전
combined = combined[combined != 0.0] if len(factors) > 1 else combined
# 수정 후
# (이 줄 삭제 - 0.0도 유효한 결합 스코어)
```

### 2. quality_scores_debt_inverse — 1개 실패

**파일**: `tests/test_quality.py`
**실패 테스트**: `test_quality_scores_debt_inverse`

**원인**: `calculate_quality_scores()`가 내부 필터링으로 3개 종목 중 1개를 제거 (로그: "2개 종목").
테스트 데이터의 ticker가 fundamentals DataFrame에 있는데, quality.py의 내부 필터 (시가총액, 거래대금 등)가 적용되거나, 또는 `calculate_quality_scores`가 DataFrame의 index를 ticker로 기대하는데 테스트가 "ticker" 컬럼으로 전달.

**수정 방향**: 두 가지 중 하나:
- (a) `calculate_quality_scores()`가 DataFrame을 받을 때 ticker column vs index 처리 확인 후 테스트 수정
- (b) quality.py의 내부 필터링이 테스트 데이터에 영향 주지 않도록 테스트 수정 (충분히 큰 market_cap 등)

**조사 필요**: `src/strategy/quality.py`의 `calculate_quality_scores()` 메서드의 정확한 시그니처와 내부 로직 확인

### 3. DualMomentum calculate_momentum — 3개 실패

**파일**: `tests/test_dual_momentum.py`
**실패 테스트**:
- `test_calculate_momentum` — TypeError: missing positional arg
- `test_calculate_momentum_insufficient_data` — 같은 이유
- `test_relative_signal_selects_best` — calculate_momentum 실패 때문에 연쇄

**원인**: 테스트가 `dms.calculate_momentum(prices)` 호출하지만, 구현은 `calculate_momentum(self, prices, lookback_months)` — lookback_months에 기본값이 없음.

**수정**: `src/strategy/dual_momentum.py`의 `calculate_momentum`에 기본값 추가:
```python
# 수정 전
def calculate_momentum(self, prices: dict[str, pd.Series], lookback_months: int) -> dict[str, float]:
# 수정 후
def calculate_momentum(self, prices: dict[str, pd.Series], lookback_months: int = None) -> dict[str, float]:
    if lookback_months is None:
        lookback_months = self.lookback_months
```

### 4. RebalanceExecutor — 2개 실패

**파일**: `tests/test_execution.py`
**실패 테스트**:
- `test_dry_run` — TypeError: unexpected keyword argument 'client'
- `test_execute_sells_before_buys` — 같은 이유

**원인**: 테스트가 `RebalanceExecutor(client=mock_client)` 호출하지만, 구현은 `__init__(self, kis_client, risk_guard=None)` — 파라미터명이 `kis_client`

**수정**: 테스트의 `client=` → `kis_client=` 로 변경:
```python
# tests/test_execution.py 수정
executor = RebalanceExecutor(kis_client=mock_client)
```

---

## 완료된 수정 사항

1. ✅ **factor_combiner.py:222** — `combined != 0.0` 필터 제거 (3개 테스트 수정)
2. ✅ **dual_momentum.py** — `calculate_momentum`, `get_relative_signal` 기본값 추가 (3개 테스트 수정)
3. ✅ **quality.py:232** — `scores[scores > 0]` → `scores.dropna()` (1개 테스트 수정)
4. ✅ **test_execution.py** — `client=` → `kis_client=`, `execute()` → `dry_run()`/`execute_rebalance()` (2개 테스트 수정)
5. ✅ **__init__.py** — execution, scheduler 패키지 export 추가
6. ✅ **requirements.txt** — opendartreader, websockets, jinja2 추가
7. ✅ **CLAUDE.md, planning_phase3.md** — 상태 업데이트
8. ✅ **git commit** — Phase 3 코드 커밋

---

## 미구현 항목 (Phase 3 P1/P2)

| # | 기능 | 우선순위 | 상태 |
|---|------|---------|------|
| 11 | 포지션 매니저 기능 보강 | P1 | 기본 구현됨 |
| 12 | WebSocket 실시간 시세 | P1 | 미구현 |
| 13 | KRX 휴장일 관리 | P1 | ✅ 구현됨 |
| 14 | 모닝/이브닝 리포트 자동화 | P1 | ✅ scheduler/main.py에 포함 |
| 15 | 장중 모니터링 | P1 | ✅ scheduler/main.py에 포함 |
| 16 | systemd + install script | P1 | 미구현 |
| 17~20 | P2 항목들 | P2 | 미구현 |
