# PR #2 종합 코드 리뷰 — Phase 7: Sharpe 개선

> 리뷰 일시: 2026-02-28
> 리뷰어: Claude(Opus 4.6) + Gemini(설계 자문) + Codex(로직 검증)
> 방법: 코드 정적 분석 + 테스트 실행 (1,657건 통과) + 금융 공식 학술 검증 + 3자 합의

---

## 리뷰 요약

| 항목 | 결과 |
|------|------|
| **테스트** | 1,657 passed / 2 skipped / 0 failed (55.56s) |
| **변경 규모** | 39 파일, +6,738 / -106 |
| **신규 모듈** | 7개 (전략 5, ML 1, 백테스트 1) |
| **신규 테스트** | 8개 파일, 2,981 LOC |
| **Feature Flags** | 7개 추가 (총 18개, 전부 기본 OFF) |
| **최종 판정** | ⛔ **보완 필요 (CRITICAL 3건 해결 후 머지 가능)** |

---

## 1. CRITICAL 이슈 (머지 차단)

> ⚠️ 이 3건은 PR에서 새로 도입된 것이 아니라 main 커밋 `9957857`에서 이미 존재하는 문제.
> main 워킹 트리에 미커밋 수정이 있으므로, PR이 이 수정을 포함(rebase)해야 함.

### C1: `lookback_days` 파라미터 부재 → 모멘텀 전략 데이터 부족

- `_fetch_price()`가 `start_date`부터만 데이터를 수집
- 모멘텀(252일), 잔차모멘텀, 52주 고점 등 전략이 첫 수개월간 시그널 생성 불가
- 백테스트 초기 구간이 현금 보유로 왜곡되어 Sharpe/CAGR 과소평가
- 테스트는 mock 데이터라 미탐지
- **수정**: main 워킹 트리의 `lookback_days=400` + `_data_start_date` 복원

### C2: 유니버스 가격 사전로딩 부재 → `data["prices"]` 비어있음

- `generate_signals()` 호출 전 `price_cache`가 비어있음
- 모멘텀/3팩터 전략의 `_filter_universe(fundamentals, prices)`가 빈 결과 반환
- Chicken-and-egg 문제 (시그널 없으면 가격 로딩 안함 → 가격 없으면 시그널 생성 불가)
- **수정**: main 워킹 트리의 `fundamentals["ticker"]` 기반 사전 로딩 복원

### C3: KOSPI 인덱스 조건부 로딩 → 잔차 모멘텀 불가

- `self.overlay is not None`일 때만 인덱스 데이터 로딩
- overlay 없이 사용 시 잔차모멘텀 계산 불가
- **수정**: main 워킹 트리의 무조건 로딩 + `reference_index` fallback 로직 복원

### 3자 합의 (Claude + Gemini + Codex)

| AI | C1~C3 판정 | 머지 차단 |
|----|-----------|---------|
| **Claude** | 확정 버그 (리그레션) | ⛔ 차단 |
| **Gemini** | P0 Must Fix — 첫 1년간 시그널 무효화 | ⛔ 차단 |
| **Codex** | 확정 버그 — `docs/fix_momentum_residual.md`와 일치 | ⛔ 차단 |

---

## 2. MAJOR 이슈 (강력 권고)

### M1: 차등 리밸런싱에서 `portfolio_value` / `current_weights` stale 문제

- 3단계 리밸런싱(sell removed → sell overweight → buy underweight)
- step 1,2 완료 후 `portfolio_value`와 `current_weights` 미재계산
- step 3의 buy 금액이 부정확
- 오차 추정 (Codex): 30% turnover → 0.45bp, 60% → 0.9bp, 100% → 1.5bp
- 정수 라운딩 + 최소주문(7만원) 결합 시 매수 누락 가능
- **권고**: Step 2 완료 후 `portfolio_value = cash + Σ(holdings[t] * price[t])` 재계산

### M2: DrawdownOverlay의 `portfolio_value_now` 계산 시 `price_cache` 불완전

- price_cache에 없는 종목의 가치 누락 → 드로다운 과대평가 가능
- C2(사전로딩) 수정 시 함께 해결될 가능성 높음

### M3: VolTargeting 첫 리밸런싱 무조건 스킵

- `_portfolio_history`가 비어있어 첫 수회 리밸런싱에서 볼타겟팅 미적용
- 의도된 동작이라면 로그 메시지 추가 권고

---

## 3. 신규 전략/모듈 리뷰 (7개) ✅ 전체 통과

| 모듈 | 판정 | 핵심 평가 |
|------|------|----------|
| `hybrid_strategy.py` | ✅ | 75/25(코어+ETF헤지) 배분 합리적. `core_weight` 범위 검증 미비(MEDIUM) |
| `low_volatility.py` | ✅ | 1/σ 스코어링 학계 표준 (Blitz & Van Vliet, 2007). 엣지케이스 양호 |
| `drawdown_overlay.py` | ✅ | DD 임계값 기관 표준. ascending 정렬+break 로직 정확 (면밀 검증 완료) |
| `vol_targeting.py` | ✅ | target/realized vol 공식 정확 (Moskowitz et al., 2012). max_exposure=1.0 적절 |
| `sector_neutral.py` | ✅ | 기본 OFF 정당 — 한국 시장 섹터 모멘텀 제거 시 수익 감소 |
| `regime_model.py` | ✅ | 2×2 레짐 규칙 기반, 과적합 방지 적절. 팩터 가중치 합리적 |
| `walk_forward.py` | ✅ | 롤링 윈도우, 자본 체이닝, OOS 지표 모두 정확 |

> **금융 공식 전부 학계 레퍼런스와 일치** (Blitz 2007, Moskowitz 2012, Novy-Marx 2016 등)

---

## 4. 기존 파일 변경 리뷰

### `three_factor.py` — N팩터 확장 ✅
- `low_vol_weight`, `sector_neutral`, `turnover_buffer`, `holding_bonus`, `regime_model` 추가
- 모두 하위호환 (기본값이 기존 동작 유지)
- **주의**: `holding_bonus`가 결합 스코어에 직접 가산 — rank 기반 결합 시 효과 과대. zscore 결합에서만 사용 권고 (MEDIUM)

### `feature_flags.py` — 7개 플래그 추가 ✅
| Flag | 기본값 | 평가 |
|------|--------|------|
| `walk_forward_backtest` | OFF | train_years:5, test_years:1, step_months:12 — 적절 |
| `low_volatility_factor` | OFF | vol_period:60, weight:0.15 — 적절 |
| `drawdown_overlay` | OFF | thresholds 3단계 + recovery_buffer:0.02 — 적절 |
| `sector_neutral` | OFF | max_sector_pct:0.25 — 역효과 확인, OFF 정당 |
| `vol_targeting` | OFF | target_vol:0.15, downside_only:True — 적절 |
| `turnover_reduction` | OFF | buffer_size:5, holding_bonus:0.1 — 보수적 |
| `regime_meta_model` | OFF | rule_based only — 적절 |

### 기타 ✅
- `dual_momentum.py` / `factor_combiner.py`: 타입 힌트 `int | None` → `Optional[int]` 변환만
- `etf_collector.py`: KODEX인버스(114800) + KOSEF국고채10년(148070) 추가, 티커 정확
- `optimization/*.py` / `execution/*.py`: `X | None` → `Optional[X]` 호환성 개선
- `sector_collector.py` (신규): pykrx WICS 섹터 분류, 방어적 import, 캐싱 미비(LOW)

---

## 5. 테스트 커버리지 — 우수 ✅

- 새 코드 ~2,500 LOC에 테스트 2,981 LOC (>1:1 비율)
- 모든 신규 모듈에 대응하는 테스트 파일 존재
- edge case 테스트 풍부 (zero peak, 경계값, 빈 데이터, flat prices 등)

### 누락 테스트 시나리오 (MEDIUM, 후속 PR 가능)
1. ETF fallback (`_fetch_price`) — `get_price_data` 빈 결과 시 `get_etf_price` 전환 미테스트
2. 3중 오버레이 순서 — MarketTiming → Drawdown → VolTargeting 통합 테스트 없음
3. `HybridStrategy` `core_weight` 범위 — `> 1.0` 또는 `< 0` 검증 없음
4. `sector_collector.py` — 테스트 파일 없음

---

## 6. 백테스트 결과 평가

### PR 자체 보고 (Walk-Forward OOS)

| 전략 | Sharpe | CAGR | MDD |
|------|--------|------|-----|
| Baseline 3F | 0.06 | 2.53% | -39.62% |
| 3F+MT | 0.09 | 3.09% | -37.48% |
| 4F+TripleOvl | 0.18 | 4.96% | -40.08% |
| **4F+Regime** | **0.28** | **7.18%** | -39.03% |

### 평가
- IS→OOS 성과 하락 정직하게 보고됨 (Sharpe 0.26 IS → 0.09 OOS for 3F+MT)
- 4F+Regime IS/OOS 역전 (IS -0.07 → OOS 0.28) — 통계적으로 불안정 가능
- **C1~C3 CRITICAL 이슈로 인해 백테스트 결과 신뢰도에 의문** — lookback 부족으로 초기 시그널 왜곡 가능
- Sector Neutral + Turnover Buffer 역효과 정직 인정 (IS Sharpe -0.07)
- 모든 신규 기능 기본 OFF — 보수적이고 올바른 접근

---

## 7. 최종 판정

### ⛔ 보완 필요 — CRITICAL 3건 해결 후 머지 가능

| 우선순위 | 항목 | 분류 |
|----------|------|------|
| **P0 필수** | C1~C3 데이터 파이프라인 복원 | 머지 전 |
| **P0 필수** | 백테스트 재실행 (수정 후 결과 갱신) | 머지 전 |
| **P1 권고** | M1 stale weights 재계산 | 머지 전 강력 권고 |
| **P2 후속** | 누락 테스트 4건 | 후속 PR 가능 |

### 권장 수정 경로
1. main에서 `engine.py` 미커밋 수정사항 커밋
2. `feature/phase7`를 main에 rebase
3. M1(stale weights) 추가 수정
4. 백테스트 재실행 후 결과 갱신

### 긍정적 평가
- **차등 리밸런싱 방향 올바름** — round-trip 거래 제거로 비용 절감
- **금융 공식 전부 정확** — 학계 레퍼런스와 일치
- **모든 신규 기능 기본 OFF** — 안전한 배포
- **테스트 커버리지 우수** (>1:1)
- **아키텍처 합리적** — 3중 오버레이 + 하이브리드 + Walk-Forward는 프로덕션 퀀트 시스템 표준 구성

---

## 부록: CRITICAL 이슈 로컬 수정 상태 (2026-02-28)

| CRITICAL | PR 브랜치 | main 커밋 | main 워킹 트리 (미커밋) |
|----------|----------|----------|----------------------|
| C1: lookback_days | ❌ 없음 | ❌ 없음 | ✅ 수정됨 (+400일 lookback) |
| C2: 유니버스 사전로딩 | ❌ 없음 | ❌ 없음 | ✅ 수정됨 (fundamentals ticker 기반) |
| C3: 인덱스 무조건 로딩 | ❌ 없음 | ❌ 없음 | ✅ 수정됨 (overlay 없어도 KOSPI 로딩) |

> 로컬 워킹 트리의 수정은 `not staged` 상태 (미커밋).
> `docs/fix_momentum_residual.md` (untracked)에 수정 내역 문서화됨.
