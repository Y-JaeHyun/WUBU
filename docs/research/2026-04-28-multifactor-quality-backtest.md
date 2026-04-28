# MultiFactor Quality 팩터 + PBR 실시간 보정 백테스트 결과

> 생성일시: 2026-04-28 (KST)  
> 이슈: [JAE-29](/JAE/issues/JAE-29)  
> 상태: **전체 백테스트 실행 중** — 아래 수치는 사전 분석 및 단기 검증 결과

---

## 1. 개요

MultiFactor 전략에 두 가지 개선을 추가하고 성과를 비교한다.

| 전략 | 설명 |
|------|------|
| **Baseline(V+M)** | Value+Momentum (0.5/0.5), adj_pbr=False, 20종목 |
| **V+M+adjPBR** | Value+Momentum (0.5/0.5), adj_pbr=True (close/BPS), 20종목 |
| **V+M+Q+adjPBR** | Value+Momentum+Quality (0.35/0.35/0.30), adj_pbr=True, 20종목 |

### PBR 실시간 보정 공식

```
adj_pbr = close / BPS
```

- `close`: 당일 시가총액 / 발행주식수 (현재 주가)
- `BPS`: 최근 분기 보고 장부가치
- `reported_pbr`: 기존 pykrx 제공 PBR (보고서 기준, 분기 lag 있음)

BPS > 0 && close > 0 조건 만족 시에만 적용; 그 외는 reported_pbr fallback.

### Quality 스코어

```
quality_score = 0.3×ROE_rank + 0.3×GPA_rank + 0.2×(1/Debt)_rank + 0.2×(1/Accrual)_rank
```

GP/A 미제공 시: `abs(ROE)/200` fallback (임시, 개선 예정)

---

## 2. 검증 결과 (2025-01-01 ~ 2025-02-28, V+M+adjPBR, 20종목→5종목)

단기 엔진 정상동작 확인용 빠른 검증. 실전 적용 파라미터와 동일한 adj_pbr 로직 사용.

| 지표 | 값 |
|------|-----|
| 기간 | 2025-01-01 ~ 2025-02-28 (2개월) |
| 초기자본 | 10,000,000원 |
| 최종가치 | 10,988,683원 |
| 총수익률 | **+9.89%** (2개월) |
| CAGR (연환산) | **81.07%** |
| Sharpe | **3.57** |
| MDD | **-3.46%** |
| 리밸런싱 | 2회 |
| 총거래 | 5건 |

> 주의: 2개월 단기 결과로 연환산 수치가 과장될 수 있음. 전체 5yr/3yr 결과 참조 필요.

---

## 3. 참조 기준선 (기존 all_backtest_results.json, 2023-01-01~2025-12-31)

JAE-29 전략은 multi_factor.py 기반이며 아래와 비교 참고한다.
(three_factor.py는 별도 구현으로 직접 비교 불가)

| 전략 | CAGR | Sharpe | MDD | 비고 |
|------|------|--------|-----|------|
| MultiFactor(V+M) | 71.0% | 2.05 | -17.8% | 10종목, adj_pbr=False (구버전) |
| ThreeFactor(V+M+Q) | 52.9% | 1.55 | -29.5% | 10종목, three_factor.py 기반 |
| RiskParity(MF) | 52.3% | 1.84 | -17.1% | 참고 |

> ThreeFactor가 V+M 대비 CAGR 낮은 원인: 섹터/재벌 분산 제약 + 레짐 메타모델 비용.
> JAE-29의 Quality 통합 방식(multi_factor.py)은 이보다 단순 — 제약 없이 Quality 가중치만 추가.

---

## 4. 전체 백테스트 결과 (실행 중)

> **현재 실행 중** — 완료 후 아래 표를 실수치로 업데이트 예정.

### 5yr 결과 (2021-01-01 ~ 2026-04-25)

| 전략 | CAGR | Sharpe | MDD | 총수익률 |
|------|------|--------|-----|----------|
| Baseline(V+M) | — | — | — | — |
| V+M+adjPBR | — | — | — | — |
| V+M+Q+adjPBR | — | — | — | — |

### 3yr 결과 (2023-01-01 ~ 2026-04-25)

| 전략 | CAGR | Sharpe | MDD | 총수익률 |
|------|------|--------|-----|----------|
| Baseline(V+M) | — | — | — | — |
| V+M+adjPBR | — | — | — | — |
| V+M+Q+adjPBR | — | — | — | — |

---

## 5. 분석 및 기대 효과

### PBR 실시간 보정 효과
- 분기 보고 PBR → 주가 반영 실시간 PBR로 밸류에이션 정확도 향상
- 주가 급등 종목의 PBR 저평가 오류 방지 (보고 시점과 현재 주가 괴리 보정)
- BPS 데이터 미제공 종목: 보정 미적용 (reported_pbr fallback) — 커버리지 모니터링 필요

### Quality 팩터 추가 효과
- 저PBR + 저Quality 종목 (밸류 트랩) 필터링
- ROE/GP/A: 수익성 검증, 부채비율: 재무건전성, 발생액: 이익품질
- 예상 트레이드오프: CAGR 소폭 감소, MDD 개선 (방어적 필터)
- 실제 효과는 전체 백테스트 결과 확인 필요

---

## 6. 구현 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/strategy/multi_factor.py` | `adj_pbr` 파라미터, `_get_value_scores()` PBR 보정, Quality 팩터 통합 |
| `src/strategy/strategy_config.py` | `quality`, `quality_live` 프로필 추가 |
| `tests/test_multi_factor.py` | TestAdjPBR, TestQualityFactor, TestQualityConfig (12개 신규) |
| `scripts/jae29_quality_pbr_backtest.py` | 3전략 × 2기간 비교 스크립트 |

### 테스트 결과
```
49 passed, 1 warning in 4.00s
```

### 브랜치 / 커밋
- 브랜치: `feature/jae29-quality-pbr-correction`
- 커밋: `2193ca6` — `feat(strategy): Quality 팩터 + PBR 실시간 보정 추가`
- 커밋: `8ebcc12` — `fix(backtest): Baseline adj_pbr=False 명시로 비교 정합성 수정`

---

## 7. 주의사항 및 후속 과제

| 항목 | 설명 |
|------|------|
| BPS 커버리지 | pykrx BPS 미제공 종목 비율 측정 필요 (>30% 시 경고) |
| Quality GP/A Fallback | `abs(ROE)/200` 임의 계수 — 별도 이슈로 개선 예정 |
| DART 부채비율 누락 | DART 부채비율 자주 누락 → Quality 스코어 불완전 가능 |
| 전체 백테스트 완료 | 결과 나오면 이 문서 업데이트 |
