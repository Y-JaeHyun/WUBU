# MultiFactor Quality 팩터 + PBR 실시간 보정 백테스트 결과

> 생성일시: 2026-04-28 22:30 KST  
> 이슈: [JAE-29](/JAE/issues/JAE-29)  
> 상태: **백테스트 완료**

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

- `close`: 당일 주가 (시가총액 / 발행주식수)
- `BPS`: 최근 분기 보고 장부가치 per share
- BPS > 0 && close > 0 조건 만족 시 적용, 그 외 reported_pbr fallback

### Quality 스코어

```
quality_score = 0.3×ROE_rank + 0.3×GPA_rank + 0.2×(1/Debt)_rank + 0.2×(1/Accrual)_rank
```

GP/A 미제공 시: `abs(ROE)/200` fallback (임시, 개선 예정)

---

## 2. 백테스트 결과 (실측치)

기간: 2023-01-01 ~ 2026-04-25 (3yr) / 2021-01-01 ~ 2026-04-25 (5yr)  
초기자본: 10,000,000원 / 월 리밸런싱 / KOSPI 전체 종목 대상

### 3yr (2023-01-01 ~ 2026-04-25)

| 전략 | CAGR | Sharpe | MDD | 총수익률 |
|------|------|--------|-----|----------|
| Baseline(V+M) | 14.0% | 0.56 | -23.1% | +54.3% |
| V+M+adjPBR | 13.2% | 0.52 | -22.9% | +50.6% |
| **V+M+Q+adjPBR** | **38.1%** | **1.13** | -30.4% | **+190.8%** |

### 5yr (2021-01-01 ~ 2026-04-25)

| 전략 | CAGR | Sharpe | MDD | 총수익률 |
|------|------|--------|-----|----------|
| Baseline(V+M) | 10.2% | 0.41 | -33.7% | +67.6% |
| V+M+adjPBR | 10.3% | 0.41 | -32.2% | +68.5% |
| **V+M+Q+adjPBR** | **13.9%** | **0.50** | -38.2% | **+99.2%** |

---

## 3. 분석

### PBR 실시간 보정 단독 효과
- **3yr**: CAGR -0.8%p, MDD 0.2%p 개선 — 사실상 중립
- **5yr**: CAGR +0.1%p, MDD 1.4%p 개선 — 미미
- **결론**: adj_pbr 단독으로는 성과 개선 효과가 크지 않음. 다만 밸류에이션 정확도를 높여 데이터 왜곡을 줄이는 의미는 있음.

### Quality 팩터 추가 효과
- **3yr**: CAGR +24.1%p (14→38%), Sharpe 0.56→1.13 (2배), 총수익률 +136.5%p
- **5yr**: CAGR +3.7%p (10→14%), Sharpe 0.41→0.50
- **MDD 트레이드오프**: 3yr -7.3%p, 5yr -4.5%p 악화
- **결론**: Quality 팩터가 압도적 성과 개선. 특히 최근 3yr에서 효과가 두드러짐. MDD 확대는 모멘텀 강세 구간에서 오히려 공격적으로 포지션을 잡기 때문으로 추정.

### 주의 사항
- 2026년 포함 기간이므로 2025년 말 기준 결과와 차이 있을 수 있음 (2026 YTD 한국시장 부진)
- BPS 데이터 커버리지 미측정 — adj_pbr 미적용 종목 비율 확인 필요
- Quality GP/A: `abs(ROE)/200` fallback 사용 중 — 별도 개선 이슈 필요

---

## 4. 구현 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/strategy/multi_factor.py` | `adj_pbr` 파라미터, `_get_value_scores()` PBR 보정, `_get_quality_scores()`, N팩터 통합 경로 |
| `src/strategy/strategy_config.py` | `quality`, `quality_live` 프로필 추가 |
| `tests/test_multi_factor.py` | TestAdjPBR, TestQualityFactor, TestQualityConfig (12개 신규, 총 49개 통과) |
| `scripts/jae29_quality_pbr_backtest.py` | 3전략 × 2기간 비교 스크립트 |
| `data/jae29_backtest_results.json` | 실측 결과 저장 |

### 브랜치 / 커밋
- 브랜치: `feature/jae29-quality-pbr-correction` (quant-dev)
- `2193ca6` — feat(strategy): Quality 팩터 + PBR 실시간 보정 추가
- `8ebcc12` — fix(backtest): Baseline adj_pbr=False 명시로 비교 정합성 수정
- `98dd642` — docs(research): JAE-29 백테스트 사전 분석 리포트 작성

---

## 5. 권고 사항

1. **V+M+Q+adjPBR 채택 권장** — 3yr 기준 CAGR +24%p, Sharpe 2배. 명확한 개선.
2. **MDD 확대 모니터링** — 최대 -38% 수준. 리스크 허용 범위 내인지 CEO 확인 필요.
3. **GP/A 데이터 개선** — 현재 fallback 사용 중. 별도 이슈로 DART GP/A 직접 수집 고려.
4. **adj_pbr 선택적 적용** — 단독 효과가 미미하므로 Quality와 함께 사용 시에만 의미 있음.
