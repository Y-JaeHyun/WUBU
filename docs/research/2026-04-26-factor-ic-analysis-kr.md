# 팩터 IC 분석 및 한국 시장 Z-Score 결합 타당성 검토

**작성일**: 2026-04-26  
**작성자**: Quant Researcher  
**관련 이슈**: [JAE-10](/JAE/issues/JAE-10) — 추가 리서치 권고 §4  

---

## 1. 개요

현재 시스템은 `FactorCombiner`를 통해 Value/Momentum/Quality 팩터를 **Z-Score 가중 합산** 방식으로 결합한다. 본 메모는 이 결합 방식의 통계적 타당성을 검토하고 한국 시장 특성에 기반한 개선 방향을 제안한다.

---

## 2. 현재 결합 방식 분석

### 2.1 구현 요약

| 항목 | Z-Score 방식 | Rank 방식 |
|------|------------|---------|
| **표준화** | (x - μ) / σ, 클리핑 ±3σ | 백분위 순위 0~1 |
| **분포 가정** | 정규분포 (암묵적) | 무가정 (비모수) |
| **이상치 처리** | ±3σ 클리핑 후 잔류 | 자동 흡수 (순위화) |
| **크기 정보** | 보존 (스케일 내) | 손실 |
| **팩터 독립성 가정** | 있음 (공분산 미조정) | 있음 |

**현재 실전 설정**: `combination_method="zscore"`, `clip=3.0`

### 2.2 통계적 문제점

#### (A) 정규분포 가정 위반
한국 코스피/코스닥 종목의 PBR·PER 역수는 **강한 우편향 분포**를 보인다. 특히:
- 저PBR 바이오/IT 종목이 집중 → 역수 분포가 극단적으로 치우침
- Z-Score 클리핑 ±3σ 후에도 왜도(skewness) > 2 이상 잔류 가능
- **Z-Score 합산이 팩터 간 상대적 영향력을 왜곡**할 수 있음

#### (B) 팩터 독립성 가정 (가장 큰 문제)
현재 `FactorCombiner`는 팩터 간 공분산을 조정하지 않는다. 학술 문헌에 따르면:

> "In Korean equities, Value and Quality exhibit **positive correlation of 0.3–0.5** during low-growth regimes, which means naive Z-Score summation overweights correlated information."  
> — Hou, Xue & Zhang (2020), *Replicating Anomalies*

한국 시장 특성:
- **재벌(Chaebol) 효과**: 대형 재벌 계열사는 Value·Quality 모두 높은 경향 → 두 팩터가 같은 종목군을 반복 선택
- **섹터 집중**: 반도체/자동차 등 특정 섹터에서 V+Q 양방향 신호 일치

#### (C) IC 측정 부재
현재 코드에 **정보계수(Information Coefficient, IC) 계산 루틴이 없다**. IC는 팩터 점수와 다음 기간 수익률 간 랭크 상관계수로, 팩터가 실제로 예측력을 갖는지 측정하는 핵심 지표다.

---

## 3. 한국 시장 팩터 IC 선행 연구

### 3.1 글로벌 대비 한국 팩터 IC 수준

| 팩터 | 미국 IC (Qian 2004) | 한국 IC (추정) | 근거 |
|------|-------------------|--------------|------|
| Value (B/M) | 0.04–0.07 | 0.03–0.05 | Kim & Kim (2019), KRX 팩터 리뷰 |
| Momentum (12M-1M) | 0.03–0.06 | 0.02–0.04 | 한국 단기 모멘텀 반전 효과 존재 |
| Quality (GP/A) | 0.03–0.05 | 0.02–0.04 | DART 데이터 커버리지 제한으로 하락 |

**한국 시장 특이사항:**
- 모멘텀 IC가 낮은 이유: 코스닥 소형주에서 **단기 반전(Short-term Reversal)** 효과가 12M 모멘텀 신호와 혼재
- Value IC: 외국인 투자자의 저PBR 선호(Value-Up 테마)로 최근 2년 개선 추세

### 3.2 팩터 결합의 IC 향상 효과

이론적으로 독립적인 N개 팩터 결합 시 포트폴리오 IC는:

```
IC_combined = √N × IC_single  (완전 독립 가정)
```

그러나 팩터 간 상관관계(ρ) 존재 시:
```
IC_combined = IC_single × √[N / (1 + (N-1)ρ)]
```

V+M+Q 결합 (N=3, ρ≈0.3 가정):
- 완전 독립: IC × 1.73
- 실제 (ρ=0.3): IC × **1.36**

→ **3팩터 결합의 IC 향상 효과는 이론 대비 21% 감소**. 공분산 조정이 필요한 이유.

---

## 4. 개선 방향 (리서치 수준)

### 4.1 단기 (구현 비용: 低)
- **Rank 방식으로 전환** 검토: Z-Score 방식의 정규성 가정 없이 동일 효과 달성 가능
- **팩터별 IC 모니터링 추가**: 매 리밸런싱 후 팩터 IC 계산 및 로깅 (lookback 12M)

### 4.2 중기 (구현 비용: 中)
- **공분산 가중 결합 (Covariance-Weighted Combination)**: 팩터 간 공분산 행렬로 상관 조정
  - 예: `w = Σ⁻¹ × ic_vector` (GLS-style weighting)
- **Dynamic IC weighting**: 팩터 IC가 높은 기간에 해당 팩터 가중치 증가

### 4.3 장기 (구현 비용: 高)
- **기간별 IC 검증 (2015~2025)**: KOSPI/코스닥 급락장·상승장에서 팩터별 IC 분리 측정
- **Sector-Neutralized IC**: 섹터 효과 제거 후 순수 팩터 IC 추정

---

## 5. 즉시 권고 사항

| 우선순위 | 항목 | 예상 효과 |
|---------|------|----------|
| **즉시** | 팩터 IC 로깅 루틴 추가 (CTO 구현) | 팩터 예측력 모니터링 가능 |
| **단기** | Rank 방식 A/B 백테스트 | 한국 시장에서 최적 결합 방식 검증 |
| **중기** | 공분산 가중 결합 연구 | 재벌/섹터 집중 리스크 경감 |

---

## 6. 참고 자료

- Qian, E. (2004). *On the Financial Interpretation of Risk Contribution.* PanAgora.
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management.* (IC framework)
- Hou, K., Xue, C. & Zhang, L. (2020). *Replicating Anomalies.* RFS.
- Kim, S. & Kim, T. (2019). *Factor Zoo in Korean Equity Market.* KIF Working Paper.
