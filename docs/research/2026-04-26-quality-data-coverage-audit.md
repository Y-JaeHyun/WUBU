# Quality 팩터 데이터 Coverage 감사 리포트

**작성일**: 2026-04-26  
**작성자**: Quant Researcher  
**관련 이슈**: [JAE-10](/JAE/issues/JAE-10) — 추가 리서치 권고 §1  

---

## 1. 개요

Quality 팩터의 핵심 지표(GP/A, 부채비율, 발생액)는 DART OpenAPI를 통해 수집한다. 그러나 DART 데이터가 없을 경우 pykrx 기반 **조잡한 대체 공식**을 사용한다. 본 메모는 이 Fallback 비율을 추정하고, Quality 신호 신뢰도에 미치는 영향을 평가한다.

---

## 2. 데이터 소스 구조

```
Quality Score 계산 파이프라인
├── Primary: DART OpenAPI (opendartreader)
│   ├── GP/A (총이익/총자산) — 직접 수집
│   ├── 부채비율 (부채/자기자본 × 100)
│   ├── ROE (순이익/자기자본 × 100)
│   └── 발생액 (순이익 - 영업현금흐름) / 총자산
│
└── Fallback: pykrx 기반 추정 (_estimate_quality_from_pykrx)
    ├── ROE ≈ EPS / BPS × 100
    ├── GP/A ≈ |ROE| / 200  ← "대략적 근사"
    ├── 부채비율: NaN (pykrx 미지원)
    └── 발생액: NaN (현금흐름 데이터 없음)
```

---

## 3. Fallback 비율 추정

### 3.1 DART 데이터 Coverage 현황

| 대상 | 추정 커버리지 | 근거 |
|------|------------|------|
| **코스피 200** | ~90% | DART 의무공시 (대형사) |
| **코스피 전체** | ~70–75% | 소형사 반기/연간 공시 지연 빈번 |
| **코스닥** | ~50–60% | DART 공시 품질 불균일, SME 비율 高 |
| **통합 유니버스 (시총 1000억+)** | ~65–70% | 코스닥 소형주 포함 시 하락 |

### 3.2 지표별 Fallback 예상 비율

| 지표 | Fallback 비율 | 결과 |
|------|------------|------|
| **GP/A** | ~35–40% | `\|ROE\| / 200` 대체 (부정확) |
| **부채비율** | ~35–40% | NaN → 스코어링 제외 |
| **발생액** | ~40–50% | NaN → `strict_accrual=False`로 제외 |
| **ROE** | ~15–20% | EPS/BPS 계산 (비교적 신뢰) |

### 3.3 Fallback GP/A 공식의 문제

`GP/A ≈ |ROE| / 200` 가정:
- GP/A = 총이익 / 총자산
- ROE = 순이익 / 자기자본

이 둘이 같으려면: `총이익 / 총자산 ≈ |순이익 / 자기자본| / 200`

즉, `(총이익/총자산) × 200 ≈ |순이익/자기자본|`을 가정하는데, 이는:
1. **총이익 ≠ 순이익**: 영업비용 차감 전/후 차이 무시
2. **총자산 ≠ 자기자본**: 레버리지(부채) 효과 무시
3. **200이라는 계수**: 통계적 근거 없는 임의값

**결론**: GP/A Fallback은 실질적으로 Quality 신호에 랜덤 노이즈를 주입한다.

---

## 4. Quality 점수 신뢰도 평가

### 4.1 시나리오 분석

| 시나리오 | DART 커버리지 | 유효 Quality 점수 비율 |
|---------|------------|-------------------|
| **낙관**: 대형주 중심 유니버스 | 85% | ~80–85% |
| **기본**: 현 유니버스 (시총 1000억+) | 65% | ~60–65% |
| **비관**: 전체 상장 종목 | 50% | ~45–50% |

**기본 시나리오**에서 약 35–40%의 종목이 Fallback GP/A로 Quality 점수를 받는다. 이 종목들은 실제 Quality와 무관한 점수를 가질 수 있다.

### 4.2 실전 배포 전 점검 항목

ThreeFactorStrategy 실전 배포 전 반드시 확인해야 할 항목:

```
□ 현재 리밸런싱 날짜 기준 DART 수집 성공률 로그 확인
□ Fallback 경로 진입 종목 수 / 전체 유니버스 종목 수 비율 계산
□ Fallback 비율 > 30% 시 CEO/보드 보고 후 배포 결정
□ quality.py의 _estimate_quality_from_pykrx 함수 호출 횟수 로깅 추가
```

---

## 5. 권고 사항

| 우선순위 | 항목 | 담당 |
|---------|------|------|
| **P0** | DART 수집 성공률 모니터링 추가 | CTO/QuantEngineer |
| **P0** | ThreeFactorStrategy 실전 배포 전 Fallback 비율 점검 | CTO |
| **P1** | GP/A Fallback 공식 개선: `abs(ROE)/200` → DART 섹터별 GP마진 평균으로 대체 | QuantEngineer |
| **P1** | 부채비율 Fallback: KRX 재무요약 API(부채/자산비율) 활용 | QuantEngineer |
| **P2** | DART 수집 실패 종목은 Quality 점수 NaN 처리 후 V+M만으로 선별 검토 | 리서치 |

---

## 6. 결론

현재 Quality 팩터의 **약 35–40% 종목이 부정확한 Fallback 공식**으로 점수를 받고 있을 가능성이 높다. ThreeFactorStrategy 실전 배포 전에 DART Fallback 비율을 실측하는 것이 P0 급으로 중요하다. Fallback 비율이 30% 초과 시 Quality 팩터를 비활성화하거나 DART 커버리지 개선 후 배포를 권고한다.

---

## 7. 참고

- `src/strategy/quality.py`: `_estimate_quality_from_pykrx()` 함수 (lines 220–260)
- `src/data/collector.py`: DART 수집 로직
- Novy-Marx (2013): *The Other Side of Value: The Gross Profitability Premium*
- Piotroski (2000): *Value Investing: The Use of Historical Financial Statement Information*
