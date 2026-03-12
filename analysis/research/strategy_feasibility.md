# 신규 전략 후보 실현 가능성 평가

> 평가일: 2026-03-12
> 평가 기준: 기존 데이터 소스 호환성, Strategy(ABC) 인터페이스 호환성, 추가 데이터 요구사항

## 평가 요약

| # | 전략 | 실현 가능성 | 데이터 충족도 | 인터페이스 호환 | 추가 데이터 필요 | 우선순위 |
|---|------|-----------|-------------|--------------|----------------|---------|
| 1 | NCAV 딥밸류 | **높음** | 90% | 완전 호환 | 유동자산/총부채 (DART) | **TOP 3** |
| 2 | Composite Quality Score | **높음** | 95% | 완전 호환 | 없음 (기존 데이터 활용) | **TOP 3** |
| 3 | Industry Momentum + Reversal | 중간 | 80% | 호환 (확장 필요) | KRX 업종 분류 (이미 수집 중) | 후순위 |
| 4 | ESG-Enhanced Multifactor | **낮음** | 30% | 호환 | ESG 등급 데이터 (KCGS 크롤링) | 보류 |
| 5 | Tactical AA + Regime Detection | 중간 | 70% | 호환 (ETF 전략 패턴) | VKOSPI, 한은 금리 | 후순위 |
| 6 | Size-Value Interaction | **높음** | 100% | 완전 호환 | 없음 (즉시 구현 가능) | **TOP 3** |

## 상세 평가

### 1. NCAV (Net Current Asset Value) 딥밸류

**실현 가능성: 높음**

- **데이터 호환성**: pykrx `get_all_fundamentals()`에서 BPS, EPS, PBR, 시가총액 제공. 유동자산/총부채는 DART `get_financial_statements()`에서 조회 가능 (ROE, 부채비율, 총자산 등 이미 수집 중). NCAV = 유동자산 - 총부채 계산에 필요한 개별 항목은 DART 연간보고서에서 직접 추출 가능.
- **인터페이스 호환성**: `generate_signals(date, data)` 완전 호환. ValueStrategy 패턴 그대로 사용 가능 (필터 → 스코어링 → 동일비중).
- **추가 데이터**: DART 재무제표에서 유동자산(current_assets), 총부채(total_liabilities) 항목 추출 필요. 기존 `dart_collector.py`에 필드 추가로 해결 가능.
- **리스크**: NCAV > MV 종목은 극소형주에 집중되어 유동성 리스크 존재. 거래대금 필터로 완화 가능.

### 2. Composite Quality Score (복합 품질 점수)

**실현 가능성: 높음**

- **데이터 호환성**: 4개 지표 모두 기존 데이터로 커버 가능.
  - ROE: DART `get_financial_statements()` 또는 pykrx EPS/BPS로 근사
  - 현금흐름/자산 (CFO/Assets): DART `accruals` 관련 데이터에서 역산
  - 부채비율: DART `debt_ratio` 필드
  - 발생액: 기존 `AccrualStrategy._get_accrual_scores()` 로직 재사용
- **인터페이스 호환성**: 완전 호환. 기존 QualityStrategy + AccrualStrategy의 상위호환 구조.
- **추가 데이터**: 없음. 기존 quality.py와 accrual.py 코드를 조합하여 즉시 구현 가능.
- **장점**: 기존 전략 대비 체계적인 복합 스코어링으로 하락장 방어력 개선 기대.

### 3. Industry Momentum with Reversal Hedge

**실현 가능성: 중간**

- **데이터 호환성**: 가격 데이터는 pykrx에서 완전 커버. KRX 업종 분류는 `get_all_fundamentals()`에서 `sector` 컬럼으로 이미 수집 중. 다만 업종별 수익률 집계 로직은 신규 구현 필요.
- **인터페이스 호환성**: `generate_signals()` 호환 가능하나, 업종 단위 모멘텀 계산을 위해 가격 데이터(`data['prices']`)를 적극 활용해야 함. 기존 전략들은 주로 fundamentals 기반이므로 가격 중심 로직은 패턴이 다름.
- **추가 데이터**: KRX 업종 분류 체계 매핑 (이미 sector 컬럼 존재), 업종 인덱스 가격 시계열.
- **리스크**: 한국 시장에서 업종 모멘텀 효과가 미국 대비 약할 수 있음. 논문 성과 재현 불확실.
- **보류 사유**: 구현 복잡도 대비 한국 시장 검증 부족. 기존 momentum 전략과 차별화 정도 불명확.

### 4. ESG-Enhanced Multifactor

**실현 가능성: 낮음**

- **데이터 호환성**: 핵심인 ESG 등급 데이터가 현재 데이터 파이프라인에 없음. KCGS(한국기업지배구조원) 또는 KRX ESG 포털 데이터는 공개 API가 없어 웹 크롤링 필요.
- **인터페이스 호환성**: 기존 MultiFactor 전략에 ESG 필터만 추가하면 되므로 구조적으로 호환.
- **추가 데이터**: ESG 등급 데이터 수집 파이프라인 전체 신규 구축 필요. 데이터 갱신 주기도 연 1회로 제한적.
- **보류 사유**: 데이터 수집 인프라 구축 비용이 전략 자체 구현보다 크다. ESG 데이터가 확보되면 재평가.

### 5. Tactical Asset Allocation with Regime Detection

**실현 가능성: 중간**

- **데이터 호환성**: KOSPI 지수는 `index_collector.py`로 수집 가능. 기존 `market_timing.py` 오버레이와 유사한 구조. 다만 VKOSPI, 한은 기준금리 등 추가 레짐 판별 시그널은 별도 수집 필요.
- **인터페이스 호환성**: ETF 로테이션 전략 패턴(`etf_universe` dict 활용)으로 구현 가능. `generate_signals()`에서 레짐별 자산배분 비중 반환.
- **추가 데이터**: VKOSPI(변동성지수), 한은 기준금리 시계열. VKOSPI는 KRX에서 조회 가능하나 API 추가 필요. 금리는 한은 ECOS API 연동 필요.
- **보류 사유**: 기존 `market_timing.py` + `etf_rotation.py` 조합과 기능 중복. 차별화를 위해선 레짐 감지 정교화가 필요하나, 추가 데이터 소스 구축 선행 필요.

### 6. Size-Value Interaction (소형 가치주 집중)

**실현 가능성: 높음**

- **데이터 호환성**: 필요 데이터 100% 기존 소스에서 제공.
  - 시가총액: pykrx `get_all_fundamentals()` → `market_cap` 컬럼
  - PBR: pykrx `get_all_fundamentals()` → `pbr` 컬럼
  - 거래량: pykrx `get_all_fundamentals()` → `volume` 컬럼
- **인터페이스 호환성**: ValueStrategy와 거의 동일한 구조. 시가총액 하위 필터 + PBR 정렬만 변경.
- **추가 데이터**: 없음. 즉시 구현 가능.
- **장점**: 한국 시장에서 학술적으로 검증된 팩터 조합. 기존 ValueStrategy 코드를 80% 이상 재사용 가능. 구현 난이도 최저.

## TOP 3 선정 및 구현 우선순위

1. **Size-Value Interaction** (소형 가치주 집중)
   - 이유: 추가 데이터 불필요, 즉시 구현 가능, 한국 시장 검증 완료
   - 예상 구현 시간: 2-3시간

2. **NCAV 딥밸류**
   - 이유: 클래식한 그레이엄 전략, DART 데이터 소폭 확장만 필요
   - 예상 구현 시간: 4-6시간 (DART 필드 추가 포함)

3. **Composite Quality Score**
   - 이유: 기존 quality + accrual 코드 재사용률 높음, 하락장 방어 포트폴리오로 분산 효과
   - 예상 구현 시간: 3-4시간

## 기각/보류 전략

- **Industry Momentum**: 한국 시장 검증 부족, 구현 복잡도 높음 → Phase 5+ 검토
- **ESG Multifactor**: 데이터 인프라 부재 → ESG 데이터 확보 후 재평가
- **Tactical AA + Regime**: 기존 오버레이와 중복 → 레짐 감지 고도화 시 검토
