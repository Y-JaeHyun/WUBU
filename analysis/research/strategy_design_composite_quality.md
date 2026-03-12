# 전략 설계: Composite Quality Score (복합 품질 점수)

## 개요
ROE, 현금흐름/자산(CFO/A), 부채비율, 발생액(Accruals) 4가지 퀄리티 지표의 백분위를 합산하여 종합 품질 점수를 산출한다. 기존 QualityStrategy와 AccrualStrategy를 체계적으로 통합한 상위호환 전략으로, 각 지표의 백분위 랭킹을 정규화하여 단일 복합 스코어로 산출한다.

## 매수 조건
- 4개 퀄리티 지표의 백분위 합산 점수(0~4.0) 계산:
  1. **ROE 백분위** (높을수록 좋음): `rank(pct=True, ascending=True)`
  2. **CFO/Assets 백분위** (높을수록 좋음): 영업CF/총자산 `rank(pct=True, ascending=True)`
  3. **부채비율 백분위** (낮을수록 좋음): `rank(pct=True, ascending=False)`
  4. **발생액 백분위** (낮을수록 좋음): `rank(pct=True, ascending=False)`
- 합산 점수 상위 `top_pct`% (기본 30%) 종목 선택
- 선택된 종목 중 상위 `num_stocks`개를 동일 비중으로 편입
- 시가총액/거래대금 최소 기준 충족 필터

## 매도 조건
- 연 1회 또는 분기 1회 리밸런싱 (기본 quarterly)
- 복합 품질 점수 하위 `bottom_pct`% (기본 30%) 진입 시 편출
- 리밸런싱 시 전체 재스코어링 → 순위 변동에 따라 교체

## 필요 데이터
- **pykrx (기존)**: EPS, BPS, PBR, PER, 시가총액, 거래량, 배당수익률
- **DART (기존)**: ROE(`roe`), 부채비율(`debt_ratio`), 발생액(`accruals`), 매출총이익/총자산(`gp_over_assets`)
  - `dart_collector.get_financial_statements()` — 이미 이 필드들을 반환하고 있음
- **Fallback (DART 없을 때)**:
  - ROE 근사: EPS / BPS (pykrx)
  - 부채비율 근사: 1/PBR 기반 역산 (정확도 낮음, 경고 로그)
  - 발생액: `AccrualStrategy._get_accrual_scores()` 기존 로직 사용 (PER 역수 proxy)
  - CFO/Assets: DART 없으면 해당 팩터 가중치를 나머지에 분배

## 파라미터
| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `num_stocks` | 20 | 10 ~ 40 | 포트폴리오 종목 수 |
| `min_market_cap` | 100_000_000_000 | 500억 ~ 5000억 | 최소 시가총액 (1000억원) |
| `min_volume` | 100_000_000 | 5000만 ~ 5억 | 최소 일 거래대금 (1억원) |
| `weights` | {"roe": 0.3, "cfo": 0.25, "debt": 0.25, "accrual": 0.2} | 합계 1.0 | 팩터별 가중치 |
| `top_pct` | 0.30 | 0.10 ~ 0.50 | 상위 종목 선택 비율 |
| `exclude_negative_roe` | True | True/False | ROE 음수 기업 제외 |
| `use_dart` | True | True/False | DART 재무데이터 사용 여부 |

## 예상 성과 (논문/백테스트 기반)
- CAGR: ~12-15% (장기, 다른 팩터와 낮은 상관관계)
- MDD: ~15-25% (하락장 방어력 우수 — 퀄리티 팩터 특성)
- Sharpe: ~0.7-1.0
- 참고: NBIM Discussion Note 기준, 퀄리티 팩터는 하락장에서 시장 대비 5-10%p 초과 성과

## 구현 계획
- **파일**: `src/strategy/composite_quality.py`
- **클래스**: `CompositeQualityStrategy(Strategy)`
- **테스트**: `tests/test_strategy_composite_quality.py`
- **기존 코드 재사용**:
  - `QualityStrategy._filter_universe()` — 유니버스 필터링 로직
  - `QualityStrategy._calculate_scores()` — ROE, GP/A, 부채비율 스코어 산출
  - `AccrualStrategy._get_accrual_scores()` — 발생액 스코어 산출
  - `dart_collector.get_financial_statements()` — DART 재무데이터 조회
  - `DEFAULT_QUALITY_WEIGHTS` — 기본 가중치 참조
- **신규 구현**:
  - `_calculate_composite_score()`: 4개 지표 백분위 합산 → 단일 복합 스코어
  - `_normalize_percentile()`: 각 지표를 0~1 백분위로 정규화
  - `_redistribute_weights()`: DART 데이터 부재 시 가중치 재분배 로직
- **기존 전략과의 차별점**:
  - QualityStrategy: 가중 합산이지만 발생액을 PER 역수로 근사
  - AccrualStrategy: 발생액 단일 팩터
  - CompositeQuality: 4개 팩터를 백분위 기반으로 정규화 후 체계적 합산, DART 직접 데이터 우선 사용
