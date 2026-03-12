# 전략 설계: Size-Value Interaction (소형 가치주 집중)

## 개요
한국 시장에서 사이즈 팩터(소형주 프리미엄)가 가장 큰 수익률 프리미엄을 보이는 점을 활용한다. 시가총액 하위 종목 중 PBR이 낮은 가치주에 집중 투자하여 사이즈×밸류 인터랙션 효과를 극대화한다. Kim et al. (Pacific-Basin Finance Journal) 논문에서 한국 시장 사이즈 팩터 프리미엄이 연 8-12%로 검증되었다.

## 매수 조건
- 시가총액 하위 `size_pct`% (기본 30%) 종목 필터
- 해당 종목 중 PBR 하위 `value_pct`% (기본 20%) 선택
- 일 거래대금 >= `min_volume` (기본 5000만원, 소형주 유동성 확보)
- PBR > 0 (자본잠식 제외)
- EPS > 0 (적자 기업 제외, 옵션)
- 선택 종목을 동일 비중으로 할당
- 종목 수 상한: `max_stocks` (기본 20개)

## 매도 조건
- 분기 리밸런싱 (기본 quarterly)
- 시가총액이 전체 시장의 상위 `exit_size_pct`% (기본 상위 30%) 진입 시 매도 (소형주 탈출)
- PBR이 `exit_pbr_threshold` (기본 1.5) 초과 시 매도 (밸류 탈출)
- 리밸런싱 시 조건 재평가 → 조건 미충족 종목 교체

## 필요 데이터
- **pykrx (기존, 100% 커버)**:
  - `market_cap`: 시가총액 — 사이즈 팩터 필터링
  - `pbr`: PBR — 밸류 팩터 스코어링
  - `volume`, `close`: 거래대금 계산
  - `eps`: 적자 기업 필터 (옵션)
- **추가 데이터**: 없음. 즉시 구현 가능.

## 파라미터
| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `size_pct` | 0.30 | 0.10 ~ 0.50 | 시가총액 하위 비율 (소형주 기준) |
| `value_pct` | 0.20 | 0.10 ~ 0.40 | PBR 하위 비율 (가치주 기준) |
| `max_stocks` | 20 | 10 ~ 30 | 최대 포트폴리오 종목 수 |
| `min_volume` | 50_000_000 | 1000만 ~ 1억 | 최소 일 거래대금 (5000만원) |
| `exit_size_pct` | 0.30 | 0.20 ~ 0.50 | 매도 시가총액 상위 기준 |
| `exit_pbr_threshold` | 1.5 | 1.0 ~ 2.0 | 매도 PBR 기준 |
| `exclude_negative_earnings` | True | True/False | 적자 기업 제외 여부 |
| `value_factor` | "pbr" | "pbr", "per", "composite" | 밸류 팩터 선택 |

## 예상 성과 (논문/백테스트 기반)
- CAGR: ~15-20% (한국 시장 소형 가치주 장기 프리미엄 기반)
- MDD: ~30-45% (소형주 집중 특성상 시장 하락기에 취약)
- Sharpe: ~0.6-0.9
- 참고: Kim et al. 논문 기준 한국 시장 사이즈 팩터 프리미엄 연 8-12%, 밸류 프리미엄 연 5-8%. 인터랙션 효과로 합산 이상 기대.

## 구현 계획
- **파일**: `src/strategy/size_value.py`
- **클래스**: `SizeValueStrategy(Strategy)`
- **테스트**: `tests/test_strategy_size_value.py`
- **기존 코드 재사용**:
  - `ValueStrategy._filter_universe()` — 거래대금/기본 필터 로직 (min_market_cap 파라미터만 변경)
  - `ValueStrategy._rank_stocks()` — PBR 기준 정렬 로직
  - `ValueStrategy.generate_signals()` — 전체 구조 (필터 → 랭킹 → 동일비중)
  - 실질적으로 ValueStrategy에 사이즈 필터를 추가한 변형
- **신규 구현**:
  - `_apply_size_filter()`: 시가총액 하위 N% 필터 (백분위 기반)
  - `_apply_value_filter()`: 사이즈 필터 통과 종목 중 PBR 하위 N% 선택
  - `_check_exit_conditions()`: 매도 조건 체크 (시총 상승, PBR 상승)
- **구현 난이도**: 최저 — ValueStrategy 코드의 80% 이상 재사용 가능. 핵심 차이는 `_filter_universe()`에서 시가총액 상한을 두어 소형주만 선택하는 것.
- **주의사항**:
  - 소형주 유동성 리스크: 거래대금 필터를 반드시 적용
  - 생존자 편향: 백테스트 시 상폐 종목 누락 가능성 인지
  - 슬리피지: 소형주 특성상 실제 매매 시 추가 비용 발생 가능
