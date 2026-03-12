# 전략 설계: NCAV (Net Current Asset Value) 딥밸류

## 개요
벤자민 그레이엄의 "net-net" 전략을 한국 시장에 적용한다. 순유동자산가치(유동자산 - 총부채)가 시가총액보다 큰 극단적 저평가 종목을 선별하여 투자한다. NCAV/시가총액 비율이 높을수록 안전마진이 크다고 판단한다.

## 매수 조건
- NCAV = 유동자산(current_assets) - 총부채(total_liabilities)
- NCAV/시가총액(market_cap) > `ncav_threshold` (기본 1.0, 보수적: 1.5)
- 시가총액 >= `min_market_cap` (기본 100억원, 극소형주 포함 위해 value 전략 대비 낮게 설정)
- 일 거래대금 >= `min_volume` (기본 5000만원, 유동성 최소 기준)
- EPS > 0 (적자 기업 제외 옵션, 기본 비활성 — 그레이엄 원전은 적자 포함)
- 조건 충족 종목 중 NCAV/MV 비율 내림차순 상위 `num_stocks`개 선택
- 동일 비중 할당

## 매도 조건
- 리밸런싱 주기(기본 quarterly)마다 전량 교체
- 리밸런싱 시 NCAV/MV < `exit_threshold` (기본 0.8) 이면 편출
- 상장폐지 위험 종목 제외 (관리종목 필터)

## 필요 데이터
- **pykrx (기존)**: 시가총액(`market_cap`), 종가(`close`), 거래량(`volume`), PBR, EPS
- **DART (확장 필요)**: 유동자산(`current_assets`), 총부채(`total_liabilities`)
  - `dart_collector.get_financial_statements()`에서 재무상태표 항목 추출
  - 연간보고서(quarter=4) 기준, 반기보고서로 업데이트 가능
- **Fallback**: DART 데이터 없을 경우 BPS 기반 근사 — NCAV ≈ BPS × 상장주식수 × 조정계수 (정확도 낮음)

## 파라미터
| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `ncav_threshold` | 1.0 | 0.5 ~ 2.0 | NCAV/MV 최소 비율. 높을수록 보수적 |
| `exit_threshold` | 0.8 | 0.5 ~ 1.0 | 매도 기준 NCAV/MV 비율 |
| `num_stocks` | 15 | 5 ~ 30 | 포트폴리오 종목 수 |
| `min_market_cap` | 10_000_000_000 | 50억 ~ 500억 | 최소 시가총액 (100억원) |
| `min_volume` | 50_000_000 | 1000만 ~ 1억 | 최소 일 거래대금 (5000만원) |
| `exclude_negative_earnings` | False | True/False | 적자 기업 제외 여부 |
| `use_bps_fallback` | True | True/False | DART 데이터 없을 때 BPS 기반 근사 사용 |

## 예상 성과 (논문/백테스트 기반)
- CAGR: ~15-25% (한국 소형주 기준, 미국 35%보다 보수적 추정)
- MDD: ~30-50% (소형주 집중 특성상 높은 편)
- Sharpe: ~0.5-0.8
- 참고: Quantpedia 기준 미국 시장 25년 백테스트 CAGR 35.3%, 다만 소형주 유동성/거래비용 미반영

## 구현 계획
- **파일**: `src/strategy/ncav.py`
- **클래스**: `NCAVStrategy(Strategy)`
- **테스트**: `tests/test_strategy_ncav.py`
- **기존 코드 재사용**:
  - `ValueStrategy._filter_universe()` 패턴 (시가총액/거래대금 필터)
  - `ValueStrategy.generate_signals()` 구조 (필터 → 스코어 → 동일비중)
  - `dart_collector.get_financial_statements()` 확장하여 유동자산/총부채 필드 추가
- **신규 구현**:
  - `_calculate_ncav()`: DART 재무데이터에서 NCAV 계산
  - `_bps_fallback_ncav()`: DART 없을 때 BPS 기반 근사
  - DART 재무제표 캐시 레이어 (분기별 갱신이므로 캐싱 효과 높음)
- **의존성**: `dart_collector.py`에 `current_assets`, `total_liabilities` 필드 추가 필요
