# 전략 배치 백테스트 비교 리포트

> 생성일: 2026-03-12
> 기간: 2022-03-12 ~ 2026-03-12 (4년)
> 초기자본: 1,500,000원
> 리밸런싱: 월간

## 전략 성과 순위표 (CAGR 기준)

| 순위 | 전략 | 옵션 | 그룹 | CAGR | MDD | Sharpe | 승률 | 거래수 | 최종가치 |
|------|------|------|------|------|-----|--------|------|--------|----------|
| 1 | enhanced_etf_rotation | default | B | 34.9% | -19.8% | 1.44 | 60% | 153 | 4,952,328원 |
| 2 | etf_rotation | default | B | 30.8% | -19.5% | 1.22 | 65% | 86 | 4,377,566원 |
| 3 | cross_asset_momentum | default | C | 22.8% | -16.6% | 1.12 | 58% | 112 | 3,406,791원 |
| 4 | three_factor | default | A | 13.3% | -29.5% | 0.53 | 62% | 612 | 2,469,871원 |
| 5 | value | default | A | 8.4% | -20.1% | 0.36 | 50% | 496 | 2,069,134원 |
| 6 | pead | default | C | 8.3% | -35.5% | 0.31 | 54% | 354 | 2,066,213원 |
| 7 | accrual | default | C | 8.2% | -36.0% | 0.31 | 54% | 355 | 2,053,052원 |
| 8 | three_factor | top10 | A | 6.4% | -43.0% | 0.24 | 58% | 367 | 1,919,760원 |
| 9 | multi_factor | backtest | A | 6.2% | -28.2% | 0.24 | 58% | 344 | 1,905,178원 |
| 10 | value | top10 | A | 2.9% | -30.1% | 0.08 | 46% | 300 | 1,681,801원 |
| 11 | shareholder_yield | default | C | 2.6% | -29.8% | 0.06 | 42% | 257 | 1,659,338원 |
| 12 | quality | default | A | 1.2% | -47.1% | 0.04 | 54% | 402 | 1,576,387원 |
| 13 | multi_factor ★ | live | A | 0.1% | -42.3% | -0.01 | 40% | 131 | 1,509,193원 |
| 14 | bb_squeeze | default | D | 0.0% | 0.0% | 0.00 | 0% | 0 | 1,500,000원 |
| 15 | dual_momentum | default | B | 0.0% | 0.0% | 0.00 | 0% | 0 | 1,500,000원 |
| 16 | high_breakout | default | D | 0.0% | 0.0% | 0.00 | 0% | 0 | 1,500,000원 |
| 17 | orb_daytrading | default | D | 0.0% | 0.0% | 0.00 | 0% | 0 | 1,500,000원 |
| 18 | swing_reversion | default | D | 0.0% | 0.0% | 0.00 | 0% | 0 | 1,500,000원 |
| 19 | low_volatility | default | C | -0.1% | -14.2% | -0.42 | 50% | 519 | 1,493,372원 |
| 20 | low_vol_quality | default | C | -0.7% | -25.1% | -0.13 | 52% | 335 | 1,458,863원 |
| 21 | momentum | top10 | A | -5.6% | -59.9% | -0.08 | 50% | 395 | 1,193,530원 |
| 22 | momentum | default | A | -8.3% | -62.2% | -0.30 | 54% | 509 | 1,062,847원 |

> ★ = 운영환경(live) 설정

## 그룹별 요약

### 그룹 A: 기본 팩터
- 전략 수: 9개
- 최고 Sharpe: three_factor_default (Sharpe=0.53, CAGR=13.3%)
- 평균 CAGR: 2.7%

### 그룹 B: ETF/자산배분
- 전략 수: 3개
- 최고 Sharpe: enhanced_etf_rotation_default (Sharpe=1.44, CAGR=34.9%)
- 평균 CAGR: 21.9%

### 그룹 C: 고급 팩터
- 전략 수: 6개
- 최고 Sharpe: cross_asset_momentum_default (Sharpe=1.12, CAGR=22.8%)
- 평균 CAGR: 6.8%

### 그룹 D: 단기 전략
- 전략 수: 4개
- 최고 Sharpe: bb_squeeze_default (Sharpe=0.00, CAGR=0.0%)
- 평균 CAGR: 0.0%

## 실패한 전략

- **risk_parity_default**: covariance 계산 오류
- **hybrid_strategy_default**: core_strategy 필수 인자 누락 (조합 전략)
- **ml_factor_default**: ml_pipeline 필수 인자 누락 (ML 파이프라인 필요)

## 운영환경 설정 비교

| 항목 | Live (★) | Backtest | Δ |
|------|----------|----------|---|
| CAGR | 0.1% | 6.2% | +6.0% |
| MDD | -42.3% | -28.2% | +14.1% |
| Sharpe | -0.01 | 0.24 | +0.25 |
| 거래수 | 131 | 344 | +213 |

**분석**: live 설정(7종목, market_timing=True)이 backtest 설정(10종목, market_timing=False) 대비 CAGR이 낮고 MDD가 높다. market_timing 오버레이가 현 장세에서 비효율적이거나, 종목 수 축소로 분산 효과가 감소한 것으로 판단.

## 핵심 인사이트

1. **ETF 로테이션 전략이 압도적 우위**: enhanced_etf_rotation(CAGR 34.9%, Sharpe 1.44)과 etf_rotation(30.8%, 1.22)이 최상위 성과
2. **자산배분 > 종목선정**: cross_asset_momentum(22.8%)이 개별종목 전략 대부분을 상회
3. **밸류 팩터 유효**: value_default(8.4%), three_factor(13.3%)가 안정적 성과
4. **모멘텀 단독 전략 부진**: momentum이 -8.3%~-5.6%로 한국 시장에서 역효과
5. **단기 전략 미작동**: Group D 전략들이 모두 거래 0건 (ShortTermBacktest 엔진 호환 문제)
6. **운영 설정 개선 필요**: multi_factor_live(0.1%) vs backtest(6.2%) 격차가 크며, market_timing 효과 재검토 필요
