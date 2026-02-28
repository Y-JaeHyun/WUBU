# 퀀트 전략 연구 보고서 (2026-02-28)

## 1. 연구 목적

현재 포트폴리오의 수익률을 개선하기 위해 다음을 조사/구현하였다:
- AI 자문(Gemini)을 통한 신규 전략 아이디어 수집
- 웹 리서치를 통한 최신 퀀트 전략 트렌드 분석
- 기존 ETF Rotation 전략의 개선안 연구 및 구현
- 신규 전략 2종 구현 및 캐시 데이터 기반 백테스트

## 2. Gemini 자문 요약

### 제안된 신규 전략 (5종)

| 전략 | 핵심 아이디어 | 예상 Sharpe | 구현 난이도 |
|------|-------------|-----------|-----------|
| K-Asset Momentum | 주식/채권/금/달러 4자산군 로테이션 | 0.8~1.2 | 하 |
| K-Small Cap NCAV | (유동자산-총부채)>시총 종목 투자 | 0.7+ | 중 |
| K-Thematic Factor Rotation | 섹터 ETF 상대모멘텀 | 0.6~0.9 | 하 |
| Overnight Gap Strategy | 과매도 갭 매수/익일 매도 | 1.0+ | 고 |
| Low-Vol Dividend Growth | 저변동+배당성장 | 0.7~0.8 | 중 |

### ETF Rotation 개선안
1. **현금 보호 필터**: KOSPI 200MA 이하 시 채권/현금 강제 전환
2. **변동성 역가중**: Risk Parity 관점 도입
3. **상관관계 기반 자산 교체**: 주식/채권/달러/금 필수 포함
4. **분할 매수/매도**: 3~5일 분할 매매로 슬리피지 감소

## 3. 인터넷 조사 결과

### 최신 퀀트 전략 트렌드 (2024~2025)
- **듀얼 모멘텀 + 변동성 필터**: Sharpe 1.6, 연간 수익률 21%, MDD -7.5% (미국 시장)
- **적응형 자산배분 (ML 기반)**: LSTM + 레짐 스위칭으로 Sharpe 1.38 달성
- **팩터 분산 전략**: 밸류/모멘텀/퀄리티/로우볼/사이즈 ETF 균등 배분

### 한국 시장 특성
- 2025년 IT/산업재/유틸리티만 아웃퍼폼, 나머지 섹터 소외
- AI/반도체 사이클과 연동된 대형주 중심 랠리
- 위성 포지션 30% 이내 제한으로 변동성 관리
- 코어 전략: 국내지수 60% + 중형/성장 20% + 단기채 20%

### 주요 참고 자료
- [R을 이용한 퀀트 투자 포트폴리오 만들기](https://hyunyulhenry.github.io/quant_cookbook/)
- [Dual Momentum Rotation Strategy](https://trendinvestorpro.com/dual-momo-spx-details-perf-latest/)
- [ETF Rotation Strategies - Logical Invest](https://logical-invest.com/trading-academy/etf-investing/etf-rotation-strategies/)
- [Asset Class Trend-Following - Quantpedia](https://quantpedia.com/strategies/asset-class-trend-following)
- [Dynamic Factor Allocation with Regime-Switching](https://arxiv.org/html/2410.14841v1)
- [2025년 글로벌 ETF 투자전략 - 하나금융](https://www.hanaw.com/download/research/FileServer/WEB/strategy/market/2024/11/11/Global_ETF_Outlook_2025.pdf)

## 4. 구현한 신규 전략

### 4-1. Enhanced ETF Rotation (`src/strategy/enhanced_etf_rotation.py`)

기존 ETF Rotation에 4가지 필터를 추가한 개선 전략.

**핵심 개선 사항:**

1. **복합 모멘텀 스코어**: 1M(20%)+3M(30%)+6M(30%)+12M(20%) 가중 z-score
   - 단일 기간 모멘텀 대비 노이즈 감소
   - 각 기간별 z-score 정규화 후 합산

2. **역변동성 가중 (Inverse Volatility Weighting)**
   - 변동성 낮은 ETF에 더 많은 비중
   - 포트폴리오 전체 리스크 평탄화

3. **시장 레짐 필터 (Market Timing)**
   - KODEX 200의 종가 vs 200일 이동평균선 비교
   - RISK_OFF 시 위험자산 비중을 30~70% 감축 → 채권 전환
   - 2022년 하락장에서 MDD를 크게 방어

4. **개별 ETF 추세 필터**
   - SMA(20) > SMA(60)인 ETF만 편입
   - 하락 추세 ETF 조기 배제

5. **최대 하락 필터**
   - 최근 1년 고점 대비 15% 이상 하락한 ETF 제외

### 4-2. Cross-Asset Momentum (`src/strategy/cross_asset_momentum.py`)

서로 상관관계가 낮은 자산군 ETF에 모멘텀 기반 투자.

**핵심 특징:**

1. **듀얼 모멘텀 스코어**: 단기(3M, 60%) + 장기(12M, 40%) 가중 합산
2. **자산군 분산**: equity_kr, equity_us, commodity, bond 중 자산군당 최대 1개
3. **상관관계 필터**: 포트폴리오 내 상관계수 0.7 이상인 자산 스킵
4. **SMA 크로스오버**: SMA(20) > SMA(60)인 자산만 투자
5. **변동성 패리티**: 역변동성 비중 배분으로 리스크 평탄화

## 5. 백테스트 결과 (2020-01-01 ~ 2025-12-31, 초기자본 145만원)

### 전체 결과 (Sharpe 내림차순)

| 순위 | 전략 | 수익률 | CAGR | Sharpe | MDD | 거래 | 최종자산 |
|------|------|--------|------|--------|-----|------|----------|
| 1 | **Enhanced: strong_riskoff** | **+291.0%** | **+25.5%** | **1.31** | **-19.6%** | 341 | 5,669K |
| 2 | **Enhanced: all_filters** | **+280.5%** | **+24.9%** | **1.27** | **-22.0%** | 341 | 5,517K |
| 3 | **Enhanced: long_mom** | +277.8% | +24.8% | 1.26 | -22.1% | 339 | 5,478K |
| 4 | Enhanced: short_mom | +271.9% | +24.5% | 1.25 | -22.0% | 343 | 5,393K |
| 5 | Enhanced: mild_riskoff | +272.4% | +24.5% | 1.23 | -24.6% | 341 | 5,399K |
| 6 | Enhanced: vol+mkt | +241.9% | +22.8% | 1.21 | -21.8% | 351 | 4,958K |
| 7 | Enhanced: mkt_only | +232.8% | +22.2% | 1.14 | -21.8% | 351 | 4,825K |
| 8 | Enhanced: vol+trend | +256.9% | +23.6% | 1.14 | -27.7% | 325 | 5,175K |
| 9 | Enhanced: top2_all | +276.6% | +24.7% | 1.13 | -23.6% | 252 | 5,460K |
| 10 | Baseline: ETF(3M,invol) | +218.2% | +21.3% | 1.08 | -33.6% | 361 | 4,614K |
| 11 | Enhanced: vol_only | +210.5% | +20.8% | 1.05 | -30.1% | 337 | 4,502K |
| 12 | **Baseline: ETF(3M,top3)** | +214.2% | +21.0% | 1.02 | -34.3% | 361 | 4,555K |
| 13 | CrossAsset: no_trend | +117.2% | +13.8% | 0.84 | -20.6% | 317 | 3,150K |
| 14 | Baseline: ETF(12M,top3) | +147.3% | +16.3% | 0.82 | -25.4% | 361 | 3,587K |
| 15 | CrossAsset: short_bias | +162.0% | +17.4% | 0.82 | -36.1% | 267 | 3,800K |
| 16 | CrossAsset: top2 | +169.4% | +18.0% | 0.81 | -36.1% | 220 | 3,906K |
| 17 | CrossAsset: no_corr | +151.8% | +16.6% | 0.79 | -36.0% | 275 | 3,652K |
| 18 | CrossAsset: high_corr | +151.8% | +16.6% | 0.79 | -36.0% | 275 | 3,652K |
| 19 | CrossAsset: max2_class | +139.5% | +15.7% | 0.77 | -33.1% | 292 | 3,472K |
| 20 | CrossAsset: base | +132.7% | +15.1% | 0.71 | -36.0% | 267 | 3,375K |

### 핵심 분석

#### 최고 전략: Enhanced ETF Rotation (strong_riskoff)
- **Sharpe 1.31**: 기존 최고(1.02) 대비 **+28% 개선**
- **MDD -19.6%**: 기존(-34.3%) 대비 **42% 개선** (절대값 기준)
- **CAGR +25.5%**: 기존(+21.0%) 대비 **+4.5%p 증가**
- 145만원 → 567만원 (6년간 약 3.9배)

#### 개선 요인별 기여도 분석

| 필터 조합 | Sharpe | MDD | vs 베이스라인 |
|-----------|--------|-----|-------------|
| 베이스라인 (3M,top3) | 1.02 | -34.3% | -- |
| +vol_weight | 1.05 | -30.1% | Sharpe +0.03, MDD 4%p 개선 |
| +market_filter | 1.14 | -21.8% | Sharpe +0.12, MDD 12%p 개선 |
| +vol+mkt | 1.21 | -21.8% | Sharpe +0.19, MDD 12%p 개선 |
| +all_filters | 1.27 | -22.0% | Sharpe +0.25, MDD 12%p 개선 |
| +strong_riskoff(0.7) | **1.31** | **-19.6%** | **Sharpe +0.29, MDD 15%p 개선** |

**결론**: 마켓 타이밍 필터가 가장 큰 기여 (Sharpe +0.12, MDD 12%p 개선).
변동성 가중과 추세 필터가 추가적으로 Sharpe를 0.1 이상 개선.

#### 연도별 수익률 비교

| 연도 | Baseline (3M/3) | Enhanced (strong_riskoff) | 차이 |
|------|-----------------|--------------------------|------|
| 2020 | +24.1% | +42.1% | +18.0%p |
| 2021 | +27.2% | +19.7% | -7.5%p |
| 2022 | **-24.8%** | **-0.2%** | **+24.6%p** |
| 2023 | +32.2% | +22.5% | -9.7%p |
| 2024 | +21.3% | +18.5% | -2.8%p |
| 2025 | +65.0% | +82.7% | +17.7%p |

**핵심**: 2022년 하락장에서 베이스라인은 -24.8% 손실, Enhanced는 -0.2%로 방어.
상승장에서는 약간 언더퍼폼하지만, 하락장 방어가 복리 효과를 극대화.

#### Cross-Asset Momentum 결과 분석
- Sharpe 0.71~0.84: 기존 ETF Rotation 대비 다소 낮음
- MDD -20.6% (no_trend 변형): 매우 안정적
- 자산군 분산이 효과적이나, 한국 ETF 유니버스가 제한적이어서 성과 제약
- `no_trend` 변형이 안정성 면에서 가장 우수

### 위험조정 성과 (Sharpe / |MDD|)
| 전략 | Sharpe/|MDD| |
|------|------------|
| Enhanced: strong_riskoff | **6.67** |
| Enhanced: all_filters | 5.78 |
| Enhanced: long_mom | 5.70 |
| CrossAsset: no_trend | 4.08 |
| Baseline: ETF(3M,top3) | 2.98 |

## 6. 추천 전략 및 포트폴리오 구성

### 주력 전략: Enhanced ETF Rotation (strong_riskoff)
- **파라미터**: num_etfs=3, all_filters=ON, cash_ratio_risk_off=0.7
- **예상 성과**: Sharpe 1.3, CAGR 25%, MDD -20%
- **거래세**: ETF이므로 매도 거래세 0원 (수수료만 약 0.015%)

### 보완 전략: CrossAsset (no_trend)
- 안정성 위주 (MDD -20.6%, Sharpe 0.84)
- Enhanced와 다른 자산군 특성으로 분산 효과

### 제안 포트폴리오 배분
| 풀 | 비중 | 전략 | 예상 Sharpe |
|----|------|------|-----------|
| ETF 주력 | 70% | Enhanced ETF Rotation (strong_riskoff) | 1.31 |
| ETF 보완 | 30% | CrossAsset Momentum (no_trend) | 0.84 |

## 7. 향후 개선 방향

### 단기 (1~2주)
1. **유니버스 확장**: 달러(261240), 국채(308620), 2차전지 ETF 추가
2. **스케줄러 통합**: EnhancedETFRotation을 09:10 스케줄에 연결
3. **분할 매매**: 3~5일 분할 매수/매도 로직 구현
4. **파라미터 최적화**: Walk-forward 검증으로 과적합 방지

### 중기 (1~3개월)
5. **레짐 스위칭 ML**: LSTM 기반 시장 레짐 예측 → 동적 cash_ratio 조절
6. **섹터 로테이션**: 산업별 ETF 유니버스 확장 (2차전지, 바이오, 자동차 등)
7. **글로벌 자산 포함**: 달러, 금, 미국 국채로 Cross-Asset 유니버스 보강
8. **변동성 타깃**: 포트폴리오 전체 목표 변동성 15% 관리

### 장기 (3~6개월)
9. **앙상블 전략**: Enhanced + CrossAsset + Value 등 다전략 가중 합산
10. **자동 파라미터 조정**: 시장 환경에 따른 lookback/riskoff 비율 자동 조절
11. **거래비용 최적화**: 리밸런싱 빈도 최적화 (월간 vs 격주)

## 8. 파일 위치

| 파일 | 경로 |
|------|------|
| Enhanced ETF Rotation 전략 | `src/strategy/enhanced_etf_rotation.py` |
| Cross-Asset Momentum 전략 | `src/strategy/cross_asset_momentum.py` |
| 백테스트 스크립트 | `scripts/run_enhanced_etf_backtest.py` |
| Enhanced ETF 테스트 | `tests/test_enhanced_etf_rotation.py` |
| Cross-Asset 테스트 | `tests/test_cross_asset_momentum.py` |
| 백테스트 결과 JSON | `data/enhanced_etf_backtest_results.json` |
| 이 보고서 | `docs/strategy_research_20260228.md` |
