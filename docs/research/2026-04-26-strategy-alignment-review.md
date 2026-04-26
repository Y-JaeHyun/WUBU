# [JAE-10] 전략 의도-구현 정합성 정밀 진단 보고서

**작성일**: 2026-04-26  
**작성자**: Quant Researcher  
**관련 이슈**: [JAE-10](/JAE/issues/JAE-10) (sub-task of [JAE-8](/JAE/issues/JAE-8))

---

## 1. 핵심 요약 (Executive Summary)

| 항목 | 경영진 의도 | 실제 실전 구현 | 상태 |
|------|-----------|--------------|------|
| **전략 명칭** | Quality top20 (V+M+Q) | MultiFactor (V+M) | **불일치** |
| **핵심 팩터** | 밸류 + 모멘텀 + **퀄리티** | 밸류 + 모멘텀 | **퀄리티 누락** |
| **종목 수** | 20종목 | 7종목 | **불일치** |
| **Sharpe** | ~1.3 (백테스트 1.55) | ~1.48 | 양호 |
| **CAGR** | ~53.5% (백테스트 52.89%) | ~36.4% | **성과 격차 −17%p** |

**결론**: 현재 시스템은 "Quality top20"이 아닌 레거시 2팩터 전략(V+M)을 실전에서 가동 중이다. 아키텍처 통합 누락으로 인해 약 17%p의 알파가 매일 손실되고 있다.

---

## 2. 전략별 가설/의도 vs 구현 매핑

### 2.1 Value Strategy (`src/strategy/value.py`)

| 항목 | 내용 |
|------|------|
| **의도된 가설** | 시장이 저평가한 종목은 장기적으로 초과수익을 낸다 (Fama-French HML 팩터) |
| **학술 표준** | B/M ratio (Fama-French 1993), PER 역수, EV/EBITDA |
| **현재 구현** | PBR 역수 + PER 역수 + 선택적 배당수익률, F-Score 필터 |
| **파라미터** | `num_stocks=20`, `min_market_cap=100B KRW`, `exclude_negative=True` |
| **갭/우려사항** | F-Score가 3항목으로 단순화(ROA>0, ROA증가, 부채비율 감소) — 원래 Piotroski (2000)는 9항목. 섹터 중립 모드는 `sector` 컬럼 미존재 시 무음 실패. |

### 2.2 Momentum Strategy (`src/strategy/momentum.py`)

| 항목 | 내용 |
|------|------|
| **의도된 가설** | 과거 12개월 수익률이 높은 종목은 향후 1개월도 아웃퍼폼 (Jegadeesh & Titman 1993) |
| **학술 표준** | 12M-1M 수익률 (최근 1달 제외), 수익률 크로스섹션 |
| **현재 구현** | 12M 수익률, skip_days=21 (1달 제외), 선택적 잔여모멘텀 · 52주신고가 비율 |
| **파라미터** | `lookback_months=12`, `momentum_cap=3.0`, 윈저라이징 1%/99% |
| **갭/우려사항** | 잔여모멘텀은 index_prices 데이터가 없을 경우 활성화 불가. 52주신고가 비율 조합은 70:30 가중치이지만 문서화 없음. |

### 2.3 Quality Strategy (`src/strategy/quality.py`)

| 항목 | 내용 |
|------|------|
| **의도된 가설** | 수익성·재무건전성이 높은 기업은 구조적 초과수익을 낸다 (Novy-Marx 2013) |
| **학술 표준** | GP/A (Novy-Marx), ROE, Piotroski F-Score, 발생액 품질 |
| **현재 구현** | ROE(30%) + GP/A(30%) + 부채비율(20%) + 발생액(20%) 합성 Z-Score |
| **파라미터** | `num_stocks=20`, `strict_accrual=False` |
| **갭/우려사항** | GP/A 데이터 부재 시 `abs(ROE)/200` 대체 — 조잡한 추정. 부채비율은 pykrx에서 자주 누락됨(DART 의존). 발생액 계산이 분기 현금흐름 데이터 지연에 취약. |

### 2.4 ThreeFactorStrategy (`src/strategy/three_factor.py`)

| 항목 | 내용 |
|------|------|
| **의도된 가설** | V+M+Q 3팩터 합성으로 단일 팩터보다 안정적 초과수익 (Asness et al. 2015) |
| **학술 표준** | AQR의 "Quality Minus Junk" + Fama-French Value + 12M Momentum |
| **현재 구현** | Value(0.33) + Momentum(0.33) + Quality(0.34) Z-Score 합성. 선택적 저변동성 4번째 팩터, 마켓타이밍 오버레이, 레짐 메타모델, 섹터/재벌 집중 제한 |
| **파라미터** | `num_stocks=20`, `combination_method="zscore"`, `max_group_weight=0.25`, `max_stocks_per_conglomerate=2` |
| **갭/우려사항** | **실전 미배포**. 레짐 모델이 팩터명 `["value","momentum","quality"]` 하드코딩. 마켓타이밍 whipsaw 필터는 월말만 신호 변경(최대 30일 지연). |

### 2.5 DualMomentumStrategy (`src/strategy/dual_momentum.py`)

| 항목 | 내용 |
|------|------|
| **의도된 가설** | 절대+상대 모멘텀 결합으로 하락장 방어 (Gary Antonacci 2012) |
| **학술 표준** | Dual Momentum: 상대 모멘텀으로 자산 선택, 절대 모멘텀으로 안전자산 전환 |
| **현재 구현** | KODEX200 vs Tiger S&P500 12M 상대모멘텀, 절대모멘텀 0% 기준, 안전자산=단기채 ETF |
| **파라미터** | `risky_assets={"domestic":"069500","us":"360750"}`, `safe_asset="214980"`, `lookback=12M` |
| **갭/우려사항** | ETF 티커 하드코딩으로 유연성 부족. 절대모멘텀 임계값 0% 고정(조정 불가). 변동성 타겟팅은 60일 창만 사용(실현변동성 추정 부족). |

### 2.6 RiskParityStrategy (`src/strategy/risk_parity.py`)

| 항목 | 내용 |
|------|------|
| **의도된 가설** | 리스크 기여도를 균등화하여 특정 종목 리스크 집중을 방지 (Qian 2005) |
| **학술 표준** | Equal Risk Contribution (ERC) 최적화 |
| **현재 구현** | 임의 selector(ThreeFactorStrategy 등) 선택 후 Ledoit-Wolf 공분산으로 ERC 가중치 |
| **파라미터** | `cov_method="ledoit_wolf"`, `lookback_days=252`, `max_weight=0.15`, `min_weight=0.01` |
| **갭/우려사항** | IPO 종목은 252일 수익률 이력 부족으로 실패 가능. 공분산 shrinkage가 꼬리 리스크 과소평가. 특이/근사특이 공분산 행렬 처리 없음. |

### 2.7 MLFactorStrategy (`src/strategy/ml_factor.py`)

| 항목 | 내용 |
|------|------|
| **의도된 가설** | ML이 비선형 팩터 관계를 학습하여 전통 팩터 대비 개선된 예측 제공 |
| **학술 표준** | Alpha prediction via regularized regression / ensemble models |
| **현재 구현** | 학습된 MLPipeline의 예측 점수로 상위 N개 종목 선택 |
| **파라미터** | `num_stocks=20`, `min_market_cap=100B KRW`, `max_weight=0.15` |
| **갭/우려사항** | **블랙박스** — 피처 중요도, 학습 데이터, 모델 스펙 문서화 전혀 없음. 학습/검증 분리 불명확(룩어헤드 바이어스 위험). 모델 재학습 주기 미명시. |

### 2.8 FactorCombiner (`src/strategy/factor_combiner.py`)

| 항목 | 내용 |
|------|------|
| **역할** | 팩터 결합 유틸리티 (전략 클래스 아님) |
| **제공 함수** | `combine_n_factors_zscore`, `combine_n_factors_rank` |
| **파라미터** | Z-Score 클리핑 ±3σ, Percentile rank 0~1 |
| **갭/우려사항** | 이상치 존재 시 Z-Score도 왜곡 가능(±3σ 후에도). Rank는 크기 정보 손실. 결측 팩터 처리 문서화 없음(기본 균등 가중치 추정). |

### 2.9 MultiFactorStrategy (`src/strategy/multi_factor.py`) — 현재 실전 전략

| 항목 | 내용 |
|------|------|
| **의도된 가설** | V+M 합성 팩터로 개별 팩터 대비 안정적 수익 |
| **현재 구현** | `factors=["value","momentum"]`, `weights=[0.5,0.5]`, 스파이크 필터, 마켓타이밍, 재벌 집중 제한 |
| **실전 파라미터** | `num_stocks=7`, `apply_market_timing=True`, `spike_threshold_1d=0.15`, `turnover_penalty=0.1` |
| **갭/우려사항** | 팩터 리스트가 `["value","momentum"]`으로 하드코딩 — Quality 추가가 config 수준에서 불가. 스파이크 필터 임계값 자의적(15%/1일, 25%/5일). 가치함정 필터는 pykrx EPS 미지원으로 오탈락 발생. |

### 2.10 MarketTimingOverlay (`src/strategy/market_timing.py`)

| 항목 | 내용 |
|------|------|
| **의도된 가설** | KOSPI MA 신호로 하락장 현금 전환하여 드로우다운 방어 |
| **학술 표준** | Faber (2007) 200일 이동평균 타이밍 |
| **현재 구현** | SMA/EMA 200일, 바이너리 또는 점진적 전환, 월말 whipsaw 필터, ±2% 밴드 |
| **파라미터** | `ma_period=200`, `switch_mode="binary"`, `cash_return_annual=0.025` |
| **갭/우려사항** | Whipsaw 필터가 월말만 신호 변경을 허용 → 최대 30일 지연. `band_pct=0.02`가 정의만 되어 있고 실제 로직에 반영 여부 불명확. |

---

## 3. 공통 리서치 결함 점검

### 3.1 룩어헤드 바이어스
- **고위험**: ML 모델(`ml_factor.py`) — 학습/검증 분리가 코드에 명시되지 않음. 미래 데이터로 학습된 모델이 백테스트 결과를 과장할 수 있음.
- **중위험**: Quality 팩터 — 분기 재무 데이터(DART)가 실제 공시일 기준으로 지연 반영되는지 불명확. DART 원자료의 `report_date` vs `signal_date` 구분 필요.
- **저위험**: Value/Momentum — 가격 데이터 기반이라 상대적으로 안전하나, 배당 수익률이나 F-Score 계산 시 전년도 결산 데이터 사용 여부 확인 필요.

### 3.2 생존 편향
- **현재 상태 불명확**: Universe 구성에서 상장폐지/관리종목/거래정지 종목 제외 로직이 `min_market_cap` + `min_volume` 필터에 의존하고 있음.
- **위험**: 백테스트 시점에 이미 상장폐지된 종목이 과거 universe에서 제외되지 않았을 경우 수익률 과대 추정.
- **권고**: pykrx 또는 KRX API에서 상장폐지 이력 데이터를 별도 수집하여 universe에서 사전 제거하는 로직 필요.

### 3.3 거래비용/슬리피지 모델링
- **백테스트 엔진**: `src/backtest/` 의 슬리피지/거래비용 가정이 실전 KIS API 수수료 구조와 정합하는지 검증 필요.
- **확인 필요 항목**:
  - 체결가: 종가 vs 익일 시가 가정?
  - 매매 수수료: HTS 기준 0.015%~0.1%?
  - 시장 충격: 소형주 유니버스에서 7종목 매매 시 영향?
- **현재 시스템**: 7종목 × 약 20만원 = 145만원 규모. 수수료 영향은 미미하나, 확장 시(20종목) 반드시 재검증 필요.

### 3.4 마켓타이밍 통계적 타당성
- **우려사항**: 200일 SMA 오버레이는 Faber(2007) 이후 반복 검증된 전략이나, **한국 시장(KOSPI) 적용 시 성과가 글로벌 결과와 상이할 수 있음**.
- **월별 신호 변경(whipsaw 필터)**은 래그를 최대 30일 허용 → 급락장에서 방어 효과 미미.
- **권고**: 마켓타이밍 오버레이 유무에 따른 KOSPI 백테스트 비교 리서치 필요. 2015/2018/2020 급락 이벤트 대응 성과 검증.

### 3.5 리밸런싱 주기와 시그널 Decay 정합성
- **현재**: 월 1회 리밸런싱
- **시그널 decay 관점**:
  - Value: 분기~연간 신호. 월 1회 적절.
  - Momentum: 1개월 성과 주기. 월 1회 적절하나, 신호 decay는 3~6개월.
  - Quality: 분기 재무 데이터 기반. 월 1회 리밸런싱 시 1~2개월은 같은 데이터로 반복 선택.
- **우려사항**: Quality 팩터의 신호 갱신이 월 단위가 아닌 분기 단위인데, 월별 리밸런싱이 거래비용만 발생시키고 알파를 훼손할 가능성.

---

## 4. 즉시 수정이 필요한 항목

| 우선순위 | 항목 | 설명 |
|---------|------|------|
| **P0** | 실전 전략 교체 | `MultiFactorStrategy(V+M)` → `ThreeFactorStrategy(V+M+Q)` 로 교체 또는 Quality 팩터 활성화 |
| **P0** | 종목 수 조정 | 7종목 → 최소 10종목 (145만원 자본에서 분산 효과 확보) |
| **P1** | Quality 데이터 검증 | DART GP/A, 부채비율 적재 상태 확인. Fallback `abs(ROE)/200` 제거 |
| **P1** | ML 모델 문서화 | 피처셋, 학습 기간, 검증 방법 명세화. 룩어헤드 바이어스 검증 |
| **P2** | 생존 편향 방지 | 상장폐지 이력 데이터 수집 및 백테스트 Universe 반영 |
| **P2** | 마켓타이밍 재검증 | KOSPI 적용 성과 독립 검증. Whipsaw 필터 개선 |

---

## 5. 추가 리서치 권고

1. **Quality 팩터 데이터 Coverage 감사**: DART 미공시 종목에서 Quality 점수가 어떻게 계산되는지 확인. Fallback 비율이 50% 이상이면 사실상 Quality가 랜덤 노이즈.
2. **리밸런싱 주기 최적화**: Value/Momentum은 월 1회, Quality는 분기 1회로 분리 운용하는 하이브리드 스케줄 실험.
3. **마켓타이밍 독립 검증**: KOSPI 2010~2025 단순 Buy&Hold vs MA200 타이밍 순수 비교.
4. **Factor Zoo 점검**: 현재 팩터 결합 방식(Z-Score 합산)이 한국 시장에서 통계적으로 유의한지 IC(Information Coefficient) 분석 필요.

---

## 6. 참고 자료

- Fama & French (1993): *Common Risk Factors in the Returns on Stocks and Bonds*
- Jegadeesh & Titman (1993): *Returns to Buying Winners and Selling Losers*
- Novy-Marx (2013): *The Other Side of Value: The Gross Profitability Premium*
- Asness, Frazzini & Pedersen (2015): *Quality Minus Junk*
- Piotroski (2000): *Value Investing: The Use of Historical Financial Statement Information*
- Gary Antonacci (2012): *Risk Premia Harvesting Through Dual Momentum*
- Mebane Faber (2007): *A Quantitative Approach to Tactical Asset Allocation*
