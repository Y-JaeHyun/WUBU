# 신규 전략 후보 목록

> 이 파일은 Ralph R&D 태스크의 입력 파일입니다.
> 대화형 Claude 세션에서 웹서칭을 통해 수집한 후보 전략을 여기에 기록합니다.
> Ralph loop는 이 파일을 읽어 실현 가능성 평가 → 설계서 작성을 수행합니다.

## 작성 가이드

각 후보 전략을 다음 형식으로 기록:

```
### {전략 이름}
- **출처**: 논문/서적/블로그 URL
- **핵심 아이디어**: 1~2문장
- **매수 조건**: ...
- **매도 조건**: ...
- **필요 데이터**: 가격, 재무제표, 거래량 등
- **알려진 성과**: CAGR ~X%, MDD ~X% (출처 기준)
- **한국 시장 적용 가능성**: 높음/중간/낮음
```

## 후보 전략

> 조사일: 2026-03-12
> 기존 구현 전략: value, momentum, quality, three_factor, multi_factor, dual_momentum,
> risk_parity, etf_rotation, enhanced_etf_rotation, cross_asset_momentum,
> low_volatility, low_vol_quality, shareholder_yield, accrual, pead,
> ml_factor, hybrid_strategy, bb_squeeze, high_breakout, swing_reversion, orb_daytrading

### 1. NCAV (Net Current Asset Value) 딥밸류
- **출처**: Benjamin Graham, "Security Analysis"; Quantpedia (https://quantpedia.com/strategies/net-current-asset-value-effect)
- **핵심 아이디어**: 순유동자산가치(유동자산 - 총부채)가 시가총액보다 큰 극단적 저평가 종목에 투자. "net-net" 전략.
- **매수 조건**: NCAV/MV > 1.0 (보수적: > 1.5), 소형주 위주, 거래대금 최소 기준 충족
- **매도 조건**: NCAV/MV < 0.8 또는 12개월 보유 후 리밸런싱
- **필요 데이터**: 재무제표(유동자산, 총부채), 시가총액, 거래량
- **알려진 성과**: CAGR ~35.3% (25년 미국 백테스트), MDD 기록 없으나 소형주 특성상 높을 수 있음
- **한국 시장 적용 가능성**: 높음 (한국 소형주에 NCAV > MV 종목 다수 존재, DART 재무데이터 활용 가능)

### 2. Composite Quality Score (복합 품질 점수)
- **출처**: NBIM Discussion Note #3-15 "The Quality Factor"; Quantpedia Earnings Quality Factor
- **핵심 아이디어**: ROE, 현금흐름/자산, 부채비율, 발생액(accruals) 4가지 지표의 백분위 합산으로 종합 품질 점수를 산출하여 상위 종목 매수.
- **매수 조건**: 4개 지표 백분위 합산 상위 30% 종목
- **매도 조건**: 연 1회 리밸런싱, 품질 점수 하위 30% 이탈 시
- **필요 데이터**: 재무제표(ROE, CFO, 총자산, 총부채, 순이익, 발생액)
- **알려진 성과**: 다른 팩터와 낮은 상관관계, 하락장 방어력 우수. 장기 CAGR ~12-15%
- **한국 시장 적용 가능성**: 높음 (기존 quality + accrual 전략의 상위호환, 동일 데이터 소스 활용)

### 3. Industry Momentum with Reversal Hedge (업종 모멘텀 + 반전 헤지)
- **출처**: Gao, Li, Yuan, Zhou "Systematic Reversal and Industry Momentum" (2024)
- **핵심 아이디어**: 업종(산업) 단위 모멘텀에 개별종목 반전(reversal) 헤지를 결합하여 샤프 비율을 대폭 개선. 순수 업종 모멘텀 Sharpe 0.56 → 반전 헤지 후 1.11.
- **매수 조건**: 최근 12개월 업종 수익률 상위 30% 업종 내 종목 매수 + 단기(1개월) 반전 하위 종목 배제
- **매도 조건**: 월간 리밸런싱, 업종 순위 변동 시 교체
- **필요 데이터**: 가격(업종별 분류 필요), KRX 업종 코드
- **알려진 성과**: Sharpe ~1.11 (미국 시장, 논문 기준)
- **한국 시장 적용 가능성**: 중간 (KRX 업종 분류 활용 가능, 다만 한국 시장에서 모멘텀 효과가 미국보다 약할 수 있음)

### 4. ESG-Enhanced Multifactor (ESG 강화 멀티팩터)
- **출처**: Amundi/Robeco ESG 퀀트 팩터 리서치; 미래에셋증권 "ESG 퀀트투자 거버넌스 스타일 전략" (2025)
- **핵심 아이디어**: 기존 멀티팩터(밸류+모멘텀+퀄리티)에 ESG/거버넌스 스코어를 추가 팩터로 결합. 수익·위험·지속가능성 3차원 최적화.
- **매수 조건**: 기존 멀티팩터 상위 종목 중 ESG 등급 B+ 이상 필터링
- **매도 조건**: 분기 리밸런싱, ESG 등급 하락 시 즉시 교체
- **필요 데이터**: 재무제표, 가격, ESG 등급 (한국거래소 ESG 포털 또는 KCGS)
- **알려진 성과**: 기존 멀티팩터 대비 MDD 개선, CAGR 유사 또는 소폭 개선
- **한국 시장 적용 가능성**: 중간 (ESG 등급 데이터 접근성이 제한적, KCGS 데이터 크롤링 필요)

### 5. Tactical Asset Allocation with Regime Detection (레짐 감지 전술적 자산배분)
- **출처**: Gresham "Systematic Strategies & Quant Trading 2025"; 다수 TAA 연구
- **핵심 아이디어**: 시장 레짐(상승/하락/횡보)을 모멘텀+변동성+금리 시그널로 판별하고, 레짐별 자산(주식/채권/현금) 배분 비중을 동적 조절.
- **매수 조건**: 레짐=상승 → 주식 비중 80%, 레짐=횡보 → 50%, 레짐=하락 → 20%
- **매도 조건**: 레짐 전환 시 즉시 리밸런싱
- **필요 데이터**: 지수 가격(KOSPI, 국채), VIX/VKOSPI, 금리(한은 기준금리)
- **알려진 성과**: TAA가 2025~2026 초 60/40 벤치마크 대비 우수한 성과
- **한국 시장 적용 가능성**: 높음 (기존 market_timing 오버레이와 유사하나, 레짐 감지를 더 정교화. ETF 로테이션과 결합 가능)

### 6. Size-Value Interaction (소형 가치주 집중)
- **출처**: Kim et al. "Enhanced Factor Investing in the Korean Stock Market" (Pacific-Basin Finance Journal)
- **핵심 아이디어**: 한국 시장에서 사이즈 팩터가 가장 큰 수익률 프리미엄을 보이는 점을 활용. 소형주 중 밸류 팩터 상위 종목에 집중 투자.
- **매수 조건**: 시가총액 하위 30% 중 PBR 하위 20% (소형 가치주)
- **매도 조건**: 분기 리밸런싱, 시총 상위 30% 진입 시 매도
- **필요 데이터**: 시가총액, PBR, 거래량
- **알려진 성과**: 한국 시장 사이즈 팩터 프리미엄 연 ~8-12% (논문 기준)
- **한국 시장 적용 가능성**: 높음 (한국 시장에서 검증된 팩터, 기존 데이터로 즉시 구현 가능)
