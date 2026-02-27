# 2026-03-02 리밸런싱 준비 점검

생성 시각: 2026-02-27 10:31 KST

---

## 1. 리밸런싱 스케줄 확인

| 항목 | 상태 |
|------|------|
| 리밸런싱일 | **2026-03-02 (월)** - 3월 첫 거래일 |
| 오늘 기준 D- | **D-3** (2/28 토, 3/1 삼일절 휴장) |
| 3/1 삼일절 | 공휴일 (일요일과 겹침) - 대체공휴일 아님 |
| 리밸런싱 주기 | monthly (매월 첫 거래일) |
| 장기 리밸런싱 시각 | 09:05 |
| ETF 로테이션 시각 | 09:10 |

## 2. 현재 포트폴리오 현황

### 잔고 요약
| 항목 | 금액 |
|------|------|
| 총 평가액 | ***원 |
| 현금 | ***원 (약 67%) |
| 보유 주식 | ***원 (약 33%) |
| 총 수익 | +**% |

### 보유 종목 (ETF 2종)
| 종목 | 코드 | 수량 | 수익률 |
|------|------|------|--------|
| TIGER 미국S&P500 | 360750 | *주 | 소폭 손실 |
| ACE AI반도체포커스 | 469150 | *주 | 대폭 수익 |

### 분석
- 현금 비중이 높음 → 리밸런싱 시 매수 여력 충분
- ACE AI반도체포커스 대폭 수익 → 모멘텀 기반이면 유지될 가능성 높음
- TIGER 미국S&P500 소폭 손실 → 모멘텀에 따라 교체 가능

## 3. 시스템 상태 점검

### KIS API
| 항목 | 상태 |
|------|------|
| 연결 | 정상 |
| 모드 | **실전투자** |
| 계좌 | ****-** |
| 토큰 만료 | 자동 갱신 |
| KIS_LIVE_CONFIRMED | 확인됨 |
| 토큰 자동 갱신 | 자동 (만료 시 재발급) |

### Feature Flags
| Flag | 상태 | 비고 |
|------|------|------|
| data_cache | ON | pykrx 캐싱 활성 |
| etf_rotation | ON | ETF 로테이션 활성 (30% 배분) |
| daily_simulation | ON | 일일 시뮬레이션 활성 |
| news_collector | ON | DART 공시 수집 활성 |
| emergency_monitor | ON | 긴급 모니터링 활성 |
| short_term_trading | OFF | 단기 비활성 (정상) |

### 스케줄러 작업 (3/2 리밸런싱일)
| 시간 | 작업 | 설명 |
|------|------|------|
| 07:00 | morning_briefing | 장전 브리핑 (리밸런싱 당일 알림) |
| 08:05 | news_check | DART 공시 체크 |
| 08:50 | premarket_check | 시그널 계산 + dry_run |
| **09:05** | **execute_rebalance** | **장기 MultiFactor 리밸런싱** |
| **09:10** | **execute_etf_rotation_rebalance** | **ETF 로테이션 리밸런싱** |
| 09~15 | monitor_positions | 포지션 모니터링 |
| 15:35 | eod_review | 장 마감 리뷰 |

## 4. 리밸런싱 플로우 검증

### 4-A. 장기 전략 리밸런싱 (09:05)
```
1. _is_trading_day() → True
2. holidays.is_rebalance_day(2026-03-02, "monthly") → True
3. _collect_strategy_data("20260302"):
   - get_all_fundamentals → 2,700+ 종목
   - get_price_data × 200종목 (시총 상위, 400일)
   - get_index_data("KOSPI", 400일)
4. strategy.generate_signals(date, data) → {ticker: weight} × 7종목
   - MultiFactor(value+momentum, zscore, top7)
   - apply_market_timing (KOSPI vs MA200)
   - turnover_penalty=0.1
   - max_group_weight=0.25 (업종 비중 25%)
   - max_stocks_per_conglomerate=2 (계열사 2종목)
5. executor.execute_rebalance(signals, pool="long_term")
   - allocator.filter_long_term_weights() → 70% 스케일링
   - position_manager.calculate_rebalance_orders() → diff-based
   - 매도 먼저 → 3초 대기 → 매수
```

### 4-B. ETF 로테이션 (09:10)
```
1. feature_flags.is_enabled("etf_rotation") → True
2. holidays.is_rebalance_day(2026-03-02, "monthly") → True
3. ETFRotationStrategy(lookback=252, num_etfs=3, max_same_sector=1)
4. _fetch_etf_prices(10개 ETF, 252일)
5. strategy.generate_signals() → 상위 3개 ETF (모멘텀 순)
   - momentum_cap=3.0 적용
   - 안전자산(단기채권) 기본 포함 조건
6. executor.execute_rebalance(signals, pool="etf_rotation")
   - allocator.filter_etf_rotation_weights() → 30% 스케일링
   - diff-based 매매 (기존 보유 ETF 유지, 변경분만 매매)
```

### 4-C. 리밸런싱 결과 예상
| Pool | 배분 |
|------|------|
| 장기 (MultiFactor) | 70% |
| ETF 로테이션 | 30% |
| 합계 | 100% |

**현재 보유 대비**:
- ETF 보유 → ETF 배분 소폭 조정 가능
- 장기 보유 0 → 전체 신규 매수
- 현금 충분 → 매수 자금 문제 없음

## 5. 전략 비교 (백테스트 기준)

| 전략 | CAGR | Sharpe | MDD | 비고 |
|------|------|--------|-----|------|
| **MultiFactor+MT** | **11.3%** | **0.38** | -60.7% | 현재 운영 전략 |
| MultiFactor(V+M) | 10.5% | 0.37 | -60.7% | MT 미적용 |
| ThreeFactor(V+M+Q) | 6.3% | 0.24 | -42.2% | MDD 우수 |
| ShareholderYield | 4.7% | 0.17 | -40.5% | 안정성 |
| **ETF 12M/3** | **11.6%** | **0.56** | **-28.3%** | **Sharpe 최고** |

**현재 설정 (MultiFactor+MT + ETF 12M/3)**:
- 장기 CAGR 11.3% + ETF CAGR 11.6% → 복합 수익률 양호
- ETF MDD -28.3%가 장기 -60.7% 보완 → 전체 MDD 완화 기대

## 6. 신규 안전 필터 적용 효과

| 필터 | 설정 | 효과 |
|------|------|------|
| 업종 비중 제한 | 25% | 동일 업종 쏠림 방지 (예: 반도체 5종목 이내) |
| 계열사 집중도 | 2종목 | 삼성/현대/SK 등 동일 그룹 2종목 제한 |
| 모멘텀 캡 | 300% | 극단적 모멘텀 종목 Z-score 왜곡 방지 |
| 회전율 페널티 | 0.1 | 불필요한 종목 교체 억제 |
| diff-based 리밸런싱 | 활성 | 유지 종목 불필요한 매도+재매수 제거 |

## 7. 리스크 체크리스트

- [x] KIS API 연결 정상
- [x] 실전 모드 확인 (KIS_LIVE_CONFIRMED=true)
- [x] 토큰 유효 (자동 갱신 가능)
- [x] feature flags 정상 (etf_rotation ON, daily_simulation ON)
- [x] 리밸런싱일 확인 (2026-03-02 = 3월 첫 거래일)
- [x] 현금 충분 (매수 자금 확보됨)
- [x] 안전 필터 적용됨 (업종/계열사/모멘텀캡)
- [x] diff-based 리밸런싱 적용됨 (불필요 매매 제거)
- [x] 긴급 모니터링 활성 (emergency_monitor ON)
- [ ] 3/2 08:50 시그널 확인 (당일 확인)
- [ ] 3/2 09:05 장기 리밸런싱 실행 (당일 확인)
- [ ] 3/2 09:10 ETF 리밸런싱 실행 (당일 확인)

## 8. 주의사항

1. **KIS 새벽 점검 (01:00~05:00)**: 이 시간대 API 사용 불가 → 07:00 부터 정상
2. **3/1 삼일절**: 일요일과 겹치지만 대체공휴일 아님 → 3/2 정상 개장
3. **ETF 토큰 만료**: 만료 시 자동 재발급됨
4. **첫 장기 리밸런싱**: 현재 장기 풀 보유 0 → 전체 신규 매수 예상
5. **최소 주문 금액**: 50,000원 미만 주문 자동 스킵

---

*이 리포트는 자동 생성되었습니다. 리밸런싱 당일(3/2) 08:50 시그널 리포트와 함께 확인하시기 바랍니다.*
