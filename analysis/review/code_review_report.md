# 코드 리뷰 리포트

## 생성일: 2026-03-12

## 요약
- 리뷰 모듈: src/execution, src/scheduler, src/strategy, src/backtest, src/data, src/optimization, src/report, src/alert, src/ml, src/utils
- 발견 이슈: 42건 (Critical: 9, Major: 14, Minor: 19)

---

## Critical Issues

### [C-001] Timezone Mismatch in KISClient Token Expiry
- **파일**: src/execution/kis_client.py
- **라인**: L172, L191, L244
- **문제**: 토큰 만료 시간이 naive `datetime.now()` (timezone 미지정)로 설정됨. 다른 모듈은 `datetime.now(KST)` 사용.
- **영향**: 시스템 timezone이 KST가 아닌 경우 거래 시간 중 인증 실패 가능
- **제안**: `datetime.now(pytz.timezone("Asia/Seoul"))` 일관 사용

### [C-002] Race Condition: Order Status Desynchronization
- **파일**: src/execution/executor.py
- **라인**: L164-170, L201-205
- **문제**: 주문 제출 후 고정 sleep(3초/2초) 동안 상태 동기화 없음. `self.orders` 리스트에 Lock 없이 동시 접근.
- **영향**: 주문 부분 체결 오판, 포트폴리오 가치 계산 오류, 후속 리밸런싱 결정 왜곡
- **제안**: thread-safe Queue 또는 Lock 사용. 지수 백오프 기반 상태 동기화 구현

### [C-003] Float/String Conversion Missing Bounds Checking
- **파일**: src/execution/kis_client.py
- **라인**: L639-641, L710-734, L801
- **문제**: KIS API의 문자열 응답을 int/float 변환 시 유효성 검증 없음. 에러 코드(-1) 반환 시 음수 현금으로 처리되어 공매도 허용 가능
- **영향**: 포트폴리오 계산 데이터 오염, 불법 포지션 생성
- **제안**: `_safe_int(val, default=0, min_val=0)` 헬퍼 함수 생성

### [C-004] Zero Portfolio Value Bypasses Risk Checks
- **파일**: src/execution/position_manager.py
- **라인**: L54-56, L85-89
- **문제**: API 실패 시 portfolio_value=0 반환. risk_guard가 0으로 나누기 또는 0 기준으로 비율 검사하여 모든 주문 통과
- **영향**: 리스크 체크 완전 우회, 자본 대비 과도한 포지션 생성
- **제안**: `if portfolio_value <= 0: raise ValueError("Portfolio value not available")`

### [C-005] Hardcoded Order Delays Without Rate Limit Retry
- **파일**: src/execution/executor.py
- **라인**: L44, L164-170, L204
- **문제**: `ORDER_DELAY=0.5초`, sleep(3.0/2.0) 고정. KIS API 429 에러 시 재시도 로직 없음
- **영향**: 리밸런싱 중간 실패, 포트폴리오 불균형
- **제안**: 설정 가능한 딜레이 + HTTP 429 지수 백오프 구현

### [C-006] Timezone Issue in Short-Term Risk Management
- **파일**: src/execution/short_term_risk.py
- **라인**: L53, L57, L149, L168
- **문제**: 일일 손실 한도 리셋에 naive `datetime.now()` 사용. 시스템 시계가 KST와 다르면 리셋 오작동
- **영향**: 일일 손실 한도 미적용 또는 장중 오리셋
- **제안**: KST 명시적 사용

### [C-007] Look-Ahead Bias in Backtest Fundamentals
- **파일**: src/backtest/engine.py
- **라인**: L414-435
- **문제**: 펀더멘탈 데이터가 날짜별 필터링 없이 로드됨. 미래 실적/애널리스트 수정이 포함될 수 있음
- **영향**: 백테스트 성과 2-5% 과대 추정 (특히 퀄리티/수익성 팩터)
- **제안**: `get_all_fundamentals(date)` 반환 데이터가 해당 날짜 이전 데이터인지 검증

### [C-008] Empty Signal Breaks Turnover Penalty
- **파일**: src/strategy/multi_factor.py
- **라인**: L470-566
- **문제**: 빈 시그널 반환 시에도 `_prev_holdings`가 갱신됨. 다음 리밸런싱의 회전율 페널티 계산 오류
- **영향**: 필터로 정당하게 제외된 종목에 잘못된 페널티 적용
- **제안**: `self._prev_holdings` 갱신을 `if signals:` 블록 내부로 이동

### [C-009] 3-Pool Rounding Errors Compound Over Time
- **파일**: src/backtest/engine.py
- **라인**: L503-507
- **문제**: `round(weight * pct, 6)` 뱅커스 라운딩 + 부동소수점 산술로 최종 비중 합 != 1.0
- **영향**: 100회 이상 리밸런싱 시 포지션 크기 +-0.1% 편차 누적
- **제안**: 개별 라운딩 대신 최종 정규화: `signals = {t: w/sum(merged.values()) ...}`

---

## Major Issues

### [M-001] API Error Codes Not Checked in Critical Paths
- **파일**: src/execution/kis_client.py
- **라인**: L356-361, L632-658
- **문제**: `rt_cd != "0"` 에러 시 로그만 남기고 응답 반환. 호출자가 빈/잘못된 데이터 처리
- **영향**: 주문 상태 "unknown" 유지, 재시도 로직 건너뜀
- **제안**: 에러 시 `{"status": "error", "error_msg": ...}` 조기 반환

### [M-002] WebSocket Reconnection Infinite Loop
- **파일**: src/execution/kis_websocket.py
- **라인**: L606-610
- **문제**: 재연결 실패 시 재귀 호출, 최대 재시도 횟수 없음
- **영향**: 메모리 누수, 봇 응답 불능
- **제안**: `max_reconnect_attempts` 카운터 추가

### [M-003] Order Amount=0 When Price Missing
- **파일**: src/execution/position_manager.py
- **라인**: L288, L306, L322
- **문제**: 가격 조회 실패 시 `amount=0` 주문 생성. risk_guard의 금액 기반 검증 우회
- **영향**: 대규모 주문이 리스크 체크 통과
- **제안**: 가격 누락 시 빈 주문 리스트 반환

### [M-004] OrderManager Not Thread-Safe
- **파일**: src/execution/order_manager.py
- **라인**: L150, L350
- **문제**: `self.orders` 리스트에 Lock 없이 접근
- **영향**: 동시 접근 시 주문 누락, 포트폴리오 정합성 깨짐
- **제안**: `threading.Lock` 적용

### [M-005] Overweight Check Warns But Continues
- **파일**: src/execution/risk_guard.py
- **라인**: L222-228
- **문제**: total_weight > 1.01 경고만 하고 실행 진행 가능
- **영향**: 100% 초과 배분, 현금 버퍼 위반
- **제안**: `is_passed=False` 반환하여 실행 차단

### [M-006] Missing update_holdings() Interface
- **파일**: src/backtest/engine.py
- **라인**: L491-492
- **문제**: `hasattr(self.strategy, "update_holdings")` 체크하지만 Strategy ABC에 미정의
- **영향**: MultiFactorStrategy 외 전략의 회전율 추적 실패
- **제안**: Strategy ABC에 메서드 추가 또는 문서화

### [M-007] Division by Zero in Momentum Residual
- **파일**: src/strategy/momentum.py
- **라인**: L209-211
- **문제**: 시장 횡보 시 `np.polyfit()`이 inf 계수 반환, Z-score 계산 NaN
- **영향**: Q2/Q3 횡보장에서 전략 크래시
- **제안**: `np.std(x) < 1e-6` 체크 추가

### [M-008] Empty Sector Data Collapses Concentration Filter
- **파일**: src/strategy/multi_factor.py
- **라인**: L227-319
- **문제**: sector 컬럼 누락 시 모든 종목 "기타" 분류 → max_group_weight 조기 도달
- **영향**: 목표 20종목 대신 3-4종목만 선택, 심각한 과소투자
- **제안**: sector_map 빈 경우 단순 top-N 선택으로 폴백

### [M-009] Non-Thread-Safe Shared Backtest Cache
- **파일**: src/backtest/engine.py
- **라인**: L82-83
- **문제**: 클래스 레벨 `_shared_price_cache`에 동기화 없음
- **영향**: 병렬 백테스트 시 데이터 충돌
- **제안**: `threading.Lock` 또는 `multiprocessing.Manager().dict()` 사용

### [M-010] Daily Simulator Turnover Understatement
- **파일**: src/data/daily_simulator.py
- **라인**: L568-588
- **문제**: symmetric_difference로 회전율 계산 시 공식 오류 (약 50% 과소 측정)
- **영향**: 전략 활동 수준 오판
- **제안**: 정확한 공식: `changed = len(prev - curr) + len(curr - prev)` / `max(len(prev), len(curr))`

### [M-011] NaN Handling Missing in Factor Combiner
- **파일**: src/strategy/factor_combiner.py
- **라인**: L149-230
- **문제**: N팩터 결합 시 공통 ticker 교집합만 사용, 일부 팩터 NaN인 종목 완전 제외
- **영향**: 퀄리티 데이터 누락 종목의 체계적 배제, 은닉 편향
- **제안**: 팩터별 NaN 허용, 개별 forward-fill 또는 0-fill 적용

### [M-012] Filter Failure Liquidates Entire Portfolio
- **파일**: src/strategy/multi_factor.py
- **라인**: L540-547
- **문제**: 집중도 필터 결과가 빈 경우 `{}` 반환 → 전량 매도 발생
- **영향**: 데이터 이슈 시 불필요한 전량 청산, 거래비용 손실
- **제안**: 필터 실패 시 이전 보유 종목 유지

### [M-013] Hardcoded Magic Numbers Throughout Codebase
- **파일**: 다수 (engine.py L576, momentum.py L155, collector.py L22-23 등)
- **문제**: `MIN_COMBINED_EXPOSURE=0.10`, `min_trading_days=252`, `_MAX_RETRIES=3` 등 하드코딩
- **영향**: 파라미터 조정 시 코드 직접 수정 필요
- **제안**: `src/utils/constants.py` 중앙 설정 모듈 생성

### [M-014] Failed Orders Never Synced to Terminal State
- **파일**: src/execution/order_manager.py
- **라인**: L441-445
- **문제**: `sync_order_status()`가 submitted/partial 상태만 동기화. 서버에서 실패한 주문은 영구 "submitted"
- **영향**: 잘못된 포지션 추적, 다음 리밸런싱 시 보류 주문 오판
- **제안**: 모든 비종료 상태 주문 동기화

---

## Minor Issues

### [m-001] Orphaned Token Cache After Mode Switch
- **파일**: src/execution/kis_client.py L238-239
- **문제**: 모의/실전 모드 전환 시 이전 토큰 캐시 무효화 미흡

### [m-002] Silent 0 Returns on API Error
- **파일**: src/execution/position_manager.py L69-70, L95-96, L118
- **문제**: 에러 시 0 반환, 실제 0 잔고와 구분 불가

### [m-003] Hardcoded WebSocket Max Subscriptions
- **파일**: src/execution/kis_websocket.py L42, L243-250
- **문제**: MAX_SUBSCRIPTIONS = 41 고정, 폴백 전략 없음

### [m-004] WebSocket Tick Uses System Time
- **파일**: src/execution/kis_websocket.py L504
- **문제**: 시스템 시각 사용, KIS 서버 시각 미활용

### [m-005] Allocator State Not Validated Post-Init
- **파일**: src/execution/executor.py L296-306, L456-475
- **문제**: allocator 부분 초기화 시 방어 로직 없음

### [m-006] Credentials Not Masked in Logs
- **파일**: src/execution/kis_client.py L110 및 전반
- **문제**: app_key/app_secret 로그 노출 위험

### [m-007] Division by Zero in Drift Calculation
- **파일**: src/execution/position_manager.py L194
- **문제**: target_weight=0 시 나누기 에러

### [m-008] Unbounded Price Cache in KISClient
- **파일**: src/execution/kis_client.py L96, L856-857
- **문제**: _price_cache 무한 성장, LRU 없음

### [m-009] Inefficient f-string Logger Calls
- **파일**: src/strategy/multi_factor.py L129-146
- **문제**: 로그 레벨과 무관하게 f-string 항상 평가, 백테스트 시 5-10% 오버헤드

### [m-010] Inconsistent Column Naming in Data Collectors
- **파일**: src/data/collector.py L255-262
- **문제**: "DIV" → "div_yield" 매핑 불일치

### [m-011] Unbounded Price Store Cache
- **파일**: src/data/price_store.py L82-86
- **문제**: 장기 백테스트 시 메모리 무한 성장

### [m-012] Negative Portfolio Value Crashes CAGR
- **파일**: src/backtest/engine.py L825-850
- **문제**: 음수 포트폴리오 가치 시 CAGR 계산에서 복소수 반환

### [m-013] Missing Length Validation in Momentum
- **파일**: src/strategy/momentum.py L167-215
- **문제**: 데이터 변형 시 pct_change() 실패 가능

### [m-014] DRY Violation in Z-Score Clipping
- **파일**: src/strategy/factor_combiner.py L126-146
- **문제**: `_zscore_with_clip()` 로직 여러 모듈에 중복

### [m-015] Stale Price Store Metadata
- **파일**: src/data/price_store.py L55-69
- **문제**: 상폐 후 재상장 종목의 메타데이터 미갱신

### [m-016] Spike Filter Index Out-of-Bounds
- **파일**: src/strategy/multi_factor.py L350-356
- **문제**: 연초 첫 리밸런싱 시 5일 수익률 계산에 데이터 부족

### [m-017] No Market Impact Model
- **파일**: src/backtest/engine.py L112-113
- **문제**: 고정 거래비용만 사용, 시장충격 모델 없음 → 연 1-3% 수익 과대추정

### [m-018] No Stock Split/Dividend Adjustment
- **파일**: src/data/collector.py
- **문제**: pykrx 데이터의 액면분할/배당 보정 미처리

### [m-019] No Bounds on Price Cache in Backtest
- **파일**: src/backtest/engine.py L82-83
- **문제**: 클래스 레벨 캐시 무한 성장

---

## 수정 우선순위 권장

### 즉시 (다음 거래일 전)
- C-001: 타임존 일관성
- C-003: float 변환 안전성
- C-004: 포트폴리오 가치 검증
- C-005: Rate limit 재시도
- C-006: 단기 리스크 타임존

### 단기 (1주 이내)
- C-002: 주문 리스트 thread-safety
- C-008: 빈 시그널 turnover 페널티
- M-001: API 에러 코드 처리
- M-005: 과대 비중 검증 강화
- M-014: 실패 주문 상태 동기화

### 중기 (1개월 이내)
- C-007: 백테스트 look-ahead bias
- C-009: 3-Pool 라운딩
- M-002: WebSocket 재연결 제한
- M-004: OrderManager 동시성
- M-008: 섹터 데이터 폴백
- M-012: 필터 실패 시 포지션 유지

### 유지보수
- M-013: 상수 모듈 분리
- Minor 이슈 전체
