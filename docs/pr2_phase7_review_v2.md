# PR #2 재리뷰 (Rebase 후) — Phase 7: Sharpe 개선

> 리뷰 일시: 2026-02-28 (리베이스 후 2차 리뷰)
> 리뷰어: Claude(Opus 4.6) + Gemini(설계 자문) + Codex(로직 검증)
> 방법: 비판적 관점 코드 정적 분석 + 테스트 실행 + 3자 합의
> 리베이스: `feature/phase7` → `main` (commit 78184f0) 충돌 해결 완료

---

## 리뷰 요약

| 항목 | 결과 |
|------|------|
| **테스트** | 1,598 passed / 2 skipped / 0 failed |
| **변경 규모** | 41 파일, +7,138 / -267 |
| **CRITICAL** | 12건 |
| **MAJOR** | 22건 |
| **MINOR** | 16건 |
| **3자 합의** | Claude: 블록 / Gemini: 블록 / Codex: 부분 블록 (핵심 3건 동의) |
| **최종 판정** | :no_entry: **머지 불가 — CRITICAL 12건 해결 필수** |

---

## CRITICAL Issues (12건)

### C1. 차등 리밸런싱: `portfolio_value`/`current_weights` Stale 문제
**파일**: `src/backtest/engine.py` (3-step pipeline)
- Step 0에서 `portfolio_value`와 `current_weights` 1회 계산
- Step 1,2에서 매도 실행 → `cash` 증가, `holdings` 변경
- Step 3에서 매수 시 **매도 전** 값으로 수량 계산
- **영향**: 매수 수량이 실제 포트폴리오 가치와 불일치, 목표 비중 미달
- **3자 동의**: Claude(C) / Gemini(C) / Codex(C)

### C2. 가격 없는 종목 Silent 삭제
**파일**: `src/backtest/engine.py` (Step 1)
- `del holdings[ticker]`가 가격 데이터 유무와 무관하게 **무조건 실행**
- 거래정지/데이터 누락 종목은 0원에 삭제 → 포지션 증발
- **3자 동의**: Claude(C) / Gemini(C) / Codex(C)

### C3. LowVolatilityStrategy: Look-Ahead Bias
**파일**: `src/strategy/low_volatility.py`
- `_compute_volatility()`가 전체 DataFrame의 `iloc[-N:]` 사용
- 백테스트 `date` 파라미터로 필터링하지 않아 **미래 데이터** 사용
- `get_scores()`에도 `date` 파라미터 부재
- **영향**: 백테스트 결과가 인위적으로 과대평가
- **3자 동의**: Claude(C) / Gemini(C) / Codex(C)

### C4. Market Timing 이중 적용
**파일**: `src/strategy/three_factor.py` + `src/backtest/engine.py`
- ThreeFactor 내부에서 `market_timing` 적용 + 엔진 외부에서 `overlay` 적용
- 동일 객체 전달 시: 50% 감소 → 25%로 제곱 효과
- **3자 판정**: Claude(C) / Gemini(C) / Codex(설정 리스크로 격하)

### C5. Turnover Buffer가 Concentration Filter 우회
**파일**: `src/strategy/three_factor.py` (merge conflict 해결 시 발생)
- `turnover_buffer > 0` 경로에서 `_apply_concentration_filter()` 미호출
- 업종 비중 제한, 계열사 집중도 제한이 비활성화됨
- Main의 안전 가드레일이 PR 기능 활성 시 무력화
- **3자 판정**: Claude(C) / Gemini(C) / Codex(확인 불가 — 코드 경로 미발견)

### C6. Sector-Neutral 경로도 Conglomerate Filter 우회
**파일**: `src/strategy/three_factor.py`
- `sector_neutral=True` 시 `select_sector_neutral()` 호출하나 `_apply_concentration_filter()` 미호출
- `max_stocks_per_conglomerate` 제한 무효화
- 삼성 계열 과집중 리스크

### C7. Turnover Buffer가 `num_stocks` 초과 선택 가능
**파일**: `src/strategy/three_factor.py`
- 현 보유종목이 `exit_threshold`(num_stocks + buffer) 내이면 모두 선택
- 이후 신규 종목 추가로 최종 포트폴리오가 `num_stocks` 초과
- 동일비중(`1/N`) 계산 시 의도치 않은 낮은 비중 발생

### C8. HybridStrategy: Hedge ETF 가격 미로딩 가능
**파일**: `src/strategy/hybrid_strategy.py`
- 백테스트 엔진이 fundamentals 기반 유니버스만 가격 프리로딩
- Hedge ETF 티커(069500 등)가 `price_cache`에 없으면 매매 불가
- 하이브리드 전략이 75% core-only로 무력화 (25% 현금화)

### C9. Overlay 스태킹: 최소 노출 하한 없음
**파일**: `src/backtest/engine.py` (cross-cutting)
- MarketTiming(50%) × DrawdownOverlay(25%) × VolTargeting(50%) = **6.25%** 노출
- HybridStrategy 적용 시: 75% × 6.25% = **4.7%** 주식 비중
- 회복 불가능한 현금 고착 상태 발생 가능
- **3자 판정**: Claude(C) / Gemini(C) / Codex(설정 리스크)

### C10. Walk-Forward 자본 체이닝 불일치
**파일**: `src/backtest/walk_forward.py`
- 실패/건너뛴 윈도우 시 자본 체인 단절
- `get_oos_portfolio_history()` 중복 날짜 `keep="last"` 처리와 자본 연속성 불일치
- OOS 총 수익률 계산이 갭 기간 미반영

### C11. `enhanced_etf_rotation` 기본값 True
**파일**: `src/utils/feature_flags.py`
- 검증되지 않은 신규 전략이 기본 활성화
- 다음 거래일에 실제 자금으로 즉시 작동
- **3자 동의**: Claude(C) / Gemini(C)

### C12. 신규 모듈 8개 테스트 파일 부재
**대상**: walk_forward, hybrid_strategy, drawdown_overlay, vol_targeting, low_volatility, regime_model, sector_neutral, turnover_reduction
- PR에서 테스트 파일 포함됐으나 리베이스 후 누락 확인 필요
- 핵심 기능에 대한 **제로 커버리지**
- **3자 동의**: Claude(C) / Gemini(C) / Codex(C)

---

## MAJOR Issues (22건)

| # | 파일 | 이슈 |
|---|------|------|
| M1 | engine.py | `min_rebalance_threshold` 누적 드리프트 — 소폭 변동 반복 시 비중 15%+ 이탈 가능 |
| M2 | engine.py | Step 2 매도 수량 계산에 거래비용 미반영 (underselling bias) |
| M3 | engine.py | ETF fallback: 부분 데이터 시 fallback 미발동, 데이터 소스 불일치 |
| M4 | engine.py | price_cache 미포함 보유종목 → invisible positions → silent 삭제 |
| M5 | walk_forward.py | 학습/테스트 분할이 영업일 기준 아닌 달력 기준 |
| M6 | engine.py | vol_targeting이 첫 리밸런싱에서 skip됨 (_portfolio_history 비어있음) |
| M7 | three_factor.py | Regime weights 부분 덮어쓰기로 가중치 합 ≠ 1.0 |
| M8 | three_factor.py | `holding_bonus` 크기 미검증 — rank 방식에서 0.1 이상이면 현 보유 고정 |
| M9 | three_factor.py | `update_holdings()` 실전 실행 경로에서 미호출 — turnover buffer 무효화 |
| M10 | hybrid_strategy.py | `core_weight` 범위 미검증 (음수, >1.0 가능) |
| M11 | drawdown_overlay.py | 비단조 threshold 설정 시 무한 진동 가능 |
| M12 | vol_targeting.py | 하방 변동성 관측치 3개로 극도로 불안정 |
| M13 | vol_targeting.py | 하방 변동성 연율화 수식 오류 (√252 → √126이 올바름) |
| M14 | vol_targeting.py | 최소 노출 하한 없음 → 포트폴리오 완전 현금화 가능 |
| M15 | sector_neutral.py | 단일 종목 섹터가 항상 최고 순위 획득 |
| M16 | sector_neutral.py | Round-robin이 num_stocks 미만 선택 가능 |
| M17 | regime_model.py | 기본 factor_names에 "low_vol" 포함 → 3팩터 사용 시 가중치 누수 |
| M18 | regime_model.py | vol_threshold=20% 고정 — 한국 시장에 부적절 (KOSPI 평균 15~25%) |
| M19 | scheduler/main.py | `emergency_monitor_check()` 구현 완료되었으나 스케줄러 미등록 |
| M20 | scheduler/main.py | `_create_etf_strategy()` 캐싱 없이 반복 호출 |
| M21 | scheduler/main.py | Enhanced 전략에 `lookback_months` 설정 미전달 |
| M22 | scripts/run_enhanced_etf_backtest.py | 자체 백테스트 엔진으로 메인 엔진과 결과 비교 불가 |

---

## MINOR Issues (16건)

| # | 파일 | 이슈 |
|---|------|------|
| m1 | engine.py | initial_capital int→float 변경 미전파 |
| m2 | engine.py | _get_price_on_date 중복 호출 (6회/종목/리밸런싱) |
| m3 | engine.py | biweekly 리밸런싱 시작점 의존적 |
| m4 | engine.py | update_holdings duck-typing (hasattr) — ABC 계약 위반 |
| m5 | engine.py | 로깅 f-string vs %s 혼용 |
| m6 | engine.py | signals 가중치 합 > 1.0 미검증 |
| m7 | three_factor.py | _current_holdings 재시작 시 소실 |
| m8 | three_factor.py | 전략명 ThreeFactor → MultiFactor 변경 (하위호환 우려) |
| m9 | three_factor.py | low_vol_weight 추가 시 가중치 합 > 1.0 미검증 |
| m10 | hybrid_strategy.py | ETF_UNIVERSE import 순환 참조 리스크 |
| m11 | sector_neutral.py | iterrows() 대규모 유니버스에서 성능 저하 |
| m12 | scheduler/main.py | auto_exit 알림이 "매도 시도" 메시지 발송하나 실제 매도 미수행 |
| m13 | feature_flags.py | n_select fallback 기본값 불일치 (플래그=3, 코드=2) |
| m14 | README.md | 테스트 수, 플래그 수 스테일 |
| m15 | scripts/*.py | 미커밋 모듈 import (cross_asset_momentum) |
| m16 | kis_websocket.py | _safe_int 부호 제거로 change 항상 양수 |

---

## 테스트 커버리지 분석

### 기존 테스트 이슈
| 파일 | 이슈 | 심각도 |
|------|------|--------|
| test_backtest.py | overlay/market timing 통합 테스트 없음 | CRITICAL |
| test_backtest.py | lookback_days 파라미터 검증 없음 | CRITICAL |
| test_backtest.py | 부분 리밸런싱(50%→30%) 미테스트 | MAJOR |
| test_backtest.py | 현금 고갈 시 매수 축소 경로 미테스트 | MAJOR |
| test_dual_momentum.py | insufficient_data 테스트가 no-op (assert 없음) | CRITICAL |
| test_dual_momentum.py | zero_volatility_target 동일 설정 비교 (자기 자신) | MAJOR |
| test_scheduler.py | ETF rotation happy-path 미테스트 | CRITICAL |
| test_scheduler.py | 통합 리밸런싱 E2E 미테스트 | CRITICAL |
| test_portfolio_allocator.py | merge_pool_targets 미테스트 | CRITICAL |
| test_portfolio_allocator.py | 비율 합 > 100% 검증 없음 | MAJOR |

### 미존재 테스트 파일 (PR에 포함됐으나 검증 필요)
- test_walk_forward.py, test_hybrid_strategy.py, test_drawdown_overlay.py
- test_vol_targeting.py, test_low_volatility.py, test_regime_model.py
- test_sector_neutral.py, test_turnover_reduction.py

---

## 3자 합의 결과

### 전체 동의 (3/3)
- C1 (stale portfolio_value), C2 (silent position deletion), C3 (look-ahead bias)
- C11 (enhanced_etf_rotation default True), C12 (테스트 부재)

### 2:1 동의 (Claude+Gemini vs Codex)
- C4 (market timing 이중 적용): Codex는 "설정 리스크"로 격하
- C9 (overlay 스태킹 무하한): Codex는 "설정 리스크"로 격하

### Claude 단독 판정
- C5~C8, C10: Codex 검증 범위 외 (PR 브랜치 코드 미확인)

---

## 머지 전 필수 수정사항

### P0 (블로커 — 반드시 수정)
1. **engine.py**: Step 2 완료 후 `portfolio_value`/`current_weights` 재계산
2. **engine.py**: `del holdings[ticker]`를 매도 성공 시에만 실행
3. **low_volatility.py**: `_compute_volatility()`에 `date` 필터링 추가, `get_scores(date)` 인터페이스 변경
4. **feature_flags.py**: `enhanced_etf_rotation` 기본값 `False`로 변경
5. **three_factor.py**: turnover_buffer/sector_neutral 경로에 `_apply_concentration_filter()` 호출 추가
6. **three_factor.py + engine.py**: market timing 이중 적용 방지 가드 추가

### P1 (강력 권고)
7. **vol_targeting.py**: 하방 변동성 연율화 수식 수정 (√252 → 적절한 값)
8. **vol_targeting.py + drawdown_overlay.py**: 최소 노출 하한(예: 10%) 추가
9. **engine.py**: overlay 적용 후 합산 노출도 로깅 + 경고
10. **three_factor.py**: `holding_bonus` 범위 검증, `core_weight` 범위 검증

### P2 (권고)
11. 8개 신규 모듈 테스트 파일 존재 확인 및 커버리지 보강
12. `emergency_monitor_check()` 스케줄러 등록 또는 플래그 비활성화
13. `update_holdings()` 실전 실행 경로에 추가

---

## 최종 판정

:no_entry: **머지 불가**

이전 리뷰(v1) 대비 main의 CRITICAL 3건(lookback_days, preloading, index loading)은 성공적으로 해결되었으나, 리베이스 과정에서 새로운 충돌 해결 문제(C5, C6)가 발생했고, 깊이 있는 재검토를 통해 추가 CRITICAL 이슈(C3 look-ahead bias, C4 이중 적용, C7~C10)가 발견되었다.

특히 C3(Look-Ahead Bias)는 PR의 백테스트 결과 자체의 신뢰성을 훼손하며, C1(stale weights)과 C2(silent position deletion)는 실전 자금 운용에 직접적인 리스크를 초래한다.

P0 항목 6건 수정 후 재리뷰 필요.
