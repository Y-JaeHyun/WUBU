# 통합 작업 진행 현황

## 완료 일시: 2026-02-19

## Step 2: DART API 수정 ✅
- `finstate()` → `finstate_all()` 업그레이드 (매출총이익, 영업현금흐름 포함)
- 계정명 매칭: 정확매칭(==) 우선, 부분매칭(contains) fallback
  - "자본과부채총계"가 "부채총계"로 오인식되던 버그 수정
- IFRS 대안 계정명 추가 (자산총계/부채와자본총계 등)
- 부채총계 미존재 시 `자산총계 - 자본총계`로 자동 계산
- 5대 종목 실데이터 검증 완료 (삼성전자, SK하이닉스, LG화학, 카카오, NAVER)

## Step 3: Telegram 알림 ✅
- 봇 토큰 + 채팅 ID 설정
- AlertManager + MddThresholdCondition + DailyMoveCondition 동작 확인
- 실제 메시지 발송 3건 성공

## Step 4: 실데이터 백테스트 ✅
- `scripts/run_backtest.py` — 5개 전략, 2023~2025 3년
- 결과: MultiFactor(V+M) 최고 37.49%, Sharpe 0.48
- 모멘텀은 데이터 부족 이슈 (상장기간 필터)

## Step 5: KIS 듀얼모드 ✅
- `KIS_TRADING_MODE=paper/live` 환경변수 지원
- 모드별 계좌 자동 선택 (KIS_PAPER_ACCOUNT_NO / KIS_REAL_ACCOUNT_NO)
- `mode_tag` 프로퍼티 → 모든 알림에 [모의]/[실전] 태그
- 실전 모드 안전장치: `KIS_LIVE_CONFIRMED=true` 필수
- RiskGuard 모드별 기본 한도: paper(10/30/15%) vs live(5/20/10%)
- 토큰 파일 캐싱 (`.kis_token.json`) — 분당 1회 발급 제한 대응
- KIS API float 파싱 수정 (`int(float(...))`)
- API 연결 테스트 완료: 토큰, 현재가, 잔고, 매수가능금액

### 계좌 현황 (2026-02-19 기준)
- 총 평가: ***,***원
- 예수금: **,***원
- 보유: TIGER 미국S&P500 *주, ACE AI반도체포커스 *주

## Step 6: 이메일 알림 → 텔레그램 대체 ✅

## Step 7: systemd 배포 ✅
- `scripts/quant-bot.service` → `/etc/systemd/system/` 등록
- PYTHONPATH 설정 추가 (모듈 import 해결)
- WatchdogSec 제거 (APScheduler BlockingScheduler 비호환 — 5분마다 강제종료 버그)
- logrotate 설정 완료 (30일 보관, 일일 회전)
- 서비스 상태: active (running), enabled (부팅 시 자동 시작)

### 일일 스케줄
| 시각 | 작업 |
|------|------|
| 07:00 | 모닝 브리핑 |
| 08:50 | 장 전 시그널 체크 |
| 09:05 | 리밸런싱 실행 (해당일만) |
| 매시(09~15) | 포트폴리오 모니터링 |
| 15:35 | EOD 리뷰 |
| 19:00 | 이브닝 종합 리포트 |

## 테스트
- 전체: 363 passed, 2 skipped
- 신규 테스트: 듀얼모드 12개, Telegram 환경변수 격리

## 다음 작업 후보
1. 계좌 규모에 맞는 전략 조정 (종목 수 축소 또는 ETF 전략)
2. 모의투자 계좌 신청 및 설정
3. Black-Litterman / 웹 대시보드 / 해외주식
