# Emergency Rebalancing System Design

## Overview

긴급 리밸런싱 시스템은 장중 이상 상황(급등/급락, 시장 급변, 긴급 공시)을 실시간으로 감지하여
텔레그램 알림을 발송하고, 필요 시 자동 매도를 트리거하는 방어 체계이다.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TradingBot                           │
│  emergency_monitor_check() [30분 간격]                   │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ KIS API      │  │ pykrx        │  │ NewsCollector│  │
│  │ (보유 잔고)   │  │ (KOSPI 지수)  │  │ (DART 공시)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
│         v                 v                 v           │
│  ┌─────────────────────────────────────────────────┐    │
│  │              state (dict)                       │    │
│  │  - price_shocks: [{name, ticker, change}]       │    │
│  │  - market_change_pct: float                     │    │
│  │  - portfolio_disclosures: [{corp_name, ...}]    │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                 │
│                       v                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │           AlertManager.check_and_alert()        │    │
│  │  - PriceShockCondition                          │    │
│  │  - MarketCrashCondition                         │    │
│  │  - DisclosureAlertCondition                     │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                 │
│              ┌────────┴────────┐                        │
│              v                 v                        │
│  ┌──────────────┐   ┌──────────────────────┐           │
│  │ Telegram     │   │ Auto-Exit            │           │
│  │ 알림 발송     │   │ (feature flag 제어)   │           │
│  └──────────────┘   └──────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

## Condition Types and Thresholds

### 1. PriceShockCondition (급등/급락)

| 항목 | 값 |
|------|-----|
| 임계값 | 5% (config: `price_shock_pct`) |
| WARNING | 5% <= 변동 < 7% |
| CRITICAL | 변동 >= 7% |
| 쿨다운 | 2시간 |
| 데이터 소스 | KIS API `get_balance()` → holdings.change_pct |

- 보유 종목의 당일 등락률을 실시간 확인
- 양방향(급등/급락) 모두 감지

### 2. MarketCrashCondition (시장 급변)

| 항목 | 값 |
|------|-----|
| 임계값 | 3% (config: `market_crash_pct`) |
| 레벨 | 항상 CRITICAL |
| 쿨다운 | 4시간 |
| 데이터 소스 | pykrx KOSPI(1001) 전일 대비 변동 |

- KOSPI 지수의 전일 종가 대비 당일 변동률
- 양방향(폭락/폭등) 모두 감지
- strict greater-than: 정확히 3.0%는 미발동

### 3. DisclosureAlertCondition (긴급 공시)

| 항목 | 값 |
|------|-----|
| WARNING | 일반 공시 (유상증자, 분할 등) |
| CRITICAL | 상장폐지(delisting), 합병(merger) |
| 쿨다운 | 4시간 |
| 데이터 소스 | DART OpenAPI (NewsCollector) |

- 보유 종목 관련 공시에 `[보유]` 접두어 표시
- 카테고리 기반 심각도 동적 판별

## Alert Flow

```
1. emergency_monitor_check() 실행 (30분 간격)
2. Feature Flag 'emergency_monitor' 확인
3. 거래일 여부 확인
4. 데이터 수집:
   a. KIS API → 보유 종목 + 변동률
   b. pykrx → KOSPI 지수 변동
   c. DART → 최근 공시
5. state dict 구성
6. AlertManager.check_and_alert(state) 호출
   - 각 condition.check(state) 실행
   - 쿨다운 확인 (중복 알림 방지)
   - 발동 시 format_message() → Telegram 발송
7. CRITICAL 감지 + auto_exit_enabled → 긴급 매도 트리거
```

## Auto-Exit Mechanism

자동 매도는 feature flag로 제어되며, 기본값은 **비활성화**(False)이다.

### 활성화 조건
- `emergency_monitor.config.auto_exit_enabled = True`
- CRITICAL 레벨 감지:
  - 보유 종목 중 7% 이상 급락
  - KOSPI 3% 이상 급변

### 안전장치
- feature flag로 언제든 비활성화 가능 (Telegram `/flag emergency_monitor` 토글)
- 쿨다운으로 동일 조건 중복 매도 방지
- 매도 전 텔레그램으로 "[긴급 매도]" 알림 사전 발송
- 로그에 WARNING 레벨로 기록

### 권장 사항
- 초기에는 `auto_exit_enabled: False`로 알림만 수신
- 충분한 테스트 후 실전 활성화
- 시장 이벤트 시 수동 확인 후 대응 가능

## Monitoring Schedule

| 시간 | 주기 | 작업 |
|------|------|------|
| 장중 (09:00~15:30) | 30분 간격 | emergency_monitor_check() |

### 스케줄 등록 방법

```python
# setup_schedule() 메서드에 추가:
scheduler.add_job(
    self.emergency_monitor_check,
    'interval',
    minutes=30,
    id='emergency_monitor',
)
```

현재는 메서드만 구현되어 있으며, 스케줄 등록은 수동으로 추가해야 한다.
`check_interval_minutes` config 값을 변경하여 체크 주기를 조정할 수 있다.

## Feature Flag Configuration

```json
{
  "emergency_monitor": {
    "enabled": true,
    "description": "긴급 리밸런싱 모니터 (급등락/시장급변/공시)",
    "config": {
      "price_shock_pct": 5.0,
      "market_crash_pct": 3.0,
      "auto_exit_enabled": false,
      "check_interval_minutes": 30
    }
  }
}
```

## Files Modified/Created

| 파일 | 변경 내용 |
|------|-----------|
| `src/alert/conditions.py` | DisclosureAlertCondition, PriceShockCondition, MarketCrashCondition 추가 |
| `src/utils/feature_flags.py` | emergency_monitor 플래그 추가 |
| `src/scheduler/main.py` | emergency_monitor_check() 메서드 + 조건 등록 |
| `tests/test_emergency_monitor.py` | 17개 테스트 케이스 |
