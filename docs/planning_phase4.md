# Phase 4 상세 기획서: 포트폴리오 최적화 + ML + 실시간 + 운영 자동화

> 작성일: 2026-02-18
> 상태: ✅ Phase 4 완료 (P0 8/8, P1 6/6, P2 3/3, 351 tests passing, 2 skipped)
> 선행 조건: Phase 3 완료 (253 tests, P0 10/10 구현)

---

## 목차

1. [Phase 4 목표 및 범위](#1-phase-4-목표-및-범위)
2. [리스크 패리티 포트폴리오 최적화](#2-리스크-패리티-포트폴리오-최적화)
3. [ML 팩터 모델](#3-ml-팩터-모델)
4. [WebSocket 실시간 시세](#4-websocket-실시간-시세)
5. [systemd 서비스 + 설치 스크립트](#5-systemd-서비스--설치-스크립트)
6. [이메일 알림](#6-이메일-알림)
7. [고급 리스크 지표](#7-고급-리스크-지표)
8. [Plotly 인터랙티브 차트](#8-plotly-인터랙티브-차트)
9. [일일 HTML 리포트 템플릿](#9-일일-html-리포트-템플릿)
10. [구현 우선순위 및 팀 배정](#10-구현-우선순위-및-팀-배정)

---

## 1. Phase 4 목표 및 범위

### 1.1 핵심 목표

| # | 목표 | 설명 |
|---|------|------|
| 1 | **포트폴리오 최적화** | 리스크 패리티(ERC) 비중 결정 |
| 2 | **ML 팩터 모델** | 기존 3팩터 스코어 기반 ML 종목 선정 |
| 3 | **실시간 데이터** | KIS WebSocket 실시간 시세 수신 |
| 4 | **운영 자동화** | systemd 배포 + 이메일 알림 |
| 5 | **시각화 고도화** | Plotly 인터랙티브 차트 + HTML 리포트 |

### 1.2 Phase 3 → Phase 4 차이

| 영역 | Phase 3 (현재) | Phase 4 (목표) |
|------|---------------|----------------|
| 비중 결정 | 동일 비중 | 리스크 패리티 최적화 |
| 종목 선정 | 팩터 스코어 상위 N개 | 팩터 스코어 + ML 예측 |
| 시세 | REST 폴링 only | REST + WebSocket 실시간 |
| 배포 | 수동 실행 | systemd 자동 서비스 |
| 알림 | 텔레그램 only | 텔레그램 + 이메일 |
| 차트 | matplotlib 정적 | + Plotly 인터랙티브 |
| 리스크 | MDD, Sharpe | + VaR, CVaR, RC |

---

## 2. 리스크 패리티 포트폴리오 최적화

### 2.1 ERC 핵심 개념

- RC_i = w_i * (Sigma * w)_i / sigma_p
- 목표: 모든 자산의 RC_i 동일화
- scipy.optimize.minimize(SLSQP) 사용

### 2.2 모듈

- `src/optimization/covariance.py` — CovarianceEstimator (sample/ledoit_wolf/ewm)
- `src/optimization/risk_parity.py` — RiskParityOptimizer (ERC, 역변동성 fallback)
- `src/strategy/risk_parity.py` — RiskParityStrategy(Strategy ABC)

---

## 3. ML 팩터 모델

### 3.1 피처 (~12-15개)

- 밸류: 1/PBR, 1/PER, 배당수익률
- 모멘텀: 12M/6M/3M return
- 퀄리티: ROE, GP/A, 부채비율, 발생액
- 기술적: 20D/60D 변동성, 거래량 비율
- 시장: log(시가총액)

### 3.2 모듈

- `src/ml/features.py` — 피처 엔지니어링
- `src/ml/pipeline.py` — MLPipeline (Ridge/RF/GBR, TimeSeriesSplit)
- `src/ml/evaluation.py` — IC, Rank IC, Long-Short Return
- `src/strategy/ml_factor.py` — MLFactorStrategy(Strategy ABC)

---

## 4. WebSocket 실시간 시세

- `src/execution/kis_websocket.py` — KISWebSocketClient (async)
- `src/execution/realtime_manager.py` — RealtimeManager
- KIS URL: ws://ops.koreainvestment.com:21000 (실전), :31000 (모의)
- 최대 41종목, PINGPONG heartbeat, 지수 백오프 재연결

---

## 5~9: 운영 자동화, 이메일, 리스크 지표, Plotly, HTML

- `scripts/quant-bot.service`, `install.sh`, `uninstall.sh`, `logrotate-quant.conf`
- `src/alert/email_sender.py` — EmailNotifier
- `src/report/risk_metrics.py` — VaR, CVaR, DR, factor_exposure
- `src/report/plotly_charts.py` — 인터랙티브 차트 5종
- `src/report/templates/daily.html` — 일일 HTML 리포트

---

## 10. 구현 우선순위

### P0 (필수): #1~#8
### P1 (중요): #9~#14
### P2 (보조): #15~#17

### 팀 배정

- **Dev Team A**: 최적화(#1-3), ML(#4-6), 리스크 지표(#14), DP+RP(#16), 팩터 노출(#17)
- **Dev Team B**: WebSocket(#7-8), systemd(#9-11), HTML(#12), 이메일(#13), Plotly(#15)
- **QA**: 전체 테스트 (~116개 신규, 목표 340+)

### 의존성 추가

```
scikit-learn>=1.3
plotly>=5.0
```
