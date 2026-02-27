# WUBU - 한국 주식시장 퀀트 트레이딩 시스템

개인용 한국 주식시장 퀀트 트레이딩 봇. 멀티팩터 전략 기반 자동 리밸런싱, 백테스트, 실시간 매매, Telegram 알림을 지원한다.

## 주요 기능

- **3-Pool 포트폴리오**: 장기(70%) + ETF 로테이션(30%) + 단기(0%) 동적 배분
- **16종 투자 전략**: 장기 11종 + 단기 4종 + ETF 로테이션
- **멀티팩터 전략**: 밸류 + 모멘텀 Z-Score 결합, 업종 비중 제한(25%) + 계열사 집중도(2종목) + 회전율 페널티
- **안전 필터**: 모멘텀 캡(300%), 업종 비중 상한, 계열사 탐지(하이브리드)
- **마켓 타이밍**: KOSPI vs MA200 기반 시장 진입/이탈 오버레이
- **ETF 로테이션**: 모멘텀 기반 상위 N개 ETF 월간 리밸런싱
- **백테스트 엔진**: diff-based 리밸런싱, 수수료/슬리피지 반영, 장기/단기 분리 엔진
- **한국투자증권 KIS API 연동**: 모의/실전 듀얼모드, WebSocket 실시간 시세
- **Telegram 실시간 알림**: 매매 시그널, 포트폴리오 현황, 커맨드 제어
- **긴급 모니터링**: 30분 간격 급등락(±5%)/시장급변(±3%)/공시 감지 + 자동매도 옵션
- **EOD 공시 교육**: 8개 카테고리별 영향 분석 + 투자 학습 팁 자동 제공
- **Feature Flag 시스템**: 11개 플래그, 런타임 토글 (재시작 없이 즉시 반영)
- **일일 시뮬레이션**: 매일 가상 리밸런싱으로 전략 검증
- **뉴스/공시 수집**: DART 공시, 매크로(ECOS/FRED) 데이터 수집
- **성과 DB**: SQLite 기반 NAV/포지션/거래 기록 추적
- **24시간 스케줄링**: 19개 작업 — 장전 리서치 + 장중 매매 + 장후 리뷰 + 야간 리서치

## 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.12 |
| 데이터 | pandas, numpy, pyarrow |
| 수집 | pykrx, FinanceDataReader, DART OpenAPI, yfinance |
| 기술분석 | ta (Technical Analysis Library) |
| ML | scikit-learn |
| 증권사 API | 한국투자증권 KIS OpenAPI (REST + WebSocket) |
| 스케줄링 | APScheduler |
| 시각화 | matplotlib, plotly |
| 알림 | Telegram Bot API |
| DB | SQLite (성과 기록) |
| 배포 | systemd |
| 테스트 | pytest (1,475+ 테스트) |

## 프로젝트 구조

```
src/
├── strategy/          # 투자 전략 (16종)
│   ├── value.py             # 밸류 팩터 (업종중립 + 주주환원 + F-Score)
│   ├── momentum.py          # 모멘텀 팩터 (잔차 모멘텀 + 52주 고점)
│   ├── quality.py           # 퀄리티 팩터 (ROE, 부채비율, strict accrual)
│   ├── multi_factor.py      # 멀티팩터 결합 (Z-Score + 업종/계열사 필터 + 회전율 페널티)
│   ├── conglomerate.py      # 계열사 탐지 (정적매핑 + 접두사 + 블랙리스트)
│   ├── three_factor.py      # 3팩터 모델
│   ├── market_timing.py     # KOSPI MA200 오버레이
│   ├── dual_momentum.py     # 듀얼 모멘텀 (변동성 타깃 + ETF 확대)
│   ├── risk_parity.py       # 리스크 패리티
│   ├── ml_factor.py         # ML 기반 팩터
│   ├── factor_combiner.py   # 팩터 결합기
│   ├── pead.py              # PEAD (실적 발표 후 드리프트)
│   ├── shareholder_yield.py # 주주환원수익률
│   ├── low_vol_quality.py   # 저변동성 + 퀄리티
│   ├── accrual.py           # 발생액 이상 전략
│   ├── etf_rotation.py      # ETF 모멘텀 로테이션
│   ├── short_term_base.py   # 단기 전략 베이스 (ATR 동적 손절)
│   ├── high_breakout.py     # 52주 신고가 돌파 (종가확인형)
│   ├── swing_reversion.py   # 스윙 평균회귀 (레짐 필터)
│   ├── bb_squeeze.py        # 볼린저 스퀴즈 (추세 필터)
│   └── orb_daytrading.py    # ORB 데이트레이딩
├── data/              # 데이터 수집/처리
│   ├── collector.py         # 주가 데이터 수집 (pykrx)
│   ├── index_collector.py   # 지수 데이터 수집
│   ├── dart_collector.py    # DART 재무제표 수집
│   ├── etf_collector.py     # ETF 데이터 수집
│   ├── global_collector.py  # 글로벌 시장 데이터 (S&P500, VIX)
│   ├── news_collector.py    # DART 공시/뉴스 수집
│   ├── macro_collector.py   # 매크로 데이터 (ECOS/FRED)
│   ├── daily_simulator.py   # 일일 리밸런싱 시뮬레이션
│   ├── performance_db.py    # 성과 DB (SQLite)
│   ├── cache.py             # parquet 데이터 캐싱
│   └── intraday_manager.py  # 장중 데이터 관리
├── execution/         # 실전 매매
│   ├── kis_client.py        # KIS API 클라이언트
│   ├── kis_websocket.py     # KIS WebSocket 실시간
│   ├── executor.py          # 주문 실행기
│   ├── order_manager.py     # 주문 관리
│   ├── position_manager.py  # 포지션 관리
│   ├── risk_guard.py        # 리스크 관리 (MDD, 일일한도)
│   ├── portfolio_allocator.py # 3-Pool 포트폴리오 배분
│   ├── short_term_trader.py # 단기 트레이딩 실행
│   └── short_term_risk.py   # 단기 리스크 관리
├── backtest/          # 백테스팅
│   ├── engine.py            # 장기 백테스트 엔진
│   ├── short_term_backtest.py # 단기 전략 백테스트
│   └── auto_runner.py       # 자동 백테스트 실행기
├── report/            # 리포트/분석
│   ├── daily_report.py      # 일일 리포트
│   ├── backtest_report.py   # 백테스트 리포트
│   ├── stock_reviewer.py    # 보유종목 리뷰
│   ├── night_research.py    # 야간 리서치
│   ├── risk_metrics.py      # VaR/CVaR 리스크
│   ├── portfolio_tracker.py # 포트폴리오 추적
│   ├── plotly_charts.py     # Plotly 인터랙티브 차트
│   └── scanner.py           # 종목 스캐너
├── optimization/      # 포트폴리오 최적화
│   ├── covariance.py        # 공분산 추정
│   └── risk_parity.py       # 리스크 패리티 최적화
├── ml/                # 머신러닝
│   ├── features.py          # 피처 엔지니어링
│   └── pipeline.py          # ML 파이프라인
├── alert/             # 알림 시스템
│   ├── telegram_bot.py      # Telegram 봇
│   ├── telegram_commander.py # Telegram 커맨드 처리
│   ├── alert_manager.py     # 알림 관리자
│   └── conditions.py        # 긴급 조건 (급등락/시장급변/공시)
├── scheduler/         # 자동 스케줄링
│   ├── main.py              # APScheduler 메인 (19개 작업)
│   └── holidays.py          # 한국 공휴일/휴장일
└── utils/             # 유틸리티
    ├── config.py            # 설정 관리
    ├── feature_flags.py     # Feature Flag 시스템 (11개)
    └── logger.py            # 로깅
```

## 설치 및 실행

### 1. 환경 설정

```bash
git clone https://github.com/Y-JaeHyun/WUBU.git
cd WUBU

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일에 실제 API 키를 입력
```

필수 환경변수:
- `KIS_APP_KEY`, `KIS_APP_SECRET`: 한국투자증권 API 키
- `KIS_TRADING_MODE`: `paper` (모의) / `live` (실전)
- `KIS_PAPER_ACCOUNT_NO` 또는 `KIS_REAL_ACCOUNT_NO`: 계좌번호
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`: Telegram 알림
- `DART_API_KEY`: DART 공시 데이터 수집 (선택)

### 3. 실행

```bash
# 직접 실행
python3 src/scheduler/main.py

# systemd 서비스로 배포
sudo bash install.sh
systemctl status quant-bot
```

### 4. 서비스 관리

```bash
systemctl start quant-bot     # 시작
systemctl stop quant-bot      # 중지
systemctl restart quant-bot   # 재시작
journalctl -u quant-bot -f    # 로그 확인
```

## 3-Pool 포트폴리오 아키텍처

```
총 자산
├── 장기 (70%) ── MultiFactor + MarketTiming → 월간 리밸런싱
├── ETF (30%) ── ETF Rotation (모멘텀) → 월간 리밸런싱
└── 단기 (0%)  ── HighBreakout/Swing → 장중 매매 (Feature Flag OFF)
```

- **PortfolioAllocator**: JSON 기반 3-pool 포지션 태깅 (`long_term`, `etf_rotation`, `short_term`)
- **소규모 자본 최적화**: 최소주문 7만원, 최대 10종목, ±20% 리밸런싱 밴드
- 배분 비율은 Feature Flag config로 동적 변경 가능

## 전략

### 장기 전략 (11종)

| 전략 | 설명 |
|------|------|
| **MultiFactor** | 밸류+모멘텀 Z-Score 결합, 업종 25%상한 + 계열사 2종목제한 + 회전율 페널티 (운영 기본) |
| Value | 업종중립 PBR/PER + 주주환원 + F-Score |
| Momentum | 잔차 모멘텀 + 52주 고점 거리 + 모멘텀 캡(300%) |
| Quality | ROE, 부채비율, strict accrual |
| ThreeFactor | V+M+Q 3팩터 결합 |
| DualMomentum | 변동성 타깃 + ETF 확대 |
| RiskParity | 리스크 기여도 균등 배분 |
| MLFactor | scikit-learn 기반 팩터 모델 |
| ShareholderYield | 배당+자사주매입+부채상환 |
| PEAD | 실적 발표 후 가격 드리프트 |
| LowVolQuality | 저변동성 + 퀄리티 결합 |
| Accrual | 발생액 이상(low accrual) |
| ETFRotation | 12M 모멘텀 기반 상위 3개 ETF 선택 + 모멘텀 캡(300%) (10개 유니버스) |

### 단기 전략 (4종)

| 전략 | 설명 |
|------|------|
| **HighBreakout** | 52주 신고가 돌파 종가확인형 (기본) |
| BBSqueeze | 볼린저밴드 스퀴즈 + 추세 필터 |
| SwingReversion | 과매도 반등 + 레짐 필터 |
| ORBDaytrading | 시가 범위 돌파 데이트레이딩 |

모든 단기 전략은 `ShortTermBase`를 상속하며, ATR 기반 동적 손절을 공유한다.

### 백테스트 결과 (2020~2025, 145만원)

**장기 전략 (Sharpe 순)**

| 전략 | 수익률 | CAGR | Sharpe | MDD |
|------|--------|------|--------|-----|
| MultiFactor+MT | 73.5% | 11.3% | 0.38 | -60.7% |
| MultiFactor(V+M) | 67.2% | 10.5% | 0.37 | -60.7% |
| Value(PBR) | 51.7% | 8.4% | 0.32 | -54.1% |
| RiskParity(MF) | 28.0% | 4.9% | 0.26 | -60.7% |
| ThreeFactor(V+M+Q) | 37.1% | 6.3% | 0.24 | -42.2% |
| ShareholderYield | 26.9% | 4.7% | 0.17 | -40.5% |

**단기 전략 (Sharpe 순, 시총 상위 100종목)**

| 전략 | 수익률 | CAGR | Sharpe | MDD | 거래수 |
|------|--------|------|--------|-----|--------|
| HighBreakout | +65.1% | 8.7% | 0.44 | -41.9% | 193 |
| BBSqueeze | -0.3% | -0.1% | 0.04 | -22.1% | 67 |
| SwingReversion | -5.4% | -0.9% | 0.04 | -35.9% | 155 |

**ETF 로테이션 (Sharpe 순, 10개 유니버스)**

| Lookback | N_ETFs | 수익률 | CAGR | Sharpe | MDD | 거래수 |
|----------|--------|--------|------|--------|-----|--------|
| 3M | 3 | +217.3% | 21.2% | 0.97 | -37.7% | 373 |
| 3M | 2 | +143.8% | 16.0% | 0.67 | -41.8% | 264 |
| 12M | 3 | +92.9% | 11.6% | 0.56 | -28.3% | 313 |
| 12M | 2 | +70.9% | 9.3% | 0.41 | -32.2% | 216 |

운영 설정: lookback=12M, n_select=3 (안정성 우선, MDD -28.3%)

## 운영 스케줄

| 시간 | 작업 | 설명 |
|------|------|------|
| 07:00 | 장전 브리핑 | 전일 시장 분석, 당일 전략 |
| 08:00 | 헬스체크 | 시스템 상태 확인 |
| 08:05 | 뉴스 수집 | DART 공시/뉴스 체크 |
| 08:50 | 시그널 계산 | 전략별 매매 시그널 생성 |
| 09:05 | 리밸런싱 실행 | 장기 포트폴리오 재조정 |
| 09:10 | ETF 로테이션 | ETF 월간 리밸런싱 |
| 09~15 | 장중 모니터링 | 보유종목 현황, MDD 추적 |
| 15:20 | 데이트레이딩 청산 | ORB 포지션 청산 |
| 09~15 | 긴급 모니터링 | 30분 간격 급등락/시장급변/공시 감지 |
| 15:35 | EOD 리뷰 | 당일 성과 요약 |
| 15:37 | 성과 DB 기록 | NAV/포지션/거래 SQLite 저장 |
| 15:40 | EOD 뉴스+교육 | 장후 공시 수집 + 투자 학습 콘텐츠 |
| 16:00 | 종목 리뷰 | 52주 고저 분석 |
| 16:05 | 일일 시뮬레이션 | 가상 리밸런싱 검증 |
| 19:00 | 이브닝 리포트 | 포트폴리오 성과, 시장 동향 |
| 22:00 | 야간 리서치 | 글로벌 동향, 시사점 |

## Telegram 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/status` | 봇 상태 확인 |
| `/portfolio` | 포트폴리오 현황 |
| `/balance` | 잔고 + 풀 배분 + ETF 리밸런싱 프리뷰 |
| `/flags` | Feature Flag 상태 |
| `/toggle <flag>` | 피처 ON/OFF 토글 |
| `/config <flag> <key> <value>` | 피처 설정 변경 |
| `/help` | 도움말 |

## Feature Flags

런타임에 기능을 켜고 끌 수 있는 Feature Flag 시스템. Telegram `/toggle` 명령으로 제어.

| Flag | 기본값 | 설명 |
|------|--------|------|
| `data_cache` | ON | pykrx API 응답 parquet 캐싱 |
| `global_monitor` | OFF | 글로벌 시장 모니터 (S&P500, VIX) |
| `stock_review` | ON | 보유종목 52주 고저 리뷰 |
| `auto_backtest` | OFF | 주간 자동 백테스트 |
| `night_research` | OFF | 야간 글로벌 동향 리서치 |
| `short_term_trading` | OFF | 단기 트레이딩 (HighBreakout) |
| `daily_simulation` | ON | 일일 리밸런싱 시뮬레이션 |
| `news_collector` | ON | DART 공시/뉴스 수집 |
| `macro_monitor` | ON | 매크로 데이터 (ECOS/FRED) |
| `etf_rotation` | ON | ETF 로테이션 전략 |
| `emergency_monitor` | ON | 긴급 모니터링 (급등락/시장급변/공시) |

## 테스트

```bash
# 전체 테스트 실행
python3 -m pytest tests/ -v

# 특정 모듈 테스트
python3 -m pytest tests/test_multi_factor.py -v
python3 -m pytest tests/test_etf_rotation.py -v

# 커버리지
python3 -m pytest tests/ --cov=src --cov-report=html
```

## 개발 이력

| Phase | 내용 | 상태 |
|-------|------|------|
| Phase 1 | 인프라 + 밸류 팩터, 데이터 파이프라인, 백테스트 엔진 | ✅ |
| Phase 2 | 모멘텀 + 마켓 타이밍, 리포트, Telegram 알림 | ✅ |
| Phase 3 | 멀티팩터 + 자산배분 + KIS API 실전매매 | ✅ |
| Phase 4 | 리스크 패리티, ML 팩터, WebSocket, systemd 배포 | ✅ |
| Phase 5-A | Feature Flag, 데이터 캐시, 24시간 스케줄링 | ✅ |
| Phase 5-B | 듀얼 포트폴리오, 단기 전략 4종, 백테스트 검증 | ✅ |
| Phase 6 | 전략 고도화 + 신규 전략 5종 + 인프라 확장 (뉴스/매크로/성과DB/시뮬레이션) | ✅ |
| Phase 6-B | 안전 필터: 모멘텀 캡, 업종 비중 제한, 계열사 집중도, 섹터 데이터 수집 | ✅ |
| Phase 6-C | 긴급 모니터링, EOD 공시 교육, 리밸런싱 시뮬레이션, 백테스트 diff-based 개선 | ✅ |

## Disclaimer

이 프로젝트는 개인 학습 및 연구 목적으로 제작되었습니다. 투자 권유가 아니며, 이 시스템을 이용한 투자 결과에 대해 어떠한 책임도 지지 않습니다. 실전 투자 시 반드시 본인의 판단과 책임 하에 진행하시기 바랍니다.

## License

MIT
