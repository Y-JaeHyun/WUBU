# WUBU - 한국 주식시장 퀀트 트레이딩 시스템

개인용 한국 주식시장 퀀트 트레이딩 봇. 멀티팩터 전략 기반 자동 리밸런싱, 백테스트, 실시간 매매, Telegram 알림을 지원한다.

## 주요 기능

- **멀티팩터 전략**: 밸류 + 모멘텀 Z-Score 결합, 상위 10종목 동일비중 투자
- **마켓 타이밍**: KOSPI vs MA200 기반 시장 진입/이탈 오버레이
- **월간 자동 리밸런싱**: 매월 첫 거래일 포트폴리오 재조정
- **백테스트 엔진**: 수수료·슬리피지 반영, 마켓타이밍 오버레이 지원
- **한국투자증권 KIS API 연동**: 모의/실전 듀얼모드, WebSocket 실시간 시세
- **Telegram 실시간 알림**: 매매 시그널, 포트폴리오 현황, 커맨드 제어
- **Feature Flag 시스템**: 런타임 기능 토글 (재시작 없이 즉시 반영)
- **단기 전략 백테스트**: SwingReversion, HighBreakout, BBSqueeze 등 4종 비교
- **24시간 스케줄링**: 장중 매매 + 장후 리뷰 + 야간 리서치 자동화

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
| 배포 | systemd |
| 테스트 | pytest (1,000+ 테스트) |

## 프로젝트 구조

```
src/
├── strategy/          # 투자 전략
│   ├── value.py             # 밸류 팩터 (1/PBR + 1/PER)
│   ├── momentum.py          # 모멘텀 팩터 (12M skip 1M)
│   ├── quality.py           # 퀄리티 팩터 (ROE, 부채비율)
│   ├── multi_factor.py      # 멀티팩터 결합 (Z-Score)
│   ├── three_factor.py      # 3팩터 모델
│   ├── market_timing.py     # KOSPI MA200 오버레이
│   ├── dual_momentum.py     # 듀얼 모멘텀
│   ├── risk_parity.py       # 리스크 패리티
│   ├── ml_factor.py         # ML 기반 팩터
│   ├── factor_combiner.py   # 팩터 결합기
│   ├── swing_reversion.py   # 스윙 평균회귀
│   ├── high_breakout.py     # 52주 신고가 돌파
│   ├── bb_squeeze.py        # 볼린저 스퀴즈
│   └── orb_daytrading.py    # ORB 데이트레이딩
├── data/              # 데이터 수집/처리
│   ├── collector.py         # 주가 데이터 수집 (pykrx)
│   ├── index_collector.py   # 지수 데이터 수집
│   ├── dart_collector.py    # DART 재무제표 수집
│   ├── etf_collector.py     # ETF 데이터 수집
│   ├── global_collector.py  # 글로벌 시장 데이터 (S&P500, VIX)
│   ├── cache.py             # parquet 데이터 캐싱
│   └── intraday_manager.py  # 장중 데이터 관리
├── execution/         # 실전 매매
│   ├── kis_client.py        # KIS API 클라이언트
│   ├── kis_websocket.py     # KIS WebSocket 실시간
│   ├── executor.py          # 주문 실행기
│   ├── order_manager.py     # 주문 관리
│   ├── position_manager.py  # 포지션 관리
│   ├── risk_guard.py        # 리스크 관리 (MDD, 일일한도)
│   ├── portfolio_allocator.py # 듀얼 포트폴리오 배분
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
│   └── alert_manager.py     # 알림 관리자
├── scheduler/         # 자동 스케줄링
│   ├── main.py              # APScheduler 메인
│   └── holidays.py          # 한국 공휴일/휴장일
└── utils/             # 유틸리티
    ├── config.py            # 설정 관리
    ├── feature_flags.py     # Feature Flag 시스템
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

# 추가 패키지 (pykrx, yfinance 등)
pip install pykrx yfinance pyarrow
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

## 운영 스케줄

| 시간 | 작업 | 설명 |
|------|------|------|
| 07:00 | 장전 브리핑 | 전일 시장 분석, 당일 전략 |
| 08:50 | 장 시작 준비 | 데이터 수집, 시그널 계산 |
| 09:01 | 리밸런싱 실행 | 매월 첫 거래일 포트폴리오 재조정 |
| 매시 정각 | 시장 모니터링 | 보유종목 현황, MDD 추적 |
| 15:40 | 장 마감 리뷰 | 당일 성과 요약 |
| 16:00 | 종목 리뷰 | 52주 고저 분석 |
| 19:00 | 일일 리포트 | 포트폴리오 성과, 시장 동향 |

## 전략

### 운영 전략: MultiFactor (장기)

밸류(1/PBR + 1/PER) 50% + 모멘텀(12개월, skip 1개월) 50%을 Z-Score로 결합하여 상위 10종목에 동일비중 투자. KOSPI MA200 마켓타이밍 오버레이 적용.

### 단기 전략 백테스트 결과 (2020~2025, 시총 상위 100종목)

| 전략 | 총수익률 | Sharpe | MDD | 거래수 |
|------|---------|--------|-----|--------|
| BBSqueeze | +3.37% | 0.11 | -18.23% | 52 |
| HighBreakout | -7.83% | 0.06 | -37.70% | 181 |
| SwingReversion | -28.49% | -0.20 | -53.79% | 167 |
| SwingReversion+OBV | -18.07% | -0.32 | -25.46% | 63 |

## 테스트

```bash
# 전체 테스트 실행
python3 -m pytest tests/ -v

# 특정 모듈 테스트
python3 -m pytest tests/test_multi_factor.py -v
python3 -m pytest tests/test_bb_squeeze.py -v

# 커버리지
python3 -m pytest tests/ --cov=src --cov-report=html
```

## Telegram 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/status` | 봇 상태 확인 |
| `/portfolio` | 포트폴리오 현황 |
| `/balance` | 계좌 잔고 조회 |
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
| `short_term_trading` | OFF | 단기 트레이딩 (BBSqueeze) |

## 개발 이력

| Phase | 내용 |
|-------|------|
| Phase 1 | 인프라 + 밸류 팩터, 데이터 파이프라인, 백테스트 엔진 |
| Phase 2 | 모멘텀 + 마켓 타이밍, 리포트, Telegram 알림 |
| Phase 3 | 멀티팩터 + 자산배분 + KIS API 실전매매 |
| Phase 4 | 리스크 패리티, ML 팩터, WebSocket, systemd 배포 |
| Phase 5-A | Feature Flag, 데이터 캐시, 24시간 스케줄링 |
| Phase 5-B | 듀얼 포트폴리오, 단기 전략 4종, 백테스트 검증 |

## Disclaimer

이 프로젝트는 개인 학습 및 연구 목적으로 제작되었습니다. 투자 권유가 아니며, 이 시스템을 이용한 투자 결과에 대해 어떠한 책임도 지지 않습니다. 실전 투자 시 반드시 본인의 판단과 책임 하에 진행하시기 바랍니다.

## License

MIT
