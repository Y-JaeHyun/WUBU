# Quant Project

## Overview
퀀트 트레이딩/분석 프로젝트. 데이터 수집, 전략 개발, 백테스팅, 실행을 포함한다.

## Tech Stack
- Python 3.12
- pandas, numpy (데이터 처리)
- 증권사 API: 한국투자증권 KIS OpenAPI (REST + WebSocket) - 1순위
- 데이터 수집: KRX Open API (공식, AUTH_KEY 기반), pykrx (fallback), FinanceDataReader, DART OpenAPI
- 기술적 분석: ta
- 스케줄링: APScheduler
- 추가 라이브러리는 필요시 설치

## Project Structure
```
/mnt/data/quant/
├── CLAUDE.md          # 프로젝트 설명
├── .env               # 환경변수 (API 키 등)
├── src/               # 소스 코드
│   ├── data/          # 데이터 수집/처리 (collector, index_collector, krx_api, krx_provider)
│   ├── strategy/      # 트레이딩 전략 (value, momentum, quality, three_factor, dual_momentum, risk_parity, ml_factor, factor_combiner, multi_factor, market_timing)
│   ├── optimization/  # 포트폴리오 최적화 (covariance, risk_parity)
│   ├── ml/            # ML 팩터 모델 (features, pipeline)
│   ├── backtest/      # 백테스팅 엔진 (오버레이 지원)
│   ├── report/        # 리포트/차트 (backtest_report, charts, plotly_charts, metrics, risk_metrics, scanner, daily_report)
│   ├── alert/         # 알림 시스템 (telegram_bot, email_sender, conditions, alert_manager)
│   ├── execution/     # 실전 매매 (kis_client, kis_websocket, realtime_manager, order_manager, position_manager, executor, risk_guard)
│   ├── scheduler/     # 자동화 스케줄러 (holidays, main)
│   └── utils/         # 유틸리티
├── tests/             # 테스트 (351+ 테스트 케이스)
├── docs/              # 리서치/문서
└── data/              # 데이터 저장
```

## Conventions
- 코드는 Python, 변수/함수명은 snake_case
- 커밋 메시지는 한국어 또는 영어
- 타임존: KST (Asia/Seoul)
- 데이터 포맷: pandas DataFrame 기본

## Agent Teams
이 프로젝트는 CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS 기능을 활용하여 다수의 에이전트가 병렬로 작업한다.

### Team 1: Research (리서치팀)
- **역할**: 기술 조사, 전략 리서치, 시장 분석
- **담당 디렉토리**: `docs/`
- **주요 업무**:
  - 증권사 API 동향 및 업데이트 조사
  - 새로운 퀀트 전략/논문 리서치
  - 한국 시장 특성 분석 (규제 변화, 세제 변경 등)
  - 데이터 소스 발굴 및 평가

### Team 2: Development (개발팀)
- **역할**: 핵심 기능 구현
- **담당 디렉토리**: `src/`
- **주요 업무**:
  - `src/data/` - 시세 크롤링, 증권사 API 연동, 재무/공시 데이터 수집
  - `src/strategy/` - 퀀트 알고리즘(밸류/모멘텀/멀티팩터 등), 스코어링 시스템
  - `src/backtest/` - 백테스팅 엔진, 성과 분석
  - `src/utils/` - 주문 실행, 알림, 로깅, 설정 관리
- **개발 원칙**:
  - 모든 전략은 공통 인터페이스를 따름
  - 백테스트 가능한 구조로 설계
  - 증권사 API 추상화 레이어 유지

### Team 3: QA (테스트팀)
- **역할**: 품질 보증, 이슈 검증
- **담당 디렉토리**: `tests/`
- **주요 업무**:
  - 단위 테스트 작성 (pytest)
  - 백테스트 결과 검증 (성과 지표 정합성)
  - 주문 로직 시뮬레이션 테스트
  - 엣지 케이스 검증 (시장 휴일, 가격제한폭, 거래정지 등)
  - 데이터 파이프라인 무결성 검증

### Team 4: Planning (기획팀)
- **역할**: 기능 확장 기획, UX 개선
- **담당 디렉토리**: `docs/`, `notebooks/`
- **주요 업무**:
  - 대시보드/리포트 기능 기획
  - 포트폴리오 리밸런싱 자동화 설계
  - 알림/모니터링 시스템 기획
  - 리스크 관리 기능 설계
  - 사용자 인터페이스 개선안

### 워크플로우
```
Research -> Planning -> Development -> QA -> 배포
    |                       ^
    +---- 피드백 루프 ------+
```

### 팀 모니터링 (tmux 분할창 모드)
에이전트 팀을 병렬 실행할 때는 반드시 tmux 4분할 모니터링 세션을 생성한다.
- 세션명: `quant-teams`
- 실행: `bash scripts/monitor.sh` 또는 자동 생성
- 접속: `tmux attach -t quant-teams`
- 레이아웃:
  ```
  ┌──────────────┬──────────────┐
  │ Research     │ Development  │
  ├──────────────┼──────────────┤
  │ Planning     │ QA           │
  └──────────────┴──────────────┘
  ```
- 각 패널은 해당 팀 에이전트의 실시간 출력을 tail -f로 스트리밍
- 팀 에이전트 시작 시 자동으로 tmux 세션 생성 및 모니터 스크립트 갱신

## System Operations (운영 개요)

이 시스템은 로컬 머신에서 구동되는 개인용 퀀트 트레이딩 시스템이다.

### 일일 운영 스케줄
| 시간 | 작업 | 산출물 |
|------|------|--------|
| 07:00 | 대형 리서치 (전일 시장 분석, 글로벌 동향) | PDF/MD 리포트 |
| 09:00~15:30 | 전략별 매매 실행 (장중) | 주문 로그 |
| 매시 정각 | 뉴스 요약 (1시간 단위) | 사용자에게 요약 전달 |
| 15:30 이후 | 장 마감 후 EOD 리뷰 | 성과 스코어링 + 피드백 |
| 19:00 | 대형 리서치 (당일 결산, 내일 전략) | PDF/MD 리포트 |

### 핵심 운영 원칙
- **리서치**: 07:00/19:00 대형, 매시 소형 뉴스 요약
- **매매**: 전략별 시그널에 따라 실행, 수동 확인 가능
- **리뷰**: 장 마감 후 일일 성과 스코어링, 전략 피드백 루프
- **배포**: install script로 패키징 → 다른 머신에 재배포 가능

## Development Roadmap
- **Phase 1**: 인프라 + 밸류 팩터 (데이터 파이프라인, 백테스팅 엔진, 밸류 전략) ✅ 완료
- **Phase 2**: 모멘텀 + 마켓 타이밍 (모멘텀 팩터, 이동평균 오버레이, 리포트, 알림) ✅ 완료
- **Phase 3**: 멀티팩터 + 자산배분 + 실전매매 (퀄리티팩터, 3팩터모델, 듀얼모멘텀, KIS API, 스케줄러) ✅ 완료
- **Phase 4**: 고도화 (리스크 패리티, ML 팩터, WebSocket, systemd, 이메일, Plotly, VaR/CVaR) ✅ 완료
