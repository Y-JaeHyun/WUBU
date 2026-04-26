# WUBU 개발 가이드 (DEV.md)

## 개요

이 문서는 `/mnt/data/quant-dev`에서 WUBU 퀀트 시스템을 개발하고 테스트하는 방법을 설명합니다.

## 환경 설정

### 1. 가상환경 활성화

```bash
cd /mnt/data/quant-dev
source /mnt/data/quant/.venv/bin/activate
```

**주의**: 운영 환경(`/mnt/data/quant`)의 venv를 공유합니다. 이는 prod와 dev의 의존성을 항상 동기화된 상태로 유지합니다.

### 1b. 의존성 관리 (Dependency Management)

**파일 구조**:
- `requirements.txt`: 개발용 (범위 지정: `pandas>=2.0`)
- `requirements-prod.txt`: 운영용 (정확한 버전: `pandas==2.3.3`)

**운영 환경 재구성**:
```bash
# 정확한 버전으로 재설치 (운영에서만, 분기당 1회)
pip install -r requirements-prod.txt --force-reinstall
```

**업데이트 주기**:
- 정상: 분기당 1회 (3개월) 보안 패치 반영
- 긴급: 보안 취약점 발견 시 즉시
- 검증: 항상 dev에서 백테스트 후 prod에 적용

### 2. 환경변수 설정

`.env` 파일에 API 키를 설정합니다:

```bash
# KIS API (한국투자증권)
KIS_API_KEY=...
KIS_API_SECRET=...
KIS_ACCOUNT=...

# 텔레그램 알림
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# 선택: 한국은행 ECOS, FRED(미 경제)
ECOS_API_KEY=...
FRED_API_KEY=...
```

## 개발과 운영의 차이점

| 항목 | Dev | Prod |
|------|-----|------|
| 디렉토리 | `/mnt/data/quant-dev` | `/mnt/data/quant` |
| Git 브랜치 | 자유로운 feature 브랜치 | `main` 고정 |
| 데이터 | 독립적 캐시 (`data/simulation/`, `analysis/`) | 운영 캐시 |
| 매매 | 시뮬레이션/백테스트만 가능 (DRY RUN) | 실전 매매 (KIS API) |
| 스케줄러 | 수동 실행 또는 테스트 스케줄 | systemd (24/7 운영) |
| 의존성 | `/mnt/data/quant/.venv` (공유) | 동일 |

## 사용 흐름

### 1. 새로운 전략/기능 개발

```bash
# 1. 브랜치 생성
git checkout -b feature/new-strategy

# 2. 코드 작성
# - src/strategy/ : 전략 클래스
# - tests/ : 단위 테스트
# - docs/ : 설계서/리서치

# 3. 로컬 백테스트
python3 src/backtest/backtest_engine.py \
  --strategy value \
  --start 20220101 \
  --end 20260426

# 4. 테스트 실행
python3 -m pytest tests/ -q --tb=short
```

### 2. 시뮬레이션 실행 (일일 신호 검증)

```bash
# 수동 시뮬레이션 (당일 데이터로 신호 생성)
python3 -m src.data.daily_simulator
```

산출물: `data/simulation/{date}/signals.json`, `{strategy}_positions.json`

### 3. 프로덕션 배포

```bash
# 1. PR 리뷰 + CEOapproval 완료
# 2. main 브랜치에 merge
git checkout main
git pull origin main

# 3. 운영 환경으로 복제 (READ-ONLY)
# → 별도 배포 스크립트 필요 (현재 미구현)

# 4. systemd 재시작 (CEO 승인 필수)
service quant-bot restart
```

## 테스트

### 단위 테스트 (필수)

```bash
# 전체 테스트
python3 -m pytest tests/ -q --tb=short

# 특정 모듈만
python3 -m pytest tests/test_strategy.py -v

# 커버리지
python3 -m pytest tests/ --cov=src
```

**현황**: 351개 테스트, 모두 패스 (2개 skip)

### 백테스트 (선택)

```bash
# 배치 백테스트 (모든 전략 비교)
python3 scripts/batch_backtest.py
```

산출물: `analysis/backtest/detailed_comparison_report.md`

## 코드 구조

```
src/
├── data/              # 데이터 수집/처리
│   ├── collector.py   # KOSPI/KOSDAQ 종목 데이터
│   ├── daily_simulator.py  # 일일 신호 생성 (시뮬레이션)
│   └── krx_provider.py     # KRX Open API / pykrx
├── strategy/          # 트레이딩 전략
│   ├── quality.py     # 퀄리티 팩터 (운영 중)
│   ├── value.py       # 밸류 팩터
│   ├── momentum.py    # 모멘텀 팩터
│   └── ...
├── backtest/          # 백테스팅 엔진
│   └── engine.py      # 성과 평가
├── execution/         # 매매 실행 (DRY RUN만 가능)
│   ├── kis_client.py  # KIS API 클라이언트
│   └── risk_guard.py  # 리스크 검증
└── report/            # 리포트/분석
```

## 안전 규칙 (Critical)

### ❌ 하면 안 되는 것

1. **실전 매매 금지**
   - `execute_order()` 호출 금지
   - 항상 DRY RUN으로 검증

2. **운영 환경 직접 수정 금지**
   - `/mnt/data/quant` 읽기만 가능
   - 배포는 PR + merge로만 가능

3. **systemd 임의 재시작 금지**
   - `service quant-bot restart` 는 CEO 승인 필수

4. **API 키 노출 금지**
   - `.env` 는 git 제외됨 (`.gitignore`)
   - 외부 파일/로그에 기록 금지

### ✅ 권장 프랙티스

1. **격일 데이터 갱신**
   ```bash
   python3 -c "from src.data.collector import Collector; Collector().refresh_all()"
   ```

2. **시뮬레이션 먼저, 백테스트 나중**
   - 일일 신호 → 시뮬레이션 → 모의 거래 → 백테스트

3. **변경사항은 브랜치로**
   - main은 항상 안정적
   - feature/* 에서 개발 후 PR

4. **PR 전 자동 검증**
   - `pytest` 통과
   - `pylint`/`black` (선택)
   - 백테스트 성과 확인

## 트러블슈팅

### 의존성 충돌

```bash
# venv 재설치 (공유 venv 사용 중)
cd /mnt/data/quant
source .venv/bin/activate
pip install -r requirements.txt --upgrade
```

### 데이터 캐시 초기화

```bash
rm -rf data/simulation/
rm -rf data/.cache/
python3 -m src.data.daily_simulator
```

### 테스트 실패

```bash
# 의존성 재확인
python3 -m pytest tests/ --tb=long -v

# 특정 테스트만 디버깅
python3 -m pytest tests/test_strategy.py::TestQuality::test_signal_generation -vv
```

## 유용한 커맨드

```bash
# 현재 branch 상태
git status

# 최근 커밋 확인
git log --oneline -10

# 운영과의 차이점 확인
git diff origin/main...HEAD

# 시뮬레이션 결과 보기
cat data/simulation/$(date +%Y-%m-%d)/signals.json | jq .

# 로그 확인
tail -f logs/quant.log
```

## 배포 체크리스트

배포 전 반드시 확인:

- [ ] 브랜치: `feature/xxx` 생성
- [ ] 테스트: `pytest` 전부 통과
- [ ] 백테스트: 성과 지표 검증
- [ ] 코드 리뷰: 팀원/CTO 리뷰 완료
- [ ] PR 생성 및 CEO 승인 획득
- [ ] main으로 merge
- [ ] systemd 재시작 (CEO 사전 승인)

## 참고 문서

- [README.md](README.md) — 프로젝트 개요
- [CLAUDE.md](CLAUDE.md) — 기술 스택, 팀 구성
- 전략 리서치: `docs/research/`
- 성과 리포트: `analysis/backtest/`
