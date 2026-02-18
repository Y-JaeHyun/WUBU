# Phase 3 상세 기획서: 실전 매매 시스템 + 자산배분

> 작성일: 2026-02-18
> 상태: ✅ Phase 3 완료 (P0 10/10 구현, 253 tests passing, 2 skipped)
> 선행 조건: Phase 2 (모멘텀 + 마켓 타이밍 + 리포트 + 알림) 완료 (168 tests)

---

## 목차

1. [Phase 3 목표 및 범위](#1-phase-3-목표-및-범위)
2. [퀄리티 팩터 전략](#2-퀄리티-팩터-전략)
3. [3팩터 모델 통합](#3-3팩터-모델-통합)
4. [듀얼 모멘텀 자산배분](#4-듀얼-모멘텀-자산배분)
5. [KIS OpenAPI 실전 매매](#5-kis-openapi-실전-매매)
6. [스케줄러 자동화](#6-스케줄러-자동화)
7. [Phase 2 잔여 항목 (P2)](#7-phase-2-잔여-항목-p2)
8. [구현 우선순위 및 팀 배정](#8-구현-우선순위-및-팀-배정)

---

## 1. Phase 3 목표 및 범위

### 1.1 핵심 목표

| # | 목표 | 설명 |
|---|------|------|
| 1 | **3팩터 모델** | 밸류 + 모멘텀 + 퀄리티 3팩터 결합 전략 |
| 2 | **듀얼 모멘텀** | ETF 기반 자산배분 (절대 + 상대 모멘텀) |
| 3 | **실전 매매** | KIS OpenAPI 연동 → 실제 주문 가능 |
| 4 | **자동화** | APScheduler + systemd로 무인 운영 |

### 1.2 Phase 3 이후 시스템 모습

```
┌─────────────────────────────────────────────────┐
│              Quant Trading System v3             │
├──────────────┬──────────────────────────────────┤
│ Strategies   │ Value, Momentum, Quality,        │
│              │ MultiFactor(3F), DualMomentum     │
├──────────────┼──────────────────────────────────┤
│ Data         │ pykrx, FDR, DART, KIS(실시간)     │
├──────────────┼──────────────────────────────────┤
│ Execution    │ KIS OpenAPI (REST + WebSocket)    │
├──────────────┼──────────────────────────────────┤
│ Automation   │ APScheduler (07:00~19:00 스케줄)  │
├──────────────┼──────────────────────────────────┤
│ Monitoring   │ Telegram + HTML Reports + Alerts  │
└──────────────┴──────────────────────────────────┘
```

### 1.3 현재 시스템과의 차이

| 영역 | Phase 2 (현재) | Phase 3 (목표) |
|------|---------------|----------------|
| 팩터 | 2팩터 (밸류+모멘텀) | 3팩터 (+퀄리티) |
| 자산 | 개별주식만 | 개별주식 + ETF |
| 매매 | 백테스트만 | 실전 매매 가능 |
| 운영 | 수동 실행 | 자동 스케줄링 |
| 팩터결합 | 2팩터 고정 | N팩터 확장 |

---

## 2. 퀄리티 팩터 전략

### 2.1 전략 개요

퀄리티 팩터는 수익성이 높고, 재무 건전성이 좋은 기업이 장기적으로 초과 수익을 올린다는 실증 근거(Novy-Marx 2013, Asness et al. 2019)에 기반한다. 한국 시장에서도 수익성 팩터의 유효성이 확인되었다.

### 2.2 퀄리티 스코어 구성요소

| 지표 | 산출 | 방향 | 가중치 | 데이터 소스 |
|------|------|------|--------|------------|
| **ROE** | 당기순이익 / 자기자본 | 높을수록 좋음 | 0.30 | DART |
| **GP/A** | 매출총이익 / 총자산 | 높을수록 좋음 | 0.30 | DART |
| **부채비율** | 총부채 / 자기자본 | 낮을수록 좋음 | 0.20 | DART |
| **발생액** | (순이익 - 영업CF) / 총자산 | 낮을수록 좋음 | 0.20 | DART |

### 2.3 퀄리티 스코어 산출

```python
class QualityStrategy(Strategy):
    """퀄리티 팩터 전략."""

    def calculate_quality_score(self, fundamentals: pd.DataFrame) -> pd.Series:
        """
        퀄리티 복합 스코어 산출.

        1. 각 지표의 cross-sectional 퍼센타일 순위 계산
        2. 부채비율/발생액은 역순위 (낮을수록 높은 점수)
        3. 가중 합산 → 복합 퀄리티 스코어
        """
        roe_rank = fundamentals["roe"].rank(pct=True)
        gpa_rank = fundamentals["gp_over_assets"].rank(pct=True)
        debt_rank = 1 - fundamentals["debt_ratio"].rank(pct=True)   # 역순위
        accrual_rank = 1 - fundamentals["accruals"].rank(pct=True)  # 역순위

        score = (
            0.30 * roe_rank
            + 0.30 * gpa_rank
            + 0.20 * debt_rank
            + 0.20 * accrual_rank
        )
        return score
```

### 2.4 데이터 요구사항

| 데이터 | 현재 수집 여부 | Phase 3 조치 |
|--------|-------------|-------------|
| ROE | ❌ | DART 재무제표에서 수집 |
| 매출총이익 | ❌ | DART 손익계산서에서 수집 |
| 총자산 | ❌ | DART 재무상태표에서 수집 |
| 총부채 | ❌ | DART 재무상태표에서 수집 |
| 영업현금흐름 | ❌ | DART 현금흐름표에서 수집 |
| 자기자본 | ❌ | DART 재무상태표에서 수집 |

→ **DART OpenAPI 재무제표 수집 모듈 신규 개발 필요** (`src/data/dart_collector.py`)

### 2.5 유니버스 필터링

Phase 2 밸류/모멘텀 필터에 추가:
- 적자 기업 제외 (ROE < 0, 3년 연속)
- 자본잠식 기업 제외
- 감사의견 '적정' 외 제외
- 재무제표 데이터 2년 이상 확보 종목만

### 2.6 리밸런싱

- **주기**: 분기 (3/6/9/12월 재무제표 공시 후)
- 재무데이터는 분기별 업데이트, 가격 데이터는 일별
- 분기 재무 공시 지연 고려 (공시 후 1~2주 대기)

### 2.7 백테스트 파라미터

| 파라미터 | 기본값 | 범위 |
|---------|--------|------|
| `roe_weight` | `0.30` | 0.1~0.5 |
| `gpa_weight` | `0.30` | 0.1~0.5 |
| `debt_weight` | `0.20` | 0.1~0.3 |
| `accrual_weight` | `0.20` | 0.1~0.3 |
| `exclude_loss_years` | `3` | 1~5 |
| `n_stocks` | `20` | 10~50 |

---

## 3. 3팩터 모델 통합

### 3.1 팩터 결합기 확장 (N팩터)

현재 `FactorCombiner`는 2팩터(밸류+모멘텀)만 지원한다. Phase 3에서 N팩터를 지원하도록 확장한다.

```python
class FactorCombiner:
    """N개 팩터의 결합 스코어 산출."""

    def combine_zscore(
        self,
        factors: dict[str, pd.Series],
        weights: dict[str, float] | None = None,
    ) -> pd.Series:
        """
        N개 팩터를 Z-Score 정규화 후 가중 합산.

        Args:
            factors: {"value": series, "momentum": series, "quality": series, ...}
            weights: {"value": 0.4, "momentum": 0.3, "quality": 0.3}
                     None이면 동일 가중
        """
        ...

    def combine_rank(
        self,
        factors: dict[str, pd.Series],
        weights: dict[str, float] | None = None,
    ) -> pd.Series:
        """N개 팩터를 순위 퍼센타일 가중 합산."""
        ...
```

### 3.2 기존 2팩터 → 3팩터 호환

- 기존 `combine_zscore(value, momentum)` 시그니처는 유지 (하위 호환)
- 새로운 dict 기반 API를 **추가**
- `MultiFactorStrategy`를 `ThreeFactorStrategy`로 확장

### 3.3 3팩터 결합 가중치

| 설정 | 밸류 | 모멘텀 | 퀄리티 | 비고 |
|------|------|--------|--------|------|
| 동일 가중 (기본) | 0.33 | 0.33 | 0.34 | 균형 |
| 퀄리티 강조 | 0.30 | 0.30 | 0.40 | 안정성 추구 |
| 모멘텀 약화 | 0.40 | 0.20 | 0.40 | 한국 시장 음의 모멘텀 고려 |

> **한국 시장 참고**: 학술 연구에 따르면 한국 시장에서 모멘텀 팩터가 특정 구간에서 음(-)의 프리미엄을 보이는 경우가 있다. 따라서 모멘텀 가중치를 상대적으로 낮추는 것도 고려한다.

### 3.4 포트폴리오 구성

| 방식 | 설명 | 장점 |
|------|------|------|
| **통합 스코어** (기본) | 3팩터 합산 스코어 상위 N개 | 심플, 분산 효과 |
| **교집합** | 각 팩터 상위 30% 교집합 | 더 엄격한 필터링 |
| **섹터 중립** | 섹터별 상위 N개 선정 | 섹터 편향 방지 |

### 3.5 성과 비교 프레임워크

Phase 3에서 비교할 전략 (Phase 2의 8개 + 신규):

| # | 전략 | 마켓 타이밍 |
|---|------|-----------|
| 1~8 | Phase 2 전략들 (유지) | ±타이밍 |
| 9 | 퀄리티 단독 | OFF |
| 10 | 퀄리티 단독 | ON |
| 11 | 3팩터 (V+M+Q) Z-Score | OFF |
| 12 | 3팩터 (V+M+Q) Z-Score | ON |
| 13 | 3팩터 (V+M+Q) 순위 합산 | OFF |
| 14 | 3팩터 (V+M+Q) 순위 합산 | ON |
| 15 | 듀얼 모멘텀 (ETF) | 내장 |

---

## 4. 듀얼 모멘텀 자산배분

### 4.1 전략 개요

Gary Antonacci(2014)의 듀얼 모멘텀은 **상대 모멘텀**(자산 간 비교)과 **절대 모멘텀**(0 초과 여부)을 결합하여 하락장 방어와 상승장 참여를 동시에 추구한다.

### 4.2 GEM 한국판 (Global Equity Momentum)

```
매월 첫 거래일:
    1. 12개월 수익률 계산
       - 국내주식: KODEX 200
       - 해외주식: TIGER 미국S&P500
    2. 상대 모멘텀: 두 자산 중 높은 쪽 선택
    3. 절대 모멘텀: 선택된 자산의 12개월 수익률 > 0?
       - YES → 해당 자산에 100% 투자
       - NO  → KODEX 단기채권PLUS로 이동 (안전자산)
```

### 4.3 ETF 유니버스

#### 위험자산 (Risky Assets)

| 자산군 | ETF | 종목코드 | 비고 |
|--------|-----|---------|------|
| 국내 대형주 | KODEX 200 | 069500 | KOSPI 200 추종 |
| 미국 대형주 | TIGER 미국S&P500 | 360750 | S&P 500 추종 |
| 미국 기술주 | TIGER 미국나스닥100 | 133690 | Nasdaq 100 추종 |

#### 안전자산 (Safe Haven)

| 자산군 | ETF | 종목코드 | 비고 |
|--------|-----|---------|------|
| 단기채 | KODEX 단기채권PLUS | 214980 | 국내 단기채 |
| 국채 | KODEX 국고채3년 | 114820 | 3년물 국채 |
| 현금성 | TIGER 단기통안채 | 157450 | 통안채 |

### 4.4 확장: 멀티 애셋 듀얼 모멘텀

단순 2자산(국내/해외) 외에 4자산 모델:

```
자산군:
  A. 국내주식 (KODEX 200)
  B. 해외주식 (TIGER 미국S&P500)
  C. 국내채권 (KODEX 국고채10년)
  D. 원자재  (KODEX 골드선물)

매월 첫 거래일:
  1. 각 자산의 12개월 수익률 순위
  2. 상위 2개 선택 (상대 모멘텀)
  3. 각각 절대 모멘텀 검증
     - 통과: 50%씩 배분
     - 미통과: 해당 비중을 단기채로 이동
```

### 4.5 데이터 요구사항

| 데이터 | 조치 |
|--------|------|
| ETF 일별 가격 | pykrx/FDR로 수집 (개별주식과 동일) |
| ETF 분배금 | pykrx 배당 데이터 또는 NAV 추적 |
| 무위험 수익률 | CD91일물 금리 → 한국은행 ECOS API |

### 4.6 모듈 설계

```python
# src/strategy/dual_momentum.py

class DualMomentumStrategy:
    """ETF 기반 듀얼 모멘텀 자산배분."""

    def __init__(
        self,
        risky_assets: dict[str, str],   # {"국내주식": "069500", "해외주식": "360750"}
        safe_asset: str = "214980",      # KODEX 단기채권PLUS
        lookback_months: int = 12,
        n_select: int = 1,               # 상대 모멘텀 선택 자산 수
    ):
        ...

    def calculate_momentum(self, prices: dict[str, pd.Series]) -> dict[str, float]:
        """각 자산의 12개월 수익률 산출."""

    def get_relative_signal(self, momentum: dict[str, float]) -> list[str]:
        """상대 모멘텀: 상위 N개 자산 선택."""

    def get_absolute_signal(self, momentum: dict[str, float], threshold: float = 0.0) -> dict[str, bool]:
        """절대 모멘텀: 수익률 > threshold 여부."""

    def generate_allocation(self, prices: dict[str, pd.Series]) -> dict[str, float]:
        """최종 자산배분 비중 산출."""
```

### 4.7 백테스트 파라미터

| 파라미터 | 기본값 | 범위 |
|---------|--------|------|
| `lookback_months` | `12` | 3, 6, 9, 12 |
| `n_select` | `1` | 1, 2 |
| `rebalance_freq` | `"monthly"` | monthly, quarterly |
| `safe_asset` | `"단기채"` | 단기채, 국채3년, 현금 |
| `absolute_threshold` | `0.0` | 0.0, 0.01, 0.02 |

---

## 5. KIS OpenAPI 실전 매매

### 5.1 개요

한국투자증권 OpenAPI를 통한 실전 주문 실행 시스템. Phase 1~2의 백테스트 시그널을 실제 주문으로 전환한다.

### 5.2 아키텍처

```
┌──────────────┐    시그널     ┌──────────────┐   REST/WS   ┌──────────────┐
│   Strategy   │ ──────────> │   Executor   │ ─────────> │   KIS API    │
│   Engine     │             │              │ <───────── │   Server     │
└──────────────┘             │  - 주문 관리   │  체결 통보   └──────────────┘
                              │  - 슬리피지 제어│
                              │  - 리스크 체크  │
                              └──────────────┘
```

### 5.3 모듈 설계

```
src/execution/
    __init__.py
    kis_client.py         # KIS API 클라이언트 (인증, 요청)
    order_manager.py      # 주문 생성/수정/취소 관리
    position_manager.py   # 포지션 조회/동기화
    executor.py           # 전략 시그널 → 실제 주문 변환
    risk_guard.py         # 사전 리스크 체크 (주문 전 검증)
```

### 5.4 KIS API 클라이언트

```python
# src/execution/kis_client.py

class KISClient:
    """한국투자증권 OpenAPI 클라이언트."""

    BASE_URL = "https://openapi.koreainvestment.com:9443"
    PAPER_URL = "https://openapivts.koreainvestment.com:29443"  # 모의투자

    def __init__(self, app_key: str, app_secret: str, account: str, is_paper: bool = True):
        self.is_paper = is_paper  # 기본: 모의투자
        ...

    # === 인증 ===
    def get_access_token(self) -> str:
        """OAuth2 액세스 토큰 발급/갱신."""

    # === 주문 ===
    def place_order(self, ticker: str, qty: int, price: int = 0, order_type: str = "시장가") -> dict:
        """매수/매도 주문."""

    def modify_order(self, order_no: str, qty: int, price: int) -> dict:
        """주문 정정."""

    def cancel_order(self, order_no: str) -> dict:
        """주문 취소."""

    # === 잔고 ===
    def get_balance(self) -> dict:
        """계좌 잔고 조회."""

    def get_positions(self) -> pd.DataFrame:
        """보유 종목 조회."""

    def get_buyable_amount(self, ticker: str) -> int:
        """매수 가능 수량 조회."""

    # === 시세 ===
    def get_current_price(self, ticker: str) -> dict:
        """현재가 조회."""

    # === WebSocket ===
    def connect_websocket(self, tickers: list[str], callback) -> None:
        """실시간 체결가 스트리밍 (최대 41종목)."""
```

### 5.5 주문 실행기

```python
# src/execution/executor.py

class OrderExecutor:
    """전략 시그널을 실제 주문으로 변환."""

    def __init__(self, kis_client: KISClient, risk_guard: RiskGuard):
        ...

    def execute_rebalance(
        self,
        target_weights: dict[str, float],
        total_capital: int,
    ) -> list[dict]:
        """
        리밸런싱 실행.

        1. 현재 포지션 조회
        2. target vs current 차이 계산
        3. 매도 먼저 → 매수 순서 (자금 확보)
        4. 각 주문에 리스크 체크 적용
        5. 주문 실행 및 체결 확인
        """

    def _calculate_orders(
        self,
        current: dict[str, int],   # {ticker: 수량}
        target: dict[str, int],    # {ticker: 수량}
    ) -> tuple[list, list]:
        """매도/매수 주문 리스트 산출. 매도 우선."""
```

### 5.6 리스크 가드

```python
# src/execution/risk_guard.py

class RiskGuard:
    """주문 전 사전 리스크 체크."""

    def __init__(self, config: dict):
        self.max_order_pct = config.get("max_order_pct", 0.10)       # 1건당 최대 10%
        self.max_daily_turnover = config.get("max_daily_turnover", 0.30)  # 일일 최대 30%
        self.max_single_stock_pct = config.get("max_single_stock_pct", 0.15)  # 개별종목 15%
        self.blocked_tickers = config.get("blocked_tickers", [])      # 매매 금지 종목

    def check_order(self, order: dict, portfolio: dict) -> tuple[bool, str]:
        """
        주문 검증. (통과, 사유) 반환.

        체크 항목:
        - 1건 주문 금액이 총자산의 max_order_pct 이하
        - 일일 누적 매매대금이 max_daily_turnover 이하
        - 개별 종목 비중이 max_single_stock_pct 이하
        - 매매 금지 종목 여부
        - 가격 제한폭(±30%) 근접 여부
        """
```

### 5.7 모의투자 → 실전 전환

| 단계 | 환경 | URL | 기간 |
|------|------|-----|------|
| 1단계 | 모의투자 | `openapivts.koreainvestment.com` | 최소 1개월 |
| 2단계 | 소액 실전 | `openapi.koreainvestment.com` | 1~3개월 |
| 3단계 | 정상 운영 | 실전 | 지속 |

### 5.8 환경변수

```env
# .env 추가
KIS_APP_KEY=
KIS_APP_SECRET=
KIS_ACCOUNT_NO=
KIS_IS_PAPER=true          # 모의투자 모드 (기본)
KIS_MAX_ORDER_PCT=0.10
KIS_MAX_DAILY_TURNOVER=0.30
```

### 5.9 WebSocket 제약 및 대응

| 제약 | 대응 |
|------|------|
| 1 세션 최대 41종목 | 포트폴리오 20~30종목이면 충분 |
| 연결 끊김 | 자동 재연결 + 상태 복구 로직 |
| 장중만 실시간 | 장외 시간은 REST 폴링으로 대체 |

---

## 6. 스케줄러 자동화

### 6.1 아키텍처

```
systemd (프로세스 관리)
  └── main.py (데몬 프로세스)
        └── APScheduler (작업 스케줄링)
              ├── 07:00 모닝 리서치/리포트
              ├── 08:50 장 전 시그널 확인
              ├── 09:00 리밸런싱 실행 (해당일)
              ├── 매시 정각: 뉴스/알림 체크
              ├── 15:35 장 마감 EOD 리뷰
              └── 19:00 이브닝 리포트
```

### 6.2 스케줄 상세

| 시간 | 작업 | 모듈 | 산출물 |
|------|------|------|--------|
| 07:00 | 모닝 리서치 | `scheduler/morning.py` | 마켓 브리핑 (텔레그램) |
| 08:50 | 장 전 시그널 | `scheduler/premarket.py` | 매매 시그널 알림 |
| 09:00 | 리밸런싱 실행 | `execution/executor.py` | 주문 체결 리포트 |
| 매시 | 포트폴리오 모니터링 | `scheduler/monitor.py` | 이상 감지 알림 |
| 15:35 | EOD 리뷰 | `scheduler/eod.py` | 일일 성과 리포트 |
| 19:00 | 이브닝 리포트 | `scheduler/evening.py` | 종합 분석 리포트 |
| 매월 1일 | 월간 리밸런싱 | `scheduler/monthly.py` | 리밸런싱 실행 |

### 6.3 모듈 설계

```
src/scheduler/
    __init__.py
    main.py               # APScheduler 초기화 + 작업 등록
    morning.py             # 모닝 리서치 작업
    premarket.py           # 장 전 시그널 체크
    eod.py                 # 장 마감 후 리뷰
    evening.py             # 이브닝 리포트
    monthly.py             # 월간 리밸런싱
    monitor.py             # 장중 모니터링
    holidays.py            # 한국 장 휴일 관리 (공휴일, 임시휴장)
```

### 6.4 한국 장 휴일 처리

```python
# src/scheduler/holidays.py

class KRXHolidays:
    """KRX 휴장일 관리."""

    def is_trading_day(self, date: datetime.date) -> bool:
        """해당일이 거래일인지 확인."""
        # 주말 체크
        # 공휴일 체크 (설날, 추석, 대체공휴일 등)
        # 임시휴장일 체크

    def next_trading_day(self, date: datetime.date) -> datetime.date:
        """다음 거래일 반환."""

    def days_to_next_rebalance(self, date: datetime.date) -> int:
        """다음 리밸런싱까지 거래일 수."""
```

### 6.5 systemd 서비스 설정

```ini
# /etc/systemd/system/quant-bot.service
[Unit]
Description=Quant Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=quant
WorkingDirectory=/mnt/data/quant
Environment="PATH=/mnt/data/quant/.venv/bin"
ExecStart=/mnt/data/quant/.venv/bin/python -u src/scheduler/main.py
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 6.6 설치 스크립트

```bash
# scripts/install.sh
#!/bin/bash
# 시스템 서비스 등록 스크립트
set -e

PROJECT_DIR="/mnt/data/quant"
SERVICE_NAME="quant-bot"

# 1. venv 확인 및 의존성 설치
python3 -m venv "$PROJECT_DIR/.venv"
source "$PROJECT_DIR/.venv/bin/activate"
pip install -r "$PROJECT_DIR/requirements.txt"

# 2. systemd 서비스 파일 복사
sudo cp "$PROJECT_DIR/scripts/$SERVICE_NAME.service" /etc/systemd/system/

# 3. 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "✅ $SERVICE_NAME 서비스가 설치되었습니다."
echo "상태 확인: sudo systemctl status $SERVICE_NAME"
echo "로그 확인: journalctl -u $SERVICE_NAME -f"
```

---

## 7. Phase 2 잔여 항목 (P2)

Phase 2에서 미구현된 P2 항목을 Phase 3에서 함께 처리한다.

| # | 항목 | 우선순위 | 비고 |
|---|------|---------|------|
| P2-18 | 일일 리포트 HTML 템플릿 | P2 | `templates/daily.html` |
| P2-19 | 이메일 알림 (SMTP) | P2 | `alert/email_sender.py` |
| P2-20 | 인터랙티브 차트 (plotly) | P2 | `report/charts.py` 확장 |

---

## 8. 구현 우선순위 및 팀 배정

### 8.1 우선순위 정의

| 등급 | 의미 |
|------|------|
| **P0** | 필수 - Phase 3 핵심 기능 |
| **P1** | 중요 - 실용성 향상 |
| **P2** | 보조 - 시간 여유 시 구현 |

### 8.2 전체 기능 우선순위표

| # | 기능 | 우선순위 | 팀 | 의존성 |
|---|------|---------|-----|--------|
| 1 | DART 재무제표 수집 모듈 | **P0** | Dev | 없음 |
| 2 | 퀄리티 팩터 전략 | **P0** | Dev | #1 |
| 3 | FactorCombiner N팩터 확장 | **P0** | Dev | #2 |
| 4 | 3팩터 통합 전략 | **P0** | Dev | #2, #3 |
| 5 | ETF 데이터 수집 확장 | **P0** | Dev | 없음 |
| 6 | 듀얼 모멘텀 전략 | **P0** | Dev | #5 |
| 7 | KIS API 클라이언트 | **P0** | Dev | 없음 |
| 8 | 주문 실행기 (Executor) | **P0** | Dev | #7 |
| 9 | 리스크 가드 | **P0** | Dev | #8 |
| 10 | APScheduler 메인 데몬 | **P0** | Dev | #8 |
| 11 | 포지션 매니저 | **P1** | Dev | #7 |
| 12 | WebSocket 실시간 시세 | **P1** | Dev | #7 |
| 13 | KRX 휴장일 관리 | **P1** | Dev | 없음 |
| 14 | 모닝/이브닝 리포트 자동화 | **P1** | Dev | #10 |
| 15 | 장중 모니터링 작업 | **P1** | Dev | #10, #12 |
| 16 | systemd 서비스 + 설치 스크립트 | **P1** | Dev | #10 |
| 17 | 일일 HTML 리포트 | **P2** | Dev | Phase 2 리포트 |
| 18 | 이메일 알림 | **P2** | Dev | Phase 2 알림 |
| 19 | 인터랙티브 차트 (plotly) | **P2** | Dev | Phase 2 차트 |
| 20 | 포트폴리오 최적화 (리스크 패리티) | **P2** | Dev | #6 |

### 8.3 팀 배정

| 팀 | Phase 3 역할 | 주요 담당 |
|----|-------------|----------|
| **Research** | DART API 조사, 한국 ETF 유니버스 분석, KIS API 문서 정리 | #1, #5, #7 사전조사 |
| **Dev Team A** | 전략 모듈 (퀄리티, 3팩터, 듀얼모멘텀) | #1~#6 |
| **Dev Team B** | 실행 모듈 (KIS API, 주문, 스케줄러) | #7~#10 |
| **QA** | 전략 테스트, 주문 실행 테스트, 통합 테스트 | 전체 |
| **Planning** | Phase 4 기획 (ML, 리스크 패리티) | Phase 4 사전 준비 |

### 8.4 구현 순서 (권장)

```
Phase 3-A: 전략 확장 (병렬 가능)
    ┌─ #1 DART 수집 → #2 퀄리티 → #3 N팩터 결합 → #4 3팩터 전략
    └─ #5 ETF 수집 → #6 듀얼 모멘텀

Phase 3-B: 실전 매매 (순차)
    #7 KIS 클라이언트 → #8 주문 실행기 → #9 리스크 가드

Phase 3-C: 자동화 (3-B 완료 후)
    #10 스케줄러 → #14 리포트 자동화 → #16 systemd 배포

Phase 3-D: 고도화 (P1/P2)
    #11~#15 (P1), #17~#20 (P2)
```

### 8.5 최종 디렉토리 구조 (Phase 3 완료 후)

```
src/
    __init__.py
    data/
        collector.py           # Phase 1: pykrx 수집
        index_collector.py     # Phase 2: 지수 데이터
        dart_collector.py      # Phase 3: DART 재무제표
        etf_collector.py       # Phase 3: ETF 가격 데이터
    strategy/
        value.py               # Phase 1: 밸류 팩터
        momentum.py            # Phase 2: 모멘텀 팩터
        market_timing.py       # Phase 2: 마켓 타이밍
        factor_combiner.py     # Phase 2→3: 2팩터→N팩터 확장
        multi_factor.py        # Phase 2: 2팩터 전략
        quality.py             # Phase 3: 퀄리티 팩터
        three_factor.py        # Phase 3: 3팩터 전략
        dual_momentum.py       # Phase 3: 듀얼 모멘텀
    backtest/
        engine.py              # Phase 1→2: 백테스트 엔진
    report/
        (Phase 2 모듈들)
    alert/
        (Phase 2 모듈들)
        email_sender.py        # Phase 3-P2: 이메일 알림
    execution/                 # Phase 3: 신규 패키지
        __init__.py
        kis_client.py          # KIS API 클라이언트
        order_manager.py       # 주문 관리
        position_manager.py    # 포지션 관리
        executor.py            # 시그널 → 주문 변환
        risk_guard.py          # 리스크 체크
    scheduler/                 # Phase 3: 신규 패키지
        __init__.py
        main.py                # APScheduler 메인
        morning.py             # 모닝 작업
        premarket.py           # 장 전 작업
        eod.py                 # EOD 리뷰
        evening.py             # 이브닝 작업
        monthly.py             # 월간 리밸런싱
        monitor.py             # 장중 모니터링
        holidays.py            # 휴장일 관리
    utils/
        config.py              # 설정 (KIS 추가)
        logger.py              # 로깅
scripts/
    monitor.sh                 # tmux 모니터
    install.sh                 # 서비스 설치
    quant-bot.service          # systemd 서비스 파일
```

### 8.6 추가 의존성 (requirements.txt)

```
# Phase 3 추가
requests>=2.31       # KIS API 호출 (이미 설치됨)
websockets>=12.0     # KIS WebSocket
apscheduler>=3.10    # 작업 스케줄러
python-dotenv>=1.0   # .env 파일 로드
opendartreader>=0.2  # DART 재무제표 수집
```

---

## 부록: Phase 3 → Phase 4 전환 고려사항

1. **ML 실험 기반**: Phase 3의 3팩터 스코어를 피처로 활용하여 ML 모델 학습 가능
2. **리스크 패리티**: 듀얼 모멘텀의 자산배분 프레임워크 위에 리스크 패리티 적용
3. **포트폴리오 최적화**: mean-variance → Black-Litterman → 리스크 패리티 순서로 고도화
4. **멀티 계좌 운영**: KIS API 클라이언트를 여러 계좌로 확장
5. **대시보드**: Flask/Streamlit 기반 웹 대시보드 (Phase 4 핵심)
