# Phase 2 상세 기획서: 모멘텀 + 마켓 타이밍

> 작성일: 2026-02-18
> 상태: ✅ Phase 2 완료 (P0 9/9, P1 8/8, 168 tests passing)
> 선행 조건: Phase 1 (인프라 + 밸류 팩터) 완료

---

## 목차

1. [모멘텀 팩터 전략 상세 스펙](#1-모멘텀-팩터-전략-상세-스펙)
2. [이동평균 마켓 타이밍 오버레이](#2-이동평균-마켓-타이밍-오버레이)
3. [밸류 + 모멘텀 2팩터 결합](#3-밸류--모멘텀-2팩터-결합)
4. [필요 데이터 및 모듈 변경 사항](#4-필요-데이터-및-모듈-변경-사항)
5. [리포트/대시보드 기능 설계](#5-리포트대시보드-기능-설계)
6. [알림/모니터링 시스템 스펙](#6-알림모니터링-시스템-스펙)
7. [구현 우선순위 및 일정](#7-구현-우선순위-및-일정)

---

## 1. 모멘텀 팩터 전략 상세 스펙

### 1.1 전략 개요

가격 모멘텀(Price Momentum)은 과거 일정 기간 동안 상대적으로 높은 수익률을 기록한 종목이 단기적으로 초과 수익을 지속하는 경향을 활용하는 전략이다. Jegadeesh & Titman(1993) 이후 글로벌/한국 시장 모두에서 유효성이 검증되었다.

### 1.2 모멘텀 스코어 산출

#### 룩백 기간 (Lookback Period)
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| 단기 모멘텀 | 3개월 (63 거래일) | 단기 추세 포착 |
| 중기 모멘텀 | 6개월 (126 거래일) | 중기 추세 포착 |
| 장기 모멘텀 | 12개월 (252 거래일) | 장기 추세 포착, 가장 널리 검증된 기간 |

#### 직전 1개월 제외 (Skip Month)
- **기본값: 직전 1개월(21 거래일) 제외**
- 사유: 단기 반전 효과(short-term reversal) 제거. 직전 1개월 수익률은 mean-reversion 경향이 강하여 모멘텀 시그널에 노이즈를 추가함
- 설정 가능: `skip_month=True/False` 파라미터로 on/off 지원

#### 모멘텀 스코어 계산 공식

```python
def calculate_momentum_score(prices: pd.Series, lookback: int, skip: int = 21) -> float:
    """
    모멘텀 스코어 = (P[t-skip] / P[t-lookback]) - 1

    Args:
        prices: 일별 종가 시계열
        lookback: 룩백 기간 (거래일)
        skip: 직전 제외 기간 (거래일, 기본 21일=1개월)

    Returns:
        모멘텀 수익률 (float)
    """
    end_price = prices.iloc[-skip] if skip > 0 else prices.iloc[-1]
    start_price = prices.iloc[-lookback]
    return (end_price / start_price) - 1
```

#### 복합 모멘텀 스코어 (선택적)
여러 룩백 기간의 모멘텀을 결합하여 안정성 향상:
```
composite_momentum = 0.33 * mom_3m + 0.33 * mom_6m + 0.34 * mom_12m
```
- 동일 가중 기본, 사용자 설정 가능

### 1.3 유니버스 및 필터링

| 항목 | 스펙 |
|------|------|
| 대상 시장 | KOSPI, KOSDAQ |
| 시가총액 필터 | 하위 20% 제외 (소형주 유동성 리스크) |
| 거래대금 필터 | 20일 평균 거래대금 1억 원 미만 제외 |
| 관리종목/정리매매 | 제외 |
| 상장 기간 | 최소 12개월 이상 (룩백 기간 확보) |
| 극단값 처리 | 모멘텀 스코어 상/하위 1% winsorizing |

### 1.4 포트폴리오 구성

| 항목 | 스펙 |
|------|------|
| 선택 종목 수 | 상위 20~30개 (파라미터화) |
| 비중 방식 | 동일 가중 (1/N), 옵션: 모멘텀 스코어 비례 가중 |
| 리밸런싱 주기 | **월간** (매월 첫 거래일) |
| 회전율 제한 | 없음 (모멘텀 전략 특성상 높은 회전율 허용) |
| 거래비용 반영 | 편도 0.015% (증권사 수수료) + 코스피 0.05%/코스닥 0.2% (세금) |

### 1.5 리밸런싱 규칙

```
매월 첫 거래일 (T):
1. T-1일 종가 기준으로 전체 유니버스 모멘텀 스코어 산출
2. 유니버스 필터링 적용
3. 모멘텀 스코어 기준 상위 N개 종목 선정
4. 기존 포트폴리오와 비교하여 매수/매도 리스트 생성
5. T일 시가(또는 VWAP) 기준 리밸런싱 실행
```

### 1.6 백테스트 파라미터

| 파라미터 | 기본값 | 범위 |
|---------|--------|------|
| `lookback_months` | `[3, 6, 12]` | 1~24 |
| `skip_month` | `True` | True/False |
| `skip_days` | `21` | 0~42 |
| `n_stocks` | `20` | 10~50 |
| `weighting` | `"equal"` | equal, score_weighted |
| `rebalance_freq` | `"monthly"` | monthly, quarterly |
| `start_date` | `"2010-01-01"` | - |
| `end_date` | `"오늘"` | - |

---

## 2. 이동평균 마켓 타이밍 오버레이

### 2.1 전략 개요

Faber(2007)의 이동평균 기반 마켓 타이밍은 시장 전체의 추세를 판단하여 하락장에서 주식 비중을 줄이는 리스크 관리 오버레이다. 단독 전략이 아닌, 기존 팩터 전략 위에 적용하는 오버레이 레이어로 설계한다.

### 2.2 시그널 정의

#### 기준 지수
- **1차**: KOSPI 지수 (코스피 전략용)
- **2차**: KOSDAQ 지수 (코스닥 전략용)
- **통합**: KOSPI 기준 단일 시그널 (기본값)

#### 이동평균 산출
```python
def calculate_ma_signal(index_prices: pd.Series, ma_period: int = 200) -> str:
    """
    200일 단순이동평균(SMA) 기반 마켓 타이밍 시그널.

    Args:
        index_prices: 지수 일별 종가
        ma_period: 이동평균 기간 (기본 200일)

    Returns:
        "RISK_ON" 또는 "RISK_OFF"
    """
    sma = index_prices.rolling(window=ma_period).mean()
    current_price = index_prices.iloc[-1]
    current_sma = sma.iloc[-1]

    if current_price > current_sma:
        return "RISK_ON"
    else:
        return "RISK_OFF"
```

### 2.3 비중 전환 규칙

#### 기본 모드 (Binary Switch)
| 시그널 | 주식 비중 | 현금(또는 채권) 비중 |
|--------|----------|-------------------|
| RISK_ON (지수 > 200SMA) | 100% | 0% |
| RISK_OFF (지수 < 200SMA) | 0% | 100% |

#### 점진적 전환 모드 (Gradual Switch, 선택적)
| 조건 | 주식 비중 |
|------|----------|
| 지수 > 200SMA + 5% | 100% |
| 200SMA < 지수 < 200SMA + 5% | 75% |
| 200SMA - 5% < 지수 < 200SMA | 50% |
| 지수 < 200SMA - 5% | 25% |

#### 현금 대체 자산
- 기본: 현금 (연 2~3% 가정, 설정 가능)
- 옵션: 단기 채권 ETF (예: KODEX 단기채권)

### 2.4 Whipsaw 방지

잦은 시그널 전환(whipsaw)을 방지하기 위한 안전 장치:

| 방법 | 스펙 |
|------|------|
| **확인 기간** | 시그널 전환 후 최소 5 거래일 유지 시 실행 (기본) |
| **밴드 필터** | 200SMA 기준 +/-2% 밴드 내에서는 시그널 변경 무시 (선택적) |
| **월말 확인** | 리밸런싱 시점(월초)에만 시그널 확인 (가장 단순) |

기본 설정: **월말 확인 방식** (모멘텀 전략의 월간 리밸런싱과 동기화)

### 2.5 마켓 타이밍 적용 흐름

```
[매월 리밸런싱 시점]
    |
    v
마켓 타이밍 시그널 확인 (KOSPI vs 200SMA)
    |
    +--- RISK_ON ---> 팩터 전략 정상 실행 (100% 주식)
    |
    +--- RISK_OFF --> 전량 현금 전환 (또는 점진적 비중 조절)
```

### 2.6 백테스트 파라미터

| 파라미터 | 기본값 | 범위 |
|---------|--------|------|
| `ma_period` | `200` | 50, 100, 150, 200 |
| `ma_type` | `"SMA"` | SMA, EMA |
| `switch_mode` | `"binary"` | binary, gradual |
| `whipsaw_filter` | `"monthly_check"` | monthly_check, confirmation_days, band |
| `confirmation_days` | `5` | 3~10 |
| `band_pct` | `0.02` | 0.01~0.05 |
| `cash_return_annual` | `0.025` | 0~0.05 |
| `reference_index` | `"KOSPI"` | KOSPI, KOSDAQ |

---

## 3. 밸류 + 모멘텀 2팩터 결합

### 3.1 결합 근거

밸류와 모멘텀은 학술적으로 **낮은 상관관계**(때로는 음의 상관관계)를 보이며, 결합 시 분산 효과로 더 높은 위험 대비 수익률(샤프 비율)을 기대할 수 있다. Asness et al.(2013) "Value and Momentum Everywhere" 참고.

### 3.2 결합 방법론

두 가지 방법을 모두 구현하여 백테스트에서 비교한다.

#### 방법 A: Z-Score 합산 (기본 추천)

```python
def combine_zscore(value_scores: pd.Series, momentum_scores: pd.Series,
                   value_weight: float = 0.5, momentum_weight: float = 0.5) -> pd.Series:
    """
    각 팩터 스코어를 Z-Score로 정규화한 뒤 가중 합산.

    장점: 팩터 간 스케일 차이를 자동 보정
    단점: 이상치에 민감 (winsorizing 필요)
    """
    value_z = (value_scores - value_scores.mean()) / value_scores.std()
    momentum_z = (momentum_scores - momentum_scores.mean()) / momentum_scores.std()

    # 이상치 처리: Z-Score +-3 기준 클리핑
    value_z = value_z.clip(-3, 3)
    momentum_z = momentum_z.clip(-3, 3)

    combined = value_weight * value_z + momentum_weight * momentum_z
    return combined
```

#### 방법 B: 순위 합산 (Rank Sum)

```python
def combine_rank(value_scores: pd.Series, momentum_scores: pd.Series,
                 value_weight: float = 0.5, momentum_weight: float = 0.5) -> pd.Series:
    """
    각 팩터 스코어를 순위 퍼센타일로 변환한 뒤 가중 합산.

    장점: 이상치에 강건, 분포 가정 불필요
    단점: 순위 간 거리 정보 손실
    """
    n = len(value_scores)
    value_rank = value_scores.rank(pct=True)
    momentum_rank = momentum_scores.rank(pct=True)

    combined = value_weight * value_rank + momentum_weight * momentum_rank
    return combined
```

### 3.3 팩터 가중치

| 설정 | 밸류 가중치 | 모멘텀 가중치 | 비고 |
|------|-----------|-------------|------|
| 동일 가중 (기본) | 0.5 | 0.5 | 가장 단순, 특별한 사전 정보 없을 때 |
| 밸류 과대 | 0.6 | 0.4 | 밸류 프리미엄이 더 안정적인 경우 |
| 모멘텀 과대 | 0.4 | 0.6 | 추세 추종 강화 |

- 가중치는 `config`에서 파라미터로 관리, 백테스트에서 최적 가중치 탐색 가능

### 3.4 포트폴리오 구성 방식

#### 방식 1: 통합 순위 (Integrated, 기본)
1. 전체 유니버스에 대해 combined_score 산출
2. combined_score 상위 N개 종목 선정
3. 단일 포트폴리오 구성

#### 방식 2: 교집합 (Intersection, 선택적)
1. 밸류 상위 30%와 모멘텀 상위 30%의 교집합
2. 교집합 내에서 combined_score 기준 상위 N개 선정
3. 더 엄격한 필터링으로 quality 개선 기대

### 3.5 리밸런싱

- **주기**: 월간 (모멘텀 기준에 맞춤)
- 밸류 스코어는 분기마다 업데이트되지만, 포트폴리오 리밸런싱은 월간으로 통일
- 분기 실적 발표 시점에 밸류 스코어 갱신

### 3.6 백테스트 비교 항목

다음 전략들을 동일 기간/유니버스에서 비교:
1. 밸류 단독
2. 모멘텀 단독
3. 밸류+모멘텀 Z-Score 합산
4. 밸류+모멘텀 순위 합산
5. 위 4개 전략 각각에 마켓 타이밍 오버레이 적용 (총 8개)

성과 지표:
- CAGR, MDD, 샤프 비율, 소르티노 비율
- 연도별 수익률, 월별 수익률 히트맵
- 벤치마크(KOSPI) 대비 초과 수익률

---

## 4. 필요 데이터 및 모듈 변경 사항

### 4.1 추가 데이터 요구사항

| 데이터 | 현재 상태 | Phase 2 요구 | 소스 |
|--------|----------|-------------|------|
| 일별 종가 (개별 종목) | Phase 1에서 수집 | 그대로 사용 | pykrx, KIS API |
| KOSPI/KOSDAQ 지수 일봉 | 미구현 | **신규 필요** | pykrx, FinanceDataReader |
| 시가총액 데이터 | Phase 1에서 수집 | 그대로 사용 | pykrx |
| 거래대금 데이터 | Phase 1에서 수집 | 그대로 사용 | pykrx |
| 단기채권 ETF 가격 | 미구현 | **선택적** (현금 대체 백테스트용) | pykrx |

### 4.2 신규/변경 모듈

#### 신규 모듈

| 모듈 | 경로 | 우선순위 | 설명 |
|------|------|---------|------|
| `MomentumStrategy` | `src/strategy/momentum.py` | **P0** | 모멘텀 팩터 전략 구현 |
| `MarketTimingOverlay` | `src/strategy/market_timing.py` | **P0** | 이동평균 마켓 타이밍 오버레이 |
| `MultiFactorStrategy` | `src/strategy/multi_factor.py` | **P0** | 2팩터 결합 전략 |
| `FactorCombiner` | `src/strategy/factor_combiner.py` | **P0** | Z-Score/순위 합산 결합 로직 |
| `IndexDataCollector` | `src/data/index_collector.py` | **P0** | KOSPI/KOSDAQ 지수 데이터 수집 |

#### 변경 모듈

| 모듈 | 변경 내용 | 우선순위 |
|------|----------|---------|
| `src/strategy/__init__.py` | 새 전략 클래스 등록 | P0 |
| `src/data/__init__.py` | 인덱스 데이터 수집기 등록 | P0 |
| `src/backtest/` (백테스트 엔진) | 마켓 타이밍 오버레이 지원, 다중 전략 비교 기능 | P0 |
| `src/utils/config.py` | Phase 2 전략 파라미터 설정 추가 | P1 |

### 4.3 추가 의존성 (requirements.txt)

```
# Phase 2 추가
scipy>=1.11          # Z-Score, 통계 검정
jinja2>=3.1          # HTML 리포트 생성
python-telegram-bot>=20.0  # 텔레그램 알림
```

### 4.4 공통 전략 인터페이스 확장

Phase 1에서 정의한 전략 인터페이스에 다음 메서드를 추가/확인:

```python
class BaseStrategy(ABC):
    """모든 전략이 구현해야 할 인터페이스."""

    @abstractmethod
    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        """팩터 스코어 산출."""
        pass

    @abstractmethod
    def select_stocks(self, scores: pd.Series, n: int) -> list[str]:
        """종목 선정."""
        pass

    @abstractmethod
    def generate_weights(self, selected: list[str], scores: pd.Series) -> dict[str, float]:
        """종목별 비중 산출."""
        pass

    def apply_overlay(self, weights: dict[str, float], overlay_signal: str) -> dict[str, float]:
        """마켓 타이밍 오버레이 적용 (선택적 오버라이드)."""
        if overlay_signal == "RISK_OFF":
            return {ticker: 0.0 for ticker in weights}
        return weights
```

---

## 5. 리포트/대시보드 기능 설계

### 5.1 일일 리포트

**우선순위: P1**
**의존성: Phase 1 백테스트 엔진, Phase 2 전략 모듈**

#### 포함 정보
| 항목 | 상세 |
|------|------|
| 포트폴리오 현황 | 보유 종목, 수량, 평가금액, 비중 |
| 일일 수익률 | 전일 대비 수익률, 누적 수익률, 벤치마크 대비 |
| 팩터 스코어 현황 | 보유 종목의 밸류/모멘텀 스코어 변동 |
| 마켓 타이밍 시그널 | KOSPI vs 200SMA 현재 상태, 이격도 |
| 리밸런싱 알림 | 다음 리밸런싱까지 D-day, 예상 변경 종목 |
| 위험 지표 | 현재 MDD, 포트폴리오 베타, 변동성 |

#### 출력 형식
- **1차 (P1)**: 터미널 텍스트 출력 (tabulate 라이브러리 활용)
- **2차 (P2)**: HTML 리포트 (Jinja2 템플릿)
- **향후**: 웹 대시보드 (Flask/Streamlit)

#### 모듈 설계
```
src/report/
    __init__.py
    daily_report.py       # 일일 리포트 생성
    templates/
        daily.html        # HTML 템플릿
```

### 5.2 백테스트 리포트

**우선순위: P0**
**의존성: Phase 1 백테스트 엔진**

#### 포함 정보
| 항목 | 상세 |
|------|------|
| 성과 지표 테이블 | CAGR, MDD, 샤프 비율, 소르티노 비율, 칼마 비율, 승률, 평균 수익률 |
| 수익률 곡선 차트 | 전략별 누적 수익률, 벤치마크 비교 |
| 드로다운 차트 | 시간에 따른 드로다운 추이 |
| 연도별 수익률 테이블 | 연도별 전략/벤치마크 수익률 비교 |
| 월별 수익률 히트맵 | 연도 x 월 히트맵 |
| 회전율 | 월별/연간 평균 회전율 |
| 전략 비교 | 다중 전략 동시 비교 (Phase 2의 8개 전략) |

#### 차트 라이브러리
- **1차 (P0)**: matplotlib (이미 requirements.txt에 포함)
- **2차 (P2)**: plotly (인터랙티브 차트, HTML 리포트용)

#### 모듈 설계
```
src/report/
    backtest_report.py    # 백테스트 리포트 생성
    charts.py             # 차트 생성 유틸리티
    metrics.py            # 성과 지표 계산 (백테스트 엔진과 공유 가능)
    templates/
        backtest.html     # 백테스트 HTML 템플릿
```

#### 성과 지표 상세 정의

```python
class PerformanceMetrics:
    """백테스트 성과 지표 산출."""

    def cagr(self, returns: pd.Series) -> float:
        """연평균 복합 수익률."""

    def max_drawdown(self, returns: pd.Series) -> float:
        """최대 낙폭."""

    def sharpe_ratio(self, returns: pd.Series, risk_free: float = 0.025) -> float:
        """샤프 비율 (무위험 수익률 기본 2.5%)."""

    def sortino_ratio(self, returns: pd.Series, risk_free: float = 0.025) -> float:
        """소르티노 비율 (하방 변동성만 사용)."""

    def calmar_ratio(self, returns: pd.Series) -> float:
        """칼마 비율 (CAGR / MDD)."""

    def win_rate(self, returns: pd.Series) -> float:
        """월간 기준 양(+) 수익률 비율."""

    def turnover(self, portfolios: list[dict]) -> float:
        """연간 평균 회전율."""

    def tracking_error(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """추적 오차."""

    def information_ratio(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """정보 비율."""
```

### 5.3 시장 스캐너

**우선순위: P1**
**의존성: Phase 2 모멘텀 전략, Phase 1 밸류 전략**

#### 포함 정보
| 항목 | 상세 |
|------|------|
| 밸류 상위 종목 | PER, PBR, PSR 기준 저평가 상위 20개 |
| 모멘텀 상위 종목 | 3/6/12개월 모멘텀 상위 20개 |
| 2팩터 상위 종목 | 밸류+모멘텀 결합 스코어 상위 20개 |
| 섹터별 분석 | GICS 섹터별 평균 밸류/모멘텀 스코어, 종목 수 |
| 신규 진입/이탈 | 전월 대비 새로 상위권 진입/이탈한 종목 |

#### 모듈 설계
```
src/report/
    scanner.py            # 시장 스캐너
```

#### 출력 예시 (터미널)
```
=== 시장 스캐너 (2026-02-18) ===

[밸류 상위 10]
순위  종목코드  종목명      PER   PBR   밸류스코어
1     005930   삼성전자    8.2   1.1   2.34
2     000660   SK하이닉스  6.5   1.3   2.21
...

[모멘텀 상위 10]
순위  종목코드  종목명      3M    6M    12M   모멘텀스코어
1     035420   NAVER     15.2% 28.3% 42.1% 1.87
...

[마켓 타이밍 현황]
KOSPI: 2,650.32 | 200SMA: 2,580.15 | 이격도: +2.72% | 시그널: RISK_ON
```

### 5.4 리포트 디렉토리 구조 (최종)

```
src/report/
    __init__.py
    daily_report.py       # P1 - 일일 리포트
    backtest_report.py    # P0 - 백테스트 리포트
    charts.py             # P0 - 차트 생성
    metrics.py            # P0 - 성과 지표
    scanner.py            # P1 - 시장 스캐너
    templates/
        daily.html        # P2 - 일일 HTML 템플릿
        backtest.html     # P1 - 백테스트 HTML 템플릿
        scanner.html      # P2 - 스캐너 HTML 템플릿
```

---

## 6. 알림/모니터링 시스템 스펙

### 6.1 알림 채널

#### 1순위: 텔레그램 봇 (P1)

| 항목 | 스펙 |
|------|------|
| 라이브러리 | `python-telegram-bot>=20.0` |
| 봇 생성 | BotFather를 통한 봇 토큰 발급 |
| 채팅방 | 개인 채팅 또는 그룹 채팅방 |
| 메시지 형식 | Markdown V2 (표, 볼드, 코드 블록 지원) |
| Rate Limit | 초당 1건, 분당 20건 |
| 환경 변수 | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |

#### 2순위: 이메일 (P2)

| 항목 | 스펙 |
|------|------|
| 방식 | SMTP (Gmail 또는 네이버 메일) |
| 형식 | HTML 메일 (리포트 첨부 가능) |
| 환경 변수 | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL_TO` |

### 6.2 알림 유형

#### 6.2.1 리밸런싱 알림 (P0)

| 시점 | 내용 |
|------|------|
| 리밸런싱 D-3 | "3 거래일 후 리밸런싱 예정. 예상 변경 종목: [목록]" |
| 리밸런싱 당일 (장 전) | "오늘 리밸런싱 실행일. 매수: [N종목], 매도: [N종목]. 상세: [목록]" |
| 리밸런싱 완료 후 | "리밸런싱 완료. 실행 결과: [요약]. 슬리피지: [X%]" |
| 마켓 타이밍 전환 | "마켓 타이밍 시그널 변경: RISK_ON -> RISK_OFF (KOSPI 2,480 < 200SMA 2,520)" |

#### 6.2.2 급등/급락 종목 알림 (P1)

| 조건 | 내용 |
|------|------|
| 보유 종목 일일 등락 >= +/-5% | "경고: [종목명] 오늘 -6.2% 하락 (현재가: 50,200원)" |
| 보유 종목 3일 연속 하락 | "경고: [종목명] 3일 연속 하락 (누적 -8.5%)" |
| 관심 종목 급등 | "알림: [종목명] 오늘 +7.3% 상승, 모멘텀 상위 진입" |

#### 6.2.3 전략 성과 이상 감지 (P1)

| 조건 | 임계값 (기본) | 알림 내용 |
|------|-------------|----------|
| MDD 임계값 초과 | -15% | "경고: 현재 MDD -16.2%로 임계값(-15%) 초과" |
| 벤치마크 대비 언더퍼폼 | 3개월 연속 | "경고: 전략이 3개월 연속 KOSPI 대비 언더퍼폼" |
| 비정상 회전율 | 월간 50% 초과 | "경고: 이번 달 회전율 62%로 비정상적 높음" |
| 전략 수익률 급감 | 월간 -10% 이하 | "경고: 이번 달 전략 수익률 -12.3%" |

#### 6.2.4 시스템 알림 (P1)

| 조건 | 알림 내용 |
|------|----------|
| 데이터 수집 실패 | "오류: 주가 데이터 수집 실패 (pykrx 타임아웃)" |
| API 토큰 만료 임박 | "알림: KIS API 토큰 만료까지 2시간" |
| 스케줄러 이상 | "오류: 일일 스캔 작업 미실행 (최근 실행: 24시간 전)" |

### 6.3 알림 모듈 설계

```
src/alert/
    __init__.py
    alert_manager.py      # 알림 통합 관리자
    telegram_bot.py       # 텔레그램 알림 발송
    email_sender.py       # 이메일 알림 발송 (P2)
    conditions.py         # 알림 조건 정의 및 검사
```

#### AlertManager 설계

```python
class AlertManager:
    """알림 통합 관리자."""

    def __init__(self, channels: list[AlertChannel]):
        self.channels = channels  # [TelegramChannel, EmailChannel, ...]
        self.conditions = []      # 등록된 알림 조건들
        self.history = []         # 알림 발송 이력

    def register_condition(self, condition: AlertCondition):
        """알림 조건 등록."""
        pass

    def check_and_alert(self, portfolio_state: dict, market_data: dict):
        """
        등록된 모든 조건을 검사하고, 해당되는 경우 알림 발송.
        중복 알림 방지: 동일 조건은 24시간 내 1회만 발송.
        """
        pass

    def send_alert(self, message: str, level: str = "INFO", channels: list = None):
        """
        알림 발송.
        level: INFO, WARNING, CRITICAL
        """
        pass
```

#### AlertCondition 인터페이스

```python
class AlertCondition(ABC):
    """알림 조건 추상 클래스."""

    @abstractmethod
    def check(self, state: dict) -> bool:
        """조건 충족 여부 확인."""
        pass

    @abstractmethod
    def format_message(self, state: dict) -> str:
        """알림 메시지 생성."""
        pass

    @property
    @abstractmethod
    def level(self) -> str:
        """알림 등급 (INFO/WARNING/CRITICAL)."""
        pass

    @property
    @abstractmethod
    def cooldown_hours(self) -> int:
        """중복 알림 방지 쿨다운 (시간)."""
        pass
```

### 6.4 환경 변수 추가 (.env)

```
# 텔레그램 알림
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# 이메일 알림 (P2)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
ALERT_EMAIL_TO=

# 알림 설정
ALERT_MDD_THRESHOLD=-0.15
ALERT_DAILY_MOVE_THRESHOLD=0.05
ALERT_UNDERPERFORM_MONTHS=3
```

---

## 7. 구현 우선순위 및 일정

### 7.1 우선순위 정의

| 등급 | 의미 | Phase 2 완료 시 상태 |
|------|------|-------------------|
| **P0** | 필수 - Phase 2 핵심 기능 | 반드시 완료 |
| **P1** | 중요 - 실용성 크게 향상 | 가능한 한 완료 |
| **P2** | 보조 - 있으면 좋음 | 시간 여유 시 구현 |

### 7.2 전체 기능 우선순위표

| # | 기능 | 우선순위 | 의존성 | 예상 범위 |
|---|------|---------|--------|----------|
| 1 | 모멘텀 팩터 전략 (`MomentumStrategy`) | **P0** | Phase 1 데이터 파이프라인 | 모멘텀 스코어 산출, 종목 선정, 리밸런싱 |
| 2 | KOSPI/KOSDAQ 지수 데이터 수집 | **P0** | Phase 1 데이터 모듈 | `IndexDataCollector` 신규 |
| 3 | 이동평균 마켓 타이밍 오버레이 | **P0** | #2 지수 데이터 | 200SMA 시그널, binary/gradual 모드 |
| 4 | 팩터 결합 모듈 (`FactorCombiner`) | **P0** | #1 모멘텀, Phase 1 밸류 | Z-Score/순위 합산 구현 |
| 5 | 2팩터 통합 전략 (`MultiFactorStrategy`) | **P0** | #4 팩터 결합 | 밸류+모멘텀 포트폴리오 구성 |
| 6 | 백테스트 엔진 확장 (오버레이 지원) | **P0** | #3 마켓 타이밍 | 오버레이 적용 로직 추가 |
| 7 | 성과 지표 계산 모듈 (`metrics.py`) | **P0** | Phase 1 백테스트 엔진 | CAGR, MDD, 샤프 등 |
| 8 | 백테스트 리포트 (차트 + 테이블) | **P0** | #7 성과 지표 | matplotlib 차트, 터미널 테이블 |
| 9 | 다중 전략 비교 백테스트 | **P0** | #1~#6 전체 | 8개 전략 동시 비교 |
| 10 | 시장 스캐너 | **P1** | #1 모멘텀, Phase 1 밸류 | 상위 종목 리스트, 섹터 분석 |
| 11 | 일일 리포트 (터미널) | **P1** | #7 성과 지표, #10 스캐너 | 포트폴리오 현황, 수익률 |
| 12 | 텔레그램 알림 (기본) | **P1** | 없음 (독립) | 봇 연동, 메시지 발송 |
| 13 | 리밸런싱 알림 | **P1** | #12 텔레그램 | 리밸런싱 사전/당일/완료 알림 |
| 14 | 급등/급락 알림 | **P1** | #12 텔레그램 | 보유 종목 변동 감지 |
| 15 | 전략 이상 감지 알림 | **P1** | #12 텔레그램, #7 성과 지표 | MDD 초과, 언더퍼폼 감지 |
| 16 | 시스템 상태 알림 | **P1** | #12 텔레그램 | 데이터 수집 실패, API 만료 |
| 17 | 백테스트 HTML 리포트 | **P1** | #8 백테스트 리포트 | Jinja2 HTML 생성 |
| 18 | 일일 리포트 HTML | **P2** | #11 일일 리포트 | HTML 템플릿 |
| 19 | 이메일 알림 | **P2** | 없음 (독립) | SMTP 발송 |
| 20 | 인터랙티브 차트 (plotly) | **P2** | #17 HTML 리포트 | plotly 기반 차트 |

### 7.3 구현 순서 (권장)

```
Phase 2-A: 핵심 전략 (P0)
    #2 지수 데이터 수집
    -> #1 모멘텀 전략
    -> #3 마켓 타이밍 오버레이
    -> #4 팩터 결합 모듈
    -> #5 2팩터 통합 전략
    -> #6 백테스트 엔진 확장

Phase 2-B: 리포트 (P0 + P1)
    #7 성과 지표 모듈
    -> #8 백테스트 리포트 (차트/터미널)
    -> #9 다중 전략 비교
    -> #10 시장 스캐너
    -> #11 일일 리포트

Phase 2-C: 알림 시스템 (P1)
    #12 텔레그램 봇 연동
    -> #13 리밸런싱 알림
    -> #14 급등/급락 알림
    -> #15 전략 이상 감지
    -> #16 시스템 상태 알림

Phase 2-D: 고도화 (P2)
    #17 백테스트 HTML 리포트
    #18 일일 HTML 리포트
    #19 이메일 알림
    #20 인터랙티브 차트
```

### 7.4 최종 디렉토리 구조 (Phase 2 완료 후)

```
src/
    __init__.py
    data/
        __init__.py
        (Phase 1 모듈들)
        index_collector.py       # 신규: 지수 데이터 수집
    strategy/
        __init__.py
        base.py                  # Phase 1: 전략 인터페이스
        value.py                 # Phase 1: 밸류 팩터
        momentum.py              # 신규: 모멘텀 팩터
        market_timing.py         # 신규: 마켓 타이밍 오버레이
        factor_combiner.py       # 신규: 팩터 결합
        multi_factor.py          # 신규: 멀티팩터 전략
    backtest/
        __init__.py
        (Phase 1 모듈들)         # 오버레이 지원 확장
    report/                      # 신규 패키지
        __init__.py
        metrics.py               # 성과 지표
        charts.py                # 차트 생성
        backtest_report.py       # 백테스트 리포트
        daily_report.py          # 일일 리포트
        scanner.py               # 시장 스캐너
        templates/
            backtest.html
            daily.html
            scanner.html
    alert/                       # 신규 패키지
        __init__.py
        alert_manager.py         # 알림 관리자
        conditions.py            # 알림 조건
        telegram_bot.py          # 텔레그램
        email_sender.py          # 이메일
    utils/
        __init__.py
        config.py                # 설정 확장
        logger.py
```

---

## 부록: Phase 2 -> Phase 3 전환 고려사항

Phase 3 (멀티팩터 + 자산배분)으로의 원활한 전환을 위해 Phase 2에서 주의할 점:

1. **팩터 결합 모듈의 확장성**: `FactorCombiner`가 2팩터뿐 아니라 N팩터를 지원하도록 설계
2. **전략 인터페이스 안정성**: `BaseStrategy` 인터페이스를 Phase 2에서 확정하여 Phase 3에서 변경 최소화
3. **자산 클래스 확장**: ETF 데이터 수집 기반을 마련하여 듀얼 모멘텀 자산배분에 활용
4. **성과 지표 표준화**: 모든 전략이 동일한 성과 지표 세트로 비교 가능하도록 통일
