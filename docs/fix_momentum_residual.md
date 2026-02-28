# Momentum(Residual) 전략 백테스트 실패 수정

## 문제 원인 분석

### 증상
백테스트 실행 시 Momentum(Residual) 전략이 수익률 0%, 거래 0건으로 완전 실패.

### 근본 원인: 가격 데이터 사전 로드 부재 (chicken-and-egg 문제)

`src/backtest/engine.py`의 `Backtest.run()` 메서드에서 가격 데이터(`price_cache`)가 **지연 로드(lazy loading)** 방식으로만 채워졌다. 즉, 전략이 시그널을 생성한 **이후에** 시그널에 포함된 종목의 가격 데이터를 로드하는 구조였다.

그러나 `MomentumStrategy._filter_universe()`는 가격 데이터를 **시그널 생성 전에** 필요로 한다:
- 상장기간 필터: `len(prices[ticker]) >= 252` (252거래일 = 약 12개월)
- 모멘텀 스코어 계산: 가격 시계열 기반

**실행 흐름:**
1. 첫 리밸런싱 시점: `price_cache = {}` (빈 딕셔너리)
2. `generate_signals()` 호출 -> `_filter_universe(fundamentals, prices)` 실행
3. 모든 종목이 `ticker in prices` 조건에서 탈락 (캐시가 비어있으므로)
4. `universe = []` -> `signals = {}`
5. 시그널이 비어있으므로 가격 데이터 로드 자체가 일어나지 않음
6. 모든 리밸런싱 날짜에서 동일 현상 반복

이 문제는 Momentum 단독 전략에서만 발현되었다. MultiFactor(V+M)에서는 모멘텀 스코어가 비어있을 때 밸류 스코어만으로 종목 선정하는 fallback 로직이 있어 정상 작동했다.

### 부차 원인: 지수 데이터 로드 범위 및 조건

기존 코드에서 `index_prices`(KOSPI 지수 데이터)는 마켓 타이밍 오버레이(`self.overlay`)가 설정된 경우에만 로드되었다. Momentum(Residual) 전략은 오버레이 없이 실행되었으므로 `index_prices`가 빈 Series였다. 잔차 모멘텀 계산에는 지수 데이터가 필수이므로, 가격 캐시 문제를 해결하더라도 잔차 모멘텀 스코어가 모두 NaN을 반환하는 문제가 있었다.

또한 지수 데이터의 수집 범위가 `start_date ~ end_date`로 설정되어 있어 lookback 기간의 과거 지수 데이터가 누락되었다.

## 수정 내용

### 수정 파일: `src/backtest/engine.py`

#### 1. 유니버스 가격 데이터 사전 로드

`run()` 메서드에서 각 리밸런싱 시점마다 `generate_signals()` 호출 **이전에** 펀더멘탈 데이터의 모든 종목에 대해 가격 데이터를 사전 로드하도록 변경.

```python
# 변경 전: generate_signals() 이후에만 가격 로드
data = {"fundamentals": fundamentals, "prices": price_cache, ...}
signals = self.strategy.generate_signals(date, data)
if signals:
    for ticker in signals:
        if ticker not in price_cache:
            price_cache[ticker] = self._fetch_price(ticker)

# 변경 후: generate_signals() 이전에 유니버스 전체 가격 사전 로드
if not fundamentals.empty and "ticker" in fundamentals.columns:
    universe_tickers = fundamentals["ticker"].tolist()
    for ticker in universe_tickers:
        if ticker not in price_cache:
            price_cache[ticker] = self._fetch_price(ticker)

data = {"fundamentals": fundamentals, "prices": price_cache, ...}
signals = self.strategy.generate_signals(date, data)
```

기존의 시그널 이후 가격 로드 코드는 fallback으로 유지 (시그널에 펀더멘탈에 없는 종목이 포함될 경우 대비).

#### 2. 지수 데이터 항상 로드 + lookback 포함

지수 데이터 로드 조건을 오버레이 존재 여부와 무관하게 변경. 수집 범위도 `_data_start_date`(lookback 포함)부터로 확장.

```python
# 변경 전: 오버레이가 있을 때만, start_date부터
if self.overlay is not None:
    index_df = get_index_data(self.overlay.reference_index, self.start_date, self.end_date)

# 변경 후: 항상 로드, _data_start_date부터 (lookback 포함)
reference_index = self.overlay.reference_index if self.overlay else "KOSPI"
index_df = get_index_data(reference_index, self._data_start_date, self.end_date)
```

## 영향 범위

### 직접 영향
- **Momentum(Residual)**: 0% -> 정상 작동 (가격 데이터 사전 로드 + 지수 데이터 제공)
- **모든 장기 전략**: 가격 데이터가 사전 로드되므로 전략의 가격 기반 필터/스코어링이 첫 리밸런싱부터 정상 작동

### 간접 영향
- **MultiFactor(V+M)**: 모멘텀 스코어가 정상 계산되어 밸류만의 fallback 대신 진정한 멀티팩터 결합이 작동
- **ThreeFactor(V+M+Q)**: 모멘텀 팩터가 포함된 3팩터 결합이 정상화
- **RiskParity(MF)**: 기반 MultiFactorStrategy의 모멘텀 팩터 정상화
- **LowVolQuality**: 가격 기반 변동성 계산이 첫 리밸런싱부터 작동 가능
- **지수 데이터**: 모든 전략에 KOSPI 지수 데이터가 제공되어 잔차 모멘텀, 마켓 타이밍 등 활용 가능

### 성능 영향
- 첫 리밸런싱 시점에 유니버스 전체(~2000종목)의 가격 데이터를 로드하므로 초기 실행 시간이 증가할 수 있음
- 그러나 `price_cache`에 한번 로드된 종목은 재로드하지 않으므로 이후 리밸런싱은 빠름
- `_data_start_date`부터 수집하므로 lookback 기간의 데이터가 확보됨

### 테스트 결과
- 전체 테스트 통과: 1563 passed, 2 skipped
- 백테스트 엔진 테스트 30건 전체 통과
- 모멘텀 전략 테스트 30건 전체 통과
