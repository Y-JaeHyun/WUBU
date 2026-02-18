# 데이터 수집 소스 상세 조사

> 조사 대상: pykrx, FinanceDataReader, DART OpenAPI
> 목적: 개발팀이 데이터 수집 모듈을 바로 구현할 수 있도록 실제 사용법 정리

---

## 목차

1. [pykrx](#1-pykrx)
2. [FinanceDataReader](#2-financedatareader)
3. [DART OpenAPI](#3-dart-openapi)
4. [3개 소스 비교 요약](#4-비교-요약)
5. [개발팀 권장 구현 가이드](#5-개발팀-권장-구현-가이드)

---

## 1. pykrx

### 1.1 개요

KRX(한국거래소) 웹사이트에서 데이터를 스크래핑하는 파이썬 라이브러리.
별도 API 키 불필요. KOSPI, KOSDAQ, KONEX 시장의 주가, 시가총액, 거래량, 펀더멘탈 지표를 제공.

### 1.2 설치

```bash
pip install pykrx
```

### 1.3 기본 임포트

```python
from pykrx import stock
from pykrx import bond  # 채권 데이터 (선택)
```

### 1.4 종목 코드 / 종목명 조회

```python
# 특정 날짜 기준 전체 종목 티커 조회
tickers = stock.get_market_ticker_list("20250117", market="KOSPI")
# -> ['005930', '000660', '005380', ...]

# KOSDAQ 종목
tickers_kq = stock.get_market_ticker_list("20250117", market="KOSDAQ")

# 티커 -> 종목명
name = stock.get_market_ticker_name("005930")
# -> '삼성전자'
```

### 1.5 OHLCV (주가 데이터) 조회

#### 개별 종목 OHLCV

```python
# 삼성전자 일별 OHLCV (기간 지정)
df = stock.get_market_ohlcv("20240101", "20250117", "005930")
```

**반환 DataFrame 구조:**

| 컬럼명 | 설명 | dtype |
|--------|------|-------|
| 날짜 (index) | 거래일 (DatetimeIndex) | datetime64 |
| 시가 | 시가 | int64 |
| 고가 | 고가 | int64 |
| 저가 | 저가 | int64 |
| 종가 | 종가 | int64 |
| 거래량 | 거래량 (주) | int64 |
| 거래대금 | 거래대금 (원) | int64 |
| 등락률 | 전일 대비 등락률 (%) | float64 |

```python
# 실제 출력 예시
#              시가    고가    저가    종가     거래량        거래대금   등락률
# 날짜
# 2024-01-02  78000  78300  77200  77800  12345678  962345678900   -0.26
# 2024-01-03  77900  78500  77600  78200   9876543  771234567890    0.51
```

#### 특정 날짜 전체 종목 OHLCV

```python
# 특정 날짜의 전 종목 OHLCV 한 번에 조회
df_all = stock.get_market_ohlcv("20250117", market="KOSPI")
```

**반환 DataFrame 구조 (전종목 조회 시):**

| 컬럼명 | 설명 |
|--------|------|
| 티커 (index) | 종목코드 (str) |
| 시가, 고가, 저가, 종가 | 가격 |
| 거래량, 거래대금 | 거래 정보 |
| 등락률 | % |

#### frequency 파라미터

```python
# 주봉 데이터
df_weekly = stock.get_market_ohlcv("20240101", "20250117", "005930", frequency="w")

# 월봉 데이터
df_monthly = stock.get_market_ohlcv("20240101", "20250117", "005930", frequency="m")
```

### 1.6 시가총액 조회

#### 개별 종목

```python
df_cap = stock.get_market_cap("20240101", "20250117", "005930")
```

**반환 DataFrame 구조:**

| 컬럼명 | 설명 | dtype |
|--------|------|-------|
| 날짜 (index) | 거래일 | datetime64 |
| 시가총액 | 시가총액 (원) | int64 |
| 거래량 | 거래량 (주) | int64 |
| 거래대금 | 거래대금 (원) | int64 |
| 상장주식수 | 상장주식수 | int64 |

#### 특정 날짜 전 종목 시가총액

```python
df_cap_all = stock.get_market_cap("20250117", market="KOSPI")
# 시가총액 기준 정렬된 전 종목 데이터
```

### 1.7 펀더멘탈 지표 (PER, PBR, 배당수익률)

#### 개별 종목 기간 조회

```python
df_fund = stock.get_market_fundamental("20240101", "20250117", "005930")
```

**반환 DataFrame 구조:**

| 컬럼명 | 설명 | dtype |
|--------|------|-------|
| 날짜 (index) | 거래일 | datetime64 |
| BPS | 주당순자산가치 | int64 |
| PER | 주가수익비율 | float64 |
| PBR | 주가순자산비율 | float64 |
| EPS | 주당순이익 | int64 |
| DIV | 주당배당금 | int64 |
| DPS | 배당수익률 (%) | float64 |

**주의**: pykrx에서 DIV는 주당배당금, DPS는 배당수익률이다. 일반적인 금융 용어와 반대이므로 혼동에 유의해야 한다.

#### 특정 날짜 전 종목 펀더멘탈

```python
df_fund_all = stock.get_market_fundamental("20250117", market="KOSPI")
# 전 종목의 BPS, PER, PBR, EPS, DIV, DPS 한번에 조회
```

### 1.8 투자자별 거래실적

```python
# 삼성전자 투자자별 순매수
df_inv = stock.get_market_trading_value_by_investor(
    "20250113", "20250117", "005930"
)
```

**반환 DataFrame 구조:**

| 컬럼명 | 설명 |
|--------|------|
| 투자자 (index) | 금융투자, 보험, 투신, 사모, 은행, 기타금융, 연기금등, 기타법인, 개인, 외국인 등 |
| 매도거래대금 | 매도금액 |
| 매수거래대금 | 매수금액 |
| 순매수거래대금 | 순매수금액 |

```python
# 전체 KOSPI 시장 투자자별 거래실적
df_inv_market = stock.get_market_trading_value_by_investor(
    "20250113", "20250117", "KOSPI"
)
```

### 1.9 인덱스(지수) 조회

```python
# KOSPI 지수 목록
indices = stock.get_index_ticker_list("20250117", market="KOSPI")
# -> ['1001', '1002', '1003', ...]

# 지수명 조회
stock.get_index_ticker_name("1001")
# -> '코스피'

# 지수 OHLCV
df_idx = stock.get_index_ohlcv("20240101", "20250117", "1001")
```

**지수 OHLCV DataFrame 구조:**

| 컬럼명 | 설명 |
|--------|------|
| 날짜 (index) | 거래일 |
| 시가, 고가, 저가, 종가 | 지수값 (float) |
| 거래량 | 거래량 |
| 거래대금 | 거래대금 |
| 상장시가총액 | 해당 지수 구성종목 시총합 |

### 1.10 ETF 조회

```python
# ETF 티커 목록
etf_tickers = stock.get_etf_ticker_list("20250117")

# ETF OHLCV
df_etf = stock.get_etf_ohlcv_by_date("20240101", "20250117", "069500")

# ETF 기본 정보 (PDF - Portfolio Deposit File)
df_etf_pdf = stock.get_etf_portfolio_deposit_file("069500", "20250117")
```

### 1.11 제한사항 및 주의사항

| 항목 | 내용 |
|------|------|
| **Rate Limit** | 공식 제한 없으나, 과도한 요청 시 KRX 서버에서 차단 가능. 요청 간 0.5~1초 sleep 권장 |
| **데이터 범위** | 약 2005년 이후 데이터 제공 (종목에 따라 다름) |
| **실시간 제공** | 불가. 전일까지의 데이터만 조회 가능 |
| **데이터 소스** | KRX 정보데이터시스템 웹 스크래핑 방식 |
| **안정성** | KRX 웹사이트 구조 변경 시 라이브러리가 동작하지 않을 수 있음 |
| **날짜 형식** | 문자열 "YYYYMMDD" 형식 필수 |
| **비거래일 처리** | 비거래일 입력 시 빈 DataFrame 반환 또는 가장 가까운 거래일 데이터 반환 |
| **병렬 요청** | 권장하지 않음. 순차적 요청 권장 |

### 1.12 실제 개발용 코드 예제

```python
"""pykrx를 활용한 데이터 수집 예제"""
import time
import pandas as pd
from pykrx import stock


def get_all_kospi_stocks_fundamental(date: str) -> pd.DataFrame:
    """특정 날짜 KOSPI 전 종목의 주가 + 펀더멘탈 데이터 통합 조회"""
    # OHLCV
    df_ohlcv = stock.get_market_ohlcv(date, market="KOSPI")
    time.sleep(1)

    # 시가총액
    df_cap = stock.get_market_cap(date, market="KOSPI")
    time.sleep(1)

    # 펀더멘탈 (PER, PBR 등)
    df_fund = stock.get_market_fundamental(date, market="KOSPI")
    time.sleep(1)

    # 통합
    df = pd.concat([df_ohlcv, df_cap[["시가총액", "상장주식수"]], df_fund], axis=1)

    # 종목명 추가
    df["종목명"] = df.index.map(lambda x: stock.get_market_ticker_name(x))

    return df


def get_stock_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """개별 종목의 OHLCV + 펀더멘탈 히스토리 조회"""
    df_ohlcv = stock.get_market_ohlcv(start, end, ticker)
    time.sleep(1)

    df_fund = stock.get_market_fundamental(start, end, ticker)
    time.sleep(1)

    df_cap = stock.get_market_cap(start, end, ticker)
    time.sleep(1)

    df = pd.concat([df_ohlcv, df_fund, df_cap[["시가총액", "상장주식수"]]], axis=1)
    return df


# 사용 예시
if __name__ == "__main__":
    # 전 종목 스냅샷
    df_snapshot = get_all_kospi_stocks_fundamental("20250117")
    print(f"종목 수: {len(df_snapshot)}")
    print(df_snapshot.head())

    # 개별 종목 히스토리
    df_samsung = get_stock_history("005930", "20240101", "20250117")
    print(f"\n삼성전자 데이터 건수: {len(df_samsung)}")
    print(df_samsung.tail())
```

---

## 2. FinanceDataReader

### 2.1 개요

한국 및 해외 금융 데이터를 통합 제공하는 파이썬 라이브러리.
Yahoo Finance, KRX, FRED, 네이버 금융 등 다양한 소스에서 데이터를 수집.
API 키 불필요.

### 2.2 설치

```bash
pip install finance-datareader
```

### 2.3 기본 임포트

```python
import FinanceDataReader as fdr
```

### 2.4 종목 리스트 조회

```python
# 한국 KOSPI 상장 종목
df_kospi = fdr.StockListing("KOSPI")

# KOSDAQ 상장 종목
df_kosdaq = fdr.StockListing("KOSDAQ")

# KRX 전체 (KOSPI + KOSDAQ + KONEX)
df_krx = fdr.StockListing("KRX")

# S&P 500 종목
df_sp500 = fdr.StockListing("S&P500")

# NASDAQ 종목
df_nasdaq = fdr.StockListing("NASDAQ")

# NYSE 종목
df_nyse = fdr.StockListing("NYSE")
```

**한국 종목 리스트 DataFrame 구조:**

| 컬럼명 | 설명 |
|--------|------|
| Code | 종목코드 (6자리) |
| ISU_CD | ISIN 코드 |
| Name | 종목명 |
| Market | 시장 (KOSPI/KOSDAQ) |
| Dept | 부서/소속 |
| Close | 종가 |
| ChangeCode | 등락코드 |
| Changes | 전일대비 |
| ChagesRatio | 등락률 |
| Open | 시가 |
| High | 고가 |
| Low | 저가 |
| Volume | 거래량 |
| Amount | 거래대금 |
| Marcap | 시가총액 |
| Stocks | 상장주식수 |
| MarketId | 시장ID |

### 2.5 주가 (OHLCV) 조회

```python
# 삼성전자 주가 (시작일 ~ 현재)
df = fdr.DataReader("005930", "2024-01-01")

# 삼성전자 주가 (기간 지정)
df = fdr.DataReader("005930", "2024-01-01", "2025-01-17")

# 미국 주식 (심볼 사용)
df_aapl = fdr.DataReader("AAPL", "2024-01-01")

# ETF
df_spy = fdr.DataReader("SPY", "2024-01-01")
```

**주가 DataFrame 구조:**

| 컬럼명 | 설명 | dtype |
|--------|------|-------|
| Date (index) | 거래일 | DatetimeIndex |
| Open | 시가 | int64/float64 |
| High | 고가 | int64/float64 |
| Low | 저가 | int64/float64 |
| Close | 종가 | int64/float64 |
| Volume | 거래량 | int64 |
| Change | 등락률 | float64 |

**참고**: 컬럼명이 영어(Open, High, Low, Close, Volume)로 pykrx(시가, 고가, 저가, 종가, 거래량)와 다르다.

### 2.6 지수 데이터 조회

```python
# KOSPI 지수
df_kospi = fdr.DataReader("KS11", "2024-01-01")

# KOSDAQ 지수
df_kosdaq = fdr.DataReader("KQ11", "2024-01-01")

# S&P 500 지수
df_sp500 = fdr.DataReader("US500", "2024-01-01")

# 다우존스
df_dji = fdr.DataReader("DJI", "2024-01-01")

# 나스닥 종합
df_ixic = fdr.DataReader("IXIC", "2024-01-01")

# 닛케이 225
df_n225 = fdr.DataReader("N225", "2024-01-01")
```

**주요 지수 심볼:**

| 심볼 | 지수 |
|------|------|
| KS11 | KOSPI |
| KQ11 | KOSDAQ |
| KS200 | KOSPI 200 |
| US500 | S&P 500 |
| DJI | 다우존스 |
| IXIC | 나스닥 종합 |
| N225 | 닛케이 225 |
| HSI | 항셍 |
| SSEC | 상해종합 |
| VIX | VIX 변동성 |

### 2.7 환율, 암호화폐, 원자재

```python
# 달러/원 환율
df_usdkrw = fdr.DataReader("USD/KRW", "2024-01-01")

# 유로/달러 환율
df_eurusd = fdr.DataReader("EUR/USD", "2024-01-01")

# 비트코인/달러
df_btc = fdr.DataReader("BTC/USD", "2024-01-01")

# 금 선물
df_gold = fdr.DataReader("GC=F", "2024-01-01")

# WTI 원유 선물
df_oil = fdr.DataReader("CL=F", "2024-01-01")
```

### 2.8 거시경제 데이터 (FRED)

```python
# 미국 GDP
df_gdp = fdr.DataReader("FRED:GDP", "2000-01-01")

# 미국 기준금리
df_rate = fdr.DataReader("FRED:DFF", "2020-01-01")

# 한국 기준금리
df_kr_rate = fdr.DataReader("FRED:IRSTCI01KRM156N", "2020-01-01")

# 미국 소비자물가지수 (CPI)
df_cpi = fdr.DataReader("FRED:CPIAUCSL", "2020-01-01")
```

### 2.9 제한사항 및 주의사항

| 항목 | 내용 |
|------|------|
| **Rate Limit** | 명시적 제한 없으나 소스에 따라 다름. 과도한 요청 시 차단 가능 |
| **데이터 범위** | 소스에 따라 다름. 한국 주가는 약 1995년 이후 |
| **실시간 제공** | 불가. 전일까지의 데이터 |
| **펀더멘탈 지표** | PER, PBR 등 개별 조회 함수 없음 (StockListing에서 시가총액 정도만 제공) |
| **데이터 소스** | 네이버 금융, KRX, Yahoo Finance 등 복합 |
| **안정성** | 데이터 소스 웹사이트 변경 시 영향받을 수 있음 |
| **날짜 형식** | "YYYY-MM-DD" 또는 "YYYY" 형식 모두 지원 |
| **해외 주식** | 티커 심볼 기반으로 바로 조회 가능 (강점) |

### 2.10 실제 개발용 코드 예제

```python
"""FinanceDataReader를 활용한 데이터 수집 예제"""
import pandas as pd
import FinanceDataReader as fdr


def get_krx_stock_list() -> pd.DataFrame:
    """KRX 전체 종목 리스트 조회"""
    df = fdr.StockListing("KRX")
    # 주요 컬럼만 선택
    cols = ["Code", "Name", "Market", "Close", "Volume", "Marcap", "Stocks"]
    return df[cols]


def get_multi_stock_prices(
    tickers: list[str],
    start: str,
    end: str = None,
) -> dict[str, pd.DataFrame]:
    """여러 종목 주가 일괄 조회"""
    result = {}
    for ticker in tickers:
        try:
            df = fdr.DataReader(ticker, start, end)
            if not df.empty:
                result[ticker] = df
        except Exception as e:
            print(f"[WARN] {ticker} 조회 실패: {e}")
    return result


def get_market_overview(start: str) -> pd.DataFrame:
    """시장 지수 + 환율 종합 조회"""
    symbols = {
        "KOSPI": "KS11",
        "KOSDAQ": "KQ11",
        "S&P500": "US500",
        "USD/KRW": "USD/KRW",
        "VIX": "VIX",
    }

    dfs = {}
    for name, symbol in symbols.items():
        try:
            df = fdr.DataReader(symbol, start)
            dfs[name] = df["Close"]
        except Exception as e:
            print(f"[WARN] {name} 조회 실패: {e}")

    return pd.DataFrame(dfs)


# 사용 예시
if __name__ == "__main__":
    # 종목 리스트
    stocks = get_krx_stock_list()
    print(f"KRX 전체 종목 수: {len(stocks)}")

    # 주요 종목 주가
    tickers = ["005930", "000660", "035420"]  # 삼성전자, SK하이닉스, NAVER
    prices = get_multi_stock_prices(tickers, "2024-01-01", "2025-01-17")
    for t, df in prices.items():
        print(f"{t}: {len(df)} rows")

    # 시장 종합
    market = get_market_overview("2024-01-01")
    print(market.tail())
```

---

## 3. DART OpenAPI

### 3.1 개요

금융감독원 전자공시시스템(DART)에서 제공하는 공식 Open API.
상장법인의 사업보고서, 재무제표, 공시 정보 등을 프로그래밍 방식으로 조회 가능.
**API 키 필요** (무료 발급).

### 3.2 API 키 발급 방법

1. DART OpenAPI 사이트 접속: https://opendart.fss.or.kr/
2. 회원가입 (공인인증서 또는 간편인증 필요)
3. 로그인 후 "인증키 신청/관리" 메뉴에서 API 키 발급
4. 발급 즉시 사용 가능 (별도 승인 대기 없음)

**Rate Limit:**

| 항목 | 제한 |
|------|------|
| 일일 요청 횟수 | 10,000회 |
| 분당 요청 횟수 | 약 1,000회 (비공식) |
| 동시 접속 | 제한 없음 (합리적 범위) |

### 3.3 주요 엔드포인트 (REST API 직접 사용 시)

**Base URL**: `https://opendart.fss.or.kr/api/`

| 엔드포인트 | 설명 | 주요 파라미터 |
|-----------|------|-------------|
| `company.json` | 기업 개황 | corp_code |
| `list.json` | 공시 목록 조회 | corp_code, bgn_de, end_de, pblntf_ty |
| `fnlttSinglAcnt.json` | 단일회사 주요계정 | corp_code, bsns_year, reprt_code |
| `fnlttSinglAcntAll.json` | 단일회사 전체 재무제표 | corp_code, bsns_year, reprt_code, fs_div |
| `fnlttMultiAcnt.json` | 다중회사 주요계정 | corp_code (콤마 구분 가능) |
| `corpCode.xml` | 고유번호 (corp_code) 전체 목록 | - |

**reprt_code (보고서 코드):**

| 코드 | 보고서 |
|------|--------|
| 11013 | 1분기 보고서 |
| 11012 | 반기 보고서 |
| 11014 | 3분기 보고서 |
| 11011 | 사업보고서 (연간) |

**fs_div (재무제표 구분):**

| 코드 | 설명 |
|------|------|
| OFS | 개별 재무제표 |
| CFS | 연결 재무제표 |

### 3.4 OpenDartReader 라이브러리

DART OpenAPI를 편리하게 사용하기 위한 파이썬 래퍼 라이브러리.

#### 설치

```bash
pip install opendartreader
```

#### 기본 설정

```python
import OpenDartReader

# API 키 직접 전달
dart = OpenDartReader("YOUR_API_KEY")

# 또는 환경변수에서 읽기 (DART_API_KEY)
import os
from dotenv import load_dotenv

load_dotenv()
dart = OpenDartReader(os.getenv("DART_API_KEY"))
```

**.env 파일 예시:**

```
DART_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3.5 기업 정보 조회

```python
# 기업 개황
company = dart.company("005930")  # 삼성전자
# -> corp_code, corp_name, stock_code, ceo_nm, corp_cls,
#    adres, hm_url, ir_url, phn_no, fax_no, est_dt, acc_mt
```

**반환 필드:**

| 필드 | 설명 |
|------|------|
| corp_code | DART 고유번호 (8자리) |
| corp_name | 회사명 |
| stock_code | 종목코드 (6자리) |
| ceo_nm | 대표자명 |
| corp_cls | 법인구분 (Y: 유가, K: 코스닥, N: 코넥스, E: 기타) |
| adres | 주소 |
| est_dt | 설립일 |
| acc_mt | 결산월 |

### 3.6 공시 목록 조회

```python
# 삼성전자 최근 공시 목록
df = dart.list("005930")

# 기간 지정
df = dart.list("005930", start="2024-01-01", end="2025-01-17")

# 공시 유형 지정 (A: 정기공시, B: 주요사항, C: 발행공시 등)
df = dart.list("005930", start="2024-01-01", kind="A")
```

**반환 DataFrame 구조:**

| 컬럼명 | 설명 |
|--------|------|
| corp_code | 고유번호 |
| corp_name | 회사명 |
| stock_code | 종목코드 |
| corp_cls | 법인구분 |
| report_nm | 보고서명 |
| rcept_no | 접수번호 |
| flr_nm | 공시 제출인명 |
| rcept_dt | 접수일자 |
| rm | 비고 |

### 3.7 재무제표 조회 (핵심)

#### 단일회사 주요계정

```python
# 삼성전자 2023년 사업보고서 주요계정 (연결)
df = dart.finstate("005930", 2023)
```

**반환 DataFrame 구조:**

| 컬럼명 | 설명 |
|--------|------|
| rcept_no | 접수번호 |
| reprt_code | 보고서 코드 |
| bsns_year | 사업연도 |
| corp_code | 고유번호 |
| sj_div | 재무제표구분 (BS: 재무상태표, IS: 손익계산서, CIS: 포괄손익계산서, CF: 현금흐름표) |
| sj_nm | 재무제표명 |
| account_id | 계정 ID |
| account_nm | 계정명 (매출액, 영업이익, 당기순이익 등) |
| account_detail | 계정상세 |
| thstrm_nm | 당기명 |
| thstrm_amount | 당기금액 |
| frmtrm_nm | 전기명 |
| frmtrm_amount | 전기금액 |
| bfefrmtrm_nm | 전전기명 |
| bfefrmtrm_amount | 전전기금액 |
| ord | 정렬순서 |

**주요 account_nm 값 예시:**
- 재무상태표 (BS): 자산총계, 부채총계, 자본총계, 유동자산, 비유동자산 등
- 손익계산서 (IS): 매출액(수익), 영업이익, 당기순이익, 법인세비용차감전순이익 등

#### 전체 재무제표 (상세)

```python
# 전체 재무제표 (연결)
df = dart.finstate_all("005930", 2023, reprt_code="11011", fs_div="CFS")

# 개별 재무제표
df = dart.finstate_all("005930", 2023, reprt_code="11011", fs_div="OFS")
```

#### 분기별 조회

```python
# 1분기
df_q1 = dart.finstate("005930", 2024, reprt_code="11013")

# 반기
df_q2 = dart.finstate("005930", 2024, reprt_code="11012")

# 3분기
df_q3 = dart.finstate("005930", 2024, reprt_code="11014")

# 사업보고서 (연간)
df_annual = dart.finstate("005930", 2023, reprt_code="11011")
```

### 3.8 PER, PBR, ROE 등 팩터 계산

DART API는 PER, PBR, ROE를 직접 제공하지 않는다. 재무제표 데이터로 직접 계산해야 한다.

```python
"""DART 재무제표에서 PER, PBR, ROE 계산 예제"""
import pandas as pd
import OpenDartReader


def calculate_valuation_factors(
    dart: OpenDartReader,
    stock_code: str,
    year: int,
    market_cap: int,  # 시가총액 (pykrx 등에서 조회)
    shares: int,       # 발행주식수 (pykrx 등에서 조회)
    current_price: int, # 현재가 (pykrx 등에서 조회)
) -> dict:
    """DART 재무제표 기반 밸류에이션 팩터 계산"""
    df = dart.finstate(stock_code, year)

    if df is None or df.empty:
        return {}

    # 연결재무제표 우선
    df_cfs = df[df["fs_div"] == "CFS"] if "fs_div" in df.columns else df

    # 금액 문자열 -> 숫자 변환 함수
    def get_amount(account_name: str, sj_div: str = None) -> int:
        mask = df_cfs["account_nm"].str.contains(account_name, na=False)
        if sj_div:
            mask = mask & (df_cfs["sj_div"] == sj_div)
        rows = df_cfs[mask]
        if rows.empty:
            return 0
        val = rows.iloc[0]["thstrm_amount"]
        if isinstance(val, str):
            val = val.replace(",", "")
        return int(val) if val else 0

    # 주요 재무 항목 추출
    net_income = get_amount("당기순이익", "IS")        # 당기순이익
    total_equity = get_amount("자본총계", "BS")         # 자본총계 (순자산)
    revenue = get_amount("매출", "IS")                  # 매출액
    operating_income = get_amount("영업이익", "IS")     # 영업이익

    # 팩터 계산
    factors = {}

    # EPS (주당순이익)
    if shares > 0:
        factors["EPS"] = net_income / shares

    # BPS (주당순자산)
    if shares > 0:
        factors["BPS"] = total_equity / shares

    # PER (주가수익비율) = 현재가 / EPS
    if factors.get("EPS", 0) > 0:
        factors["PER"] = current_price / factors["EPS"]

    # PBR (주가순자산비율) = 현재가 / BPS
    if factors.get("BPS", 0) > 0:
        factors["PBR"] = current_price / factors["BPS"]

    # ROE (자기자본이익률) = 당기순이익 / 자본총계
    if total_equity > 0:
        factors["ROE"] = (net_income / total_equity) * 100

    # ROA = 당기순이익 / 자산총계
    total_assets = get_amount("자산총계", "BS")
    if total_assets > 0:
        factors["ROA"] = (net_income / total_assets) * 100

    # 영업이익률
    if revenue > 0:
        factors["영업이익률"] = (operating_income / revenue) * 100

    factors["매출액"] = revenue
    factors["영업이익"] = operating_income
    factors["당기순이익"] = net_income
    factors["자본총계"] = total_equity

    return factors
```

### 3.9 REST API 직접 사용 (OpenDartReader 없이)

```python
"""DART API 직접 호출 예제 (requests 사용)"""
import requests
import pandas as pd
import zipfile
import io
import xml.etree.ElementTree as ET


DART_API_KEY = "YOUR_API_KEY"
BASE_URL = "https://opendart.fss.or.kr/api"


def get_corp_codes() -> pd.DataFrame:
    """전체 기업 고유번호(corp_code) 목록 조회"""
    url = f"{BASE_URL}/corpCode.xml"
    params = {"crtfc_key": DART_API_KEY}
    response = requests.get(url, params=params)

    # ZIP 파일 압축 해제
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    xml_data = zf.read(zf.namelist()[0])

    # XML 파싱
    root = ET.fromstring(xml_data)
    rows = []
    for child in root.findall("list"):
        rows.append({
            "corp_code": child.findtext("corp_code"),
            "corp_name": child.findtext("corp_name"),
            "stock_code": child.findtext("stock_code"),
            "modify_date": child.findtext("modify_date"),
        })

    df = pd.DataFrame(rows)
    # 상장사만 필터 (stock_code가 빈 문자열이 아닌 것)
    df_listed = df[df["stock_code"].str.strip() != ""]
    return df_listed


def get_financial_statements(
    corp_code: str,
    year: int,
    reprt_code: str = "11011",
) -> pd.DataFrame:
    """단일회사 주요계정 조회"""
    url = f"{BASE_URL}/fnlttSinglAcnt.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "000":
        print(f"Error: {data.get('message')}")
        return pd.DataFrame()

    return pd.DataFrame(data["list"])


def get_disclosure_list(
    corp_code: str,
    start: str,
    end: str,
    page_count: int = 100,
) -> pd.DataFrame:
    """공시 목록 조회"""
    url = f"{BASE_URL}/list.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de": start.replace("-", ""),
        "end_de": end.replace("-", ""),
        "page_count": page_count,
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "000":
        return pd.DataFrame()

    return pd.DataFrame(data["list"])
```

### 3.10 주의사항 및 팁

| 항목 | 내용 |
|------|------|
| **corp_code vs stock_code** | DART는 자체 고유번호(corp_code, 8자리)를 사용. 종목코드(stock_code, 6자리)와 다름. corpCode.xml로 매핑 필요 |
| **일일 요청 한도** | 10,000회/일. 전 종목 재무제표 조회 시 주의 필요 |
| **데이터 시차** | 재무제표는 보고서 제출 후 반영. 사업보고서는 회계연도 종료 후 약 90일 내 제출 |
| **금액 단위** | 원(KRW) 단위. 문자열로 반환되는 경우가 있으므로 int 변환 필요 |
| **연결 vs 개별** | 연결재무제표(CFS) 사용 권장. 자회사 없는 기업은 개별(OFS)만 존재 |
| **분기 보고서** | 1분기/반기/3분기 보고서는 누적 금액. 해당 분기만의 실적은 직접 계산 필요 |
| **결산월** | 대부분 12월이나 일부 기업은 3월, 6월 등 다른 결산월 사용 |

---

## 4. 비교 요약

### 4.1 데이터 소스별 제공 데이터

| 데이터 | pykrx | FinanceDataReader | DART OpenAPI |
|--------|:-----:|:----------------:|:------------:|
| 한국 주가 (OHLCV) | **O** | **O** | X |
| 해외 주가 | X | **O** | X |
| 시가총액 | **O** | O (리스트) | X |
| PER/PBR | **O** (일별) | X | 계산 필요 |
| EPS/BPS | **O** | X | 계산 필요 |
| 배당수익률 | **O** | X | X |
| 재무제표 상세 | X | X | **O** |
| 공시 정보 | X | X | **O** |
| 투자자별 매매 | **O** | X | X |
| 지수 | O | **O** | X |
| 환율 | X | **O** | X |
| 거시경제 (FRED) | X | **O** | X |
| ETF | O | **O** | X |
| API 키 필요 | 불필요 | 불필요 | **필요** |

### 4.2 장단점 비교

| 항목 | pykrx | FinanceDataReader | DART OpenAPI |
|------|-------|-------------------|-------------|
| **장점** | PER/PBR 일별 제공, 투자자별 매매, 시가총액 상세 | 해외주식/지수/환율 통합, 심플한 API | 공식 재무제표, 공시 데이터, 신뢰도 높음 |
| **단점** | 한국 시장만, 스크래핑 방식 | 펀더멘탈 지표 미제공 | PER/PBR 직접 계산 필요, API 키 관리 |
| **안정성** | 중 (KRX 사이트 의존) | 중 (다중 소스 의존) | 상 (정부 공식 API) |
| **속도** | 중 | 상 | 중~하 |

### 4.3 퀀트 전략별 데이터 소스 매핑

| 퀀트 전략 | 주 데이터 소스 | 보조 소스 |
|----------|-------------|----------|
| 밸류 팩터 (PER, PBR) | pykrx (일별 PER/PBR) | DART (재무제표 검증) |
| 모멘텀 팩터 | pykrx 또는 FDR (OHLCV) | - |
| 퀄리티 팩터 (ROE, 영업이익률) | DART (재무제표) | pykrx (시가총액) |
| 듀얼 모멘텀 (해외 ETF 포함) | FDR (해외 자산) | pykrx (국내) |
| 수급 분석 | pykrx (투자자별 매매) | - |
| 멀티팩터 | pykrx + DART + FDR 통합 | - |

---

## 5. 개발팀 권장 구현 가이드

### 5.1 requirements.txt 추가 패키지

```
pykrx>=1.0.45
finance-datareader>=0.9.66
opendartreader>=0.2.1
```

### 5.2 데이터 수집 모듈 구조 제안

```
src/data/
├── __init__.py
├── base.py              # 데이터 수집 기본 클래스
├── krx_collector.py     # pykrx 기반 수집기
├── fdr_collector.py     # FinanceDataReader 기반 수집기
├── dart_collector.py    # DART OpenAPI 기반 수집기
├── unified.py           # 통합 데이터 인터페이스
└── cache.py             # 로컬 캐싱 (중복 API 호출 방지)
```

### 5.3 통합 데이터 조회 예제

```python
"""3개 소스를 통합하여 전 종목 팩터 데이터를 수집하는 예제"""
import time
import pandas as pd
from pykrx import stock
import FinanceDataReader as fdr
import OpenDartReader


def build_factor_universe(
    date: str,
    dart_api_key: str,
    year: int,
    market: str = "KOSPI",
) -> pd.DataFrame:
    """
    전 종목 팩터 유니버스 구축

    데이터 소스 역할:
    - pykrx: 주가, 시가총액, PER, PBR, 배당수익률
    - FDR: 종목 리스트 (보조 검증)
    - DART: 재무제표 (ROE, 영업이익률 등)
    """
    # 1. pykrx: 전 종목 기본 데이터
    df_ohlcv = stock.get_market_ohlcv(date, market=market)
    time.sleep(1)
    df_cap = stock.get_market_cap(date, market=market)
    time.sleep(1)
    df_fund = stock.get_market_fundamental(date, market=market)
    time.sleep(1)

    # 통합
    df = pd.concat([
        df_ohlcv[["종가", "거래량", "거래대금"]],
        df_cap[["시가총액", "상장주식수"]],
        df_fund[["PER", "PBR", "EPS", "BPS", "DIV", "DPS"]],
    ], axis=1)

    # 종목명 추가
    tickers = df.index.tolist()
    df["종목명"] = [stock.get_market_ticker_name(t) for t in tickers]

    # 필터링: 거래량 0, 종가 0 제외
    df = df[(df["종가"] > 0) & (df["거래량"] > 0)]

    # 2. DART: 재무제표 기반 팩터 추가 (시간이 오래 걸리므로 배치 처리 권장)
    dart = OpenDartReader(dart_api_key)

    # corp_code 매핑 (stock_code -> corp_code)
    # OpenDartReader는 내부적으로 종목코드 -> corp_code 변환을 지원
    roe_list = []
    for ticker in tickers[:10]:  # 예시: 상위 10종목만
        try:
            fs = dart.finstate(ticker, year)
            if fs is not None and not fs.empty:
                # 당기순이익, 자본총계 추출
                ni_row = fs[fs["account_nm"].str.contains("당기순이익", na=False)]
                eq_row = fs[fs["account_nm"].str.contains("자본총계", na=False)]

                if not ni_row.empty and not eq_row.empty:
                    ni = int(str(ni_row.iloc[0]["thstrm_amount"]).replace(",", ""))
                    eq = int(str(eq_row.iloc[0]["thstrm_amount"]).replace(",", ""))
                    roe = (ni / eq * 100) if eq > 0 else None
                    roe_list.append({"ticker": ticker, "ROE": roe})
            time.sleep(0.5)  # DART Rate Limit 준수
        except Exception as e:
            print(f"[WARN] {ticker} DART 조회 실패: {e}")

    if roe_list:
        df_roe = pd.DataFrame(roe_list).set_index("ticker")
        df = df.join(df_roe, how="left")

    return df


# 사용 예시
if __name__ == "__main__":
    result = build_factor_universe(
        date="20250117",
        dart_api_key="YOUR_DART_API_KEY",
        year=2023,
        market="KOSPI",
    )
    print(f"유니버스 종목 수: {len(result)}")
    print(result[["종목명", "종가", "시가총액", "PER", "PBR"]].head(20))
```

### 5.4 캐싱 전략 권장

```python
"""API 호출 최소화를 위한 로컬 캐싱 패턴"""
import os
import pandas as pd
from datetime import datetime


CACHE_DIR = "/mnt/data/quant/data/cache"


def cached_fetch(
    key: str,
    fetch_fn,
    cache_hours: int = 24,
) -> pd.DataFrame:
    """
    데이터를 로컬 parquet로 캐싱.
    cache_hours 이내의 데이터가 있으면 API 호출 생략.
    """
    cache_path = os.path.join(CACHE_DIR, f"{key}.parquet")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 캐시 확인
    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        if age_hours < cache_hours:
            return pd.read_parquet(cache_path)

    # 새로 조회
    df = fetch_fn()
    if df is not None and not df.empty:
        df.to_parquet(cache_path)

    return df


# 사용 예시
# df = cached_fetch(
#     key="kospi_ohlcv_20250117",
#     fetch_fn=lambda: stock.get_market_ohlcv("20250117", market="KOSPI"),
#     cache_hours=24,
# )
```

### 5.5 에러 처리 및 재시도 패턴

```python
"""API 호출 실패 시 재시도 로직"""
import time
import logging

logger = logging.getLogger(__name__)


def retry_fetch(
    fetch_fn,
    max_retries: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
):
    """지수 백오프 재시도"""
    for attempt in range(max_retries):
        try:
            result = fetch_fn()
            return result
        except Exception as e:
            wait = delay * (backoff ** attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                logger.error(f"All {max_retries} attempts failed.")
                raise
```

---

## 부록: 컬럼명 대응표

개발 시 데이터 소스 간 컬럼명 통일이 필요하다. 아래 매핑을 참고.

| 통합 컬럼명 | pykrx | FinanceDataReader | 설명 |
|-----------|-------|-------------------|------|
| date | 날짜 (index) | Date (index) | 거래일 |
| open | 시가 | Open | 시가 |
| high | 고가 | High | 고가 |
| low | 저가 | Low | 저가 |
| close | 종가 | Close | 종가 |
| volume | 거래량 | Volume | 거래량 |
| amount | 거래대금 | - | 거래대금 |
| change_pct | 등락률 | Change | 등락률 |
| market_cap | 시가총액 | Marcap | 시가총액 |
| shares | 상장주식수 | Stocks | 발행주식수 |
| per | PER | - | 주가수익비율 |
| pbr | PBR | - | 주가순자산비율 |
| eps | EPS | - | 주당순이익 |
| bps | BPS | - | 주당순자산 |
| div_yield | DPS | - | 배당수익률 |
| dps | DIV | - | 주당배당금 |

> **주의**: pykrx의 DIV/DPS 명칭은 일반적인 금융 용어와 반대이다. 통합 시 반드시 확인 필요.
