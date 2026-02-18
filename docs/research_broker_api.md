# 한국 증권사 API 비교 분석

> 조사 기준: Python + Linux 서버 환경 퀀트 개발 관점 (2025.05 기준)

## 종합 비교표

| 항목 | 한국투자증권 | 키움증권 | eBest (신규) | 대신증권 |
|------|:---:|:---:|:---:|:---:|
| **API 방식** | REST+WS | COM/OCX | REST+WS | COM |
| **커뮤니티 Python 라이브러리** | mojito2, pykis | pykiwoom | 성장중 | 일부 |
| **Linux/서버 지원** | **O** | **X** | **O** | **X** |
| **Docker/Cloud 배포** | **O** | **X** | **O** | **X** |
| **실시간 시세** | O (WS) | O (COM) | O (WS) | O (COM) |
| **주문 기능** | O | O | O | O |
| **해외주식** | **O (강점)** | 제한적 | 제한적 | 제한적 |
| **모의투자** | O | O | O | 제한적 |
| **문서화 수준** | 상 | 중 | 중 | 중하 |
| **커뮤니티 크기** | 중상 | 최대 | 중 | 중하 |
| **Rate Limit** | ~20/초 | ~5/초 | ~25/초 | ~4/초 |
| **기술 현대성** | 최신 | 레거시 | 최신 | 레거시 |

## 추천 순위

### 1위: 한국투자증권 (KIS Developers OpenAPI)
- REST + WebSocket -> Linux/서버 완벽 지원
- 문서화 우수, 커뮤니티 Python 패키지(mojito2, pykis) 활성화
- 해외주식 API 지원, 모의투자 제공
- OAuth 인증, 토큰 유효기간 24시간

### 2위: eBest투자증권 (eBest OpenAPI)
- REST + WebSocket -> Linux 서버 지원
- 기존 xingAPI의 풍부한 TR 종류를 REST로 전환
- 파생상품(선물/옵션) 지원 양호
- Rate Limit 초당 25회로 가장 넉넉

### 3위: 키움증권 (Open API+)
- 가장 큰 커뮤니티, 학습 자료 풍부
- **COM/OCX 기반 Windows 전용** -> Linux 서버 불가
- Rate Limit 초당 5회로 제한적

### 4위: 대신증권 (크레온 API)
- COM 기반 Windows 전용, Linux 불가
- 커뮤니티 축소 추세

## 데이터 수집 보조 도구
- **pykrx**: KRX 데이터 스크래핑 (증권사 API Rate Limit 부담 없이 과거 데이터 확보)
- **DART OpenAPI**: 전자공시 데이터
- **FinanceDataReader**: 다양한 금융 데이터 소스 통합

## 결론
**Python + Linux 서버 환경에서는 한국투자증권(KIS)을 1순위로 선택.**
파생상품 위주라면 eBest도 좋은 대안.
