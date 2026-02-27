# EOD 공시 모니터링 + 투자 교육 기능 리뷰

## 개요

장 마감 후(15:40) EOD 공시 모니터링에 투자 교육 콘텐츠를 추가하여,
각 공시 유형별 영향도 분석과 대응 방안을 자동으로 제공하는 기능이다.

## 아키텍처

```
스케줄러 (15:40 EOD)
  └── NewsCollector.format_eod_with_education()
        ├── fetch_recent_disclosures()  ← DART API
        ├── filter_important()          ← 키워드 기반 필터링
        ├── analyze_disclosure_impact() ← 교육 정보 + 우선순위
        │     ├── _categorize()         ← 카테고리 판별
        │     ├── _DISCLOSURE_EDUCATION ← 8개 카테고리 교육 데이터
        │     └── priority scoring      ← risk + 보유종목 부스트
        ├── format (보유종목/기타 분리)
        └── _generate_learning_tip()    ← 당일 핵심 학습 팁
```

## 8개 공시 카테고리 및 영향 분석

| 카테고리 | 리스크 | 영향 요약 | 권장 조치 |
|----------|--------|-----------|-----------|
| **delisting** | critical | 투자금 전액 손실 위험 | 즉시 매도 강력 권고 |
| **merger** | high | 기업가치 재평가, 희석 가능 | 합병비율/시너지 분석 |
| **rights_issue** | high | 주식 가치 희석 | 증자 목적 확인 |
| **earnings** | medium | 주가 직접 영향 | 컨센서스 대비 확인 |
| **major_shareholder** | medium | 경영권 이슈 | 지분율 변화 방향 확인 |
| **investment** | medium | 내용에 따라 다름 | 공시 전문 확인 |
| **buyback** | low | 주주환원, 하방 지지 | 소각 여부 확인 |
| **dividend** | low | 주주환원 의지 | 배당 증감 확인 |

## 우선순위 스코어링

기본 리스크 점수:
- critical = 4, high = 3, medium = 2, low = 1

보유종목 부스트:
- 보유종목(stock_code 일치) → **+3 가산**

최종 priority 매핑:
- 4+ → "critical"
- 3 → "high"
- 2 → "medium"
- 1 → "low"

예시:
- earnings(medium=2) + 보유종목(+3) = 5 → **critical** (최우선 체크)
- delisting(critical=4) + 비보유 = 4 → **critical**
- buyback(low=1) + 보유종목(+3) = 4 → **critical**
- rights_issue(high=3) + 비보유 = 3 → **high**

## 교육 콘텐츠

각 공시에 다음 3가지 교육 정보가 추가된다:

1. **영향(impact)**: 해당 공시 유형이 주가/기업가치에 미치는 일반적 영향
2. **조치(action)**: 투자자로서 취해야 할 구체적 행동 가이드
3. **리스크(risk)**: critical / high / medium / low 등급

분류되지 않은 공시는 기본값("공시 전문 확인 필요", risk=low)이 적용된다.

## 스케줄러 통합

기존 15:40 EOD 뉴스 작업(`_eod_news_check`)에서 `format_eod_with_education()`을
호출하여 교육 콘텐츠가 포함된 리포트를 텔레그램으로 전송할 수 있다.

```python
# 기존: format_eod_news()
# 확장: format_eod_with_education(disclosures, holdings)
```

## 출력 포맷 예시

```
[EOD 공시 모니터링] 2026-02-27
================================
[보유종목 공시] (우선 체크)
  1. [보유] 삼성전자 - 분기보고서 (earnings)
     영향: 실적 공시는 주가에 직접적 영향. 컨센서스 대비 서프라이즈/쇼크 확인 필요
     조치: 서프라이즈 → 추가매수 검토, 쇼크 → 손절/비중축소 검토
     리스크: medium

[기타 주요 공시]
  2. SK하이닉스 - 유상증자결정 (rights_issue)
     영향: 유상증자는 주식 가치 희석. 기존 주주 가치 하락 가능
     조치: 증자 목적(시설/운영/차환) 확인, 투기적 증자는 부정적
     리스크: high

[투자 학습 팁]
  오늘의 핵심: 유상증자 공시가 나온 경우, ...
```

## 테스트 커버리지

- `TestDisclosureEducation` (4 tests): 카테고리별 교육 정보 매핑 검증
- `TestEodFormat` (2 tests): 보유종목 유/무에 따른 출력 포맷 검증
- `TestAnalyzeImpact` (2 tests): 보유종목 우선순위 부스트 및 빈 입력 처리
