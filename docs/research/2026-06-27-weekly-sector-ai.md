# [주간 섹터 리포트] AI — 2026-06-27

**데이터 기준일**: 2026-06-20 ~ 2026-06-26 (직전 7일 영업일 1순위, [JAE-88](/JAE/issues/JAE-88) 규칙 7~9 적용)
**레이어 구조**: 공통 문서 `valuechain-layers-reference.md` A섹션(5개 레이어) 전 레이어 포함
**관점**: 산업 분석 (투자 추천 아님)

---

## 0. 섹터 전체 요약 및 금주 White List

이번 주 AI 섹터는 **한국 메모리가 글로벌 HBM 사이클의 중심**임을 다시 각인시킨 주였다. **SK하이닉스(000660)가 사상 처음 코스피 시가총액 1위(보통주 기준)로 올라서며 25년 7개월 만의 대장주 교체**가 발생했고([한국경제, 2026-06-22](https://www.hankyung.com/amp/202606220331i)), **Micron(MU)이 6/24 회계 Q3 어닝 서프라이즈로 시간외 약 +14.6% 급등**([Investing.com, 2026-06-24](https://www.investing.com/news/transcripts/earnings-call-transcript-micron-tops-q3-2026-estimates-shares-jump-146-93CH-4759504))하며 HBM 수요 구조적 강세를 재확인했다. 반면 클라우드(L2)·파운데이션 모델(L3)·AI SaaS(L4)·데이터 인프라(L5)는 7일 내 단독 catalyst가 약했다.

### White List — 1순위 (직전 7일 명확한 catalyst)

| 종목 | 레이어 | 근거 (직전 7일) | 출처 |
|------|--------|------------------|------|
| **SK하이닉스(000660)** | L1 | 6/22 코스피 시총 1위 첫 등극(보통주 기준), HBM 수익성 동력 | 한국경제, 2026-06-22 |
| **Micron(MU)** | L1 | 6/24 회계 Q3 서프라이즈, 시간외 +14.6%, HBM3E/4 완판 언급 | CNBC/Investing.com, 2026-06-24 |
| **한미반도체(042700)** | L1 | TC본더(HBM 패키징) 수요 강세, 약 29만원대 | 핀포인트뉴스, 2026-06 |
| **이오테크닉스(039030)** | L1 | 레이저 공정장비 수요 기대 +8%대 | 핀포인트뉴스, 2026-06 |

### White List — 2순위 관찰 후보 (신뢰도 낮음 / 7일 외 데이터 의존)

| 종목 | 레이어 | 사유 (데이터 부족) |
|------|--------|---------------------|
| Snowflake(SNOW) | L5 | 월 +31.6% 모멘텀·BofA H2 톱픽, 단 실적은 5월말~6월초(배경) |
| 더존비즈온(012510) | L4 | ONE AI 교체수요 펀더멘털 양호하나 7일 내 단독 트리거 미확인 |
| NVIDIA(NVDA) | L1 | 6/22 유럽 35개 AI 슈퍼컴·Halos 발표이나 단독 주가 catalyst 약함 |

---

## L1. 반도체·HBM·AI 가속기

> **요약**: 한국 메모리가 주의 핵심. SK하이닉스 코스피 시총 1위 첫 등극 + Micron 호실적으로 HBM 사이클 강세 재확인. 국내 HBM 장비주(한미·이오테크닉스) 동반 강세. 단기 쏠림·과열 경계.

**주요 회사 (글로벌)**
- **Micron(MU)**: 6/24 회계 Q3 어닝 서프라이즈, 시간외 **+14.6%**. HBM3E/HBM4 2027년까지 완판, HBM4 고객 인증 진행 언급. *(매출 절대치는 검색 소스 간 모순 → 수치 미검증, 주가 반응만 채택)* [CNBC/Investing.com, 2026-06-24]
- **NVIDIA(NVDA)**: 6/23 종가 $200.04. 6/22 유럽 35개 신규 AI 슈퍼컴 발표, Halos(로보틱스 안전 풀스택), 6/23 BioNeMo Agent Toolkit 발표 [NVIDIA Newsroom, 2026-06-22~23].
- **AMD·Broadcom(AVGO)·Marvell(MRVL)**: 6/25 동반 하락(AMD -2.45%, AVGO -2.54%, MRVL -2.87%) — 주중 칩주 단기 변동성 [heygotrade, 2026-06-25]. Broadcom Q2 실적은 6/3 (배경).
- **TSMC**: 28nm 레거시 25%+ 감산, 2/3nm 선단공정 집중 전략 보도 [디지털타임스, 2026-06-25]. 1Q 파운드리 점유율 TSMC 72.3% vs 삼성 6.5% [서울신문, 2026-06-12 (배경)].

**주요 회사 (한국)**
- **SK하이닉스(000660)**: 6/22 삼성전자(보통주) 시총을 추월하며 사상 첫 코스피 시총 1위 등극(SK하이닉스 약 ₩2,911,000 / 시총 약 2,083.9조 vs 삼성 보통주 약 2,081.3조). MR-MUF 패키징 기반 HBM 수율·영업이익률이 동력 [한국경제/Businesskorea, 2026-06-22]. *주의: 삼성 보통주+우선주 합산 기준으로는 역전 여부 이견 존재 → 헤드라인 리스크* [라이센스뉴스].
- **삼성전자(005930)**: 시총 1위 내주었으나 HBM4 엔비디아·AMD 테스트 통과·6월 공급 개시 스토리 잔존(배경). HBM4 수율 60% 미만으로 SK하이닉스 추격 중.
- **한미반도체(042700)**: HBM TC본더 수요 확대 기대로 강세, 약 29만원대 [핀포인트뉴스, 2026-06].
- **이오테크닉스(039030)**: 레이저 공정장비 수요 기대로 +8%대 상승 [핀포인트뉴스, 2026-06].
- **리노공업(058470)**: 7일 내 단독 catalyst 미확인 (테스트 소켓 구조적 수혜는 배경).

**금주 주요 이슈·이벤트**
- (+) SK하이닉스 시총 1위 + Micron 호실적 → HBM 수요 구조적 강세 재확인, 국내 HBM 장비주로 온기 확산.
- (−) 시총 역전 "보통주 vs 합산" 논란 → 헤드라인 리스크. 미국 칩주 6/25 동반 조정 → 단기 차익실현 압력.
- 차주 영향: Micron 가이던스가 7월 초 SK하이닉스·삼성 HBM 센티먼트에 긍정 스필오버 가능. 코스피 반도체 쏠림(9,100선)의 피로도 주시.

**관련 ETF**
- 글로벌: SOXX·SMH·SOXL — 6/25 미국 칩주 조정 영향 단기 약세.
- 한국: ACE AI반도체포커스(469150)·KODEX 반도체(091160) — SK하이닉스·한미·이오테크닉스 비중 수혜. **단 SK하이닉스 단일종목 쏠림 심화로 코스피200·반도체 ETF 집중도 리스크 확대**.

**차주 투자 관점 레이어 점수: 9/10** — HBM catalyst 밀도 최고, 단 단기 과열·쏠림 경계.

---

## L2. 클라우드·하이퍼스케일러·인프라

> **요약**: 2026 하이퍼스케일러 capex 합산 $660~725B(전년比 약 +77% 추정)로 AI 인프라 투자 사이클 지속 확인. 단 7일 윈도 내 신규 단독 catalyst는 약함.

**주요 회사 (글로벌)**
- **2026 capex 집계(보도)**: Amazon ~$200B, Microsoft ~$190B, Alphabet $175–185B(클라우드 백로그 $460B), Meta $125–145B, Oracle ~$50B(+136%). 합산 $660–690B, 약 75% AI 인프라 [Futurum/Tom's Hardware/CreditSights]. *집계성 자료, 특정일 catalyst 아님*.
- **Oracle(ORCL)**: Q4 FY26 매출 +21% $19.2B, OCI +93%, RPO $638B. FY27 capex 최대 $95B + $40B 조달 계획에 6/11 -10% 급락 — **6/10~6/11, 윈도우 밖(배경)**.

**주요 회사 (한국)**
- **네이버(035420)**: 엔비디아 DSX 기반 '소버린 AI 팩토리' 55MW→1GW 로드맵 발표(6/8, **윈도우 밖**). AI 투자비(연 1조원+ GPU) → 수익성 딜레마 지속 보도.
- **카카오(035720)**: 2026 '투자의 해'로 비용 증가 우려 지속.
- **KT(030200)**: 7일 내 단독 catalyst 미확인.

**금주 주요 이슈·이벤트**
- (+) 글로벌 capex 상향 사이클 → L1 메모리/HBM 수요의 펀더멘털 뒷받침.
- (−) 국내 빅테크 AI 투자비 부담·수익성 희석 우려 지속.
- 차주 영향: 7월 말 美 빅테크 2Q 실적 시즌 전 capex 가이던스 변동 주시.

**관련 ETF**
- 글로벌: CLOU·SKYY·WCLD — capex 상향 수혜 구조.
- 한국: TIGER 미국나스닥100(133690) 간접 노출. 국내 인터넷 ETF는 네이버·카카오 수익성 우려로 상대 부진 가능.

**차주 투자 관점 레이어 점수: 6/10** — 펀더멘털 견조하나 7일 단독 catalyst 부재.

---

## L3. 파운데이션 모델·LLM·기반 AI

> **요약**: 7일 내 메이저 모델 출시·펀딩 단독 catalyst 제한적. 가장 큰 흐름은 OpenAI IPO 준비설(9월 가능성, 밸류 $730B)과 Gemini 3.5 Pro 6월 출시 예고이나, 정확한 발표일이 7일 윈도에 떨어진다고 확정하기 어려움.

**주요 회사 (글로벌)**
- **OpenAI(비상장)**: Goldman·Morgan Stanley와 IPO 비공개 제출 준비, 9월 상장 가능성, 사적 밸류 $730B (검색 종합, *날짜 특정 불명확*).
- **Alphabet Gemini**: I/O(5/19)에서 Gemini 3.5 Pro "다음달(6월)" 출시 예고. 현재 3.5 Flash만 가용, Pro 정식 출시일 미공개.
- **xAI Grok 5**: Colossus 2에서 학습 중, 6/30 이전 출시 확률 낮음(Polymarket 12~33%).
- **Anthropic**: Claude Fable 5(6/9, **윈도우 밖**).

**주요 회사 (한국)**
- **네이버 HyperCLOVA X(035420)**: 소버린 AI 전략으로 검색·커머스 옴니채널 통합 진행(6월 맥락). 신규 모델 catalyst는 윈도우 내 미확인.
- **LG AI연구원(LG, 003550 우회)**: 7일 내 catalyst 미확인.

**금주 주요 이슈·이벤트**
- (+) OpenAI IPO·Gemini Pro 임박 기대 → AI 플랫폼 센티먼트 우호.
- (−) Grok 5 지연, 모델 출시일 불확실. **상장 모델주 기준 7일 내 catalyst 미확인** (OpenAI·xAI 비상장).
- 차주 영향: Gemini 3.5 Pro 실제 출시일·OpenAI IPO 공식화 여부가 트리거.

**관련 ETF**
- 글로벌: CHAT·AIQ — 비상장 비중 커 직접 노출 제한, Alphabet·Meta 통해 간접.
- 한국: TIGER 글로벌AI&로보틱스INDXX(464310) 간접.

**차주 투자 관점 레이어 점수: 5/10** — 기대는 크나 7일 내 확정 catalyst 부족(상장사 직접 수혜 약함).

---

## L4. AI 응용·SaaS·생산성 (Copilot·Agentic AI)

> **요약**: 미국 엔터프라이즈 SaaS는 2026년 약세 지속(PLTR·NOW·ADBE 큰 폭 하락), 6/25 밸류에이션 비교 기사로 저점 매수 논의 등장. 국내는 더존비즈온 ONE AI 교체수요 스토리 부각이나 7일 내 신규 단독 catalyst 약함.

**주요 회사 (글로벌)**
- **밸류에이션 비교(6/25)**: ADBE forward P/E ~10.3(YTD -31%, 5년 신저가권), CRM ~14.1, PLTR ~104.9(2026 EPS +74.7% 기대), NOW·PLTR 연초比 각각 -33%/-23% [24/7 Wall St., 2026-06-25].
- **Salesforce(CRM)**: Agentforce ARR $1.2B(+205%) (실적은 윈도우 밖, 6/25 비교기사로 재부각).
- **Palantir(PLTR)**: Rule of 40 = 145이나 실적 후 -22.28% (배경).

**주요 회사 (한국)**
- **더존비즈온(012510)**: ONE AI 통합 모듈 교체수요로 3Q 매출 +18%·영업이익 +73%, 목표주가 ₩116,500(BUY) (실적/목표가는 누적 자료·배경). 7일 내 단독 트리거 미확인.
- **한글과컴퓨터(030520)**: 7일 내 catalyst 미확인.

**금주 주요 이슈·이벤트**
- (+) ADBE/CRM 저PER 부각 기사로 저점 매수 논의.
- (−) PLTR·NOW·ADBE YTD 큰 폭 하락 지속, 모멘텀 약세.
- 차주 영향: 7월 말 SaaS 2Q 실적 시즌 전 Agentforce/AI 에이전트 ARR 모멘텀 확인 필요.

**관련 ETF**
- 글로벌: AIQ·WTAI·IGV — 2026 약세 지속으로 상대 부진.
- 한국: HANARO Fn K-AI플러스(417810).

**차주 투자 관점 레이어 점수: 5/10** — 밸류 매력은 생겼으나 주가 모멘텀·7일 catalyst 약함.

---

## L5. 데이터·MLOps·벡터DB·검색

> **요약**: 소프트웨어 섹터 반등 분위기. Snowflake 강세(월 +31.6%), BofA가 SNOW·MDB H2 톱픽 선정. 단 대부분 실적/업그레이드는 5월말~6월초로 7일 윈도 경계선.

**주요 회사 (글로벌)**
- **Snowflake(SNOW)**: 최근 1개월 +31.6%, product revenue +34% YoY, BofA H2 2026 톱픽 [Motley Fool, 2026-06-03; CoinCentral] — 실적은 5/31~6/3 경계, 모멘텀은 주중까지 연속.
- **MongoDB(MDB)**: Q4 FY26 매출 $695M(+27%), Atlas +29%, 벡터서치 고객 YoY 2배 (실적 윈도우 밖).
- **Confluent(CFLT)**: CapitalOne Overweight 상향 (날짜 윈도우 경계).
- **Elastic(ESTC)**: 6/24 종가 $58.71, 단독 catalyst 약함 [MacroTrends/Google Finance, 2026-06-24].

**주요 회사 (한국)**
- 직접 상장 노출 부재. 7일 내 catalyst 미확인.

**금주 주요 이슈·이벤트**
- (+) 벡터DB·AI 워크로드 확장 + BofA 톱픽으로 데이터 인프라 재평가.
- (−) 개별 catalyst 대부분 윈도우 밖, 7일 내 신규 트리거 제한.
- 차주 영향: 벡터서치/AI 데이터 워크로드 수요가 SNOW·MDB 모멘텀 지속 여부 결정.

**관련 ETF**
- 글로벌: AIQ·WTAI — SNOW·MDB·CFLT·ESTC 분산 노출, 소프트웨어 반등 시 베타 수혜.
- 한국: 직접 노출 부재.

**차주 투자 관점 레이어 점수: 6/10** — 섹터 반등 우호적, 단 7일 내 단독 catalyst는 SNOW 모멘텀 외 약함.

---

## 관련 출처

**직전 7일 (1순위)**
- [한국경제 — SK하이닉스 삼성전자 시총 첫 추월 (2026-06-22)](https://www.hankyung.com/amp/202606220331i)
- [Businesskorea — SK하이닉스 시총 1위 등극 (2026-06-22)](https://www.businesskorea.co.kr/news/articleView.html?idxno=271751)
- [CNBC — Micron Q3 2026 earnings (2026-06-24)](https://www.cnbc.com/2026/06/24/micron-mu-earnings-report-q3-2026.html)
- [Investing.com — Micron Q3, shares +14.6% (2026-06-24)](https://www.investing.com/news/transcripts/earnings-call-transcript-micron-tops-q3-2026-estimates-shares-jump-146-93CH-4759504)
- [디지털타임스 — TSMC 2나노, 삼성 파운드리 (2026-06-25)](https://www.dt.co.kr/article/12069444)
- [heygotrade — AI chip stocks AMD/AVGO/MRVL (2026-06-25)](https://www.heygotrade.com/en/news/ai-chip-stocks-nvidia-broadcom-marvell-rally/)
- [24/7 Wall St. — Enterprise software metric (2026-06-25)](https://247wallst.com/investing/2026/06/25/look-past-the-ai-hype-the-cold-hard-financial-metric-telling-you-exactly-which-enterprise-software-giant-to-buy-right-now/)
- [핀포인트뉴스 — 한미반도체·이오테크닉스 HBM (2026-06)](https://www.pinpointnews.co.kr/news/articleView.html?idxno=457452)

**배경 (7일 외)**
- [라이센스뉴스 — 시총 1위 역전 오보 해프닝](https://www.lcnews.co.kr/news/articleView.html?idxno=203804)
- [서울신문 — TSMC 1Q 점유율 70%+ (2026-06-12)](https://www.seoul.co.kr/news/economy/2026/06/12/20260612500275)
- [Oracle IR — Record Q4 FY2026 (2026-06-10)](https://investor.oracle.com/investor-news/news-details/2026/Oracle-Announces-Record-Q4-and-FY-2026-Results-Driven-by-Cloud-Infrastructure--Cloud-Applications/default.aspx)
- [AI매터스 — 네이버 NVIDIA DSX (2026-06-08)](https://aimatters.co.kr/news-report/44935/)
- [Motley Fool — Snowflake vs MongoDB 2026 (2026-06-03)](https://www.fool.com/coverage/better-buy/2026/06/03/snowflake-vs-mongodb-which-technology-stock-is-a-better-buy-in-2026/)

*레이어 구조는 `valuechain-layers-reference.md` A섹션 정의를 따른다. 본 리포트는 산업 분석이며 투자 추천이 아니다.*
