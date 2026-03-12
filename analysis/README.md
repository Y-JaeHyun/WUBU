# Analysis — Ralph 분석 결과 디렉토리

이 디렉토리는 Ralph Loop가 생성하는 **커밋 대상 분석 결과물**을 보관한다.
환경 설정/프롬프트/spec 등 비커밋 파일은 `ralph/` 디렉토리에 위치한다.

## 구조

```
analysis/
├── backtest/          # 전략 백테스트 비교 분석 결과
│   ├── comparison_report.md    # 전략별 순위 및 운영환경 비교
│   └── {strategy}_detail.md    # 전략별 리밸런싱 상세 로그
├── review/            # 코드 리뷰 리포트
│   └── code_review_report.md
└── research/          # 전략 R&D 결과
    ├── strategy_candidates.md  # 후보 전략 목록 (입력)
    ├── strategy_design_*.md    # 전략 설계서 (출력)
    └── rnd_comparison_report.md
```
