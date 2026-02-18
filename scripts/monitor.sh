#!/bin/bash
# Phase 3 팀별 에이전트 모니터링 - tmux 4분할

SESSION="quant-teams"
TASK_DIR="/tmp/claude-0/-mnt-data-quant/tasks"

# 에이전트 ID (Phase 3)
DEV_A="ac98d4e"    # Dev Team A: 전략+데이터
DEV_B="a65c6eb"    # Dev Team B: 실행+스케줄러
QA="ac525b5"       # QA Team: 테스트

# 로그 파싱 함수
parse_agent() {
    python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        msg = d.get('message', {})
        if msg.get('role') == 'assistant':
            for c in msg.get('content', []):
                if isinstance(c, dict) and c.get('type') == 'text':
                    t = c['text'][:300]
                    if t.strip(): print(t)
                elif isinstance(c, dict) and c.get('type') == 'tool_use':
                    name = c.get('name', '')
                    inp = str(c.get('input',{}))[:80]
                    print(f'  [Tool] {name}({inp})')
    except: pass
" 2>/dev/null
}

# 기존 세션 종료
tmux kill-session -t "$SESSION" 2>/dev/null

# 새 세션 생성
tmux new-session -d -s "$SESSION" -x 200 -y 50

# Dev A (좌상) - 전략+데이터
tmux send-keys -t "$SESSION" "echo -e '\\033[1;32m=== Dev Team A: 전략+데이터 모듈 ===\\033[0m'; tail -f $TASK_DIR/$DEV_A.output 2>/dev/null | python3 -c \"
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        msg = d.get('message', {})
        if msg.get('role') == 'assistant':
            for c in msg.get('content', []):
                if isinstance(c, dict) and c.get('type') == 'text':
                    t = c['text'][:300]
                    if t.strip(): print(t)
                elif isinstance(c, dict) and c.get('type') == 'tool_use':
                    print(f'  [Tool] {c.get(\\\"name\\\", \\\"\\\")}({str(c.get(\\\"input\\\",{}))[:80]})')
    except: pass
\"" Enter

# Dev B (우상) - 실행+스케줄러
tmux split-window -h -t "$SESSION"
tmux send-keys -t "$SESSION" "echo -e '\\033[1;36m=== Dev Team B: 실행+스케줄러 ===\\033[0m'; tail -f $TASK_DIR/$DEV_B.output 2>/dev/null | python3 -c \"
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        msg = d.get('message', {})
        if msg.get('role') == 'assistant':
            for c in msg.get('content', []):
                if isinstance(c, dict) and c.get('type') == 'text':
                    t = c['text'][:300]
                    if t.strip(): print(t)
                elif isinstance(c, dict) and c.get('type') == 'tool_use':
                    print(f'  [Tool] {c.get(\\\"name\\\", \\\"\\\")}({str(c.get(\\\"input\\\",{}))[:80]})')
    except: pass
\"" Enter

# QA (좌하) - 테스트
tmux select-pane -t "$SESSION:0.0"
tmux split-window -v -t "$SESSION"
tmux send-keys -t "$SESSION" "echo -e '\\033[1;31m=== QA Team: 테스트 작성 ===\\033[0m'; tail -f $TASK_DIR/$QA.output 2>/dev/null | python3 -c \"
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        msg = d.get('message', {})
        if msg.get('role') == 'assistant':
            for c in msg.get('content', []):
                if isinstance(c, dict) and c.get('type') == 'text':
                    t = c['text'][:300]
                    if t.strip(): print(t)
                elif isinstance(c, dict) and c.get('type') == 'tool_use':
                    print(f'  [Tool] {c.get(\\\"name\\\", \\\"\\\")}({str(c.get(\\\"input\\\",{}))[:80]})')
    except: pass
\"" Enter

# Status (우하) - 파일 현황
tmux select-pane -t "$SESSION:0.2"
tmux split-window -v -t "$SESSION"
tmux send-keys -t "$SESSION" "echo -e '\\033[1;35m=== Phase 3 진행 현황 ===\\033[0m'; watch -n 3 'echo \"--- 생성된 파일 ---\"; ls -la /mnt/data/quant/src/execution/*.py /mnt/data/quant/src/scheduler/*.py /mnt/data/quant/src/strategy/quality.py /mnt/data/quant/src/strategy/three_factor.py /mnt/data/quant/src/strategy/dual_momentum.py /mnt/data/quant/src/data/dart_collector.py /mnt/data/quant/src/data/etf_collector.py /mnt/data/quant/tests/test_quality.py /mnt/data/quant/tests/test_three_factor.py /mnt/data/quant/tests/test_dual_momentum.py /mnt/data/quant/tests/test_n_factor_combiner.py /mnt/data/quant/tests/test_execution.py /mnt/data/quant/tests/test_scheduler.py 2>/dev/null | awk \"{print \\$5, \\$NF}\"; echo \"\"; echo \"--- Task Output Sizes ---\"; wc -l /tmp/claude-0/-mnt-data-quant/tasks/{ac98d4e,a65c6eb,ac525b5}.output 2>/dev/null'" Enter

echo ""
echo "========================================="
echo "  Phase 3 tmux 모니터링 세션 생성 완료"
echo "========================================="
echo ""
echo "  다른 터미널에서 접속:"
echo "    tmux attach -t quant-teams"
echo ""
echo "  팀 구성:"
echo "    좌상: Dev A (전략+데이터)"
echo "    우상: Dev B (실행+스케줄러)"
echo "    좌하: QA (테스트)"
echo "    우하: 진행 현황"
echo ""
