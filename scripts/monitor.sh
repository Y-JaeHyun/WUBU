#!/bin/bash
# 팀별 에이전트 모니터링 - tmux 4분할

SESSION="quant-teams"
TASK_DIR="/tmp/claude-0/-mnt-data-quant/tasks"

# 로그에서 텍스트만 추출하는 함수 (jq 사용)
parse_cmd() {
    local file="$1"
    local label="$2"
    echo "echo -e '\\033[1;33m=== $label ===\\033[0m'; tail -f $file 2>/dev/null | while IFS= read -r line; do echo \"\$line\" | python3 -c \"
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        msg = d.get('message', {})
        role = msg.get('role', '')
        if role == 'assistant':
            for c in msg.get('content', []):
                if isinstance(c, dict) and c.get('type') == 'text':
                    print(c['text'][:200])
                elif isinstance(c, dict) and c.get('type') == 'tool_use':
                    print(f'  [Tool] {c[\\\"name\\\"]}')
        elif 'toolUseResult' in d:
            r = str(d['toolUseResult'])[:100]
            if r and r != 'None':
                print(f'  > {r}')
    except:
        pass
\" 2>/dev/null; done"
}

# 기존 세션 종료
tmux kill-session -t "$SESSION" 2>/dev/null

# 새 세션 생성
tmux new-session -d -s "$SESSION" -x 200 -y 50

# Research (좌상)
tmux send-keys -t "$SESSION" "echo -e '\\033[1;36m[Research Team] 데이터소스 조사\\033[0m'; tail -f $TASK_DIR/a214e4e.output 2>/dev/null | python3 -c \"
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

# Dev (우상)
tmux split-window -h -t "$SESSION"
tmux send-keys -t "$SESSION" "echo -e '\\033[1;32m[Dev Team] 핵심 모듈 개발\\033[0m'; tail -f $TASK_DIR/a76035c.output 2>/dev/null | python3 -c \"
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

# Planning (좌하)
tmux select-pane -t "$SESSION:0.0"
tmux split-window -v -t "$SESSION"
tmux send-keys -t "$SESSION" "echo -e '\\033[1;35m[Planning Team] Phase 2 기획\\033[0m'; tail -f $TASK_DIR/a7d375f.output 2>/dev/null | python3 -c \"
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

# QA Status (우하)
tmux select-pane -t "$SESSION:0.2"
tmux split-window -v -t "$SESSION"
tmux send-keys -t "$SESSION" "echo -e '\\033[1;31m[QA Team] 대기중 - Dev 완료 후 시작\\033[0m'; echo ''; watch -n 5 'echo \"=== Task Status ===\"; ls -la $TASK_DIR/*.output 2>/dev/null | awk \"{print \\$NF, \\$5, \\$6, \\$7, \\$8}\"'" Enter

echo ""
echo "========================================="
echo "  tmux 모니터링 세션이 생성되었습니다"
echo "========================================="
echo ""
echo "  다른 터미널에서 접속:"
echo "    tmux attach -t quant-teams"
echo ""
echo "  tmux 조작법:"
echo "    Ctrl+B, 방향키  : 패널 이동"
echo "    Ctrl+B, z       : 패널 최대화/복원"
echo "    Ctrl+B, d       : 세션 분리(detach)"
echo ""
