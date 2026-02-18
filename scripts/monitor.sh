#!/bin/bash
# Phase 4 팀별 에이전트 모니터링 - tmux 4분할

SESSION="quant-teams"
TASK_DIR="/tmp/claude-0/-mnt-data-quant/tasks"

# 에이전트 ID (Phase 4)
DEV_A="a4aaf80"    # Dev Team A: 최적화+ML+리스크
DEV_B="ab0bbf8"    # Dev Team B: WebSocket+인프라+시각화
QA="ab00e19"       # QA Team: 테스트

# 기존 세션 종료
tmux kill-session -t "$SESSION" 2>/dev/null

# 새 세션 생성
tmux new-session -d -s "$SESSION" -x 200 -y 50

# Dev A (좌상) - 최적화+ML
tmux send-keys -t "$SESSION" "echo -e '\\033[1;32m=== Dev Team A: 최적화+ML+리스크 ===\\033[0m'; tail -f $TASK_DIR/$DEV_A.output 2>/dev/null | python3 -c \"
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

# Dev B (우상) - WebSocket+인프라
tmux split-window -h -t "$SESSION"
tmux send-keys -t "$SESSION" "echo -e '\\033[1;36m=== Dev Team B: WebSocket+인프라+시각화 ===\\033[0m'; tail -f $TASK_DIR/$DEV_B.output 2>/dev/null | python3 -c \"
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
tmux send-keys -t "$SESSION" "echo -e '\\033[1;31m=== QA Team: Phase 4 테스트 ===\\033[0m'; tail -f $TASK_DIR/$QA.output 2>/dev/null | python3 -c \"
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
tmux send-keys -t "$SESSION" "echo -e '\\033[1;35m=== Phase 4 진행 현황 ===\\033[0m'; watch -n 3 'echo \"--- Phase 4 신규 파일 ---\"; ls -la /mnt/data/quant/src/optimization/*.py /mnt/data/quant/src/ml/*.py /mnt/data/quant/src/strategy/risk_parity.py /mnt/data/quant/src/strategy/ml_factor.py /mnt/data/quant/src/report/risk_metrics.py /mnt/data/quant/src/report/plotly_charts.py /mnt/data/quant/src/execution/kis_websocket.py /mnt/data/quant/src/execution/realtime_manager.py /mnt/data/quant/src/alert/email_sender.py /mnt/data/quant/scripts/quant-bot.service /mnt/data/quant/scripts/install.sh 2>/dev/null | awk \"{print \\$5, \\$NF}\"; echo \"\"; echo \"--- Task Output Sizes ---\"; wc -l /tmp/claude-0/-mnt-data-quant/tasks/{$DEV_A,$DEV_B,$QA}.output 2>/dev/null'" Enter

echo ""
echo "========================================="
echo "  Phase 4 tmux 모니터링 세션 생성 완료"
echo "========================================="
echo ""
echo "  다른 터미널에서 접속:"
echo "    tmux attach -t quant-teams"
echo ""
echo "  팀 구성:"
echo "    좌상: Dev A (최적화+ML+리스크)"
echo "    우상: Dev B (WebSocket+인프라+시각화)"
echo "    좌하: QA (테스트)"
echo "    우하: 진행 현황"
echo ""
