#!/bin/bash
# Quant Agent Teams - tmux 4분할 모니터링
# 사용법: bash scripts/monitor.sh
# 접속: tmux attach -t quant-teams

SESSION="quant-teams"

# 기존 세션 종료
tmux kill-session -t "$SESSION" 2>/dev/null

# 새 세션 생성 (pane 0)
tmux new-session -d -s "$SESSION" -x 220 -y 55

# pane 0 → split-h → pane 0(left), pane 1(right)
tmux split-window -h -t "$SESSION:0.0"

# pane 0(left) → split-v → pane 0(top-left), pane 1(bottom-left), pane 2(right)
tmux split-window -v -t "$SESSION:0.0"

# pane 2(right) → split-v → pane 2(top-right), pane 3(bottom-right)
tmux split-window -v -t "$SESSION:0.2"

# ────────────────────────────────────────
# pane 0 (좌상): Researcher
# ────────────────────────────────────────
tmux send-keys -t "$SESSION:0.0" "watch -t -n 5 bash -c '
echo -e \"\\033[1;34m══════════════════════════════════════\\033[0m\"
echo -e \"\\033[1;34m  🔵 RESEARCHER (리서치팀)\\033[0m\"
echo -e \"\\033[1;34m══════════════════════════════════════\\033[0m\"
echo \"\"
echo \"📋 담당: 기술 조사, 전략 리서치, 시장 분석\"
echo \"📂 디렉토리: docs/\"
echo \"\"
echo \"── 할당된 태스크 ──\"
python3 -c \"
import json, os, glob
task_dir = os.path.expanduser(\\\"~/.claude/tasks/peppy-twirling-nova\\\")
found = False
for f in sorted(glob.glob(os.path.join(task_dir, \\\"*.json\\\"))):
    try:
        t = json.load(open(f))
        if t.get(\\\"owner\\\") == \\\"researcher\\\":
            s = t.get(\\\"status\\\", \\\"\\\")
            icon = {\\\"pending\\\": \\\"⏳\\\", \\\"in_progress\\\": \\\"🔄\\\", \\\"completed\\\": \\\"✅\\\"}.get(s, \\\"❓\\\")
            print(f\\\"  {icon} [{s}] {t.get(\\\"subject\\\", \\\"\\\")}\\\")
            found = True
    except: pass
if not found: print(\\\"  (할당된 태스크 없음)\\\")
\"
echo \"\"
echo \"── 최근 docs/ 변경 ──\"
ls -lt /mnt/data/quant/docs/*.md 2>/dev/null | head -5 | awk \"{print \\\"  \\\" \\\$6, \\\$7, \\\$8, \\\$NF}\"
'" Enter

# ────────────────────────────────────────
# pane 1 (좌하): Tester
# ────────────────────────────────────────
tmux send-keys -t "$SESSION:0.1" "watch -t -n 5 bash -c '
echo -e \"\\033[1;33m══════════════════════════════════════\\033[0m\"
echo -e \"\\033[1;33m  🟡 TESTER (QA팀)\\033[0m\"
echo -e \"\\033[1;33m══════════════════════════════════════\\033[0m\"
echo \"\"
echo \"📋 담당: 단위 테스트, 품질 보증\"
echo \"📂 디렉토리: tests/\"
echo \"\"
echo \"── 할당된 태스크 ──\"
python3 -c \"
import json, os, glob
task_dir = os.path.expanduser(\\\"~/.claude/tasks/peppy-twirling-nova\\\")
found = False
for f in sorted(glob.glob(os.path.join(task_dir, \\\"*.json\\\"))):
    try:
        t = json.load(open(f))
        if t.get(\\\"owner\\\") == \\\"tester\\\":
            s = t.get(\\\"status\\\", \\\"\\\")
            icon = {\\\"pending\\\": \\\"⏳\\\", \\\"in_progress\\\": \\\"🔄\\\", \\\"completed\\\": \\\"✅\\\"}.get(s, \\\"❓\\\")
            print(f\\\"  {icon} [{s}] {t.get(\\\"subject\\\", \\\"\\\")}\\\")
            found = True
    except: pass
if not found: print(\\\"  (할당된 태스크 없음)\\\")
\"
echo \"\"
echo \"── 테스트 현황 ──\"
echo \"  총 테스트 파일: \$(find /mnt/data/quant/tests -name test_*.py | wc -l)개\"
echo \"\"
echo \"── 최근 tests/ 변경 ──\"
ls -lt /mnt/data/quant/tests/test_*.py 2>/dev/null | head -5 | awk \"{print \\\"  \\\" \\\$6, \\\$7, \\\$8, \\\$NF}\"
'" Enter

# ────────────────────────────────────────
# pane 2 (우상): Developer
# ────────────────────────────────────────
tmux send-keys -t "$SESSION:0.2" "watch -t -n 5 bash -c '
echo -e \"\\033[1;32m══════════════════════════════════════\\033[0m\"
echo -e \"\\033[1;32m  🟢 DEVELOPER (개발팀)\\033[0m\"
echo -e \"\\033[1;32m══════════════════════════════════════\\033[0m\"
echo \"\"
echo \"📋 담당: 핵심 기능 구현 (src/)\"
echo \"📂 디렉토리: src/data, src/strategy, src/backtest\"
echo \"\"
echo \"── 할당된 태스크 ──\"
python3 -c \"
import json, os, glob
task_dir = os.path.expanduser(\\\"~/.claude/tasks/peppy-twirling-nova\\\")
found = False
for f in sorted(glob.glob(os.path.join(task_dir, \\\"*.json\\\"))):
    try:
        t = json.load(open(f))
        if t.get(\\\"owner\\\") == \\\"developer\\\":
            s = t.get(\\\"status\\\", \\\"\\\")
            icon = {\\\"pending\\\": \\\"⏳\\\", \\\"in_progress\\\": \\\"🔄\\\", \\\"completed\\\": \\\"✅\\\"}.get(s, \\\"❓\\\")
            print(f\\\"  {icon} [{s}] {t.get(\\\"subject\\\", \\\"\\\")}\\\")
            found = True
    except: pass
if not found: print(\\\"  (할당된 태스크 없음)\\\")
\"
echo \"\"
echo \"── Git 상태 ──\"
cd /mnt/data/quant && git diff --stat HEAD 2>/dev/null | tail -5
echo \"\"
echo \"── 최근 src/ 변경 ──\"
ls -lt \$(find /mnt/data/quant/src -name \"*.py\" -type f 2>/dev/null) 2>/dev/null | head -5 | awk \"{print \\\"  \\\" \\\$6, \\\$7, \\\$8, \\\$NF}\"
'" Enter

# ────────────────────────────────────────
# pane 3 (우하): Planner + 전체 현황
# ────────────────────────────────────────
tmux send-keys -t "$SESSION:0.3" "watch -t -n 5 bash -c '
echo -e \"\\033[1;35m══════════════════════════════════════\\033[0m\"
echo -e \"\\033[1;35m  🟣 PLANNER + 전체 현황\\033[0m\"
echo -e \"\\033[1;35m══════════════════════════════════════\\033[0m\"
echo \"\"
echo \"── 전체 태스크 현황 ──\"
python3 -c \"
import json, os, glob
task_dir = os.path.expanduser(\\\"~/.claude/tasks/peppy-twirling-nova\\\")
stats = {\\\"pending\\\": 0, \\\"in_progress\\\": 0, \\\"completed\\\": 0}
for f in sorted(glob.glob(os.path.join(task_dir, \\\"*.json\\\"))):
    try:
        t = json.load(open(f))
        s = t.get(\\\"status\\\", \\\"pending\\\")
        stats[s] = stats.get(s, 0) + 1
        icon = {\\\"pending\\\": \\\"⏳\\\", \\\"in_progress\\\": \\\"🔄\\\", \\\"completed\\\": \\\"✅\\\"}.get(s, \\\"❓\\\")
        owner = t.get(\\\"owner\\\", \\\"미배정\\\")
        tid = os.path.basename(f).replace(\\\".json\\\", \\\"\\\")
        print(f\\\"  #{tid} {icon} [{owner:10s}] {t.get(\\\"subject\\\", \\\"\\\")[:40]}\\\")
    except: pass
print(f\\\"\\\")
print(f\\\"  📊 대기: {stats[\\\"pending\\\"]} | 진행: {stats[\\\"in_progress\\\"]} | 완료: {stats[\\\"completed\\\"]}\\\")
\"
echo \"\"
echo \"── 팀 멤버 ──\"
python3 -c \"
import json, os
team_f = os.path.expanduser(\\\"~/.claude/teams/peppy-twirling-nova/config.json\\\")
if os.path.exists(team_f):
    cfg = json.load(open(team_f))
    colors = {\\\"blue\\\": \\\"🔵\\\", \\\"green\\\": \\\"🟢\\\", \\\"yellow\\\": \\\"🟡\\\", \\\"purple\\\": \\\"🟣\\\"}
    for m in cfg.get(\\\"members\\\", []):
        c = colors.get(m.get(\\\"color\\\",\\\"\\\"), \\\"⚪\\\")
        print(f\\\"  {c} {m.get(\\\"name\\\",\\\"\\\")} ({m.get(\\\"agentType\\\",\\\"\\\")})\\\")
\"
echo \"\"
echo \"── 서비스 ──\"
echo \"  quant-bot: \$(systemctl is-active quant-bot 2>/dev/null || echo unknown)\"
echo \"\"
echo \"  \$(date \"+%Y-%m-%d %H:%M:%S\") KST\"
echo \"  업타임: \$(uptime -p 2>/dev/null)\"
'" Enter

echo ""
echo "========================================="
echo "  Quant Teams tmux 모니터링 세션 생성 완료"
echo "========================================="
echo ""
echo "  접속: tmux attach -t quant-teams"
echo ""
echo "  ┌──────────────────┬──────────────────┐"
echo "  │ 🔵 Researcher    │ 🟢 Developer     │"
echo "  ├──────────────────┼──────────────────┤"
echo "  │ 🟡 Tester        │ 🟣 Planner+현황  │"
echo "  └──────────────────┴──────────────────┘"
echo ""
