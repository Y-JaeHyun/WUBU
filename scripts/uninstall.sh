#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Quant Trading Bot — 제거 스크립트
# 사용법: sudo bash scripts/uninstall.sh
# ─────────────────────────────────────────────────────────────

SERVICE_FILE="quant-bot.service"

echo "=== Quant Trading Bot 제거 시작 ==="

# 1. 서비스 중지
echo "[1/4] 서비스 중지..."
if systemctl is-active --quiet ${SERVICE_FILE} 2>/dev/null; then
    systemctl stop ${SERVICE_FILE}
    echo "  서비스 중지 완료."
else
    echo "  서비스가 실행 중이 아닙니다."
fi

# 2. 서비스 비활성화 및 파일 제거
echo "[2/4] 서비스 비활성화..."
if systemctl is-enabled --quiet ${SERVICE_FILE} 2>/dev/null; then
    systemctl disable ${SERVICE_FILE}
fi

if [ -f "/etc/systemd/system/${SERVICE_FILE}" ]; then
    rm -f "/etc/systemd/system/${SERVICE_FILE}"
    echo "  서비스 파일 제거 완료."
fi

# 3. systemd 데몬 리로드
echo "[3/4] systemd 데몬 리로드..."
systemctl daemon-reload

# 4. logrotate 설정 제거
echo "[4/4] logrotate 설정 제거..."
if [ -f "/etc/logrotate.d/quant-bot" ]; then
    rm -f "/etc/logrotate.d/quant-bot"
    echo "  logrotate 설정 제거 완료."
else
    echo "  logrotate 설정이 없습니다."
fi

echo ""
echo "=== 제거 완료 ==="
echo ""
echo "참고: 프로젝트 디렉토리, 가상환경, 로그, 데이터 파일은 보존됩니다."
echo "완전 삭제가 필요하면 수동으로 제거하세요:"
echo "  rm -rf /mnt/data/quant/.venv"
echo "  rm -rf /mnt/data/quant/logs"
