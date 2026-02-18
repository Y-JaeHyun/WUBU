#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Quant Trading Bot — 설치 스크립트
# 사용법: sudo bash scripts/install.sh
# ─────────────────────────────────────────────────────────────

PROJECT_DIR="/mnt/data/quant"
VENV_DIR="${PROJECT_DIR}/.venv"
SERVICE_FILE="quant-bot.service"
LOGROTATE_CONF="logrotate-quant.conf"

echo "=== Quant Trading Bot 설치 시작 ==="

# 1. 시스템 사용자 생성 (없으면)
if ! id -u quant &>/dev/null; then
    echo "[1/6] 시스템 사용자 'quant' 생성..."
    useradd --system --no-create-home --shell /usr/sbin/nologin quant
else
    echo "[1/6] 사용자 'quant' 이미 존재합니다."
fi

# 2. 가상환경 생성 및 패키지 설치
echo "[2/6] Python 가상환경 설정..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet

if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
    "${VENV_DIR}/bin/pip" install -r "${PROJECT_DIR}/requirements.txt" --quiet
    echo "  requirements.txt 설치 완료."
else
    echo "  requirements.txt가 없습니다. 수동 설치가 필요할 수 있습니다."
fi

# 3. 로그 디렉토리 생성
echo "[3/6] 로그 디렉토리 생성..."
mkdir -p "${PROJECT_DIR}/logs"
chown -R quant:quant "${PROJECT_DIR}/logs"

# 4. 데이터 디렉토리 생성
echo "[4/6] 데이터 디렉토리 생성..."
mkdir -p "${PROJECT_DIR}/data"
chown -R quant:quant "${PROJECT_DIR}/data"

# 5. systemd 서비스 등록
echo "[5/6] systemd 서비스 등록..."
cp "${PROJECT_DIR}/scripts/${SERVICE_FILE}" /etc/systemd/system/${SERVICE_FILE}
systemctl daemon-reload
systemctl enable ${SERVICE_FILE}
echo "  서비스 등록 완료. 시작하려면: sudo systemctl start ${SERVICE_FILE}"

# 6. logrotate 설정
echo "[6/6] logrotate 설정..."
if [ -f "${PROJECT_DIR}/scripts/${LOGROTATE_CONF}" ]; then
    cp "${PROJECT_DIR}/scripts/${LOGROTATE_CONF}" /etc/logrotate.d/quant-bot
    echo "  logrotate 설정 완료."
else
    echo "  logrotate 설정 파일이 없습니다. 건너뜁니다."
fi

# 권한 설정
echo "프로젝트 디렉토리 권한 설정..."
chown -R quant:quant "${VENV_DIR}"

echo ""
echo "=== 설치 완료 ==="
echo ""
echo "다음 단계:"
echo "  1. .env 파일을 확인하세요: ${PROJECT_DIR}/.env"
echo "  2. 서비스 시작: sudo systemctl start ${SERVICE_FILE}"
echo "  3. 상태 확인: sudo systemctl status ${SERVICE_FILE}"
echo "  4. 로그 확인: sudo journalctl -u ${SERVICE_FILE} -f"
