"""프로젝트 설정 관리."""

import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")

# 타임존
TIMEZONE = "Asia/Seoul"

# 증권사 API 설정 (추후 선정 후 채움)
BROKER_API_KEY = os.getenv("BROKER_API_KEY", "")
BROKER_API_SECRET = os.getenv("BROKER_API_SECRET", "")
BROKER_ACCOUNT_NO = os.getenv("BROKER_ACCOUNT_NO", "")

# 데이터 설정
DEFAULT_MARKET = "KRX"
CACHE_EXPIRE_HOURS = 24
