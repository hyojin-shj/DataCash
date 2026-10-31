"""경로와 설정. 키는 .env(MODEL_API_KEY)에서만 읽는다 — 코드·저장소에 넣지 않는다."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATACASH_DATA", ROOT / "data"))
MODELS = DATA / "models"          # MediaPipe .task 파일 (git 밖)
OPENDATA = DATA / "opendata"      # 라이선스 확인된 Wikimedia Commons 영상 (git 밖)
OUT = Path(os.environ.get("DATACASH_OUT", ROOT / "out"))

API_BASE = "https://api.meta.ai/v1"
API_BUDGET_USD = float(os.environ.get("DATACASH_API_BUDGET_USD", "100"))  # 이 장부 기준 누적 사용 한도 ($10 → 10-27 오후 $8.6 사용 시점에서 $20 더 → 10-31 $2 더 → 10-31 계정 잔액 $121 확인, $28.9 사용 시점에서 $100 으로)
LEDGER = DATA / "api_ledger.jsonl"

# 브랜드 (기획 이미지 docs/1026_datacash_brand.png)
BRAND = {"name": "DataCash", "name_ko": "데이터캐시", "primary": "#FF5F5F", "accent": "#FFAF5F",
         "pure": "#FFFFFF", "charcoal": "#333333"}


def load_env() -> None:
    """저장소 루트의 .env 를 환경 변수로 읽는다 (이미 있는 값은 덮지 않는다)."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def api_key() -> str:
    load_env()
    key = os.environ.get("MODEL_API_KEY", "")
    if not key:
        raise RuntimeError("MODEL_API_KEY 가 없습니다 (.env 에 넣어 주세요)")
    return key
