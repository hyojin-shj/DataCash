"""Space 비밀값 넣기 — .env 와 data/server 의 토큰 파일에서 읽어 Hugging Face Space Secrets 에 (값은 화면에 찍지 않는다).

  .venv/bin/python scripts/set_space_secrets.py
"""

from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE = "simhyojin024/datacash"

env = {}
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
sec = {"DATACASH_TOKEN": "admin",      # 운영자 화면 — 심사 기간(11/5~8)엔 README 에 적은 'admin'. 끝나면 data/server/.admin_token 으로 되돌린다
       "DATACASH_APP_TOKEN": (ROOT / "data/server/.app_token").read_text().strip()}    # 앱 (APK 에 든 것과 같아야 한다)
for k in ("MODEL_API_KEY", "FIREWORKS_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "DC_S3_BUCKET", "baseten_api", "baseten_deploy", "DATACASH_BASETEN_URL"):
    if env.get(k):
        sec[k] = env[k]
api = HfApi()
for k, v in sec.items():
    api.add_space_secret(SPACE, k, v)
print("넣음:", ", ".join(sorted(sec)))
