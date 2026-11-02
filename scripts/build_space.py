"""Hugging Face Space 묶음 만들기 · 올리기 — 서버 + 구매 기업 웹을 클라우드에 (맥을 꺼 둬도 심사위원이 열 수 있게).

  .venv/bin/python scripts/build_space.py            data/space_build/ 에 묶음만
  .venv/bin/python scripts/build_space.py --push     만들고 simhyojin024/datacash Space 에 올린다

넣는 것: 코드(datacash · web) · MediaPipe/OpenCV 모델 · 시연 데이터 중 이미 판매용으로 처리된 것만
  (인증서 · 처리 기록 · LeRobot 묶음, 웹 예시 녹화의 얼굴 바꾼 영상, 의뢰 미션).
넣지 않는 것: 원본 녹화(sessions) · 얼굴 특징값(people) · 계좌(payout) · 이의 신청 · 토큰 · .env.
비밀값(토큰·API 키)은 Space 설정의 Secrets 에 따로 넣는다.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "server"
OUT = ROOT / "data" / "space_build"
SPACE = "simhyojin024/datacash"

REQUIREMENTS = """numpy==2.5.4
scipy==1.18.1
pandas==3.1.0
scikit-learn==1.9.1
opencv-contrib-python-headless==5.0.0.93
mediapipe==1.1.1
pyarrow==26.0.0
numba==0.68.0
openai==3.37.0
python-dotenv==1.2.4
fastapi==0.162.0
uvicorn[standard]==0.54.0
python-multipart==0.0.32
jinja2==3.1.6
httpx==0.28.1
soundfile==0.14.0
imageio-ffmpeg==0.6.0
meta-sam-parser==0.1.0
boto3==1.43.123
pyyaml==6.0.3
"""


def build() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    shutil.copytree(ROOT / "datacash", OUT / "datacash", ignore=ign)
    shutil.copytree(ROOT / "web", OUT / "web", ignore=ign)
    shutil.copy(ROOT / "pyproject.toml", OUT)
    for f in ("Dockerfile", "start.sh", "README.md"):
        shutil.copy(ROOT / "deploy" / "space" / f, OUT)
    (OUT / "requirements.txt").write_text(REQUIREMENTS)

    seed = OUT / "seed"
    shutil.copytree(ROOT / "data" / "models", seed / "models", ignore=ign)
    for f in ("sync_bias.json", "device_latency.json"):
        if (ROOT / "data" / f).exists():
            shutil.copy(ROOT / "data" / f, seed / f)
    srv = seed / "server"
    samples = json.loads((SRC / "samples.json").read_text())
    srv.mkdir(parents=True)
    shutil.copy(SRC / "samples.json", srv / "samples.json")
    n = 0
    for cert in sorted((SRC / "out").glob("*/certificate.json")):
        sid = cert.parent.name
        o = srv / "out" / sid
        o.mkdir(parents=True)
        for f in ("certificate.json", "certificate.md", "job.json"):
            if (cert.parent / f).exists():
                shutil.copy(cert.parent / f, o / f)
        if (cert.parent / "lerobot").exists():
            shutil.copytree(cert.parent / "lerobot", o / "lerobot", ignore=ign)
        if sid in samples:                     # 웹 예시 장면: 얼굴 바꾸고 가린 영상만 (원본·SAM 입력 영상은 넣지 않는다)
            v = next((cert.parent / f for f in ("masked_swap.mp4", "masked.mp4") if (cert.parent / f).exists()), None)
            if v:
                shutil.copy(v, o / v.name)
        st = SRC / "status" / f"{sid}.json"
        if st.exists():
            (srv / "status").mkdir(exist_ok=True)
            shutil.copy(st, srv / "status" / st.name)
        n += 1
    apps = {"datacash.apk": ROOT / "app/app/build/outputs/apk/debug/app-debug.apk",          # 이 Space 주소로 빌드한 앱 (local.properties dc.server)
            "datacash-wear.apk": ROOT / "app/wear/build/outputs/apk/debug/wear-debug.apk"}
    (srv / "dist").mkdir()
    for name, f in apps.items():
        shutil.copy(f, srv / "dist" / name)
    if (SRC / "requests").exists():
        shutil.copytree(SRC / "requests", srv / "requests", ignore=ign)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"묶음: {OUT} · 녹화 {n}편 (예시 영상 {len(samples)}편) · {size:.0f} MB")
    return OUT


def push(folder: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(SPACE, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=SPACE, repo_type="space", commit_message="DataCash 서버 + 구매 기업 웹",
                      delete_patterns=["*"])
    print(f"올림: https://huggingface.co/spaces/{SPACE}  →  https://{SPACE.replace('/', '-')}.hf.space")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    f = build()
    if a.push:
        push(f)
