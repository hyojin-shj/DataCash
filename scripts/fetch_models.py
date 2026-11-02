"""직접 서버를 켜는 사람용: 모델 파일 내려받기 (data/models + 얼굴 치환 LivePortrait 를 data/liveportrait 에). 이미 있으면 건너뛴다.

  .venv/bin/python scripts/fetch_models.py            # 전부 (약 700MB)
  .venv/bin/python scripts/fetch_models.py --no-swap  # 얼굴 치환 빼고 (약 70MB, 얼굴은 흐리게)
"""

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datacash import config  # noqa: E402

MP = "https://storage.googleapis.com/mediapipe-models"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
FILES = {   # 이름: (주소, 크기)
    "hand_landmarker.task": (f"{MP}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", 7819105),
    "pose_landmarker_full.task": (f"{MP}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task", 9398198),
    "face_landmarker.task": (f"{MP}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", 3758596),
    "efficientdet_lite0.tflite": (f"{MP}/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite", 13836895),
    "yunet.onnx": (f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx", 232589),
    "sface.onnx": (f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx", 38696353),
}
LP_REPO, LP_COMMIT = "https://github.com/KwaiVGI/LivePortrait", "9b294b3d0536135442ea73cb01e6cb3ca7029dd3"   # MIT


def models():
    config.MODELS.mkdir(parents=True, exist_ok=True)
    for name, (url, size) in FILES.items():
        p = config.MODELS / name
        if p.exists() and p.stat().st_size == size:
            print("있음", name)
            continue
        print("받는 중", name)
        urllib.request.urlretrieve(url, p)
        if p.stat().st_size != size:
            sys.exit(f"{name} 크기가 다름 ({p.stat().st_size} != {size})")


def liveportrait():
    lp = config.DATA / "liveportrait"
    if not (lp / ".git").exists():
        print("받는 중 LivePortrait 코드")
        subprocess.run(["git", "clone", "-q", LP_REPO, str(lp)], check=True)
    subprocess.run(["git", "-C", str(lp), "checkout", "-q", LP_COMMIT], check=True)
    w = lp / "pretrained_weights"
    if (w / "liveportrait" / "base_models" / "spade_generator.pth").exists():
        print("있음 LivePortrait 가중치")
        return
    print("받는 중 LivePortrait 가중치 (약 630MB)")
    from huggingface_hub import snapshot_download

    snapshot_download("KwaiVGI/LivePortrait", local_dir=str(w), allow_patterns=["liveportrait/*", "insightface/*"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-swap", action="store_true", help="얼굴 치환(LivePortrait) 은 받지 않는다")
    a = ap.parse_args()
    models()
    if not a.no_swap:
        liveportrait()
    print("끝:", config.MODELS, "" if a.no_swap else config.DATA / "liveportrait")
