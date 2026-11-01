"""가상 인물마다 여러 각도 사진을 Muse Image 편집(POST /v1/images/edits)으로 만든다 — 옆모습·숙인 고개를 자연스럽게 치환하려고.

  python scripts/make_donor_views.py               # 모든 가상 인물, 아직 없는 사진만 (사람당 24장 · 장당 $0.01)
  python scripts/make_donor_views.py donor0 donor2 # 고른 사람만

정면 사진을 넣고 '같은 사람, 고개만 돌림'으로 편집한다 (점·귀·머리 모양이 그대로 유지된다).
무표정·같은 조명만 쓴다: 표정이 있는 사진을 원본으로 쓰면 그 표정이 따라오고, 조명이 다르면 원본이 바뀔 때 색이 튄다.
결과는 datacash/donors/views/<이름>/<각도>.jpg (512px, 저장소에 넣으려고 작게 — 34명 × 24장 약 40MB). 치환은 장면 고개 각도에 가장 가까운 사진을 원본으로 쓴다
(datacash.privacy.faceswap.LivePortraitSwap.add_views)."""

import base64
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacash.meta.client import MetaClient, check_budget, record  # noqa: E402
from datacash.meta.image import MODEL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DONORS = ROOT / "datacash" / "donors"
KEEP = ("Same person, same face, same identity, same skin, same hair, same clothes, same lighting, "
        "plain light gray background, neutral relaxed expression, mouth closed. ")
VIEWS = {
    "l20": "Head turned about 20 degrees to the person's left.", "r20": "Head turned about 20 degrees to the person's right.",
    "l45": "Head turned 45 degrees to the person's left, three-quarter view.",
    "r45": "Head turned 45 degrees to the person's right, three-quarter view.",
    "l70": "Head turned 70 degrees to the person's left, almost side profile.",
    "r70": "Head turned 70 degrees to the person's right, almost side profile.",
    "l90": "Full side profile facing the person's left.", "r90": "Full side profile facing the person's right.",
    "up": "Head tilted up, looking slightly upward.", "down": "Head tilted down, looking down at a table in front.",
    "down_l30": "Head tilted down and turned 30 degrees to the person's left, looking at hands.",
    "down_r30": "Head tilted down and turned 30 degrees to the person's right, looking at hands.",
}
for yaw, (k, desc) in itertools.product((30, 50, 65, 80), (("lv", "head level"), ("dn", "head tilted down 25 degrees as if looking at the floor"),
                                                          ("dd", "head bowed down 45 degrees, looking at the hands below"))):
    VIEWS[f"y{yaw}_{k}"] = f"Head turned {yaw} degrees to the side, {desc}, soft studio light."


def make(name: str, client) -> int:
    out = DONORS / "views" / name
    out.mkdir(parents=True, exist_ok=True)
    todo = [k for k in VIEWS if not (out / f"{k}.jpg").exists()]
    if not todo:
        return 0
    check_budget(0.01 * len(todo))

    def one(k):
        for _ in range(2):
            try:
                with open(DONORS / f"{name}.png", "rb") as f:
                    r = client.images.edit(model=MODEL, image=f, prompt=KEEP + VIEWS[k], n=1, size="1024x1024", response_format="b64_json")
                img = cv2.imdecode(np.frombuffer(base64.b64decode(r.data[0].b64_json), np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(str(out / f"{k}.jpg"), cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
                record(MODEL, 0.01, f"{name} view {k}")
                return 1
            except Exception as e:  # noqa: BLE001 — 한 장 실패는 다음 실행에서 다시
                print(name, k, e)
        return 0

    with ThreadPoolExecutor(6) as ex:
        return sum(ex.map(one, todo))


def main():
    names = sys.argv[1:] or sorted(p.stem for p in DONORS.glob("donor*.png"))
    c = MetaClient(timeout=180).client   # 연결이 끊긴 요청에 몇 시간 멈춰 있던 적이 있다
    for n in names:
        print(n, make(n, c), "장")


if __name__ == "__main__":
    main()
