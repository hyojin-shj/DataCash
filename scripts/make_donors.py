"""Muse Image 로 실존하지 않는 가상 인물(얼굴 치환용)을 더 만든다 — 전 세계 참여자에게 맞는 사람을 고를 수 있게.

  python scripts/make_donors.py            # DONORS 가운데 아직 없는 사람만 만든다 (장당 $0.01)
원본(1024px)은 data/donors/, 치환용(768px)은 datacash/donors/, 성별·나이·지역은 datacash/donors/donors.json.
얼굴을 찾지 못한 사진은 한 번 다시 만든다."""

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacash.meta.image import generate  # noqa: E402
from datacash.privacy.faceswap import FaceLandmarks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, RAW = ROOT / "datacash" / "donors", ROOT / "data" / "donors"
INFO = OUT / "donors.json"
BASE = ("Photorealistic passport-style portrait photo of a fictional {who}, {age}, facing the camera straight on, "
        "head and upper shoulders, neutral expression, eyes open, mouth closed, no glasses, no hat, no jewelry, "
        "natural skin texture, even soft studio lighting, plain light gray background, sharp focus.")
GROUPS = [("east_asian", "East Asian"), ("southeast_asian", "Southeast Asian"), ("south_asian", "South Asian"),
          ("black", "Black African"), ("white", "white European"), ("latino", "Hispanic Latino"),
          ("middle_eastern", "Middle Eastern")]
AGES = [(22, "in their early twenties"), (65, "in their mid sixties with grey hair and natural wrinkles")]
EXISTING = {"donor0": ("female", 30, "east_asian"), "donor1": ("male", 30, "east_asian"), "donor2": ("female", 20, "east_asian"),
            "donor3": ("male", 50, "east_asian"), "donor4": ("female", 40, "white"), "donor5": ("male", 20, "white")}


# 참여자 대부분이 한국 사람이라 동아시아 가상 인물을 나이대·성별마다 더 (11-01) — 같은 성별·나이대·무리 안에서
# '덜 닮은 사람'을 고를 수 있게 (셋뿐이면 고를 것이 없어 재식별이 그대로였다). 얼굴형·머리 모양을 달리한다.
EA_AGES = [(25, "in their mid twenties"), (35, "in their mid thirties"), (45, "in their mid forties"), (55, "in their mid fifties")]
EA_LOOKS = {"female": ["round face, straight black hair past the shoulders", "oval face with high cheekbones, short bob haircut",
                       "long face, wavy dark brown hair tied back"],
            "male": ["round face, short black hair", "angular face with a strong jaw, buzz cut", "oval face, side-parted medium length hair"]}


def plan():
    k = 6
    for g, gdesc in GROUPS:
        for age, adesc in AGES:
            for sex, word in (("female", "woman"), ("male", "man")):
                yield f"donor{k}", sex, age, g, BASE.format(who=f"{gdesc} {word}", age=adesc)
                k += 1
    for age, adesc in EA_AGES:
        for sex, word in (("female", "woman"), ("male", "man")):
            for look in EA_LOOKS[sex]:
                yield f"donor{k}", sex, age, "east_asian", BASE.format(who=f"East Asian {word}, {look}", age=adesc)
                k += 1


def main():
    info = json.loads(INFO.read_text()) if INFO.exists() else {
        k: {"gender": g, "age": a, "group": r} for k, (g, a, r) in EXISTING.items()}
    fl = FaceLandmarks()
    for name, sex, age, group, prompt in plan():
        if (OUT / f"{name}.png").exists():
            continue
        for attempt in range(2):
            raw = generate(prompt, RAW, what=f"{name}")[0]
            img = cv2.imread(str(raw))
            if img is not None and fl(img):
                break
            print(name, "얼굴을 못 찾아 다시")
        cv2.imwrite(str(OUT / f"{name}.png"), cv2.resize(img, (768, 768), interpolation=cv2.INTER_AREA))
        info[name] = {"gender": sex, "age": age, "group": group}
        INFO.write_text(json.dumps(info, indent=1, ensure_ascii=False))
        print(name, sex, age, group)


if __name__ == "__main__":
    main()
