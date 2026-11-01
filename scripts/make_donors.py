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

# 참여자가 거의 20대라 20대 동아시아 가상 인물을 이목구비가 서로 다르게 더 (11-01) — 후보가 다섯뿐이라 상자 녹화 둘은
# 가장 덜 닮은 사람도 바꾼 얼굴이 9~10% 장면에서 참여자로 보였다. 이목구비(눈·눈썹·코·입·얼굴형)를 크게 달리해 고를 여지를 넓힌다.
EA_YOUNG = {"female": ["monolid narrow eyes, thick straight eyebrows, wide nose, full lips, square jaw, short black hair",
                       "large double-lidded eyes, thin arched eyebrows, small pointed nose, small mouth, heart-shaped face, long straight hair",
                       "round face with full cheeks, small eyes, low nose bridge, freckles on the nose, shoulder-length hair",
                       "long narrow face, prominent cheekbones, hooded eyes, high nose bridge, thin lips, hair in a bun",
                       "wide face, wide-set eyes, flat brows, broad nose, wide mouth, pixie cut",
                       "slim face, upturned almond eyes, thick eyebrows, straight nose, plump lips, mole under the left eye, long wavy hair",
                       "soft round jaw, droopy eyes, short eyebrows, button nose, wide smile lines, dyed light brown hair",
                       "angular face, deep-set eyes, strong brow ridge, aquiline nose, thin wide mouth, very short undercut hair"],
            "male": ["monolid narrow eyes, thick eyebrows, wide nose, square jaw, short hair",
                     "large double-lidded eyes, thin face, high nose bridge, small mouth, curtain-parted hair",
                     "round full face, small eyes, low nose bridge, light stubble, buzz cut",
                     "long face, hooded eyes, prominent cheekbones, thin lips, medium wavy hair"]}


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
    for sex, word in (("female", "woman"), ("male", "man")):
        for i, look in enumerate(EA_YOUNG[sex]):
            age = (23, 26, 28)[i % 3]
            yield f"donor{k}", sex, age, "east_asian", BASE.format(who=f"East Asian {word}, {look}", age=f"{age} years old")
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
