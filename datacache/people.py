"""동의 인원 — 녹화하는 본인과 함께 찍히는 데 동의한 사람 (폰마다 한 명부).

폰(기기 id)마다 한 파일: data/server/people/<기기 id>.json  {"people": [...]}
  사람  id · 이름 · 관계(본인·배우자·가족·동거인·기타) · 성인 확인 · 연령대 · 동의(영상에 나오기·판매, 버전·시각)
        · 생체 동의(본인만: 심박·HRR·피부온도 변화·미션 후 응답) · 얼굴 상태(none·enrolled·failed) · 얼굴 특징값(사진 3장)
        · 가상 인물 (직접 고름 또는 auto)
  '본인' 이 녹화의 참여자 — 녹화 1단계 얼굴 확인과 녹화 속 참여자 고르기에 쓴다.
사진은 저장하지 않는다 — 받자마자 OpenCV YuNet(얼굴 찾기) + SFace(특징값 128개)로 바꾸고 버린다.
같은 사람 기준은 SFace 코사인 0.45 + 다음으로 닮은 다른 등록자보다 0.05 이상 높을 것.
OpenCV 권장값(0.363)보다 엄격하게 잡았다 — 저장소의 Muse Image 가상 인물 6명끼리도 0.33~0.60 이 나와서,
모르는 사람을 동의한 가족으로 잘못 알아보면(가리지 않게 되면) 안 되기 때문.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config

SAME_PERSON = 0.45
MARGIN = 0.05
DEVICE = re.compile(r"^[A-Za-z0-9_\-]{6,64}$")
RELATIONS = ("본인", "배우자", "가족", "동거인", "기타")
AGE_BANDS = ("19~29세", "30대", "40대", "50대", "60대 이상")
BIO_SCOPE = ["heart_rate", "heart_rate_reserve", "skin_temperature_change", "post_mission_survey"]
DONORS_DIR = Path(__file__).parent / "donors"
GROUP_KO = {"east_asian": "동아시아", "southeast_asian": "동남아시아", "south_asian": "남아시아", "black": "아프리카계",
            "white": "유럽계", "latino": "라틴계", "middle_eastern": "중동"}


def _donor_labels() -> dict:
    """가상 인물 이름표 '여 · 20대 · 동아시아' (donors/donors.json)."""
    p = DONORS_DIR / "donors.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    return {k: f"{'여' if v.get('gender') == 'female' else '남'} · {v.get('age', 0) // 10 * 10}대 · {GROUP_KO.get(v.get('group'), '')}"
            for k, v in d.items()}


DONOR_LABELS = _donor_labels()
_lock = threading.Lock()
_models = {}


def _reid():
    if "r" not in _models:
        from .privacy.faceswap import Reid

        _models["r"] = Reid()
    return _models["r"]


def embed_image(bgr: np.ndarray, box: Optional[Tuple[float, float, float, float]] = None,
                min_px: int = 40) -> Tuple[Optional[np.ndarray], str]:
    """사진(또는 box 안)에서 가장 큰 얼굴의 특징값. (특징값, 문제) — 문제가 있으면 특징값은 None."""
    r = _reid()
    img = bgr
    if box is not None:
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < min_px or y1 - y0 < min_px:
            return None, "too_small"
        img = bgr[y0:y1, x0:x1]
    h, w = img.shape[:2]
    r.det.setInputSize((w, h))
    _, faces = r.det.detect(img)
    if faces is None or not len(faces):
        return None, "no_face"
    f = max(faces, key=lambda f: f[2] * f[3])
    if min(f[2], f[3]) < min_px * 0.6:
        return None, "too_small"
    return r.rec.feature(r.rec.alignCrop(img, f)).ravel().astype(np.float32), ""


def embed_jpeg(data: bytes) -> Tuple[Optional[np.ndarray], str]:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "bad_image"
    return embed_image(img, min_px=60)


def sim(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def best_sim(refs: List[list], emb) -> float:
    return max((sim(e, emb) for e in refs), default=-1.0)


def donors() -> List[dict]:
    return [{"id": p.stem, "label": DONOR_LABELS.get(p.stem, "")}
            for p in sorted(DONORS_DIR.glob("donor*.png"), key=lambda p: int(p.stem[5:]))]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class People:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, device: str) -> Path:
        if not DEVICE.match(device or ""):
            raise ValueError("기기 id")
        return self.root / f"{device}.json"

    def load(self, device: str) -> dict:
        p = self._path(device)
        return json.loads(p.read_text()) if p.exists() else {"people": []}

    def save(self, device: str, d: dict) -> None:
        with _lock:
            self._path(device).write_text(json.dumps(d, ensure_ascii=False))

    @staticmethod
    def _find(d: dict, pid: str) -> dict:
        for p in d["people"]:
            if p["id"] == pid:
                return p
        raise KeyError(pid)

    def add(self, device: str, name: str, relation: str, adult: bool, age_band: str, agreed: bool,
            bio: bool = False) -> dict:
        if relation not in RELATIONS:
            raise ValueError("관계")
        if not adult:
            raise ValueError("만 19세 이상 성인만 추가할 수 있어요 (아이는 항상 가려져요)")
        if age_band not in AGE_BANDS:
            raise ValueError("연령대")
        if not agreed:
            raise ValueError("영상에 나오는 것과 판매에 동의해야 추가할 수 있어요")
        if bio and relation != "본인":
            raise ValueError("생체 신호는 본인만 제공할 수 있어요")
        d = self.load(device)
        if relation == "본인" and any(p["relation"] == "본인" for p in d["people"]):
            raise ValueError("본인은 이미 있어요")
        p = {"id": secrets.token_hex(8), "name": name.strip()[:20], "relation": relation, "adult": True, "age_band": age_band,
             "consent": {"video": True, "sale": True, "version": "datacache-people-v1", "at": now()},
             "bio_consent": bio_record(bool(bio)), "face": "none", "embeddings": [], "donor": "auto"}
        d["people"].append(p)
        self.save(device, d)
        return self.public(p)

    def enroll(self, device: str, pid: str, photos: List[bytes]) -> dict:
        """얼굴 사진 여러 장(정면·왼쪽·오른쪽) → 특징값들. 두 장 이상에서 얼굴이 나와야 등록 (사진은 버린다)."""
        d = self.load(device)
        p = self._find(d, pid)
        embs, why = [], []
        for ph in photos:
            e, w = embed_jpeg(ph)
            if e is not None:
                embs.append([round(float(v), 6) for v in e])
            else:
                why.append(w)
        if photos and len(embs) >= min(2, len(photos)):
            p.update(face="enrolled", embeddings=embs, face_at=now(), face_error="")
        else:
            p.update(face="failed", embeddings=[], face_error={"no_face": "얼굴을 찾지 못한 사진이 있어요",
                                                               "too_small": "얼굴이 너무 작게 나왔어요"}.get(why[0] if why else "", "다시 찍어 주세요"))
        self.save(device, d)
        return self.public(p)

    def set_donor(self, device: str, pid: str, donor: str) -> dict:
        if donor != "auto" and donor not in {x["id"] for x in donors()}:
            raise ValueError("가상 인물")
        d = self.load(device)
        p = self._find(d, pid)
        p["donor"] = donor
        self.save(device, d)
        return self.public(p)

    def set_bio(self, device: str, pid: str, agreed: bool) -> dict:
        d = self.load(device)
        p = self._find(d, pid)
        if agreed and p["relation"] != "본인":
            raise ValueError("생체 신호는 본인만 제공할 수 있어요")
        p["bio_consent"] = bio_record(agreed)
        self.save(device, d)
        return self.public(p)

    def remove(self, device: str, pid: str) -> bool:
        d = self.load(device)
        n = len(d["people"])
        d["people"] = [p for p in d["people"] if p["id"] != pid]
        self.save(device, d)
        return len(d["people"]) < n

    @staticmethod
    def public(p: dict) -> dict:
        return {k: v for k, v in p.items() if k != "embeddings"}

    def listing(self, device: str) -> dict:
        d = self.load(device)
        me = next((p for p in d["people"] if p["relation"] == "본인"), None)
        return {"owner_registered": bool(me and me["face"] == "enrolled"), "people": [self.public(p) for p in d["people"]],
                "donors": donors()}

    def check_owner(self, device: str, emb: np.ndarray) -> dict:
        me = next((p for p in self.load(device)["people"] if p["relation"] == "본인" and p["face"] == "enrolled"), None)
        if not me:
            return {"registered": False, "match": False}
        s = best_sim(me["embeddings"], emb)
        return {"registered": True, "match": s >= SAME_PERSON, "similarity": round(s, 3)}

    def donor_for(self, device: Optional[str], pid: str) -> Optional[Path]:
        """그 사람이 고른 가상 인물 사진 (auto 면 None). pid 'owner' 는 본인."""
        if not device or not DEVICE.match(device):
            return None
        for p in self.load(device)["people"]:
            if (p["id"] == pid or (pid == "owner" and p["relation"] == "본인")) and p.get("donor", "auto") != "auto":
                f = DONORS_DIR / f"{p['donor']}.png"
                return f if f.exists() else None
        return None

    def matcher(self, device: Optional[str], session_owner=None):
        """녹화 처리용: 특징값 → ('owner' | 'consented' | None, 사람 id, 닮음). 등록이 하나도 없으면 None.
        session_owner: 이 녹화 1단계(얼굴 확인)에서 뽑은 얼굴 — 본인 등록이 없어도 참여자를 알아볼 수 있게."""
        d = self.load(device) if device and DEVICE.match(device) else None
        return matcher_from(d, session_owner)


def bio_record(agreed: bool) -> dict:
    return {"agreed": agreed, "scope": BIO_SCOPE if agreed else [], "version": "bio-v1",
            ("consented_at" if agreed else "revoked_at"): now()}


def matcher_from(d: Optional[dict], session_owner=None):
    refs: List[Tuple[str, str, list]] = []          # (종류, 사람 id, 특징값들)
    for p in (d or {}).get("people", []):
        if p.get("face") != "enrolled":
            continue
        refs.append(("owner" if p["relation"] == "본인" else "consented", p["id"], p["embeddings"]))
    if session_owner is not None:
        refs.append(("owner", "face_check", [list(np.asarray(session_owner, np.float32))]))
    if not refs:
        return None

    def match(emb) -> Tuple[Optional[str], str, float]:
        ranked = sorted(((best_sim(es, emb), kind, pid) for kind, pid, es in refs), key=lambda x: -x[0])
        best = ranked[0]
        # 본인 기준이 둘(등록·얼굴 확인)이어도 서로 경쟁하지 않게, 다른 사람끼리만 차이를 본다
        rivals = [r for r in ranked[1:] if r[1] != best[1] or (best[1] == "consented" and r[2] != best[2])]
        clear = not rivals or best[0] - rivals[0][0] >= MARGIN
        return (best[1], best[2], best[0]) if best[0] >= SAME_PERSON and clear else (None, "", best[0])

    return match


def default_people() -> People:
    return People(config.DATA / "server" / "people")
