"""사람 알아보기 — 녹화하는 본인(계정 주인)과 함께 찍히는 데 동의한 가족.

폰마다(기기 id) 한 파일: data/server/people/<기기 id>.json
  owner   본인 얼굴 특징값 (녹화 전 '얼굴 확인'과 녹화 속 참여자 고르기에 쓴다)
  people  동의 인원: 이름 · 관계 · 동의(영상에 나오기, 판매) · 얼굴 특징값
사진은 저장하지 않는다 — 받자마자 OpenCV YuNet(얼굴 찾기) + SFace(특징값 128개)로 바꾸고 버린다.
같은 사람 기준은 SFace 코사인 0.45 + 다음으로 닮은 등록자보다 0.05 이상 높을 것.
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
        return json.loads(p.read_text()) if p.exists() else {"owner": None, "people": []}

    def save(self, device: str, d: dict) -> None:
        with _lock:
            self._path(device).write_text(json.dumps(d, ensure_ascii=False))

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def set_owner(self, device: str, emb: np.ndarray) -> dict:
        d = self.load(device)
        d["owner"] = {"embedding": [round(float(v), 6) for v in emb], "registered_at": self.now()}
        self.save(device, d)
        return {"registered": True}

    def check_owner(self, device: str, emb: np.ndarray) -> dict:
        o = self.load(device).get("owner")
        if not o:
            return {"registered": False, "match": False}
        s = sim(o["embedding"], emb)
        return {"registered": True, "match": s >= SAME_PERSON, "similarity": round(s, 3)}

    def add(self, device: str, name: str, relation: str, consent_video: bool, consent_sale: bool, emb: np.ndarray) -> dict:
        d = self.load(device)
        p = {"id": secrets.token_hex(4), "name": name[:20], "relation": relation[:10], "consent_video": bool(consent_video),
             "consent_sale": bool(consent_sale), "embedding": [round(float(v), 6) for v in emb], "added_at": self.now()}
        d["people"].append(p)
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
        return {k: v for k, v in p.items() if k != "embedding"}

    def listing(self, device: str) -> dict:
        d = self.load(device)
        return {"owner_registered": bool(d.get("owner")), "people": [self.public(p) for p in d["people"]]}

    def matcher(self, device: Optional[str], session_owner=None):
        """녹화 처리용: 특징값 → ('owner' | 'consented' | None, 이름표, 닮음). 등록이 하나도 없으면 None.
        session_owner: 이 녹화 1단계(얼굴 확인)에서 뽑은 얼굴 — 본인 등록이 없어도 참여자를 알아볼 수 있게."""
        d = self.load(device) if device and DEVICE.match(device) else None
        return matcher_from(d, session_owner)


def matcher_from(d: Optional[dict], session_owner=None):
    refs: List[Tuple[str, str, list]] = []
    if d and d.get("owner"):
        refs.append(("owner", "owner", d["owner"]["embedding"]))
    if session_owner is not None:
        refs.append(("owner", "face_check", list(np.asarray(session_owner, np.float32))))
    for p in (d or {}).get("people", []):
        if p.get("consent_video") and p.get("consent_sale"):
            refs.append(("consented", p["relation"] or "family", p["embedding"]))
    if not refs:
        return None

    def match(emb) -> Tuple[Optional[str], str, float]:
        ranked = sorted(((sim(e, emb), kind, tag) for kind, tag, e in refs), key=lambda x: -x[0])
        best = ranked[0]
        rivals = [r for r in ranked[1:] if r[1] != best[1]]      # 본인 기준이 둘(등록·얼굴 확인)이어도 서로 경쟁하지 않게
        clear = not rivals or best[0] - rivals[0][0] >= MARGIN
        return (best[1], best[2], best[0]) if best[0] >= SAME_PERSON and clear else (None, "", best[0])

    return match


def default_people() -> People:
    return People(config.DATA / "server" / "people")
