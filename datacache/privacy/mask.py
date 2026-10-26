"""SAM 3.1 로 영상 가리기.

1. 명사구마다 SAM 3.1 영상 요청 한 번 (물체를 장면 사이로 추적, 물체 id 유지)
     사람: "person" · 아이: "child" · 민감 물체: "television screen", "computer monitor", "mirror", "picture frame", "document"
     조작 물체(미션별): 예) "plate", "cup", "sponge" — 가리지 않고 영역만
2. 참여자 고르기: 사람 추적 중 (지정이 없으면) 영상에 가장 오래·크게 나온 한 명. 손 21점이 있으면 손이 겹치는 사람.
   아이로 잡힌 영역은 참여자와 겹쳐도 가린다 (아이는 참여자가 될 수 없다).
   참여자의 손: 손끝이 참여자 영역 안이거나 참여자 상자 가까이 있는 손. 1인칭 영상에서 SAM 이 두 팔을 따로 잡아도
   참여자의 손이 든 '사람' 조각은 참여자 몸으로 보고 가리지 않으며, 참여자 상자는 장면마다 두 손을 모두 덮게 넓힌다.
3. 참여자가 아닌 사람 + 아이 + 민감 물체 영역을 회색으로 칠하고, 경계를 몇 픽셀 넓혀 윤곽이 새지 않게 한다.
4. 보고: 명사구별 장면 수·물체 수, 가린 넓이 비율, 참여자 id, 비용.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..meta.client import BudgetExceeded
from ..meta.sam import Sam, SamObject

PEOPLE = ("person",)
CHILD = ("child",)
SENSITIVE = ("television screen", "computer monitor", "mirror", "picture frame", "document")
GRAY = (128, 128, 128)


@dataclass
class PrivacyResult:
    masked_video: Path
    participant: Optional[str]
    objects: List[Dict[int, tuple]]                 # 장면마다 {물체 칸: 상자} (조작 물체, LeRobot 용)
    object_names: List[str]
    report: dict = field(default_factory=dict)
    participant_boxes: Dict[int, tuple] = field(default_factory=dict)   # 장면마다 참여자 상자 (재검사·접촉용)


def _hands(frame_pts) -> List[np.ndarray]:
    """장면 하나의 손 점들 → 손마다 (k, 2) 배열 목록. 손 목록이나 이어 붙인 배열 둘 다 받는다."""
    if frame_pts is None:
        return []
    if isinstance(frame_pts, (list, tuple)):
        return [np.asarray(h, np.float32)[:, :2] for h in frame_pts if h is not None and len(h)]
    a = np.asarray(frame_pts, np.float32)
    return [a[:, :2]] if len(a) else []


def _box_of(pts: np.ndarray, pad: float) -> tuple:
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    p = pad * max(x2 - x1, y2 - y1, 20.0)
    return (float(x1 - p), float(y1 - p), float(x2 + p), float(y2 + p))


def _union(a: Optional[tuple], b: tuple) -> tuple:
    return b if a is None else (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _inside(mask: np.ndarray, pts: np.ndarray) -> float:
    h, w = mask.shape
    xy = np.round(pts).astype(int)
    return float(np.mean([mask[min(h - 1, max(0, y)), min(w - 1, max(0, x))] for x, y in xy]))


def _near(box: tuple, hb: tuple, frac: float = 0.5) -> bool:
    """손 상자가 참여자 상자를 크기의 frac 만큼 넓힌 범위와 겹치나."""
    dx, dy = frac * (box[2] - box[0]), frac * (box[3] - box[1])
    return not (hb[2] < box[0] - dx or hb[0] > box[2] + dx or hb[3] < box[1] - dy or hb[1] > box[3] + dy)


def participant_hands(p_obj: Optional[SamObject], frame_pts) -> List[np.ndarray]:
    """이 장면에서 참여자의 손: 손끝이 참여자 영역 안이거나 참여자 상자 가까이. 참여자가 안 보이면 (1인칭) 모든 손."""
    hs = _hands(frame_pts)
    if p_obj is None:
        return hs
    out = []
    for hp in hs:
        if (p_obj.mask is not None and _inside(p_obj.mask, hp) > 0) or _near(tuple(p_obj.box), _box_of(hp, 0.0)):
            out.append(hp)
    return out


def _choose_participant(people: Dict[int, List[SamObject]], hand_pts: Optional[Sequence] = None) -> Optional[str]:
    score: Dict[str, float] = {}
    for fi, objs in people.items():
        for o in objs:
            s = o.area
            if hand_pts is not None and fi < len(hand_pts) and hand_pts[fi] is not None and o.mask is not None:
                hs = _hands(hand_pts[fi])
                if hs:
                    s *= 1 + 4 * _inside(o.mask, np.concatenate(hs))
            score[o.object_id] = score.get(o.object_id, 0.0) + s
    return max(score, key=score.get) if score else None


def mask_video(video, out_path, manipulated: Sequence[str] = (), participant: Optional[str] = None,
               hand_pts: Optional[Sequence] = None, sam: Optional[Sam] = None, dilate_px: int = 6,
               concepts_people=PEOPLE, concepts_child=CHILD, concepts_sensitive=SENSITIVE) -> PrivacyResult:
    sam = sam or Sam()
    video = Path(video)
    file_id = sam.c.upload(video)
    tracks: Dict[str, Dict[int, List[SamObject]]] = {}
    report = {"concepts": {}, "errors": []}
    for c in (*concepts_people, *concepts_child, *concepts_sensitive, *manipulated):
        try:
            tracks[c] = sam.video(video, c, file_id=file_id)
            ids = {o.object_id for objs in tracks[c].values() for o in objs}
            report["concepts"][c] = {"frames": len(tracks[c]), "objects": len(ids)}
        except BudgetExceeded as e:
            report["errors"].append(f"{c}: {e}")
            tracks[c] = {}
        except Exception as e:  # 한 명사구가 실패해도 나머지로 계속 (보고에 남긴다)
            report["errors"].append(f"{c}: {type(e).__name__}: {str(e)[:200]}")
            tracks[c] = {}
    people = {}
    for c in concepts_people:
        for fi, objs in tracks[c].items():
            people.setdefault(fi, []).extend(objs)
    pid = participant or _choose_participant(people, hand_pts)
    report["participant"] = pid

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    objects: List[Dict[int, tuple]] = []
    slots: Dict[tuple, int] = {}
    masked_frac, fi = [], 0
    pboxes: Dict[int, tuple] = {}
    kept_parts = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        hide = np.zeros((h, w), bool)
        p_obj = next((o for o in people.get(fi, []) if o.object_id == pid), None)
        fp = hand_pts[fi] if hand_pts is not None and fi < len(hand_pts) else None
        mine = participant_hands(p_obj, fp)
        pb = tuple(float(v) for v in p_obj.box) if p_obj is not None else None
        for hp in mine:
            pb = _union(pb, _box_of(hp, 0.6))      # 손끝만 있으므로 손·손목까지 덮게 넉넉히
        if pb is not None:
            pboxes[fi] = (max(0.0, pb[0]), max(0.0, pb[1]), min(float(w), pb[2]), min(float(h), pb[3]))
        for c in concepts_people:
            for o in tracks[c].get(fi, []):
                if o.object_id == pid or o.mask is None:
                    continue
                if mine and any(_inside(o.mask, hp) >= 0.4 for hp in mine):
                    kept_parts += 1        # 참여자의 다른 팔 (SAM 이 따로 잡은 조각) → 가리지 않음
                    continue
                hide |= o.mask
        for c in (*concepts_child, *concepts_sensitive):
            for o in tracks[c].get(fi, []):
                if o.mask is not None:
                    hide |= o.mask
        if hide.any():
            hide = cv2.dilate(hide.astype(np.uint8), kernel) > 0
            f[hide] = GRAY
        masked_frac.append(float(hide.mean()))
        fo = {}
        for c in manipulated:
            for o in tracks[c].get(fi, []):
                key = (c, o.object_id)
                if key not in slots and len(slots) < 4:
                    slots[key] = len(slots)
                if key in slots:
                    fo[slots[key]] = tuple(float(v) for v in o.box)
        objects.append(fo)
        vw.write(f)
        fi += 1
    cap.release()
    vw.release()
    from ..media import run

    run(["-i", raw, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path])
    raw.unlink()
    report.update(frames=fi, masked_area_mean_pct=round(100 * float(np.mean(masked_frac)) if masked_frac else 0.0, 2),
                  frames_with_masking_pct=round(100 * float(np.mean(np.array(masked_frac) > 0)) if masked_frac else 0.0, 1),
                  participant_parts_kept=kept_parts)
    names = [f"{c}#{oid}" for (c, oid), _ in sorted(slots.items(), key=lambda kv: kv[1])]
    return PrivacyResult(out_path, pid, objects, names, report, pboxes)
