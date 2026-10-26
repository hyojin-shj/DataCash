"""SAM 3.1 로 영상 가리기.

1. 명사구마다 SAM 3.1 영상 요청 한 번 (물체를 장면 사이로 추적, 물체 id 유지)
     사람: "person" · 아이: "child" · 민감 물체: "television screen", "computer monitor", "mirror", "picture frame", "document"
     조작 물체(미션별): 예) "plate", "cup", "sponge" — 가리지 않고 영역만
2. 참여자 고르기: 사람 추적 중 (지정이 없으면) 영상에 가장 오래·크게 나온 한 명. 손 21점이 있으면 손이 겹치는 사람.
   아이로 잡힌 영역은 참여자와 겹쳐도 가린다 (아이는 참여자가 될 수 없다).
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


def _choose_participant(people: Dict[int, List[SamObject]], hand_pts: Optional[Sequence] = None) -> Optional[str]:
    score: Dict[str, float] = {}
    for fi, objs in people.items():
        for o in objs:
            s = o.area
            if hand_pts is not None and fi < len(hand_pts) and hand_pts[fi] is not None and o.mask is not None:
                p = np.asarray(hand_pts[fi])[:, :2].astype(int)
                h, w = o.mask.shape
                inside = [o.mask[min(h - 1, max(0, y)), min(w - 1, max(0, x))] for x, y in p]
                s *= 1 + 4 * float(np.mean(inside))
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
    while True:
        ok, f = cap.read()
        if not ok:
            break
        hide = np.zeros((h, w), bool)
        for c in concepts_people:
            for o in tracks[c].get(fi, []):
                if o.object_id != pid and o.mask is not None:
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
                  frames_with_masking_pct=round(100 * float(np.mean(np.array(masked_frac) > 0)) if masked_frac else 0.0, 1))
    names = [f"{c}#{oid}" for (c, oid), _ in sorted(slots.items(), key=lambda kv: kv[1])]
    return PrivacyResult(out_path, pid, objects, names, report)
