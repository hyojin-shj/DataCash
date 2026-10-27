"""영상으로 읽는 접촉 단계 — 손가락 끝(손 21점)과 물체 영역(SAM 3.1)이 겹치는지, 물체가 손과 같이 움직이는지.

장면마다
  phase   0 없음 · 1 다가감 · 2 닿음 · 3 잡고 있음 · 4 미끄러짐 · 5 놓음
  contact 0~1 (손끝이 물체 안 1, 물체 근처 0.5, 아니면 0)
  object  잡은/닿은 물체 칸 번호 (-1 없음)
조용히 집는 동작은 소리·진동이 거의 없어서, 잡음·놓음 시각은 영상이 주력이다 (장면 단위 약 33ms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

FINGERTIPS = [4, 8, 12, 16, 20]
PHASES = ["none", "approach", "contact", "hold", "slip", "release"]


@dataclass
class ContactFrame:
    phase: int
    contact: float
    object: int


def _box_dist(pts: np.ndarray, box) -> float:
    x1, y1, x2, y2 = box
    dx = np.maximum(0, np.maximum(x1 - pts[:, 0], pts[:, 0] - x2))
    dy = np.maximum(0, np.maximum(y1 - pts[:, 1], pts[:, 1] - y2))
    return float(np.min(np.hypot(dx, dy)))


def contact_phases(hand_pts: Sequence[Optional[np.ndarray]], objects: Sequence[Dict[int, tuple]], fps: float,
                   move_px: float = 0.6, window_s: float = 0.2) -> List[ContactFrame]:
    """hand_pts[i]: (k, 2) 손 점(손끝 위주) 또는 None. objects[i]: {물체 칸: (x1, y1, x2, y2)}."""
    n = len(hand_pts)
    w = max(1, int(round(window_s * fps)))
    centers = [{k: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) for k, b in o.items()} for o in objects]
    hand_c = [None if p is None else np.mean(p[:, :2], axis=0) for p in hand_pts]
    raw = []
    for i in range(n):
        p, objs = hand_pts[i], objects[i]
        best, bd, size = -1, np.inf, 1.0
        if p is not None:
            for k, b in objs.items():
                d = _box_dist(p[:, :2], b)
                if d < bd:
                    best, bd, size = k, d, max(b[2] - b[0], b[3] - b[1])
        touching = p is not None and best >= 0 and bd <= 0.08 * size
        near = p is not None and best >= 0 and bd <= 1.0 * size
        raw.append((best if near else -1, touching, near))
    out: List[ContactFrame] = []
    held_prev = False
    for i in range(n):
        k, touching, near = raw[i]
        phase, c = 0, 1.0 if touching else (0.5 if near else 0.0)
        if touching:
            a, b = max(0, i - w), min(n - 1, i + w)
            oa, ob = centers[a].get(k), centers[b].get(k)
            ha, hb = hand_c[a], hand_c[b]
            if oa is not None and ob is not None and ha is not None and hb is not None:
                om, hm = ob - oa, hb - ha
                o_moving = np.linalg.norm(om) > move_px * (b - a)
                rel = np.linalg.norm(om - hm)
                if o_moving and rel <= 0.35 * max(np.linalg.norm(om), 1e-6) + move_px * 2:
                    phase = 3                                   # 물체가 손을 따라 움직인다
                elif o_moving:
                    phase = 4                                   # 닿아 있는데 물체가 손과 따로 움직인다
                elif np.linalg.norm(hm) > move_px * (b - a):
                    phase = 2                                   # 물체는 멈췄는데 손만 움직인다 → 놓는 중
                    held_prev = False
                else:
                    phase = 3 if held_prev else 2
            else:
                phase = 2
        elif near:
            phase = 5 if held_prev else 1
        held_prev = phase in (3, 4) or (held_prev and touching and phase != 2)
        out.append(ContactFrame(phase, c, k))
    # '놓음'은 잡고 있다가 손끝이 떨어진 첫 장면만
    for i in range(1, n):
        if out[i].phase == 5 and out[i - 1].phase == 5:
            out[i] = ContactFrame(1 if out[i].contact > 0 else 0, out[i].contact, out[i].object)
    return out


def phase_events(frames: List[ContactFrame], frame_t_s: Sequence[float]) -> List[dict]:
    """단계가 바뀐 시각만 뽑는다 (잡음 시작·놓음·미끄러짐)."""
    ev, prev = [], 0
    for f, t in zip(frames, frame_t_s):
        if f.phase != prev and f.phase in (2, 3, 4, 5):
            ev.append({"t_s": round(float(t), 3), "phase": PHASES[f.phase], "object": int(f.object)})
        prev = f.phase
    return ev
