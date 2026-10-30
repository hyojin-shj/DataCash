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


def _hands(p) -> list:
    """장면 하나의 손 점 → 손마다 (k, 2). 배열 하나면 손 하나로 본다."""
    if p is None:
        return []
    if isinstance(p, (list, tuple)):
        return [np.asarray(h, float)[:, :2] for h in p if h is not None and len(h)]
    return [np.asarray(p, float)[:, :2]]


def _smooth(c: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(c, np.nan)
    for i in range(len(c)):
        s = c[max(0, i - w): i + w + 1]
        s = s[np.isfinite(s).all(1)]
        if len(s):
            out[i] = np.median(s, 0)
    return out


def contact_phases(hand_pts: Sequence, objects: Sequence[Dict[int, tuple]], fps: float,
                   window_s: float = 0.2, touch: float = 0.08, move: float = 0.008, rel: float = 0.5,
                   smooth_s: float = 0.1, keep_s: float = 0.15) -> List[ContactFrame]:
    """hand_pts[i]: 손 점 (k, 2) 하나, 또는 손마다 (k, 2) 의 목록, 또는 None. objects[i]: {물체 칸: (x1, y1, x2, y2)}.

    - 장면마다 물체에 가장 가까운 손 하나만 본다 (두 손을 한데 평균 내면 손 중심이 두 손 사이로 가서
      물체가 손과 따로 움직이는 것처럼 보였다 — 녹화 27편에서 장면의 33% 가 '미끄러짐', 단계가 1초에 3.9번 바뀌었다).
    - 거리·움직임 기준은 물체 크기에 비례한다 (손이 카메라에 가까울수록 픽셀이 커진다).
    - 한 번 '잡고 있음'이 되면 손끝이 물체에서 keep_s 넘게 떨어질 때까지 유지한다 (물체 상자가 흔들려도 끊기지 않게)."""
    n = len(hand_pts)
    w = max(1, int(round(window_s * fps)))
    keep = int(round(keep_s * fps))
    best = []                                   # (물체 칸, 그 손, 거리, 물체 크기)
    for i in range(n):
        b = (-1, None, np.inf, 1.0)
        for h in _hands(hand_pts[i]):
            for k, bx in objects[i].items():
                d, sz = _box_dist(h, bx), max(bx[2] - bx[0], bx[3] - bx[1], 1.0)
                if d / sz < b[2] / b[3]:
                    b = (k, h, d, sz)
        best.append(b)
    hc = _smooth(np.array([np.mean(b[1], 0) if b[1] is not None else [np.nan, np.nan] for b in best]),
                 max(1, int(round(smooth_s * fps))))

    def center(j: int, k: int):
        bx = objects[j].get(k)
        return None if bx is None else np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2])

    out: List[ContactFrame] = []
    held, away = False, 0
    for i in range(n):
        k, _, d, sz = best[i]
        touching = k >= 0 and d <= touch * sz
        near = k >= 0 and d <= 1.0 * sz
        c = 1.0 if touching else (0.5 if near else 0.0)
        if touching:
            away = 0
            a, b = max(0, i - w), min(n - 1, i + w)
            oa, ob = center(a, k), center(b, k)
            phase = 3 if held else 2
            if oa is not None and ob is not None and np.isfinite(hc[a]).all() and np.isfinite(hc[b]).all():
                om, hm = ob - oa, hc[b] - hc[a]
                thr = move * sz * (b - a)
                gap = np.linalg.norm(om - hm)
                if np.linalg.norm(om) > thr and gap <= rel * max(np.linalg.norm(om), np.linalg.norm(hm)):
                    held, phase = True, 3                       # 물체가 손을 따라 움직인다
                elif held and np.linalg.norm(om) > thr and gap > 2 * rel * np.linalg.norm(om):
                    phase = 4                                   # 잡고 있는데 물체가 손과 따로 움직인다
        else:
            away += 1
            if held and away > keep:
                held, phase = False, 5
            elif held:
                phase = 3
            else:
                phase = 1 if near else 0
        out.append(ContactFrame(phase, c, k if near else -1))
    return out


def phase_events(frames: List[ContactFrame], frame_t_s: Sequence[float]) -> List[dict]:
    """단계가 바뀐 시각만 뽑는다 (잡음 시작·놓음·미끄러짐)."""
    ev, prev = [], 0
    for f, t in zip(frames, frame_t_s):
        if f.phase != prev and f.phase in (2, 3, 4, 5):
            ev.append({"t_s": round(float(t), 3), "phase": PHASES[f.phase], "object": int(f.object)})
        prev = f.phase
    return ev
