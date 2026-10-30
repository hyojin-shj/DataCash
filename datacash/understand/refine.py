"""단계 경계 다듬기 — 모델은 동작 이름은 대체로 맞히지만 경계가 수백 ms 밀린다 (docs/label_eval.md).
동작 이름과 순서는 그대로 두고, 이웃한 두 단계 사이의 경계 시각만 옮긴다.

  snap        경계를 가장 가까운 '사건' 시각으로 (±tol 안에 있을 때만). 사건:
                손 속도가 잠깐 멈춘 순간 (hand_events — 손이 닿거나 놓을 때 거의 멈춘다, 영상만으로)
                영상 접촉 단계가 바뀐 순간 · 접촉음 · 손목 충격 (우리 녹화에서는 처리 중에 이미 구한다)
  pick        경계 앞뒤 장면 8장에 시각을 붙여 Muse Spark Open 에게 '다음 동작이 시작된 첫 장면'을 고르게 한다
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def hand_events(hands: Sequence[Sequence[Optional[np.ndarray]]], fps: float, width: float,
                smooth_s: float = 0.15, win_s: float = 0.25) -> List[float]:
    """hands[i] = [왼손 (21, 3) 또는 None, 오른손 ...] → 손 속도가 앞뒤 win_s 안에서 가장 낮고 중앙값의 절반보다 낮은 시각들.
    두 손 중 빠른 쪽 속도를 쓴다 (한 손이 멈춰도 다른 손이 일하면 사건이 아니다 — 대신 각 손의 멈춤도 따로 넣는다)."""
    n = len(hands)
    if n < 3:
        return []
    out = []
    for k in (0, 1):
        pos = np.full((n, 2), np.nan)
        for i, h in enumerate(hands):
            if h[k] is not None:
                pos[i] = np.mean(np.asarray(h[k])[[0, 4, 8, 12], :2], axis=0)   # 손목 + 엄지·검지·중지 끝
        v = np.r_[np.nan, np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps / width]
        w = max(1, int(round(smooth_s * fps)))
        vs = np.array([np.nanmean(v[max(0, i - w): i + w + 1]) if np.isfinite(v[max(0, i - w): i + w + 1]).any() else np.nan
                       for i in range(n)])
        ok = np.isfinite(vs)
        if ok.sum() < 3:
            continue
        med = float(np.nanmedian(vs))
        r = max(1, int(round(win_s * fps)))
        for i in np.nonzero(ok)[0]:
            seg = vs[max(0, i - r): i + r + 1]
            if vs[i] <= np.nanmin(seg) and vs[i] < 0.5 * med:      # 거의 멈춘 순간만 (고르게 움직이는 중의 작은 흔들림은 뺀다)
                out.append(round(i / fps, 3))
    return sorted(set(out))


def snap(steps: List[dict], events: Sequence[float], tol: float = 0.4) -> List[dict]:
    """이웃한 단계 사이 경계(앞 단계 끝 ≈ 뒤 단계 시작)를 ±tol 안의 가장 가까운 사건으로. 순서는 지킨다."""
    ev = np.asarray(sorted(events), float)
    out = [dict(s) for s in sorted(steps, key=lambda s: s["start_s"])]
    if not len(ev):
        return out
    for a, b in zip(out, out[1:]):
        t = b["start_s"]
        j = int(np.argmin(np.abs(ev - t)))
        if abs(ev[j] - t) > tol:
            continue
        nt = float(ev[j])
        if a["start_s"] < nt < b["end_s"]:
            if abs(a["end_s"] - t) < 0.3:          # 붙어 있던 경계면 같이 옮긴다 (틈은 그대로)
                a["end_s"] = nt
            b["start_s"] = nt
    return out


PICK = """You are fixing the timing of step labels in a first-person hand-manipulation video, mission: "{mission}".
For each boundary below you get one image with 8 frames in time order (left-to-right, then top-to-bottom), each with its time.
The step before the boundary is labeled A and the step after is labeled B. Pick the FIRST frame in which B has begun
(e.g. for grasp: fingers first touch the object; for lift: the object first leaves the surface; for release: fingers first open;
for reach: the hand first starts moving toward the next object). If B has already begun in frame 1, answer 1; if it has not begun
in any frame, answer 8. Gray areas are privacy masks; ignore them.
Return ONLY JSON {{"boundaries": [{{"k": <boundary number>, "frame": <1-8>}}]}}."""


def pick(video, steps: List[dict], mission: str, span_s: float = 0.7, per_call: int = 8, side: int = 320) -> List[dict]:
    """경계마다 [t − span, t + span] 장면 8장 → Spark Open 이 고른 장면 시각으로 경계를 옮긴다 (동작이 같은 이웃은 건너뛴다)."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    import cv2

    from . import spark_open

    out = [dict(s) for s in sorted(steps, key=lambda s: s["start_s"])]
    bounds = [k for k in range(1, len(out)) if out[k]["action"] != out[k - 1]["action"]]

    def grid(t: float):
        times = [t - span_s + 2 * span_s * j / 7 for j in range(8)]
        fr = spark_open._grab(video, [max(0.0, x) for x in times])
        if len(fr) != 8:
            return None, None
        imgs = []
        for x, f in fr:
            s = side / max(f.shape[:2])
            f = cv2.resize(f, None, fx=s, fy=s)
            cv2.putText(f, f"{x:.2f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            imgs.append(f)
        rows = [np.hstack(imgs[r * 4:(r + 1) * 4]) for r in range(2)]
        return [x for x, _ in fr], np.vstack(rows)

    moves = {}

    def one(g):
        chunk = bounds[g:g + per_call]
        parts = [{"type": "text", "text": PICK.format(mission=mission)}]
        times = {}
        for j, k in enumerate(chunk):
            ts, img = grid(out[k]["start_s"])
            if img is None:
                continue
            times[j + 1] = (k, ts)
            parts += [{"type": "text", "text": f"Boundary {j + 1}: A = {out[k - 1]['action']}, B = {out[k]['action']}"},
                      spark_open._jpg(img)]
        if not times:
            return
        prov, r = spark_open._ask(lambda mx, p=parts: p)
        spark_open._record(getattr(r, "usage", None), f"spark-open boundary pick {Path(video).name} {len(times)}", prov)
        text = r.choices[0].message.content or "{}"
        try:
            ans = json.loads(text[text.find("{"): text.rfind("}") + 1]).get("boundaries") or []
        except ValueError:
            ans = []
        for x in ans:
            try:
                j, f = int(x["k"]), int(x["frame"])
            except (KeyError, TypeError, ValueError):
                continue
            if j in times and 1 <= f <= 8:
                k, ts = times[j]
                moves[k] = ts[f - 1]

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, range(0, len(bounds), per_call)))
    for k, nt in moves.items():
        a, b = out[k - 1], out[k]
        if a["start_s"] < nt < b["end_s"]:
            if abs(a["end_s"] - b["start_s"]) < 0.3:
                a["end_s"] = nt
            b["start_s"] = nt
    return out
