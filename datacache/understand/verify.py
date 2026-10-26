"""Muse Spark 라벨을 다른 감각으로 확인한다 — 모델 라벨을 그대로 믿지 않는다.

단계마다 증거 (있는 감각만):
  motion   그 구간에 손목(또는 손 관절) 움직임이 평소보다 크다 → 무언가 하고 있었다
  contact  잡기·옮기기·내려놓기류 단계면 그 구간에 영상 접촉(닿음·잡고 있음) 또는 쥐는 힘이 있다
  sound    내려놓기·문지르기·헹구기류면 그 구간 둘레에 접촉음이 있다
  boundary 단계 시작이 접촉 단계 변화·소리 사건과 0.5초 안에 맞는다
verified = 쓸 수 있는 증거 중 절반 이상이 맞고, 최소 하나는 맞음 (증거가 하나도 없으면 'unverifiable').
실수 후보: 참여자가 누른 '실수했어요'와 2초 안이면 confirmed_by_participant.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

GRASPY = {"grasp", "lift", "carry", "place", "release", "pour", "push", "open", "close", "wipe", "scrub", "cut", "stir"}
NOISY = {"place", "release", "wipe", "scrub", "rinse", "cut", "stir", "pour"}


def verify_steps(steps: List[dict], t_s: np.ndarray, motion: Optional[np.ndarray] = None,
                 contact: Optional[np.ndarray] = None, grip: Optional[np.ndarray] = None,
                 sound_t_s: Optional[List[float]] = None, contact_change_t_s: Optional[List[float]] = None) -> List[dict]:
    """t_s: 장면 시각(초). motion·contact·grip: 장면마다 값(없으면 None)."""
    base = float(np.median(motion)) if motion is not None and len(motion) else None
    out = []
    for st in steps:
        sel = (t_s >= st["start_s"]) & (t_s < st["end_s"])
        ev = {}
        if motion is not None and sel.any():
            ev["motion"] = bool(np.mean(motion[sel]) > 1.2 * base) if base and base > 0 else bool(np.mean(motion[sel]) > 0)
        if st.get("action") in GRASPY and sel.any() and (contact is not None or grip is not None):
            c = contact[sel].max() if contact is not None else 0
            g = grip[sel].max() if grip is not None else 0
            ev["contact"] = bool(c >= 1.0 or g > 0.5)
        if st.get("action") in NOISY and sound_t_s is not None:
            ev["sound"] = any(st["start_s"] - 0.5 <= t <= st["end_s"] + 0.5 for t in sound_t_s)
        marks = list(contact_change_t_s or []) + list(sound_t_s or [])
        if marks:
            ev["boundary"] = any(abs(st["start_s"] - t) <= 0.5 for t in marks)
        n_ok = sum(ev.values())
        verified = "unverifiable" if not ev else (n_ok >= 1 and n_ok * 2 >= len(ev))
        out.append({**st, "evidence": ev, "verified": verified})
    return out


def check_mistakes(mistakes: List[dict], self_marks_s: List[float], tol_s: float = 2.0) -> List[dict]:
    return [{**m, "confirmed_by_participant": any(abs(m["t_s"] - t) <= tol_s for t in self_marks_s)} for m in mistakes]
