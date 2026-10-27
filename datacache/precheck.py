"""돈 드는 처리(SAM·Spark) 전에 이 맥에서 공짜로 하는 검수 — 쓸 수 없는 녹화는 여기서 반려해 API 값을 쓰지 않는다.

  동작 길이   얼굴 확인(1단계) 뒤 동작 구간이 MIN_ACTION_S 초 이상
  손          동작 구간에서 손이 보인 장면이 MIN_HANDS 이상 (손으로 하는 미션만 받는다)
  밝기        장면 평균 밝기 MIN_LUMA 이상 (너무 어두우면 관절·물체를 못 찾는다)
반려 사유는 참여자에게 보이고, 이의 신청으로 다시 볼 수 있다."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

MIN_ACTION_S = 3.0
MIN_HANDS = 0.15
MIN_LUMA = 30.0


def precheck(video, labels: Optional[list], action_start_s: float = 0.0) -> dict:
    """{'ok': bool, 'reasons': [반려 사유], 'action_s', 'hands_frac', 'luma'}."""
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(7))
    i0 = min(n, int(action_start_s * fps))
    lumas: List[float] = []
    for i in np.linspace(i0, max(i0, n - 1), 8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            lumas.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
    cap.release()
    action_s = (n - i0) / fps
    part = (labels or [])[i0:]
    hands = sum(any(h is not None for h in f.hands) for f in part) / len(part) if part else None
    luma = float(np.median(lumas)) if lumas else 0.0
    reasons = []
    if action_s < MIN_ACTION_S:
        reasons.append(f"동작 부분이 너무 짧아요 ({action_s:.1f}초 · {MIN_ACTION_S:.0f}초 이상 찍어 주세요)")
    if hands is not None and hands < MIN_HANDS:
        reasons.append(f"손이 거의 안 보여요 (동작 중 {hands * 100:.0f}% · 손이 화면에 나오게 찍어 주세요)")
    if luma < MIN_LUMA:
        reasons.append("너무 어두워요 · 불을 켜고 찍어 주세요")
    return {"ok": not reasons, "reasons": reasons, "action_s": round(action_s, 2),
            "hands_frac": None if hands is None else round(hands, 3), "luma": round(luma, 1)}
