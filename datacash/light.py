"""어두운 영상 밝히기 — 사람마다 찍는 곳이 달라(저녁 방·그늘진 창고) 어두워도 손·물체를 찾게 한다.

  장면 평균 밝기(0~255)가 DARK 아래면 밝기를 TARGET 쯤까지 곱해 올린다 (최대 MAX_GAIN 배).
  관절(MediaPipe)·Spark 1.3 클립·Spark Open 장면에 쓴다. 판매 영상(얼굴 바꾼 영상)은 원래 밝기 그대로 둔다.
  실험 (컵 옮기기 녹화를 일부러 어둡게, 손 찾은 장면 비율 · 원래 0.59):
    1/4 밝기(평균 33)  안 밝힘 0.16 · 감마 0.26 · 감마+CLAHE 0.35 · 곱하기 0.57  ← 곱하기
    1/8 밝기(평균 15)  안 밝힘 0.00 · 곱하기 0.19 (정보가 거의 없어 한계)
"""

from __future__ import annotations

import cv2
import numpy as np

DARK = 85.0         # 이보다 어두우면 밝힌다
TARGET = 120.0
MAX_GAIN = 6.0


def luma(bgr) -> float:
    small = cv2.resize(bgr, (64, 36), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def gain_for(l) -> float:
    """평균 밝기 l → 곱할 배율 (1 이면 그대로)."""
    if l is None or l >= DARK:
        return 1.0
    return float(np.clip(TARGET / max(l, 1.0), 1.0, MAX_GAIN))


_LUT: dict = {}


def lift(bgr, g: float = None):
    """어두우면 밝힌 장면, 아니면 그대로."""
    g = gain_for(luma(bgr)) if g is None else g
    if g <= 1.01:
        return bgr
    k = round(g, 1)
    if k not in _LUT:
        _LUT[k] = np.clip(np.arange(256) * k, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, _LUT[k])


def video_luma(video, n: int = 8, start_s: float = 0.0) -> float:
    cap = cv2.VideoCapture(str(video))
    fps, total = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(7))
    i0 = min(max(0, total - 1), int(start_s * fps))
    ls = []
    for i in np.linspace(i0, max(i0, total - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            ls.append(luma(f))
    cap.release()
    return float(np.median(ls)) if ls else None
