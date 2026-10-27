"""SAM 을 부르기 전에 이 맥에서 공짜로 훑어보기 — 무엇을 SAM 에 얼마나 자주 물을지 정한다 (SAM 은 장면 × 명사구마다 값).

훑기: MediaPipe 물체 찾기 (EfficientDet-Lite0, COCO 80종, Apache 2.0) 를 초당 2장면.
  사람이 한 장면에 둘 이상   → 'person'·'child' 를 SAM 에 초당 10장면 (움직이는 다른 사람은 촘촘히)
                   혼자면   → 묻지 않는다 (참여자는 가리지 않고 얼굴만 바꾼다 — 얼굴은 이 맥에서 찾는다)
  tv·laptop 이 보이면      → 'television screen'·'computer monitor' 초당 1장면 (화면은 잘 안 움직인다)
  거울·액자·문서            → COCO 에 없어 못 거른다 → 늘 초당 1장면 (개인정보가 먼저)
  조작 물체(라벨용 상자)    → 초당 2장면
모델 파일이 없거나 훑기가 실패하면 예전처럼 모두 묻는다."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2

from .. import config

MODEL = config.MODELS / "efficientdet_lite0.tflite"
SCAN_FPS = 2.0
SCORE = 0.3              # 낮게: 놓치면 개인정보가 새고, 잘못 잡으면 값만 조금 더 든다
PEOPLE_FPS, SCREEN_FPS, STATIC_FPS, OBJECT_FPS = 10.0, 1.0, 1.0, 2.0
SCREENS = ("television screen", "computer monitor")
ALWAYS = ("mirror", "picture frame", "document")


def scan(video, fps: float = SCAN_FPS) -> Optional[dict]:
    """{'frames': 본 장면 수, 'max_people': 한 장면 최대 사람 수, 'seen': {COCO 이름: 장면 수}}. 못 하면 None."""
    if not Path(MODEL).exists():
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        det = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)), score_threshold=SCORE, max_results=20))
        cap = cv2.VideoCapture(str(video))
        step = max(1, int(round((cap.get(cv2.CAP_PROP_FPS) or 30.0) / fps)))
        seen: Dict[str, int] = {}
        frames = max_people = i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, f = cap.retrieve()
                if ok:
                    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                    names = [d.categories[0].category_name for d in r.detections if d.categories]
                    for n in set(names):
                        seen[n] = seen.get(n, 0) + 1
                    max_people = max(max_people, names.count("person"))
                    frames += 1
            i += 1
        cap.release()
        return {"frames": frames, "max_people": max_people, "seen": seen}
    except Exception:
        return None


def sam_plan(found: Optional[dict], people=("person",), child=("child",), sensitive: Sequence[str] = (),
             manipulated: Sequence[str] = ()) -> Dict[str, float]:
    """명사구 → SAM 에 보낼 초당 장면 수 (0 = 묻지 않음)."""
    plan: Dict[str, float] = {}
    many = found is None or found["max_people"] >= 2
    for c in (*people, *child):
        plan[c] = PEOPLE_FPS if many else 0.0
    screens = found is None or any(found["seen"].get(k) for k in ("tv", "laptop"))
    for c in sensitive:
        plan[c] = (SCREEN_FPS if screens else 0.0) if c in SCREENS else STATIC_FPS
    for c in manipulated:
        plan[c] = OBJECT_FPS
    return plan


def sam_frames(plan: Dict[str, float], fps0: float, n0: int) -> int:
    """이 계획으로 SAM 이 셀 장면 수 (요청마다 최소 50)."""
    return sum(max(50, -(-n0 // max(1, int(round(fps0 / f))))) for f in plan.values() if f > 0)
