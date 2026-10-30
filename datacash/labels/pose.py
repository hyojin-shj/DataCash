"""영상 → 장면마다 손·몸 관절.

손: HandLandmarker(VIDEO 모드, 추적) — 양손 21점 (x, y 픽셀, z 상대), 손잡이(왼/오른). 0.2초 이하로 끊긴 손은 앞뒤로 보간하고 표시한다.
몸: PoseLandmarker(VIDEO 모드) — 33점 화면 좌표(x, y 픽셀, 보임) + 3D 월드 좌표(미터, 골반 가운데 원점).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2

from ..light import lift
import numpy as np

from .. import config

MAX_GAP_S = 0.2


@dataclass
class FrameLabels:
    hands: List[Optional[np.ndarray]] = field(default_factory=lambda: [None, None])   # [왼, 오른] (21, 3) px,px,z
    hands_state: List[float] = field(default_factory=lambda: [0.0, 0.0])              # 1 검출, 0.5 보간, 0 없음
    body: Optional[np.ndarray] = None          # (33, 3) px, px, 보임
    body_world: Optional[np.ndarray] = None    # (33, 3) m


def _landmarkers(max_hands: int = 2):
    from mediapipe.tasks.python import BaseOptions, vision

    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.MODELS / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=max_hands,
        min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2, min_tracking_confidence=0.2))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.MODELS / "pose_landmarker_full.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1, min_pose_detection_confidence=0.3))
    return hand, pose


def extract_labels(video_path, max_frames: Optional[int] = None, start_frame: int = 0) -> List[FrameLabels]:
    import mediapipe as mp

    hand, pose = _landmarkers()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out: List[FrameLabels] = []
    i = 0
    while max_frames is None or i < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        bgr = lift(bgr)                     # 어두운 곳에서 찍어도 손·몸을 찾게
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = int((start_frame + i) * 1000 / fps)
        fl = FrameLabels()
        hr = hand.detect_for_video(img, ts)
        for lms, side in zip(hr.hand_landmarks, hr.handedness):
            k = 0 if side[0].category_name == "Left" else 1
            if fl.hands[k] is None:
                fl.hands[k] = np.array([[p.x * w, p.y * h, p.z] for p in lms], np.float32)
                fl.hands_state[k] = 1.0
        pr = pose.detect_for_video(img, ts)
        if pr.pose_landmarks:
            fl.body = np.array([[p.x * w, p.y * h, p.visibility] for p in pr.pose_landmarks[0]], np.float32)
            if pr.pose_world_landmarks:
                fl.body_world = np.array([[p.x, p.y, p.z] for p in pr.pose_world_landmarks[0]], np.float32)
        out.append(fl)
        i += 1
    cap.release()
    hand.close()
    pose.close()
    fill_hand_gaps(out, int(round(MAX_GAP_S * fps)))
    return out


def fill_hand_gaps(frames: List[FrameLabels], max_gap: int) -> None:
    for k in (0, 1):
        idx = [i for i, f in enumerate(frames) if f.hands_state[k] == 1.0]
        for a, b in zip(idx, idx[1:]):
            if 1 < b - a <= max_gap + 1:
                for j in range(a + 1, b):
                    t = (j - a) / (b - a)
                    frames[j].hands[k] = (1 - t) * frames[a].hands[k] + t * frames[b].hands[k]
                    frames[j].hands_state[k] = 0.5
