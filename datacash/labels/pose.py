"""영상 → 장면마다 손·몸 관절.

손: HandLandmarker(VIDEO 모드, 추적) — 양손 21점 (x, y 픽셀, z 상대), 손잡이(왼/오른). 0.2초 이하로 끊긴 손은 앞뒤로 보간하고 표시한다.
몸: PoseLandmarker(VIDEO 모드) — 33점 화면 좌표(x, y 픽셀, 보임) + 3D 월드 좌표(미터, 골반 가운데 원점).
얼굴: FaceLandmarker(VIDEO 모드) — 표정 52값(blendshape 0~1) · 고개 방향(도) · 두 눈 시선(눈 안에서 눈동자 위치 -1~1).
  얼굴을 가상 인물로 바꾸기 전 원래 영상에서 뽑는다 — 치환한 영상은 표정을 줄이거나(재식별 검사) 옆모습을 원래대로 두지만, 이 값은 그대로 남는다.
  얼굴 점(478점)은 사람을 알아볼 수 있어 남기지 않는다. 화면에서 가장 큰 얼굴 하나만.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2

from ..light import lift
import numpy as np

from .. import config

MAX_GAP_S = 0.2
BLENDSHAPES = ["_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
               "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
               "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
               "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
               "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
               "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
               "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
               "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"]
FACE_VALUES = BLENDSHAPES + ["head_yaw", "head_pitch", "head_roll", "gaze_right_x", "gaze_right_y", "gaze_left_x", "gaze_left_y"]
# 눈마다 (눈동자 가운데, 바깥 꼬리, 안쪽 꼬리, 윗눈꺼풀, 아랫눈꺼풀) — MediaPipe 478점, 오른쪽·왼쪽은 찍힌 사람 기준
EYES = ((468, 33, 133, 159, 145), (473, 263, 362, 386, 374))


@dataclass
class FrameLabels:
    hands: List[Optional[np.ndarray]] = field(default_factory=lambda: [None, None])   # [왼, 오른] (21, 3) px,px,z
    hands_state: List[float] = field(default_factory=lambda: [0.0, 0.0])              # 1 검출, 0.5 보간, 0 없음
    body: Optional[np.ndarray] = None          # (33, 3) px, px, 보임
    body_world: Optional[np.ndarray] = None    # (33, 3) m
    face: Optional[np.ndarray] = None          # (len(FACE_VALUES),) 표정 52 · 고개 방향 3 · 시선 4


def _landmarkers(max_hands: int = 2):
    from mediapipe.tasks.python import BaseOptions, vision

    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.MODELS / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=max_hands,
        min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2, min_tracking_confidence=0.2))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.MODELS / "pose_landmarker_full.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1, min_pose_detection_confidence=0.3))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.MODELS / "face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
        output_facial_transformation_matrixes=True, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3))
    return hand, pose, face


def head_angles(M) -> tuple:
    """얼굴 변환 행렬 → 고개 방향 (yaw 좌우 · pitch 위아래 · roll 기울기, 도)."""
    R = np.asarray(M, np.float64)[:3, :3]
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(yaw), float(pitch), float(roll)


def gaze(pts: np.ndarray) -> List[float]:
    """두 눈 시선: 눈동자 가운데가 눈 안에서 어디 있나 — x 는 두 눈 꼬리 가운데에서 (-1 바깥 꼬리 쪽 ~ 1 안쪽 꼬리 쪽),
    y 는 두 눈꺼풀 가운데에서 (-1 위 ~ 1 아래). 고개가 기울어도 눈 꼬리 선을 기준으로 잰다."""
    out = []
    for c, outer, inner, up, low in EYES:
        a, b = pts[outer, :2], pts[inner, :2]
        u = (b - a) / (np.linalg.norm(b - a) + 1e-6)
        v = np.array([-u[1], u[0]])
        if np.dot(v, pts[low, :2] - pts[up, :2]) < 0:   # v 가 아래(아랫눈꺼풀 쪽)를 향하게
            v = -v
        half_w = np.linalg.norm(b - a) / 2
        half_h = max(abs(np.dot(pts[low, :2] - pts[up, :2], v)) / 2, half_w * 0.15)   # 눈을 감아도 0 으로 나누지 않게
        out += [float(np.clip(np.dot(pts[c, :2] - (a + b) / 2, u) / half_w, -1.5, 1.5)),
                float(np.clip(np.dot(pts[c, :2] - (pts[up, :2] + pts[low, :2]) / 2, v) / half_h, -1.5, 1.5))]
    return out


def extract_labels(video_path, max_frames: Optional[int] = None, start_frame: int = 0) -> List[FrameLabels]:
    import mediapipe as mp

    hand, pose, face = _landmarkers()
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
        fr = face.detect_for_video(img, ts)
        if fr.face_landmarks and fr.face_blendshapes and fr.facial_transformation_matrixes:
            pts = np.array([[p.x * w, p.y * h] for p in fr.face_landmarks[0]], np.float32)
            bs = {c.category_name: c.score for c in fr.face_blendshapes[0]}
            fl.face = np.array([bs.get(k, 0.0) for k in BLENDSHAPES] + list(head_angles(fr.facial_transformation_matrixes[0]))
                               + (gaze(pts) if len(pts) >= 478 else [0.0] * 4), np.float32)
        out.append(fl)
        i += 1
    cap.release()
    hand.close()
    pose.close()
    face.close()
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
