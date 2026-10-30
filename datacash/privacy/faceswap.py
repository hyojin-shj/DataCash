"""참여자 얼굴을 실존하지 않는 가상 인물(Muse Image 로 만든 사진)로 바꾼다 — LivePortrait(MIT) + MediaPipe 얼굴 자르기.

LivePortrait 기본 얼굴 찾기·자르기는 InsightFace(비상업 라이선스)를 쓰므로 부르지 않는다. 대신
  자르기   MediaPipe FaceLandmarker 478점 → 눈 선으로 기울기를 맞춘 정사각형 (LivePortrait 의 자르기 비율과 비슷하게)
  움직임   원래 영상의 참여자 얼굴을 driving 으로, 가상 인물 사진을 source 로 (상대 움직임: 고개·표정을 첫 장면 대비 차이로 옮김)
  되붙이기 LivePortrait 의 얼굴 마스크 틀로 가장자리를 부드럽게
  재식별 검사  OpenCV YuNet + SFace(Apache 2.0): 바꾼 얼굴이 원래 사람과 닮으면(REID_MAX 이상) 표정을 줄여 다시 만든다
  옆얼굴·먼 얼굴  MediaPipe 가 점을 못 찾아도 YuNet 의 5점(눈·코·입꼬리)으로 자르기를 만들어 치환한다 — 모자이크로 끊기지 않게.
              그래도 못 바꾸면 얼굴 모양대로 부드럽게 흐린다 (굵은 모자이크는 쓰지 않는다)
  자연스럽게  붙일 얼굴의 밝기·대비·색을 장면 얼굴 자리에 맞추고, 자르기 위치를 앞 장면과 섞어 떨림을 줄인다
원본(가리기 전) 영상이 필요하다 — 해커톤 시연은 동의자 본인 녹화로만. LivePortrait 위치는 LP_ROOT 환경 변수.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .. import config

LP_ROOT = Path(os.environ.get("LP_ROOT", Path.home() / "home-data-mission" / "models" / "liveportrait"))
REID_MAX = 0.363           # SFace 의 '같은 사람' 기준 — 이보다 덜 닮으면 다른 사람 (닮은 가상 인물이 자연스럽다는 의견으로 0.30 → 기준 그대로)
EXP_RAMP = 0.25            # 표정을 줄인 뒤 RAMP_EVERY 장면마다 이만큼씩 되돌린다
RAMP_EVERY = 5
CROP = 512
EYE_L, EYE_R, CHIN, MOUTH = (33, 133), (362, 263), 152, (13, 14)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
             150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


@dataclass
class SwapReport:
    frames: int = 0
    swapped: int = 0
    no_face: int = 0
    damped: int = 0                 # 바꾼 얼굴이 닮아 표정을 줄여 다시 만든 장면 (모자이크 대신 — 영상이 이어진다)
    over_reid: int = 0              # 표정을 다 빼도 같은 사람 기준(REID_MAX)을 넘은 장면 — 가장 덜 닮은 것을 그대로 썼다 (인증서에 남김)
    small_face_swapped: int = 0     # 작은 얼굴: YuNet 으로 찾고 그 둘레를 키워 478점을 찾아 치환한 장면
    small_face_covered: int = 0     # 478점도 YuNet 5점 치환도 안 된 얼굴 → 얼굴 모양대로 부드럽게 흐림 (원래 얼굴이 남지 않게)
    yunet_swapped: int = 0          # MediaPipe 가 못 찾은 옆얼굴·먼 얼굴을 YuNet 5점 자르기로 치환한 장면
    family: int = 0                 # 동의한 가족 수 (사람마다 다른 가상 인물)
    reid_max: float = 0.0
    reid_mean: float = 0.0
    seconds: float = 0.0
    donor: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class FaceLandmarks:
    def __init__(self, max_faces: int = 4):
        from mediapipe.tasks.python import BaseOptions, vision

        self.lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.MODELS / "face_landmarker.task")),
            num_faces=max_faces, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))

    def __call__(self, bgr) -> List[np.ndarray]:
        import mediapipe as mp

        h, w = bgr.shape[:2]
        r = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        return [np.array([[p.x * w, p.y * h] for p in f], np.float32) for f in r.face_landmarks]


def crop_matrix(lmk: np.ndarray, size: int = CROP, scale: float = 1.9, vy: float = -0.06) -> np.ndarray:
    """crop(size×size) → 원본 좌표 2×3 행렬. 눈 선이 수평이 되게 돌리고, 얼굴 점 범위의 scale 배 (LivePortrait 자르기 비율 근처)."""
    el, er = lmk[list(EYE_L)].mean(0), lmk[list(EYE_R)].mean(0)
    ang = np.arctan2(er[1] - el[1], er[0] - el[0])
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, s], [-s, c]])                       # 원본 → 똑바로 세운 좌표
    p = (lmk - lmk.mean(0)) @ rot.T
    lo, hi = p.min(0), p.max(0)
    side = float(max(hi - lo)) * scale
    up = np.array([np.sin(ang), -np.cos(ang)])
    center = lmk.mean(0) + rot.T @ ((lo + hi) / 2) + up * (-vy * side)
    k = side / size
    R = np.array([[c, -s], [s, c]]) * k
    t = center - R @ np.array([size / 2, size / 2])
    return np.hstack([R, t[:, None]]).astype(np.float32)


def warp_in(img, M, size=CROP):
    return cv2.warpAffine(img, cv2.invertAffineTransform(M), (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class Reid:
    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(str(config.MODELS / "yunet.onnx"), "", (320, 320), 0.6)
        self.rec = cv2.FaceRecognizerSF.create(str(config.MODELS / "sface.onnx"), "")

    def embed(self, bgr, box_hint=None) -> Optional[np.ndarray]:
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or not len(faces):
            return None
        if box_hint is not None:
            cx, cy = box_hint
            f = min(faces, key=lambda f: (f[0] + f[2] / 2 - cx) ** 2 + (f[1] + f[3] / 2 - cy) ** 2)
        else:
            f = max(faces, key=lambda f: f[2] * f[3])
        return self.rec.feature(self.rec.alignCrop(bgr, f)).ravel()

    def sim(self, a, b) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


ALIGN = [33, 133, 362, 263, 1, 61, 291, 152, 234, 454]   # 눈꼬리 4 · 코끝 · 입꼬리 2 · 턱 · 양 볼 (MediaPipe 얼굴 점)
D0_FRAMES = 15        # 움직임 기준 후보 수
EXP_CLIP = 0.07       # 키포인트마다 표정 차이 한도 (LivePortrait exp 단위)


class LivePortraitSwap:
    def __init__(self):
        sys.path.insert(0, str(LP_ROOT))
        import torch
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.camera import get_rotation_matrix

        self.torch, self.rot = torch, get_rotation_matrix
        self.w = LivePortraitWrapper(InferenceConfig())
        self.half = self.w.device == "mps"
        if self.half:       # 맥 GPU: 큰 Conv3d 를 2D 로 + 변형·생성은 fp16 (장면당 1.65초 → 0.17초, fp32 와 PSNR 51dB)
            from .mps_speed import speed_up

            speed_up(self.w.warping_module)
            self.w.warping_module.half()
            self.w.spade_generator.half()
        m = cv2.imread(str(LP_ROOT / "src" / "utils" / "resources" / "mask_template.png"), cv2.IMREAD_GRAYSCALE)
        self.mask = cv2.resize(m, (CROP, CROP)).astype(np.float32) / 255.0 if m is not None else None

    def source(self, donor_bgr, lmk):
        M = crop_matrix(lmk)
        crop = warp_in(donor_bgr, M)
        x = self.w.prepare_source(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        info = self.w.get_kp_info(x)
        self.s_info, self.s_feat = info, self.w.extract_feature_3d(x)
        self.s_kp = self.w.transform_keypoint(info)
        self.s_R = self.rot(info["pitch"], info["yaw"], info["roll"])
        self.d0, self.d0_buf = None, []
        return crop

    def set_reference(self, crops: list) -> None:
        """움직임 기준을 영상 곳곳의 얼굴 장면(MediaPipe 로 자른 crop 들) 가운데 표정이 가운데인 장면으로 미리 정한다
        (첫 장면들로만 정하면 녹화 처음의 표정에 끌려간다)."""
        for c in crops:
            if len(self.d0_buf) >= D0_FRAMES:
                break
            x = self.w.prepare_source(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            d = self.w.get_kp_info(x)
            self.d0_buf.append((d, self.rot(d["pitch"], d["yaw"], d["roll"])))
        if self.d0_buf:
            t = self.torch
            ex = t.stack([b[0]["exp"].flatten() for b in self.d0_buf])
            self.d0 = self.d0_buf[int(((ex - ex.median(0).values) ** 2).sum(1).argmin())]

    def state(self) -> tuple:
        """지금 가상 인물·움직임 기준 (여러 사람을 바꿀 때 사람마다 따로 둔다)."""
        return self.s_info, self.s_feat, self.s_kp, self.s_R, self.d0, list(self.d0_buf)

    def use(self, st: tuple) -> None:
        self.s_info, self.s_feat, self.s_kp, self.s_R, self.d0, buf = st
        self.d0_buf = list(buf)

    def drive(self, drive_crop_bgr, exp_scale: float = 1.0, anchor: bool = True) -> np.ndarray:
        """원래 얼굴 crop → 가상 인물이 같은 움직임을 한 512 얼굴 (BGR).
        exp_scale: 표정을 얼마나 옮길지 (1 = 그대로, 0 = 고개 움직임만). 표정이 원래 사람을 닮게 만드는 장면에서 줄인다.
        anchor: 이 장면을 움직임 기준(d0) 후보로 써도 되는가 — MediaPipe 점으로 제대로 자른 얼굴만.
          (상자 옮기기 녹화: 기준이 YuNet·작은 얼굴로 다르게 자른 장면에서 잡혀, 이후 모든 장면의 표정 차이가 커져
           눈이 튀어나온 '놀란 얼굴' 이 됐다.) 기준은 처음 D0_FRAMES 개 후보 가운데 표정이 가운데인 장면, 표정 차이는 EXP_CLIP 로 묶는다."""
        t = self.torch
        x = self.w.prepare_source(cv2.cvtColor(drive_crop_bgr, cv2.COLOR_BGR2RGB))
        d = self.w.get_kp_info(x)
        R_d = self.rot(d["pitch"], d["yaw"], d["roll"])
        if anchor and len(self.d0_buf) < D0_FRAMES:
            self.d0_buf.append((d, R_d))
            ex = t.stack([b[0]["exp"].flatten() for b in self.d0_buf])
            med = ex.median(0).values
            self.d0 = self.d0_buf[int(((ex - med) ** 2).sum(1).argmin())]
        if self.d0 is None:                   # 아직 기준이 없으면 이 장면이 기준인 것처럼 (표정 차이 0, 고개만)
            d0, R_d0 = d, R_d
        else:
            d0, R_d0 = self.d0
        s = self.s_info
        R_new = (R_d @ R_d0.permute(0, 2, 1)) @ self.s_R
        delta = (d["exp"] - d0["exp"]).clamp(-EXP_CLIP, EXP_CLIP)
        exp_new = s["exp"] + exp_scale * delta
        scale_new = s["scale"] * (d["scale"] / d0["scale"])
        t_new = s["t"] + (d["t"] - d0["t"])
        t_new[..., 2] = 0
        kp_c = s["kp"].view(1, -1, 3)
        x_d = scale_new[..., None] * (kp_c @ R_new + exp_new.view(1, -1, 3)) + t_new[:, None, :]
        x_d = self.w.stitching(self.s_kp, x_d)
        with t.no_grad():
            if self.half:
                warped = self.w.warping_module(self.s_feat.half(), kp_source=self.s_kp.half(), kp_driving=x_d.half())["out"]
                out = self.w.spade_generator(feature=warped).float()
            else:
                out = self.w.warp_decode(self.s_feat, self.s_kp, x_d)["out"]
        img = (out.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()[0] * 255).astype(np.uint8)
        return cv2.cvtColor(cv2.resize(img, (CROP, CROP)), cv2.COLOR_RGB2BGR)

    def paste(self, frame, face512, M, gen_lmk: Optional[np.ndarray], clip=None, target: Optional[np.ndarray] = None) -> np.ndarray:
        """만든 얼굴의 얼굴 윤곽(이마는 조금 위로) 안만, 가장자리를 흐리게, 색을 장면에 맞춰 붙인다.
        clip: YuNet 얼굴(점 없이 자른 경우) — 붙이는 자리를 원래 얼굴 타원 안으로 묶는다 (얼굴 밖으로 튀어나오지 않게).
        target: 원래 얼굴 점(장면 좌표) — 만든 얼굴의 눈·코·입·턱을 이 자리에 맞춰 옮긴다 (가상 인물의 얼굴 비율이 달라
          눈코입이 얼굴 폭의 10%쯤, 많게는 20% 어긋나 이마에 눈이 붙은 것처럼 보였다)."""
        h, w = frame.shape[:2]
        if gen_lmk is not None and target is not None:
            tc = cv2.transform(target[None, :, :2].astype(np.float32), cv2.invertAffineTransform(M))[0]
            A, _ = cv2.estimateAffine2D(gen_lmk[ALIGN, :2].astype(np.float32), tc[ALIGN], method=cv2.LMEDS)
            if A is not None and 0.7 < np.sqrt(abs(np.linalg.det(A[:, :2]))) < 1.4:
                face512 = cv2.warpAffine(face512, A, (CROP, CROP), borderMode=cv2.BORDER_REPLICATE)
                gen_lmk = gen_lmk.copy()
                gen_lmk[:, :2] = cv2.transform(gen_lmk[None, :, :2].astype(np.float32), A)[0]
        mk = np.zeros((CROP, CROP), np.float32)
        if gen_lmk is not None:
            poly = gen_lmk[FACE_OVAL].copy()
            c = poly.mean(0)
            poly = c + (poly - c) * 0.97
            cv2.fillPoly(mk, [poly.astype(np.int32)], 1.0)
        elif self.mask is not None:
            mk = self.mask.copy()
        k = max(3, int(CROP * 0.035)) | 1
        mk = cv2.GaussianBlur(cv2.erode(mk, np.ones((k, k), np.uint8)), (2 * k + 1, 2 * k + 1), 0)
        bg = warp_in(frame, M)
        sel = mk > 0.5
        if sel.sum() > 100:                                  # 밝기·대비·색을 원래 장면 얼굴 자리에 맞춘다 (LAB 평균·퍼짐)
            a = cv2.cvtColor(face512, cv2.COLOR_BGR2LAB).astype(np.float32)
            b = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
            for ch in range(3):                              # 밝기는 퍼짐도 조금만(그늘진 장면에서 얼룩지지 않게), 색은 평균만
                am, asd = a[..., ch][sel].mean(), a[..., ch][sel].std() + 1e-3
                bm, bsd = b[..., ch][sel].mean(), b[..., ch][sel].std() + 1e-3
                k = float(np.clip(bsd / asd, 0.9, 1.1)) if ch == 0 else 1.0
                a[..., ch] = (a[..., ch] - am) * k + am + (bm - am) * (0.85 if ch == 0 else 1.0)
            face512 = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        warped = cv2.warpAffine(face512, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mw = cv2.warpAffine(mk, M, (w, h), flags=cv2.INTER_LINEAR)
        if clip is not None:
            x, y, fw, fh = (float(v) for v in clip[:4])
            el = np.zeros((h, w), np.float32)
            cv2.ellipse(el, (int(x + fw / 2), int(y + fh / 2)), (int(fw * 0.55), int(fh * 0.62)), 0, 0, 360, 1.0, -1)
            kk = max(3, int(fw * 0.25)) | 1
            mw = mw * cv2.GaussianBlur(el, (kk, kk), 0)
        mw = mw[..., None]
        return (warped * mw + frame * (1 - mw)).astype(np.uint8)


SMALL_TARGET = 360   # 작은 얼굴 둘레를 이 크기(px)로 키워 다시 찾는다


def find_small_faces(frame, fl: "FaceLandmarks", reid: "Reid", participant_box=None) -> List[np.ndarray]:
    """MediaPipe 가 못 찾는 작은 얼굴: YuNet 으로 상자를 찾고, 상자 3배 둘레를 SMALL_TARGET 크기로 키운 뒤
    478점을 찾아 원래 좌표로 되돌린다 (먼 장면의 40px 얼굴도 치환할 수 있게)."""
    h, w = frame.shape[:2]
    reid.det.setInputSize((w, h))
    _, boxes = reid.det.detect(frame)
    out = []
    for f in boxes if boxes is not None else []:
        x, y, fw, fh = (float(v) for v in f[:4])
        cx, cy = x + fw / 2, y + fh / 2
        if participant_box is not None and not (participant_box[0] <= cx <= participant_box[2] and participant_box[1] <= cy <= participant_box[3]):
            continue
        side = 3.0 * max(fw, fh)
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        k = SMALL_TARGET / max(x1 - x0, y1 - y0)
        roi = cv2.resize(frame[y0:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        found = fl(roi)
        if found:
            lm = max(found, key=lambda a: np.ptp(a[:, 0]) * np.ptp(a[:, 1]))
            out.append((lm / k + np.array([x0, y0], np.float32)).astype(np.float32))
    return out


def yunet_faces(frame, reid: "Reid", participant_box=None) -> List[np.ndarray]:
    """YuNet 얼굴 [x, y, w, h, 오른눈x, y, 왼눈x, y, 코x, y, 입오른x, y, 입왼x, y, 점수] (참여자 상자가 있으면 그 안만)."""
    h, w = frame.shape[:2]
    reid.det.setInputSize((w, h))
    _, faces = reid.det.detect(frame)
    out = []
    for f in faces if faces is not None else []:
        cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
        if participant_box is None or (participant_box[0] <= cx <= participant_box[2] and participant_box[1] <= cy <= participant_box[3]):
            out.append(f)
    return out


def crop_from_yunet(f, size: int = CROP) -> np.ndarray:
    """MediaPipe 점이 없을 때: YuNet 상자·눈 두 점으로 crop_matrix 와 비슷한 자르기 (눈 선 수평, 상자의 1.9배 근처)."""
    re, le = np.array(f[4:6], np.float32), np.array(f[6:8], np.float32)
    ang = np.arctan2(le[1] - re[1], le[0] - re[0])            # 화면 왼쪽 눈 → 오른쪽 눈
    c, s = np.cos(ang), np.sin(ang)
    side = float(max(f[2], f[3])) * 1.95
    center = np.array([f[0] + f[2] / 2, f[1] + f[3] / 2], np.float32) + np.array([np.sin(ang), -np.cos(ang)]) * (0.06 * side)
    k = side / size
    R = np.array([[c, -s], [s, c]]) * k
    t = center - R @ np.array([size / 2, size / 2])
    return np.hstack([R, t[:, None]]).astype(np.float32)


PROFILE_EYES = 0.36    # YuNet 두 눈 사이 / 얼굴 폭이 이보다 작으면 옆얼굴 — LivePortrait 가 옆얼굴은 못 만들어 흐리기로


def is_profile(f) -> bool:
    return float(np.hypot(f[6] - f[4], f[7] - f[5])) / max(1.0, float(f[2])) < PROFILE_EYES


def soften(img, f) -> None:
    """치환할 수 없는 얼굴: 얼굴 모양(타원)대로, 가장자리를 흐려 부드럽게 (굵은 모자이크처럼 튀지 않게)."""
    h, w = img.shape[:2]
    x, y, fw, fh = (float(v) for v in f[:4])
    cx, cy, ax, ay = x + fw / 2, y + fh / 2, fw * 0.62, fh * 0.75
    x0, y0, x1, y1 = int(max(0, cx - ax * 1.4)), int(max(0, cy - ay * 1.4)), int(min(w, cx + ax * 1.4)), int(min(h, cy + ay * 1.4))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return
    roi = img[y0:y1, x0:x1]
    k = max(5, int(max(fw, fh) * 0.45)) | 1
    blur = cv2.GaussianBlur(cv2.resize(cv2.resize(roi, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
                                       (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR), (k, k), 0)
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.ellipse(m, (int(cx - x0), int(cy - y0)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (k, k), 0)[..., None]
    img[y0:y1, x0:x1] = (blur * m + roi * (1 - m)).astype(np.uint8)


def smooth_crop(prev: Optional[np.ndarray], M: np.ndarray) -> np.ndarray:
    """자르기 위치를 앞 장면과 섞는다 — 얼굴 점이 장면마다 조금씩 흔들려 붙인 얼굴이 떨리는 것을 줄인다.
    많이 움직였으면(얼굴 크기의 4% 넘게) 그대로 따라간다 (늦게 따라가 어긋나지 않게)."""
    if prev is None:
        return M
    scale = float(np.hypot(M[0, 0], M[1, 0])) * CROP
    move = float(np.hypot(*(M[:, 2] - prev[:, 2])))
    a = 0.55 if move < 0.04 * scale else 0.0
    return (prev * a + M * (1 - a)).astype(np.float32)


def swap_video(original, base, out_path, donor_img, participant_boxes: Optional[dict] = None,
               log=print, consented_boxes: Optional[dict] = None, extra_donors: Optional[list] = None,
               family_donors: Optional[dict] = None) -> SwapReport:
    """original: 원래 영상(얼굴·움직임 읽기용), base: 가린 영상(같은 장면 수) 위에 바꾼 얼굴을 붙인다.
    participant_boxes: 장면 → 참여자 상자. consented_boxes: 장면 → [(추적 id, 상자)] 동의한 가족 (사람마다 다른 가상 인물).
    family_donors: 추적 id → 그 사람이 직접 고른 가상 인물 사진 (없으면 덜 닮은 순서대로 자동)."""
    import time

    t0 = time.time()
    rep = SwapReport(donor=Path(donor_img).name)
    fl, lp, reid = FaceLandmarks(), LivePortraitSwap(), Reid()
    donors = [Path(donor_img)] + [Path(d) for d in (extra_donors or []) if Path(d) != Path(donor_img)]
    states: dict = {}

    def person_state(key, k):
        """사람(key)마다 가상 인물 하나: 참여자는 첫째, 가족은 고른 사람 또는 다음 사람부터 차례로."""
        if key not in states:
            chosen = (family_donors or {}).get(key.split(":", 1)[1]) if key.startswith("family:") else None
            img = cv2.imread(str(chosen or donors[min(k, len(donors) - 1)]))
            dl = fl(img)
            if not dl:
                raise RuntimeError("가상 인물 사진에서 얼굴을 못 찾음")
            lp.source(img, dl[0])
            states[key] = lp.state()
        return states[key]

    person_state("participant", 0)
    ref, cap = [], cv2.VideoCapture(str(original))          # 참여자 움직임 기준: 영상 곳곳 정면에 가까운 얼굴 장면들로 미리
    nf = int(cap.get(7))
    for j in np.linspace(0, max(0, nf - 1), 3 * D0_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
        ok, f = cap.read()
        found = fl(f) if ok else []
        if found:
            lmk = max(found, key=lambda a: np.ptp(a[:, 0]) * np.ptp(a[:, 1]))
            if np.ptp(lmk[:, 0]) > 80:                     # 너무 작은 얼굴은 빼고
                ref.append(warp_in(f, crop_matrix(lmk)))
    cap.release()
    if ref:
        lp.use(states["participant"])
        lp.set_reference(ref[:: max(1, len(ref) // D0_FRAMES)])
        states["participant"] = lp.state()
    co, cb = cv2.VideoCapture(str(original)), cv2.VideoCapture(str(base))
    fps = cb.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cb.get(3)), int(cb.get(4))
    out_path = Path(out_path)
    raw = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    sims, scales, crops, i = [], {}, {}, 0
    family_order: list = []
    while True:
        ok1, fo = co.read()
        ok2, fb = cb.read()
        if not (ok1 and ok2):
            break
        if fo.shape[:2] != (h, w):
            fo = cv2.resize(fo, (w, h))
        rep.frames += 1
        faces = fl(fo)
        targets = [("participant", (participant_boxes or {}).get(i))]
        for oid, box in (consented_boxes or {}).get(i, []):
            if oid not in family_order:
                family_order.append(oid)
            targets.append((f"family:{oid}", box))
        res, any_face, used = fb, False, set()
        for key, box in targets:
            inside = [k for k, f in enumerate(faces) if k not in used and box is not None
                      and box[0] <= f[:, 0].mean() <= box[2] and box[1] <= f[:, 1].mean() <= box[3]]
            if not inside and box is None and key == "participant":
                inside = [k for k in range(len(faces)) if k not in used]      # 참여자 상자가 없으면 가장 큰 얼굴
            small, yf = False, None
            if inside:
                kk = max(inside, key=lambda k: np.ptp(faces[k][:, 0]) * np.ptp(faces[k][:, 1]))
                used.add(kk)
                lmk = faces[kk]
            else:
                found = find_small_faces(fo, fl, reid, box)
                if found:
                    lmk, small = found[0], True
                else:                                      # 옆얼굴·먼 얼굴: YuNet 5점으로 자른다 (모자이크로 끊기지 않게)
                    yfs = yunet_faces(fo, reid, box)
                    if not yfs:
                        continue
                    yf = max(yfs, key=lambda f: f[2] * f[3])
                    lmk = None
                    if is_profile(yf):                     # 옆얼굴은 만들면 얼굴이 튀어나와 더 어색하다 → 얼굴 모양대로 흐림
                        soften(res, yf)
                        rep.small_face_covered += 1
                        any_face = True
                        crops.pop(key, None)
                        continue
            any_face = True
            M = smooth_crop(crops.get(key), crop_matrix(lmk) if lmk is not None else crop_from_yunet(yf))
            crops[key] = M
            crop = warp_in(fo, M)
            c = lmk.mean(0) if lmk is not None else np.array([yf[0] + yf[2] / 2, yf[1] + yf[3] / 2], np.float32)
            e0 = reid.embed(fo, c)
            # 재식별 검사를 장면마다: 바꾼 얼굴이 원래 사람과 닮으면 모자이크(영상이 끊긴다) 대신 표정을 줄여 다시 만든다
            # 바로 앞 장면에서 통과한 표정 세기부터 (매번 원래 표정부터 다시 하면 장면마다 2~3번 만들어 느렸다),
            # 줄였던 표정은 RAMP_EVERY 장면마다 한 번씩만 조금 올려 본다 (깜빡이지 않게, 약 0.7초에 원래대로)
            prev = scales.get(key, 1.0)
            start = min(1.0, prev + EXP_RAMP) if prev < 1.0 and i % RAMP_EVERY == 0 else prev
            tries = []                                     # (닮음, 결과, 표정 비율)
            for es in dict.fromkeys((start, prev, 0.0) if start > prev else (start, 0.0)):
                lp.use(person_state(key, 0 if key == "participant" else 1 + family_order.index(key.split(":", 1)[1])))
                gen = lp.drive(crop, es, anchor=lmk is not None and not small)
                gl = fl(gen)
                out = lp.paste(res, gen, M, gl[0] if gl else None, clip=yf, target=lmk)
                e1 = reid.embed(out, c) if e0 is not None else None
                sv = reid.sim(e0, e1) if e1 is not None else None
                tries.append((-1.0 if sv is None else sv, out, es))
                if sv is None or sv < REID_MAX:
                    break
            sv, out, used_scale = min(tries, key=lambda t: t[0])   # 다 넘으면 가장 덜 닮은 것 — 굳히거나 모자이크하지 않는다 (연속성)
            if sv >= 0:
                sims.append(sv)
            states[key] = lp.state()                       # 움직임 기준(d0)을 사람마다 이어 간다
            scales[key] = used_scale
            rep.swapped += 1
            rep.damped += used_scale < 1.0
            rep.over_reid += sv >= REID_MAX
            rep.small_face_swapped += small
            rep.yunet_swapped += yf is not None
            res = out
        if not any_face:
            rep.no_face += 1
            covered = 0
            for _, box in targets:                         # 남은 얼굴(치환도 못 한 것)은 부드럽게 흐린다
                for f in yunet_faces(fo, reid, box):
                    soften(res, f)
                    covered += 1
            rep.small_face_covered += covered > 0
            crops.clear()                                  # 얼굴이 끊겼다 다시 나오면 앞 위치와 섞지 않는다
        vw.write(res)
        i += 1
        if i % 60 == 0:
            log(f"얼굴 치환 {i}장면")
    vw.release()
    from ..media import run

    run(["-i", raw, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path])
    raw.unlink()
    if sims:
        rep.reid_max, rep.reid_mean = float(np.max(sims)), float(np.mean(sims))
    rep.family = len(family_order)
    rep.seconds = time.time() - t0
    return rep




def _donor_info() -> dict:
    """가상 인물 이름 → (성별, 나이) — Muse Image 로 만들 때 정한 값 (donors/donors.json, scripts/make_donors.py)."""
    import json

    p = Path(__file__).parents[1] / "donors" / "donors.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    return {k: (v.get("gender"), v.get("age")) for k, v in d.items()}


DONOR_INFO = _donor_info()
AGE_SPAN = 25             # 나이를 알면 이만큼 안쪽만 후보 (20대 참여자에게 60대 얼굴이 붙지 않게)


def _age_num(band: Optional[str]) -> Optional[int]:
    if not band:
        return None
    digits = "".join(ch for ch in band if ch.isdigit())
    return 20 if band.startswith("19") else int(digits[:2]) if digits else None


OVER_LIMIT = 0.5          # 시험 장면 가운데 같은 사람 기준을 넘는 비율이 이보다 크면 다음으로 닮은 사람
TRIAL_FRAMES = 20
TRIAL_DONORS = 4


def over_rate(original_video, donor: Path, frames: int = TRIAL_FRAMES, lp: Optional["LivePortraitSwap"] = None,
              fl: Optional["FaceLandmarks"] = None, reid: Optional["Reid"] = None) -> Optional[float]:
    """영상 곳곳 몇 장면을 이 가상 인물로 실제처럼(표정 그대로 → 반 → 고개만) 바꿔 보고, 셋 다 같은 사람 기준을 넘는 장면 비율."""
    fl, reid, lp = fl or FaceLandmarks(), reid or Reid(), lp or LivePortraitSwap()
    img = cv2.imread(str(donor))
    dl = fl(img) if img is not None else []
    if not dl:
        return None
    cap = cv2.VideoCapture(str(original_video))
    n = int(cap.get(7))
    over = seen = 0
    st = None
    for i in np.linspace(0, max(0, n - 10), frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        found = fl(f) if ok else []
        if not found:
            continue
        lmk = max(found, key=lambda a: np.ptp(a[:, 0]) * np.ptp(a[:, 1]))
        c = lmk.mean(0)
        e0 = reid.embed(f, c)
        if e0 is None:
            continue
        M = crop_matrix(lmk)
        crop = warp_in(f, M)
        if st is None:                       # 실제 치환처럼 첫 장면을 움직임 기준으로 이어 간다
            lp.source(img, dl[0])
        else:
            lp.use(st)
        best = None
        for es in (1.0, 0.5, 0.0):
            g = lp.drive(crop, es)
            gl = fl(g)
            e1 = reid.embed(lp.paste(f, g, M, gl[0] if gl else None), c)
            sv = reid.sim(e0, e1) if e1 is not None else 0.0
            best = sv if best is None else min(best, sv)
            if st is None:
                st = lp.state()
            if sv < REID_MAX:
                break
        seen += 1
        over += best >= REID_MAX
    cap.release()
    return over / seen if seen else None


def pick_donor(original_video, ranked: List[Path], log=print) -> tuple:
    """닮은 순서(rank_donors)대로 시험해, 같은 사람 기준을 넘는 장면이 OVER_LIMIT 이하인 첫 사람.
    (가장 닮은 사람이 어떤 영상에서는 거의 본인처럼 나왔다 — 점심 녹화 티슈 ②, 87% 장면이 기준을 넘음.)
    모두 넘으면 시험한 가운데 가장 덜 넘은 사람. (고른 사람, {이름: 넘은 비율})"""
    fl, reid, lp = FaceLandmarks(), Reid(), LivePortraitSwap()
    tried = {}
    for d in ranked[:TRIAL_DONORS]:
        r = over_rate(original_video, d, lp=lp, fl=fl, reid=reid)
        tried[d.stem] = None if r is None else round(r, 2)
        log(f"가상 인물 시험 {d.stem}: 기준 넘음 {tried[d.stem]}")
        if r is not None and r <= OVER_LIMIT:
            return d, tried
    ok = [d for d in ranked[:TRIAL_DONORS] if tried.get(d.stem) is not None]
    return (min(ok, key=lambda d: tried[d.stem]) if ok else ranked[0]), tried


def rank_donors(original_video, donors_dir=None, gender: Optional[str] = None, age_band: Optional[str] = None) -> List[Path]:
    """가상 인물 순서 (첫째는 참여자, 다음은 동의한 가족에게).
    같은 성별 · 나이 차 AGE_SPAN 안에서 사진이 가장 닮은 사람부터 — 닮을수록 치환이 자연스럽다 (점심 녹화 의견:
    '닮아도 어차피 내가 아니다'). 단 같은 사람 기준(REID_MAX)을 넘게 닮은 사진은 빼고, 치환 중 장면마다 재식별 검사가 지킨다."""
    donors = sorted(Path(donors_dir or Path(__file__).parents[1] / "donors").glob("*.png"))
    if not donors:
        return []
    reid = Reid()
    cap = cv2.VideoCapture(str(original_video))
    n = int(cap.get(7))
    embs = []
    for i in np.linspace(0, max(0, n - 10), 8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        e = reid.embed(f) if ok else None
        if e is not None:
            embs.append(e)
    if not embs:
        return donors

    def worst(d):
        e = reid.embed(cv2.imread(str(d)))
        return 1.0 if e is None else max(reid.sim(e, x) for x in embs)

    sims = {d: worst(d) for d in donors}
    age = _age_num(age_band)

    def fit(d):
        g, a = DONOR_INFO.get(d.stem, (None, None))
        return (0 if sims[d] < REID_MAX else 1,                        # 같은 사람만큼 닮은 사진은 맨 뒤
                0 if gender in (None, "") or g == gender else 1,
                0 if not age or not a or abs(a - age) <= AGE_SPAN else 1,
                -sims[d] if sims[d] < REID_MAX else sims[d])           # 그 안에서 가장 닮은 사람

    return sorted(donors, key=fit)
