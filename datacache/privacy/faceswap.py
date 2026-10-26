"""참여자 얼굴을 실존하지 않는 가상 인물(Muse Image 로 만든 사진)로 바꾼다 — LivePortrait(MIT) + MediaPipe 얼굴 자르기.

LivePortrait 기본 얼굴 찾기·자르기는 InsightFace(비상업 라이선스)를 쓰므로 부르지 않는다. 대신
  자르기   MediaPipe FaceLandmarker 478점 → 눈 선으로 기울기를 맞춘 정사각형 (LivePortrait 의 자르기 비율과 비슷하게)
  움직임   원래 영상의 참여자 얼굴을 driving 으로, 가상 인물 사진을 source 로 (상대 움직임: 고개·표정을 첫 장면 대비 차이로 옮김)
  되붙이기 LivePortrait 의 얼굴 마스크 틀로 가장자리를 부드럽게
  재식별 검사  OpenCV YuNet + SFace(Apache 2.0): 바꾼 얼굴이 원래 사람과 같은 사람 기준(0.363)에 가까우면(REID_MAX 이상) 그 장면은 얼굴을 가린다
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
REID_MAX = 0.30            # SFace 같은 사람 기준 0.363 에 여유
CROP = 512
EYE_L, EYE_R, CHIN, MOUTH = (33, 133), (362, 263), 152, (13, 14)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
             150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


@dataclass
class SwapReport:
    frames: int = 0
    swapped: int = 0
    no_face: int = 0
    reid_blocked: int = 0
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


class LivePortraitSwap:
    def __init__(self):
        sys.path.insert(0, str(LP_ROOT))
        import torch
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.camera import get_rotation_matrix

        self.torch, self.rot = torch, get_rotation_matrix
        self.w = LivePortraitWrapper(InferenceConfig())
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
        self.d0 = None
        return crop

    def drive(self, drive_crop_bgr) -> np.ndarray:
        """원래 얼굴 crop → 가상 인물이 같은 움직임을 한 512 얼굴 (BGR)."""
        t = self.torch
        x = self.w.prepare_source(cv2.cvtColor(drive_crop_bgr, cv2.COLOR_BGR2RGB))
        d = self.w.get_kp_info(x)
        R_d = self.rot(d["pitch"], d["yaw"], d["roll"])
        if self.d0 is None:
            self.d0 = (d, R_d)
        d0, R_d0 = self.d0
        s = self.s_info
        R_new = (R_d @ R_d0.permute(0, 2, 1)) @ self.s_R
        exp_new = s["exp"] + (d["exp"] - d0["exp"])
        scale_new = s["scale"] * (d["scale"] / d0["scale"])
        t_new = s["t"] + (d["t"] - d0["t"])
        t_new[..., 2] = 0
        kp_c = s["kp"].view(1, -1, 3)
        x_d = scale_new[..., None] * (kp_c @ R_new + exp_new.view(1, -1, 3)) + t_new[:, None, :]
        x_d = self.w.stitching(self.s_kp, x_d)
        with t.no_grad():
            out = self.w.warp_decode(self.s_feat, self.s_kp, x_d)["out"]
        img = (out.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()[0] * 255).astype(np.uint8)
        return cv2.cvtColor(cv2.resize(img, (CROP, CROP)), cv2.COLOR_RGB2BGR)

    def paste(self, frame, face512, M, gen_lmk: Optional[np.ndarray]) -> np.ndarray:
        """만든 얼굴의 얼굴 윤곽(이마는 조금 위로) 안만, 가장자리를 흐리게, 색을 장면에 맞춰 붙인다."""
        h, w = frame.shape[:2]
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
        if sel.sum() > 100:                                  # 색·밝기를 원래 장면 얼굴 자리에 맞춘다 (LAB 평균)
            a = cv2.cvtColor(face512, cv2.COLOR_BGR2LAB).astype(np.float32)
            b = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
            for ch in range(3):
                a[..., ch] += (b[..., ch][sel].mean() - a[..., ch][sel].mean()) * (0.6 if ch == 0 else 1.0)
            face512 = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        warped = cv2.warpAffine(face512, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mw = cv2.warpAffine(mk, M, (w, h), flags=cv2.INTER_LINEAR)[..., None]
        return (warped * mw + frame * (1 - mw)).astype(np.uint8)


def swap_video(original, base, out_path, donor_img, participant_boxes: Optional[dict] = None, reid_every: int = 3,
               log=print) -> SwapReport:
    """original: 원래 영상(얼굴·움직임 읽기용), base: 가린 영상(같은 장면 수) 위에 바꾼 얼굴을 붙인다."""
    import time

    t0 = time.time()
    rep = SwapReport(donor=Path(donor_img).name)
    fl, lp, reid = FaceLandmarks(), LivePortraitSwap(), Reid()
    donor = cv2.imread(str(donor_img))
    dl = fl(donor)
    if not dl:
        raise RuntimeError("가상 인물 사진에서 얼굴을 못 찾음")
    lp.source(donor, dl[0])
    co, cb = cv2.VideoCapture(str(original)), cv2.VideoCapture(str(base))
    fps = cb.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cb.get(3)), int(cb.get(4))
    out_path = Path(out_path)
    raw = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    sims, last_ok, i = [], True, 0
    while True:
        ok1, fo = co.read()
        ok2, fb = cb.read()
        if not (ok1 and ok2):
            break
        if fo.shape[:2] != (h, w):
            fo = cv2.resize(fo, (w, h))
        rep.frames += 1
        faces = fl(fo)
        pb = (participant_boxes or {}).get(i)
        if pb is not None and faces:
            faces = [f for f in faces if pb[0] <= f[:, 0].mean() <= pb[2] and pb[1] <= f[:, 1].mean() <= pb[3]] or faces
        if not faces:
            rep.no_face += 1
            vw.write(fb)
            i += 1
            continue
        lmk = max(faces, key=lambda f: np.ptp(f[:, 0]) * np.ptp(f[:, 1]))
        M = crop_matrix(lmk)
        gen = lp.drive(warp_in(fo, M))
        gl = fl(gen)
        res = lp.paste(fb, gen, M, gl[0] if gl else None)
        if i % reid_every == 0:                       # 재식별 검사: 바꾼 얼굴 ↔ 원래 얼굴
            c = lmk.mean(0)
            e0, e1 = reid.embed(fo, c), reid.embed(res, c)
            if e0 is not None and e1 is not None:
                sv = reid.sim(e0, e1)
                sims.append(sv)
                last_ok = sv < REID_MAX
        if not last_ok:                               # 알아봐지면 얼굴 둘레를 회색으로
            x0, y0 = lmk.min(0)
            x1, y1 = lmk.max(0)
            pad = 0.25 * max(x1 - x0, y1 - y0)
            cv2.rectangle(res, (int(x0 - pad), int(y0 - pad)), (int(x1 + pad), int(y1 + pad)), (128, 128, 128), -1)
            rep.reid_blocked += 1
        else:
            rep.swapped += 1
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
    rep.seconds = time.time() - t0
    return rep


def pick_donor(original_video, donors_dir=None) -> Optional[Path]:
    """참여자 얼굴과 가장 덜 닮은 가상 인물 (SFace 유사도 최소). 참여자 얼굴은 영상 앞 몇 장면에서."""
    donors = sorted(Path(donors_dir or Path(__file__).parents[1] / "donors").glob("*.png"))
    if not donors:
        return None
    reid = Reid()
    cap = cv2.VideoCapture(str(original_video))
    n = int(cap.get(7))
    embs = []
    for i in np.linspace(0, max(0, n - 1), 8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        e = reid.embed(f) if ok else None
        if e is not None:
            embs.append(e)
    if not embs:
        return donors[0]
    def worst(d):
        e = reid.embed(cv2.imread(str(d)))
        return 1.0 if e is None else max(reid.sim(e, x) for x in embs)
    return min(donors, key=worst)
