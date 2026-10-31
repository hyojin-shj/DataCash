"""참여자 얼굴을 실존하지 않는 가상 인물(Muse Image 로 만든 사진)로 바꾼다 — LivePortrait(MIT) + MediaPipe 얼굴 자르기.

LivePortrait 기본 얼굴 찾기·자르기는 InsightFace(비상업 라이선스)를 쓰므로 부르지 않는다. 대신
  자르기   MediaPipe FaceLandmarker 478점 → 눈 선으로 기울기를 맞춘 정사각형 (LivePortrait 의 자르기 비율과 비슷하게)
  움직임   원래 영상의 참여자 얼굴을 driving 으로, 가상 인물 사진을 source 로 (상대 움직임: 고개·표정을 첫 장면 대비 차이로 옮김)
  되붙이기 LivePortrait 의 얼굴 마스크 틀로 가장자리를 부드럽게
  재식별 검사  OpenCV YuNet + SFace(Apache 2.0): 바꾼 얼굴이 원래 사람과 닮으면(REID_MAX 이상) 표정을 줄여 다시 만든다
  옆얼굴·먼 얼굴  MediaPipe 가 점을 못 찾아도 YuNet 의 5점(눈·코·입꼬리)으로 자르기를 만들어 치환한다 — 모자이크로 끊기지 않게.
              그래도 못 바꾸면 얼굴 모양대로 부드럽게 흐린다 (굵은 모자이크는 쓰지 않는다)
  자연스럽게  붙일 얼굴의 밝기·대비·색을 장면 얼굴 자리에 맞추고, 자르기 위치를 앞 장면과 섞어 떨림을 줄인다
  이어 붙이기  얼굴을 잠깐(BRIDGE 장면) 아예 못 찾으면 앞 장면의 바꾼 얼굴을 얼굴 움직임(optical flow)을 따라 옮겨 붙이며 서서히 빼고,
              흐림도 끝난 뒤 BLUR_HOLD 장면 동안 같은 자리를 서서히 덜 흐린다.
              해 보고 뺀 것: 원래 얼굴 → 바꾼 얼굴 서서히 섞기(섞는 동안 원래 정면 얼굴이 비쳤다), 옆모습이 돼도 몇 장면 억지로 치환하기
              (깨진 회색 얼굴이 나오고, 손을 얼굴로 잘못 찾은 자리에도 얼굴이 붙었다), 옆모습에 앞 얼굴 따라 붙이기·한두 장면 메우기
              (정면 얼굴이 떠 보이거나 어둡게 깨졌다) — 옆모습은 옆모습을 만들 수 있는 치환이 있어야 풀린다.
              끊김(flips): 참여자 얼굴이 한 장면 사이에 반 넘게 바뀐 횟수 (바꾼 얼굴 ↔ 원래 얼굴 · 흐림 시작·끝)
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
BRIDGE = 4                 # 얼굴을 아예 못 찾은 장면에 앞 장면의 바꾼 얼굴을 따라 붙이는 장면 수 (30fps 약 0.13초) — 그동안 서서히 뺀다
                           # 옆모습으로 알아보고 원래대로 둔 장면에는 붙이지 않는다: 정면 얼굴이 옆얼굴·머리카락 위에 떠 보였다
VIEW_MARGIN = 8.0          # 각도 사진: 지금 쓰는 원본보다 이만큼(도) 더 가까울 때만 바꾼다 (원본이 장면마다 바뀌어 깜빡이지 않게)
OPEN_PROFILES = os.environ.get("DATACASH_OPEN_PROFILES") == "1"   # 알아보지 못하는 옆모습도 각도 사진으로 바꿀지 — 기본은 끔:
                           #   녹화 4편에서 치환 장면은 조금 늘지만(컵 녹화 +19장면) 끊김 117 → 122, 새로 바꾼 옆모습의 약 40% 가
                           #   머리카락 위에 겹치거나 반쯤 어둡게 떠 원래 옆모습보다 어색했다 (docs/privacy_eval.md)
OPEN_PROFILE_EYES = 0.22   # 알아보지 못하는 옆모습은 얼굴이 이만큼(눈 사이/얼굴 폭) 보일 때만 바꾼다 — 더 돌면 만든 얼굴이 옆얼굴에 붙거나
                           #   머리카락 위에 회색 얼굴이 떴다 (230651 상자 녹화: 눈 비율 0.08~0.18 이 대부분 → 원래대로)
SKIN_DARK = 0.55           # 색 맞추기: 얼굴 자리 밝기(70번째 백분위)의 이 비율보다 어두운 곳(머리카락·그늘)은 빼고 잰다
NEAR_FRAMES = 45           # 알아보지 못하는 옆모습은 이 장면 안에 확인한 참여자 얼굴 곁(얼굴 폭의 NEAR_DIST 배 안)일 때만 바꾼다 —
NEAR_DIST = 0.6            #   컵을 쥔 주먹을 YuNet 이 얼굴로 찾아(점수 0.65~0.76) 바닥에 가상 얼굴이 붙었다 (225537)
OPEN_MIN_RUN = 5           # 알아보지 못하는 옆모습 치환이 이보다 짧게 끝나면 되돌린다 (OpenRunFilter)
PROFILE_EXP = 0.3          # 옆모습은 표정을 이만큼만 옮긴다 (옆에서 읽은 표정은 틀리기 쉬워 일그러진 얼굴이 됐다)
BLUR_HOLD = 10             # 흐린 얼굴은 바로 끄지 않고 이만큼 같은 자리를 서서히 덜 흐린다 (흐림이 켜졌다 꺼졌다 하지 않게)
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
    kept_why: dict = field(default_factory=dict)   # 원래대로 둔 까닭: side(거의 옆·알아보지 못함) · far(참여자 얼굴 곁이 아님 — 주먹 등) · failed(만들지 못함) · short(짧게 되다 말아 되돌림) · small(점 없는 작은·번진 얼굴) · other
    view_fallback: int = 0          # 각도 사진으로 만든 옆모습이 참여자를 닮아 정면 사진으로 다시 만든 장면
    views: int = 0                  # 가상 인물 각도 사진 수 (좌우 뒤집기 포함) — 0 이면 정면 한 장으로만
    profile_kept: int = 0           # 옆모습인데 재식별 모델이 참여자로 알아보지 못함(닮음 < REID_MAX) → 원래대로 둠 (흐리지 않는다)
    profile_swapped: int = 0        # 참여자로 알아보는 옆모습 → 5점(눈·코·입꼬리)에 맞춰 치환, 치환 뒤 재식별 검사 통과
    bridged: int = 0                # 얼굴을 놓친 장면에 앞 장면의 바꾼 얼굴을 따라 붙인 장면 (BRIDGE)
    flips: int = 0                  # 끊김: 참여자 얼굴이 한 장면 사이에 반 넘게 바뀐 횟수 (바꾼 얼굴 ↔ 원래 얼굴 · 흐림 시작·끝)
    flips_per_min: float = 0.0
    family: int = 0                 # 동의한 가족 수 (사람마다 다른 가상 인물)
    reid_max: float = 0.0
    reid_mean: float = 0.0
    seconds: float = 0.0
    donor: str = ""
    notes: List[str] = field(default_factory=list)

    def keep(self, why: str) -> None:
        self.profile_kept += 1
        self.kept_why[why] = self.kept_why.get(why, 0) + 1

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
        self.views, self.cur = [], None
        return crop

    def add_views(self, imgs) -> int:
        """같은 가상 인물의 여러 각도 사진(정면 무표정이 아닌 고개 돌림·숙임). 장면 각도에 가장 가까운 사진을 원본으로 쓴다 —
        정면 한 장으로 옆모습을 만들면 LivePortrait 가 크게 돌려야 해 눈이 뭉개지고 얼굴이 일그러졌다.
        회전 기준(s_R)은 정면 사진 것을 그대로 쓴다 (돌린 만큼은 원래 사람 고개에서 온다). 좌우를 뒤집은 사진도 넣는다."""
        fl = FaceLandmarks(max_faces=1)
        for img in imgs:
            for im in (img, np.ascontiguousarray(img[:, ::-1])):
                l = fl(im)
                if not l:
                    continue
                x = self.w.prepare_source(cv2.cvtColor(warp_in(im, crop_matrix(l[0])), cv2.COLOR_BGR2RGB))
                info = self.w.get_kp_info(x)
                self.views.append((float(info["yaw"]), float(info["pitch"]), info, self.w.extract_feature_3d(x), self.w.transform_keypoint(info)))
        return len(self.views)

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
        return self.s_info, self.s_feat, self.s_kp, self.s_R, self.d0, list(self.d0_buf), getattr(self, "views", []), getattr(self, "cur", None)

    def use(self, st: tuple) -> None:
        self.s_info, self.s_feat, self.s_kp, self.s_R, self.d0, buf, self.views, self.cur = st
        self.d0_buf = list(buf)

    def _pick(self, d, d0):
        """장면 고개 각도에 맞는 원본 (정면 사진 또는 각도 사진). 지금 것보다 VIEW_MARGIN 넘게 가까울 때만 바꾼다."""
        s = self.s_info
        ty = float(d["yaw"]) - float(d0["yaw"]) + float(s["yaw"])
        tp = float(d["pitch"]) - float(d0["pitch"]) + float(s["pitch"])
        cands = [(float(s["yaw"]), float(s["pitch"]), s, self.s_feat, self.s_kp)] + list(self.views or [])
        dist = lambda v: abs(v[0] - ty) + 0.5 * abs(v[1] - tp)
        best = min(range(len(cands)), key=lambda k: dist(cands[k]))
        cur = self.cur if self.cur is not None and self.cur < len(cands) else 0
        if dist(cands[best]) + VIEW_MARGIN < dist(cands[cur]):
            cur = best
        self.cur = cur
        return cands[cur][2:]

    def drive(self, drive_crop_bgr, exp_scale: float = 1.0, anchor: bool = True, views: bool = False) -> np.ndarray:
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
        # 각도 사진은 옆모습에만: 정면에 가까운 장면까지 쓰면 바꾼 얼굴이 참여자를 조금 더 닮았다 (상자 녹화 3편 알아봄 +0.4~2.2%p)
        s, s_feat, s_kp = self._pick(d, d0) if views and self.views else (self.s_info, self.s_feat, self.s_kp)
        R_new = (R_d @ R_d0.permute(0, 2, 1)) @ self.s_R
        delta = (d["exp"] - d0["exp"]).clamp(-EXP_CLIP, EXP_CLIP)
        exp_new = s["exp"] + exp_scale * delta
        scale_new = s["scale"] * (d["scale"] / d0["scale"])
        t_new = s["t"] + (d["t"] - d0["t"])
        t_new[..., 2] = 0
        kp_c = s["kp"].view(1, -1, 3)
        x_d = scale_new[..., None] * (kp_c @ R_new + exp_new.view(1, -1, 3)) + t_new[:, None, :]
        x_d = self.w.stitching(s_kp, x_d)
        with t.no_grad():
            if self.half:
                warped = self.w.warping_module(s_feat.half(), kp_source=s_kp.half(), kp_driving=x_d.half())["out"]
                out = self.w.spade_generator(feature=warped).float()
            else:
                out = self.w.warp_decode(s_feat, s_kp, x_d)["out"]
        img = (out.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()[0] * 255).astype(np.uint8)
        return cv2.cvtColor(cv2.resize(img, (CROP, CROP)), cv2.COLOR_RGB2BGR)

    def paste(self, frame, face512, M, gen_lmk: Optional[np.ndarray], clip=None, target: Optional[np.ndarray] = None,
              target5: Optional[np.ndarray] = None) -> np.ndarray:
        """만든 얼굴의 얼굴 윤곽(이마는 조금 위로) 안만, 가장자리를 흐리게, 색을 장면에 맞춰 붙인다.
        clip: YuNet 얼굴(점 없이 자른 경우) — 붙이는 자리를 원래 얼굴 타원 안으로 묶는다 (얼굴 밖으로 튀어나오지 않게).
        target: 원래 얼굴 점(장면 좌표) — 만든 얼굴의 눈·코·입·턱을 이 자리에 맞춰 옮긴다 (가상 인물의 얼굴 비율이 달라
          눈코입이 얼굴 폭의 10%쯤, 많게는 20% 어긋나 이마에 눈이 붙은 것처럼 보였다)."""
        h, w = frame.shape[:2]
        if gen_lmk is not None and target is None and target5 is not None:   # 옆모습: YuNet 5점(눈 둘·코·입꼬리 둘)에만 맞춘다
            t5 = cv2.transform(np.asarray(target5, np.float32).reshape(1, 5, 2), cv2.invertAffineTransform(M))[0]
            g5 = np.array([gen_lmk[[33, 133]].mean(0)[:2], gen_lmk[[362, 263]].mean(0)[:2], gen_lmk[1, :2],
                           gen_lmk[61, :2], gen_lmk[291, :2]], np.float32)
            A, _ = cv2.estimateAffinePartial2D(g5, t5, method=cv2.LMEDS)
            if A is not None and 0.6 < np.sqrt(abs(np.linalg.det(A[:, :2]))) < 1.6:
                face512 = cv2.warpAffine(face512, A, (CROP, CROP), borderMode=cv2.BORDER_REPLICATE)
                gen_lmk = gen_lmk.copy()
                gen_lmk[:, :2] = cv2.transform(gen_lmk[None, :, :2].astype(np.float32), A)[0]
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
        a = cv2.cvtColor(face512, cv2.COLOR_BGR2LAB).astype(np.float32)
        b = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
        if sel.sum() > 100:                                  # 피부만으로 색을 잰다: 얼굴 자리에 들어온 머리카락·그늘이 섞이면 얼굴이 회색으로 떴다
            # (머리카락을 바꾼 얼굴 위로 남기는 것도 해 봤지만, 머리카락에 이어진 원래 눈·눈썹까지 남아 알아보는 장면이 3.8% → 17.4% — 뺐다)
            skin = sel & (b[..., 0] > SKIN_DARK * np.percentile(b[..., 0][sel], 70)) & (a[..., 0] > SKIN_DARK * np.percentile(a[..., 0][sel], 70))
            if skin.sum() > 0.3 * sel.sum():
                sel = skin
        if sel.sum() > 100:                                  # 밝기·대비·색을 원래 장면 얼굴 자리에 맞춘다 (LAB 평균·퍼짐)
            for ch in range(3):                              # 밝기는 퍼짐도 조금만(그늘진 장면에서 얼룩지지 않게), 색은 평균만
                am, asd = a[..., ch][sel].mean(), a[..., ch][sel].std() + 1e-3
                bm, bsd = b[..., ch][sel].mean(), b[..., ch][sel].std() + 1e-3
                k = float(np.clip(bsd / asd, 0.85, 1.15)) if ch == 0 else float(np.clip(bsd / asd, 0.8, 1.25))
                a[..., ch] = (a[..., ch] - am) * k + bm      # 밝기·색 평균을 장면 얼굴에 그대로 (0.85 만 맞추면 어두운·초록빛 장면에서 얼굴만 희게 떴다)
            face512 = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        warped = cv2.warpAffine(face512, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mw = cv2.warpAffine(mk, M, (w, h), flags=cv2.INTER_LINEAR)
        if clip is not None:
            x, y, fw, fh = (float(v) for v in clip[:4])
            el = np.zeros((h, w), np.float32)
            cv2.ellipse(el, (int(x + fw / 2), int(y + fh / 2)), (int(fw * 0.55), int(fh * 0.62)), 0, 0, 360, 1.0, -1)
            kk = max(3, int(fw * 0.25)) | 1
            mw = mw * cv2.GaussianBlur(el, (kk, kk), 0)
        warped = match_texture(warped, frame, mw)
        mw = mw[..., None]
        return (warped * mw + frame * (1 - mw)).astype(np.uint8)


BLUR_STEPS = (0.0, 0.6, 1.0, 1.5, 2.2, 3.0)   # 만든 얼굴을 장면만큼 흐리게 할 때 시험하는 가우스 폭 (px)


def _lap_var(gray, sel) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F)[sel].var())


def _noise(gray, sel) -> float:
    return float((gray - cv2.medianBlur(gray, 3).astype(np.float32))[sel].std())


def match_texture(face, frame, mw, rng=None) -> np.ndarray:
    """만든 얼굴(장면 좌표로 옮긴 것)의 선명도·입자를 원래 장면 얼굴 자리에 맞춘다.
    만든 얼굴은 512px 로 깨끗한데 폰 영상은 흔들림·번짐·노이즈가 있어, 얼굴만 매끈하게 떠 보였다.
    선명도: 라플라시안 분산이 장면과 가장 가까워지는 만큼 흐린다. 입자: 장면이 더 거칠면 그 차이만큼 회색 노이즈를 더한다 (장면마다 새로)."""
    ys, xs = np.nonzero(mw > 0.5)
    if len(ys) < 200:
        return face
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    f = face[y0:y1, x0:x1]
    g_scene = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    sel = mw[y0:y1, x0:x1] > 0.5
    target = _lap_var(g_scene, sel)
    best, out = None, f
    for s in BLUR_STEPS:
        c = cv2.GaussianBlur(f, (0, 0), s) if s > 0 else f
        d = abs(np.log((_lap_var(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), sel) + 1) / (target + 1)))
        if best is None or d < best:
            best, out = d, c
    n_scene = _noise(g_scene.astype(np.uint8), sel)
    n_face = _noise(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), sel)
    add = float(np.sqrt(max(0.0, n_scene ** 2 - n_face ** 2)))
    if add > 0.3:
        rng = rng or np.random.default_rng()
        out = np.clip(out.astype(np.float32) + rng.normal(0, add, out.shape[:2])[..., None], 0, 255).astype(np.uint8)
    face = face.copy()
    face[y0:y1, x0:x1] = out
    return face


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


def donor_views(donor_png) -> List[np.ndarray]:
    """가상 인물의 여러 각도 사진 (datacash/donors/views/<이름>/*.jpg 512px — scripts/make_donor_views.py 로 Muse Image 편집)."""
    d = Path(donor_png).parent / "views" / Path(donor_png).stem
    return [im for im in (cv2.imread(str(p)) for p in sorted(d.glob("*.[jp][pn]g"))) if im is not None]


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


def soften(img, f, strength: float = 1.0) -> None:
    """치환할 수 없는 얼굴: 얼굴 모양(타원)대로, 가장자리를 흐려 부드럽게 (굵은 모자이크처럼 튀지 않게). strength: 흐림 세기 0~1."""
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
    m = cv2.GaussianBlur(m, (k, k), 0)[..., None] * float(strength)
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


def track_crop(g0, g1, pts, M):
    """앞 장면(g0)의 얼굴 점들(pts)을 이번 장면(g1)으로 따라가(Lucas-Kanade) 자르기 M 을 그만큼 옮긴다. 못 따라가면 None."""
    if pts is None or len(pts) < 8:
        return None
    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(21, 21), maxLevel=3)
    ok = ok.ravel().astype(bool)
    if ok.sum() < 8:
        return None
    T, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if T is None or inl.sum() < 8 or not 0.8 < np.sqrt(abs(np.linalg.det(T[:, :2]))) < 1.25:
        return None
    M1 = np.hstack([T[:, :2] @ M[:, :2], (T[:, :2] @ M[:, 2] + T[:, 2])[:, None]]).astype(np.float32)
    return M1, nxt[ok][inl.ravel().astype(bool)].reshape(-1, 1, 2)


def face_points(gray, M):
    """자르기 M 의 가운데(얼굴 안쪽)에서 따라갈 점들."""
    h, w = gray.shape
    q = cv2.transform(np.array([[[CROP * a, CROP * b] for a, b in ((.3, .3), (.7, .3), (.7, .75), (.3, .75))]], np.float32), M)[0]
    mk = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mk, q.astype(np.int32), 255)
    return cv2.goodFeaturesToTrack(gray, 40, 0.01, 4, mask=mk)


class OpenRunFilter:
    """장면을 OPEN_MIN_RUN 장 늦게 내보내며, 알아보지 못하는 옆모습을 바꾼 장면이 '원래대로 둠' 사이에 OPEN_MIN_RUN 장보다 짧게 끼면
    바꾸기 전 장면으로 되돌린다 — 옆모습 치환이 한두 장면만 되다 말아 얼굴이 깜빡였다 (225537: 끊김 86 → 96).
    되돌려도 개인정보는 그대로다 (그 장면들은 원래도 참여자로 알아보지 못하는 옆모습)."""

    def __init__(self, emit, rep):
        from collections import deque

        self.buf, self.emit, self.rep, self.before, self.last = deque(), emit, rep, None, None

    def push(self, frame, alt, kind: str, tv) -> None:
        if kind == "open":
            if not self.buf or self.buf[-1][2] != "open":
                self.before = self.buf[-1][2] if self.buf else self.last
        else:
            run = 0
            while run < len(self.buf) and self.buf[-1 - run][2] == "open":
                run += 1
            if 0 < run < OPEN_MIN_RUN and kind == "orig" and self.before == "orig":
                for e in list(self.buf)[-run:]:
                    e[0], e[2], e[3] = e[1], "orig", (0.0, 0.0)
                    self.rep.profile_swapped -= 1
                    self.rep.swapped -= 1
                    self.rep.keep("short")
        self.buf.append([frame, alt, kind, tv])
        while len(self.buf) > OPEN_MIN_RUN:
            self._out()

    def _out(self):
        e = self.buf.popleft()
        self.last = e[2]
        self.emit(e[0], e[3])

    def flush(self) -> None:
        while self.buf:
            self._out()


def count_flips(trace: list) -> int:
    """trace: 장면마다 참여자 얼굴 (바꾼 얼굴 비율 0~1, 흐림 세기 0~1) 또는 None(얼굴 없음). 얼굴 없는 장면은 건너뛰고 잇는다."""
    n, prev = 0, None
    for v in trace:
        if v is None:
            continue
        if prev is not None and (abs(v[0] - prev[0]) > 0.5 or abs(float(v[1]) - float(prev[1])) > 0.5):
            n += 1
        prev = v
    return n


def swap_video(original, base, out_path, donor_img, participant_boxes: Optional[dict] = None,
               log=print, consented_boxes: Optional[dict] = None, extra_donors: Optional[list] = None,
               family_donors: Optional[dict] = None, bridge: int = BRIDGE) -> SwapReport:
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
            path = Path(chosen or donors[min(k, len(donors) - 1)])
            img = cv2.imread(str(path))
            dl = fl(img)
            if not dl:
                raise RuntimeError("가상 인물 사진에서 얼굴을 못 찾음")
            lp.source(img, dl[0])
            rep.views = max(rep.views, lp.add_views(donor_views(path)))
            states[key] = lp.state()
        return states[key]

    person_state("participant", 0)
    ref, ref_emb, cap = [], [], cv2.VideoCapture(str(original))   # ref_emb: 참여자 정면 얼굴 (옆모습을 알아보는지 잴 때)          # 참여자 움직임 기준: 영상 곳곳 정면에 가까운 얼굴 장면들로 미리
    nf = int(cap.get(7))
    for j in np.linspace(0, max(0, nf - 1), 3 * D0_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
        ok, f = cap.read()
        found = fl(f) if ok else []
        if found:
            lmk = max(found, key=lambda a: np.ptp(a[:, 0]) * np.ptp(a[:, 1]))
            if np.ptp(lmk[:, 0]) > 80:                     # 너무 작은 얼굴은 빼고
                ref.append(warp_in(f, crop_matrix(lmk)))
                e = reid.embed(f, lmk[:, :2].mean(0))
                if e is not None:
                    ref_emb.append(e)
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
    seen: dict = {}            # 사람마다 마지막으로 확인한 얼굴 (MediaPipe 점, 가운데·폭·장면) — 알아보지 못하는 옆모습은 이 곁일 때만 바꾼다
    family_order: list = []
    last: dict = {}            # 사람마다 마지막으로 바꾼 얼굴 {gen, gl, M, pts, age} — 놓친 장면에 따라 붙인다
    alpha: dict = {}           # 사람마다 이번 장면에 붙인 바꾼 얼굴 비율 (끊김 세기용)
    trace: list = []           # 참여자 얼굴 장면마다 (바꾼 얼굴 비율, 흐림 세기) — 끊김 세기
    blurred: dict = {}         # 사람마다 마지막으로 흐린 얼굴 {f, age} — 흐림을 서서히 끈다
    g_prev = None
    delay = OpenRunFilter(lambda f, tv: (vw.write(f), trace.append(tv)), rep)
    while True:
        ok1, fo = co.read()
        ok2, fb = cb.read()
        if not (ok1 and ok2):
            break
        if fo.shape[:2] != (h, w):
            fo = cv2.resize(fo, (w, h))
        rep.frames += 1
        gray = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY)
        faces = fl(fo)
        now: dict = {}         # 이번 장면 사람마다 swap · orig(원래대로 둠) · blur · bridge(따라 붙임) · fading(흐림을 서서히 끔)

        def put(key, out, gen, gl, M):
            """바꾼 얼굴을 붙였다 — 놓친 장면에 이어 붙일 수 있게 남긴다."""
            alpha[key], now[key] = 1.0, "swap"
            last[key] = {"gen": gen, "gl": gl, "M": M, "pts": face_points(gray, M) if bridge > 0 else None, "age": 0}
            return out

        targets = [("participant", (participant_boxes or {}).get(i))]
        for oid, box in (consented_boxes or {}).get(i, []):
            if oid not in family_order:
                family_order.append(oid)
            targets.append((f"family:{oid}", box))
        res, any_face, used = fb, False, set()
        alt = None             # 알아보지 못하는 참여자 옆모습을 바꿨을 때, 바꾸기 전 장면 (짧게 끝나면 이것으로 되돌린다)
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
                    if is_profile(yf):
                        # 옆모습: ① 재식별 모델이 참여자로 알아보지 못하면 원래대로 (흐리면 모자이크처럼 튀어 어색하다)
                        #         ② 알아보면 5점에 맞춰 치환하고, 치환 뒤에도 알아보면 그때만 흐린다
                        any_face = True
                        crops.pop(key, None)
                        yc = np.array([yf[0] + yf[2] / 2, yf[1] + yf[3] / 2], np.float32)
                        ep = reid.embed(fo, yc)
                        sim_p = max(reid.sim(ep, r) for r in ref_emb) if ep is not None and ref_emb else 1.0
                        eyes = float(np.hypot(yf[6] - yf[4], yf[7] - yf[5])) / max(1.0, float(yf[2]))
                        # 거의 완전한 옆모습(눈 사이가 얼굴 폭의 22% 미만): 만들면 깨지고, 재식별 모델은 머리카락·윤곽만 보고 높게 잰다
                        # (물병 상자 녹화 580·600장면: 눈 비율 0.05~0.15) → 원래대로 둔다. 귀·머리카락·턱선만 보인다.
                        known = eyes >= 0.22 and sim_p >= REID_MAX
                        # 각도 사진이 있으면 알아보지 못하는 옆모습도 바꾼다 (원래 얼굴로 두면 치환이 끊겨 보인다) — 거의 완전한 옆모습만 원래대로
                        near = seen.get(key)
                        near = near is not None and i - near[2] <= NEAR_FRAMES and \
                            float(np.hypot(*(yc - near[0]))) < NEAR_DIST * max(near[1], float(yf[2]))
                        if not known and not (OPEN_PROFILES and lp.views and eyes >= OPEN_PROFILE_EYES and near):
                            if near or eyes < OPEN_PROFILE_EYES:
                                rep.keep("side")
                                now[key] = "orig"
                            else:                          # 참여자 얼굴 곁이 아니다 (주먹·물건을 얼굴로 찾음) — 참여자 얼굴이 없는 장면으로 센다
                                rep.keep("far")
                            continue
                        Mp = crop_from_yunet(yf)
                        lp.use(person_state(key, 0 if key == "participant" else 1 + family_order.index(key.split(":", 1)[1])))
                        gp = lp.drive(warp_in(fo, Mp), PROFILE_EXP if lp.views else 0.5, anchor=False, views=True)
                        states[key] = lp.state()
                        gpl = fl(gp)
                        outp = lp.paste(res, gp, Mp, gpl[0] if gpl else None, clip=yf, target5=np.asarray(yf[4:14], np.float32).reshape(5, 2)) \
                            if gpl else None
                        e1 = reid.embed(outp, yc) if outp is not None else None
                        if lp.views and e1 is not None and ref_emb and max(reid.sim(e1, r) for r in ref_emb) >= REID_MAX:
                            # 각도 사진으로 만든 옆모습이 참여자 정면 사진을 닮으면 예전처럼 정면 사진으로 다시 만든다
                            # (각도 사진만 쓰면 컵 녹화에서 알아보는 장면 4.0 → 6.5%)
                            gp = lp.drive(warp_in(fo, Mp), 0.5, anchor=False)
                            gpl = fl(gp)
                            outp = lp.paste(res, gp, Mp, gpl[0] if gpl else None, clip=yf, target5=np.asarray(yf[4:14], np.float32).reshape(5, 2)) \
                                if gpl else None
                            e1 = reid.embed(outp, yc) if outp is not None else None
                            rep.view_fallback += 1
                        if outp is not None and (e1 is None or ep is None or reid.sim(ep, e1) < REID_MAX):
                            if not known and key == "participant" and len(targets) == 1:
                                alt = res
                            res = put(key, outp, gp, gpl[0], Mp)
                            if near or known:              # 참여자로 알아본 옆모습이나 확인한 얼굴에서 이어진 옆모습만 다음 장면의 기준으로
                                # (컵을 쥔 주먹은 참여자 닮음 0.0~0.23 이라 시작점이 되지 못한다)
                                seen[key] = (yc, float(yf[2]), i)
                            rep.profile_swapped += 1
                            rep.swapped += 1
                        elif not known:                    # 원래도 알아보지 못하는 옆모습: 못 만들었으면 흐리지 않고 원래대로
                            rep.keep("failed")
                            now[key] = "orig"
                        else:
                            soften(res, yf)
                            rep.small_face_covered += 1
                            now[key], alpha[key] = "blur", 0.0
                            last.pop(key, None)
                            blurred[key] = {"f": yf, "age": 0}
                        continue
                    # 점을 못 찾은 작은·번진 얼굴: 참여자로 알아보지 못하면 원래대로 (번진 얼굴에 또렷한 얼굴을 붙이면 더 어색하다)
                    ey = reid.embed(fo, np.array([yf[0] + yf[2] / 2, yf[1] + yf[3] / 2], np.float32))
                    if ey is not None and ref_emb and max(reid.sim(ey, r) for r in ref_emb) < REID_MAX:
                        rep.keep("small")
                        any_face = True
                        crops.pop(key, None)
                        now[key] = "orig"
                        continue
            any_face = True
            pidx = 0 if key == "participant" else 1 + family_order.index(key.split(":", 1)[1])
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
                lp.use(person_state(key, pidx))
                gen = lp.drive(crop, es, anchor=lmk is not None and not small)
                gl = fl(gen)
                out = lp.paste(res, gen, M, gl[0] if gl else None, clip=yf, target=lmk,
                               target5=np.asarray(yf[4:14], np.float32).reshape(5, 2) if yf is not None else None)
                e1 = reid.embed(out, c) if e0 is not None else None
                sv = reid.sim(e0, e1) if e1 is not None else None
                tries.append((-1.0 if sv is None else sv, out, es, gen, gl[0] if gl else None))
                if sv is None or sv < REID_MAX:
                    break
            sv, out, used_scale, gen, gl = min(tries, key=lambda t: t[0])   # 다 넘으면 가장 덜 닮은 것 — 굳히거나 모자이크하지 않는다 (연속성)
            if sv >= 0:
                sims.append(sv)
            states[key] = lp.state()                       # 움직임 기준(d0)을 사람마다 이어 간다
            scales[key] = used_scale
            rep.swapped += 1
            rep.damped += used_scale < 1.0
            rep.over_reid += sv >= REID_MAX
            rep.small_face_swapped += small
            rep.yunet_swapped += yf is not None
            res = put(key, out, gen, gl, M)
            if lmk is not None and not small:              # 얼굴 점 478개를 찾은 얼굴에서 시작한다 (아무 옆모습 치환으로나 이으면 주먹으로 옮겨 갔다)
                seen[key] = (c, float(np.ptp(lmk[:, 0])), i)
        if not any_face:
            rep.no_face += 1
            covered = 0
            for key, box in targets:                       # 남은 얼굴(치환도 못 한 것): 참여자로 알아보는 얼굴만 부드럽게 흐린다
                for f in yunet_faces(fo, reid, box):
                    ef = reid.embed(fo, np.array([f[0] + f[2] / 2, f[1] + f[3] / 2], np.float32))
                    if ef is not None and ref_emb and max(reid.sim(ef, r) for r in ref_emb) < REID_MAX:
                        rep.keep("other")
                        continue
                    soften(res, f)
                    covered += 1
                    now[key], alpha[key] = "blur", 0.0
                    last.pop(key, None)
                    blurred[key] = {"f": f, "age": 0}
            rep.small_face_covered += covered > 0
            crops.clear()                                  # 얼굴이 끊겼다 다시 나오면 앞 위치와 섞지 않는다
        for key, _ in targets:                             # 얼굴을 아예 못 찾은 사람: 앞 장면의 바꾼 얼굴을 따라 붙이며 서서히 뺀다
            if now.get(key) is not None:                   # 바꿈 · 흐림 · 옆모습이라 원래대로 둠 → 붙이지 않는다
                if now[key] == "orig":
                    last.pop(key, None)
                    alpha[key] = 0.0
                continue
            lb = last.get(key)
            tr = track_crop(g_prev, gray, lb["pts"], lb["M"]) if lb and g_prev is not None and lb["age"] < bridge else None
            if tr is None:
                last.pop(key, None)
                alpha.pop(key, None)
                continue
            lb["M"], lb["pts"] = tr
            lb["age"] += 1
            a = 1.0 - lb["age"] / (bridge + 1)
            out = lp.paste(res, lb["gen"], lb["M"], lb["gl"])
            res = cv2.addWeighted(out, a, res, 1 - a, 0)
            alpha[key], now[key] = a, "bridge"
            rep.bridged += 1
        blur_s = {}
        for key, _ in targets:                             # 흐림이 끝난 사람: 같은 자리를 BLUR_HOLD 장면 동안 서서히 덜 흐린다
            bl = blurred.get(key)
            if now.get(key) == "blur":
                blur_s[key] = 1.0
            elif bl and now.get(key) != "swap" and bl["age"] < BLUR_HOLD:
                bl["age"] += 1
                blur_s[key] = 1.0 - bl["age"] / (BLUR_HOLD + 1)
                soften(res, bl["f"], blur_s[key])
                now.setdefault(key, "fading")
            else:
                blurred.pop(key, None)
        st = now.get("participant")
        tv = None if st is None else (alpha.get("participant", 0.0) if st in ("swap", "bridge") else 0.0, blur_s.get("participant", 0.0))
        delay.push(res, alt, "open" if alt is not None else ("orig" if st == "orig" else "other"), tv)
        g_prev = gray
        i += 1
        if i % 60 == 0:
            log(f"얼굴 치환 {i}장면")
    delay.flush()
    vw.release()
    from ..media import run

    run(["-i", raw, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path])
    raw.unlink()
    if sims:
        rep.reid_max, rep.reid_mean = float(np.max(sims)), float(np.mean(sims))
    rep.family = len(family_order)
    rep.flips = count_flips(trace)
    rep.flips_per_min = rep.flips / max(1e-6, rep.frames / fps / 60)
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
TRIAL_DONORS = 6         # 닮은 순서로 이만큼 시험해, 표정을 줄여야 하는 장면이 가장 적은 사람을 고른다


def trial_rates(original_video, donor: Path, frames: int = TRIAL_FRAMES, lp: Optional["LivePortraitSwap"] = None,
                fl: Optional["FaceLandmarks"] = None, reid: Optional["Reid"] = None) -> Optional[tuple]:
    """영상 곳곳 몇 장면을 이 가상 인물로 실제처럼(표정 그대로 → 반 → 고개만) 바꿔 본다 →
    (표정 그대로는 같은 사람 기준을 넘어 표정을 줄여야 하는 장면 비율, 셋 다 넘는 장면 비율)."""
    fl, reid, lp = fl or FaceLandmarks(), reid or Reid(), lp or LivePortraitSwap()
    img = cv2.imread(str(donor))
    dl = fl(img) if img is not None else []
    if not dl:
        return None
    cap = cv2.VideoCapture(str(original_video))
    n = int(cap.get(7))
    over = damp = seen = 0
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
        best, first = None, None
        for es in (1.0, 0.5, 0.0):
            g = lp.drive(crop, es)
            gl = fl(g)
            e1 = reid.embed(lp.paste(f, g, M, gl[0] if gl else None, target=lmk), c)   # 실제처럼 눈코입을 맞춰 붙인 얼굴로 (맞추면 더 닮는다)
            sv = reid.sim(e0, e1) if e1 is not None else 0.0
            best = sv if best is None else min(best, sv)
            first = sv if first is None else first
            if st is None:
                st = lp.state()
            if sv < REID_MAX:
                break
        seen += 1
        over += best >= REID_MAX
        damp += first >= REID_MAX
    cap.release()
    return (damp / seen, over / seen) if seen else None


def pick_donor(original_video, ranked: List[Path], log=print) -> tuple:
    """닮은 순서(rank_donors) 앞쪽 TRIAL_DONORS 명을 시험해, 표정을 줄여야 하는 장면이 가장 적은 사람 (같으면 더 닮은 사람).
    가장 닮은 사람을 바로 쓰면 바꾼 얼굴이 원래 사람과 닮아 표정을 줄인 장면이 32% 였다 (녹화 26편) — 표정이 굳어 부자연스럽다.
    표정을 다 빼도 기준을 넘는 장면이 OVER_LIMIT 넘는 사람은 뺀다 (가장 닮은 사람이 어떤 영상에서는 거의 본인처럼 나왔다 — 점심 녹화 티슈 ②, 87%).
    (고른 사람, {이름: 표정을 줄여야 하는 장면 비율})"""
    fl, reid, lp = FaceLandmarks(), Reid(), LivePortraitSwap()
    tried, over = {}, {}
    for d in ranked[:TRIAL_DONORS]:
        r = trial_rates(original_video, d, lp=lp, fl=fl, reid=reid)
        tried[d.stem] = None if r is None else round(r[0], 2)
        over[d.stem] = None if r is None else r[1]
        log(f"가상 인물 시험 {d.stem}: 표정 줄임 {tried[d.stem]} · 표정 빼도 넘음 {None if r is None else round(r[1], 2)}")
        if r is not None and r[0] == 0:                    # 표정을 줄일 장면이 없으면 더 볼 것 없다 (닮은 순서라 이 사람이 가장 자연스럽다)
            return d, tried
    ok = [d for d in ranked[:TRIAL_DONORS] if tried.get(d.stem) is not None]
    fine = [d for d in ok if over[d.stem] <= OVER_LIMIT] or ok
    return (min(fine, key=lambda d: tried[d.stem]) if fine else ranked[0]), tried


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
