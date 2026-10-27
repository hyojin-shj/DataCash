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
REID_MAX = 0.363           # SFace 의 '같은 사람' 기준 — 이보다 덜 닮으면 다른 사람 (닮은 가상 인물이 자연스럽다는 의견으로 0.30 → 기준 그대로)
EXP_RAMP = 0.1             # 표정을 줄인 뒤 장면마다 이만큼씩 되돌린다 (30fps 에서 약 0.3초에 원래대로)
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
    small_face_covered: int = 0     # 그래도 478점을 못 찾은 작은 얼굴 → 모자이크로 가림 (원래 얼굴이 남지 않게)
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
        self.d0 = None
        return crop

    def state(self) -> tuple:
        """지금 가상 인물·움직임 기준 (여러 사람을 바꿀 때 사람마다 따로 둔다)."""
        return self.s_info, self.s_feat, self.s_kp, self.s_R, self.d0

    def use(self, st: tuple) -> None:
        self.s_info, self.s_feat, self.s_kp, self.s_R, self.d0 = st

    def drive(self, drive_crop_bgr, exp_scale: float = 1.0) -> np.ndarray:
        """원래 얼굴 crop → 가상 인물이 같은 움직임을 한 512 얼굴 (BGR).
        exp_scale: 표정을 얼마나 옮길지 (1 = 그대로, 0 = 고개 움직임만). 표정이 원래 사람을 닮게 만드는 장면에서 줄인다."""
        t = self.torch
        x = self.w.prepare_source(cv2.cvtColor(drive_crop_bgr, cv2.COLOR_BGR2RGB))
        d = self.w.get_kp_info(x)
        R_d = self.rot(d["pitch"], d["yaw"], d["roll"])
        if self.d0 is None:
            self.d0 = (d, R_d)
        d0, R_d0 = self.d0
        s = self.s_info
        R_new = (R_d @ R_d0.permute(0, 2, 1)) @ self.s_R
        exp_new = s["exp"] + exp_scale * (d["exp"] - d0["exp"])
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


def pixelate(img, x0: int, y0: int, x1: int, y1: int, cells: int = 7) -> None:
    """상자 안을 가로 cells 칸 정도의 굵은 모자이크로 (얼굴 크기와 상관없이 알아볼 수 없게)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    cw = max(1, cells)
    ch = max(1, round(cells * (y1 - y0) / (x1 - x0)))
    small = cv2.resize(img[y0:y1, x0:x1], (cw, ch), interpolation=cv2.INTER_AREA)
    img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


def cover_small_faces(base, original, reid: "Reid", participant_box=None) -> int:
    """치환할 수 없는 작은 얼굴을 YuNet 으로 찾아 모자이크로 덮는다 (참여자 상자가 있으면 그 안의 얼굴만). 덮은 개수."""
    h, w = original.shape[:2]
    reid.det.setInputSize((w, h))
    _, faces = reid.det.detect(original)
    n = 0
    for f in faces if faces is not None else []:
        x, y, fw, fh = (int(v) for v in f[:4])
        cx, cy = x + fw / 2, y + fh / 2
        if participant_box is not None and not (participant_box[0] <= cx <= participant_box[2] and participant_box[1] <= cy <= participant_box[3]):
            continue
        pad = int(0.35 * max(fw, fh))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + fw + pad), min(h, y + fh + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        pixelate(base, x0, y0, x1, y1)
        n += 1
    return n


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
    co, cb = cv2.VideoCapture(str(original)), cv2.VideoCapture(str(base))
    fps = cb.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cb.get(3)), int(cb.get(4))
    out_path = Path(out_path)
    raw = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    sims, scales, i = [], {}, 0
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
            small = False
            if inside:
                kk = max(inside, key=lambda k: np.ptp(faces[k][:, 0]) * np.ptp(faces[k][:, 1]))
                used.add(kk)
                lmk = faces[kk]
            else:
                found = find_small_faces(fo, fl, reid, box)
                if not found:
                    continue
                lmk, small = found[0], True
            any_face = True
            M = crop_matrix(lmk)
            crop = warp_in(fo, M)
            c = lmk.mean(0)
            e0 = reid.embed(fo, c)
            # 재식별 검사를 장면마다: 바꾼 얼굴이 원래 사람과 닮으면 모자이크(영상이 끊긴다) 대신 표정을 줄여 다시 만든다
            start = min(1.0, scales.get(key, 1.0) + EXP_RAMP)       # 줄였던 표정은 조금씩 되돌린다 (깜빡이지 않게)
            tries = []                                     # (닮음, 결과, 표정 비율)
            for es in dict.fromkeys((start, start * 0.5, 0.0)):
                lp.use(person_state(key, 0 if key == "participant" else 1 + family_order.index(key.split(":", 1)[1])))
                gen = lp.drive(crop, es)
                gl = fl(gen)
                out = lp.paste(res, gen, M, gl[0] if gl else None)
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
            res = out
        if not any_face:
            rep.no_face += 1
            covered = 0
            for _, box in targets:
                covered += cover_small_faces(res, fo, reid, box)
            rep.small_face_covered += covered > 0
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
