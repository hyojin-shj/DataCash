"""참여자 얼굴 → 실존하지 않는 가상 인물 (jobs.py 2-1 단계). 어디서 돌릴지는 DATACASH_FACESWAP:

  1 · local  이 컴퓨터에서 (맥 GPU, 기본)
  baseten    Baseten GPU 에서 (클라우드 서버에는 GPU 가 없다) — 영상은 S3 임시 주소로 주고받는다 (요청 본문에 넣기엔 크다)
  blur       얼굴만 흐리게 (GPU 없이 몇 초) — baseten 이 실패할 때도 이것으로
  0          끔

돌려주는 값: jobs.py report 에 더할 것 {"face_swap": {...}, "donor_trial": [...]} 또는 {"face_blur": {...}}.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .. import config


def mode() -> str:
    m = os.environ.get("DATACASH_FACESWAP", "1")
    return "local" if m == "1" else m


def run_local(video, masked, out_path, gender=None, age_band=None, mine=None, family=None,
              participant_boxes=None, consented_boxes=None, log=print) -> dict:
    from .faceswap import pick_donor, rank_donors, swap_video

    out: dict = {}
    donors = rank_donors(video, gender=gender, age_band=age_band)
    if not donors:
        raise RuntimeError("고를 가상 인물이 없음")
    first = Path(mine) if mine else None
    if first is None:                      # 본인이 고르지 않았으면: 같은 성별·나이대에서 덜 닮은 순서대로 시험해 표정을 줄일 일이 적은 사람
        first, tried = pick_donor(video, donors, log=log)
        out["donor_trial"] = tried
        donors = [first] + [d for d in donors if d != first]
    sr = swap_video(video, masked, out_path, first, participant_boxes=participant_boxes, log=log,
                    consented_boxes=consented_boxes, extra_donors=donors[1:],
                    family_donors={k: v for k, v in (family or {}).items() if v})
    out["face_swap"] = sr.to_dict()
    return out


def blur(video, masked, out_path, log=print) -> dict:
    """원래 영상에서 찾은 얼굴 자리를 가린 영상 위에서 흐리게 (GPU 없는 서버용). 못 찾은 장면은 앞 장면 자리를 몇 장면 더 흐린다."""
    from ..media import run

    t0 = time.time()
    det = cv2.FaceDetectorYN.create(str(config.MODELS / "yunet.onnx"), "", (320, 320), 0.5)
    co, cb = cv2.VideoCapture(str(video)), cv2.VideoCapture(str(masked))
    fps = cb.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cb.get(3)), int(cb.get(4))
    out_path = Path(out_path)
    raw = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frames = blurred = hold = 0
    last: list = []
    while True:
        ok_o, fo = co.read()
        ok_b, fb = cb.read()
        if not ok_b:
            break
        boxes = []
        if ok_o:
            if fo.shape[:2] != fb.shape[:2]:
                fo = cv2.resize(fo, (w, h))
            det.setInputSize((w, h))
            _, faces = det.detect(fo)
            boxes = [f[:4] for f in faces] if faces is not None else []
        if boxes:
            last, hold = boxes, 6
        elif hold > 0:                       # 얼굴을 잠깐 놓친 장면도 흐린 채로 (0.2초)
            boxes, hold = last, hold - 1
        for x, y, bw, bh in boxes:
            m = 0.35                            # 머리카락·턱까지
            x0, y0 = max(0, int(x - bw * m)), max(0, int(y - bh * m))
            x1, y1 = min(w, int(x + bw * (1 + m))), min(h, int(y + bh * (1 + m)))
            if x1 - x0 > 4 and y1 - y0 > 4:
                k = max(31, (max(x1 - x0, y1 - y0) // 3) | 1)
                fb[y0:y1, x0:x1] = cv2.GaussianBlur(fb[y0:y1, x0:x1], (k, k), 0)
        blurred += bool(boxes)
        frames += 1
        vw.write(fb)
    co.release(), cb.release(), vw.release()
    run(["-i", raw, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path])
    raw.unlink()
    log(f"얼굴 흐리게: {blurred}/{frames}장면")
    return {"face_blur": {"frames": frames, "blurred": blurred, "seconds": round(time.time() - t0, 1)}}


def _jsonable(boxes) -> Optional[dict]:
    if not boxes:
        return None
    return json.loads(json.dumps({str(k): v for k, v in boxes.items()}, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


def run_baseten(video, masked, out_path, gender=None, age_band=None, mine=None, family=None,
                participant_boxes=None, consented_boxes=None, log=print) -> dict:
    """Baseten 의 datacash-faceswap 모델(deploy/baseten_faceswap)에 맡긴다. 영상은 S3 에 올려 1시간짜리 주소로 주고, 결과도 그 주소로 받는다."""
    import boto3
    import httpx

    url = os.environ["DATACASH_BASETEN_URL"]           # https://model-xxxx.api.baseten.co/environments/production/predict
    key = os.environ.get("BASETEN_API_KEY") or os.environ["baseten_deploy"]   # Personal 키 (해커톤 키 baseten_api 는 Spark Open 만)
    bucket = os.environ["DC_S3_BUCKET"]
    s3 = boto3.client("s3")
    pre = f"faceswap/{uuid.uuid4().hex}"
    s3.upload_file(str(video), bucket, f"{pre}/video.mp4")
    s3.upload_file(str(masked), bucket, f"{pre}/masked.mp4")
    get = lambda k: s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": f"{pre}/{k}"}, ExpiresIn=3600)
    put = s3.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": f"{pre}/out.mp4", "ContentType": "video/mp4"}, ExpiresIn=3600)
    body = {"video_url": get("video.mp4"), "masked_url": get("masked.mp4"), "out_put_url": put,
            "gender": gender, "age_band": age_band, "mine": Path(mine).stem if mine else None,
            "family": {k: Path(v).stem for k, v in (family or {}).items() if v},
            "participant_boxes": _jsonable(participant_boxes), "consented_boxes": _jsonable(consented_boxes)}
    try:
        t0 = time.time()
        log("얼굴 치환: Baseten GPU 에 보냄")
        r = httpx.post(url, json=body, headers={"Authorization": f"Api-Key {key}"}, timeout=1800)   # 처음 깨어날 때 몇 분 더
        r.raise_for_status()
        res = r.json()
        if res.get("error"):
            raise RuntimeError(res["error"])
        s3.download_file(bucket, f"{pre}/out.mp4", str(out_path))
        log(f"얼굴 치환: Baseten {time.time() - t0:.0f}초")
        return {k: v for k, v in res.items() if k in ("face_swap", "donor_trial")}
    finally:
        for k in ("video.mp4", "masked.mp4", "out.mp4"):   # 원본이 남지 않게
            try:
                s3.delete_object(Bucket=bucket, Key=f"{pre}/{k}")
            except Exception:
                pass


def run(video, masked, out_path, log=print, **kw) -> dict:
    m = mode()
    if m == "blur":
        return blur(video, masked, out_path, log=log)
    if m == "baseten":
        try:
            return run_baseten(video, masked, out_path, log=log, **kw)
        except Exception as e:                  # GPU 가 안 되면 흐리게라도 — 녹화가 멈추지 않게
            log(f"Baseten 얼굴 치환 실패 → 흐리게: {type(e).__name__}: {str(e)[:200]}")
            out = blur(video, masked, out_path, log=log)
            out["face_blur"]["fallback_from"] = f"baseten: {type(e).__name__}"
            return out
    return run_local(video, masked, out_path, log=log, **kw)
