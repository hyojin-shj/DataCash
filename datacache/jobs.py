"""업로드된 녹화 하나를 끝까지: SAM 3.1 가리기 → 관절 → Muse Spark 이해 → 4감 내보내기·검증 → 보상.

API 를 쓸 수 없으면(한도·오류) 그 단계만 건너뛰고 나머지는 계속한다 — 보고서에 무엇을 건너뛰었는지 남긴다.
가린 영상이 없으면(가리기 실패) 판매용 데이터셋을 만들지 않는다 (원본을 내보내는 길은 없다). 합성 세션만 예외.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import pricing
from .meta.client import BudgetExceeded
from .pipeline import FINGERTIPS, process


def _mission(meta: dict, missions_path: Optional[Path] = None) -> dict:
    p = missions_path or Path(__file__).with_name("missions.json")
    ms = {m["id"]: m for m in json.loads(p.read_text())["missions"]}
    mid = (meta.get("mission") or {}).get("id")
    return ms.get(mid, {"id": mid or "unknown", "title": (meta.get("mission") or {}).get("title", "task"),
                        "title_en": "household task", "objects": []})


def run_session(session_dir, out_dir, use_api: bool = True, level: Optional[str] = None, log=print) -> dict:
    sd, out = Path(session_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((sd / "meta.json").read_text())
    mission = _mission(meta)
    report = {"session_id": meta.get("session_id", sd.name), "mission": mission["id"], "skipped": []}
    video = sd / "video.mp4"
    synthetic = bool(meta.get("synthetic"))

    # 1. 관절 (원본에서 — 가리기 전에 뽑아야 손·몸이 온전하다)
    labels = None
    if not synthetic:
        from .labels import extract_labels

        labels = extract_labels(video)
        log(f"[{sd.name}] 관절 {len(labels)}장면")

    # 2. SAM 3.1 가리기 + 조작 물체 영역
    masked, objects, privacy = None, None, None
    if use_api and not synthetic:
        try:
            from .privacy import mask_video

            pts = [None if all(h is None for h in f.hands) else
                   np.concatenate([h[FINGERTIPS, :2] for h in f.hands if h is not None]) for f in labels]
            r = mask_video(video, out / "masked.mp4", manipulated=mission.get("objects", [])[:3], hand_pts=pts)
            masked, objects, privacy = r.masked_video, r.objects, r.report
            log(f"[{sd.name}] 가리기: {json.dumps(r.report, ensure_ascii=False)[:300]}")
        except BudgetExceeded as e:
            report["skipped"].append(f"privacy: {e}")
        except Exception as e:
            report["skipped"].append(f"privacy: {type(e).__name__}: {str(e)[:200]}")
    # 2-1. 참여자 얼굴 → 실존하지 않는 가상 인물 (LivePortrait + MediaPipe 자르기, 재식별 검사)
    if masked is not None and use_api is not None and not synthetic and os.environ.get("DATACACHE_FACESWAP", "1") == "1":
        try:
            from .privacy.faceswap import pick_donor, swap_video

            donor = pick_donor(video)
            if donor is not None:
                sr = swap_video(video, masked, out / "masked_swap.mp4", donor,
                                participant_boxes=getattr(r, "participant_boxes", None), log=log)
                masked = out / "masked_swap.mp4"
                privacy = {**(privacy or {}), "face_swap": sr.to_dict()}
        except Exception as e:
            report["skipped"].append(f"face_swap: {type(e).__name__}: {str(e)[:200]}")
    if masked is None and not synthetic:
        report["error"] = "가린 영상이 없어 판매용 데이터셋을 만들지 않음"
        (out / "job.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    # 3. Muse Spark 이해 (가린 영상만)
    steps, mistakes, scene = None, None, None
    if use_api:
        try:
            from .media import clip_mp4
            from .understand.spark import understand_clip

            src = masked or video
            cap = cv2.VideoCapture(str(src))
            dur = cap.get(7) / (cap.get(5) or 30)
            cap.release()
            clip = clip_mp4(src, out / "understand_clip.mp4", 0, min(dur, 90), max_side=640, audio=False)
            u = understand_clip(clip, f"{mission.get('title_en')} ({mission.get('title')})", min(dur, 90))
            steps, mistakes = u["steps"], u["mistakes"]
            scene = {"task": u["task"], "completed": u["completed"], "environment": u["environment"], "model": u["model"]}
            clip.unlink()
        except BudgetExceeded as e:
            report["skipped"].append(f"understand: {e}")
        except Exception as e:
            report["skipped"].append(f"understand: {type(e).__name__}: {str(e)[:200]}")

    # 4. 4감 내보내기
    has_watch = (sd / "watch_imu.csv").exists()
    bio = bool(((meta.get("participant") or {}).get("consent") or {}).get("bio")) and (sd / "watch_hr.csv").exists()
    lvl = level or ("bio" if bio else "premium" if has_watch else "basic")
    kw = {}
    if synthetic and (sd / "tracks.json").exists():
        tr = json.loads((sd / "tracks.json").read_text())
        kw["hand_points"] = [np.array([[h[0] + 16 * np.cos(a), h[1] + 16 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)])
                             for h in [x["hand"] for x in tr]]
        objects = [{0: tuple(x["cup"])} for x in tr]
    cert = process(sd, out, level=lvl, masked_video=masked, labels=labels, objects=objects, steps=steps,
                   mistakes=mistakes, scene=scene, privacy=privacy, extract=False, **kw)
    minutes = cert["dataset"]["frames"] / cert["dataset"]["fps"] / 60
    report.update(level=lvl, minutes=round(minutes, 2), reward=pricing.reward(minutes, lvl),
                  sync=cert["sync"]["chosen"], certificate=str(out / "certificate.md"))
    (out / "job.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
