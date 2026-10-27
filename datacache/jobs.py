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
                        "title_en": "manipulation task", "objects": []})


def face_check_embedding(sd: Path, meta: dict):
    """녹화 1단계(얼굴 확인) 구간의 장면들에서 가장 큰 얼굴 특징값 (가장 잘 나온 것). 없으면 None."""
    fc = meta.get("face_check") or {}
    if not fc.get("start_ns") or not fc.get("end_ns") or not (sd / "video_frames.csv").exists():
        return None
    from .people import embed_image

    t = np.loadtxt(sd / "video_frames.csv", delimiter=",", skiprows=1, ndmin=2)[:, 1]
    idx = np.where((t >= fc["start_ns"]) & (t <= fc["end_ns"]))[0]
    if not len(idx):
        return None
    cap = cv2.VideoCapture(str(sd / "video.mp4"))
    try:
        for i in idx[np.linspace(len(idx) - 1, 0, min(8, len(idx))).astype(int)]:   # 끝 쪽(통과 직전)부터
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            emb = embed_image(f)[0] if ok else None
            if emb is not None:
                return emb
        return None
    finally:
        cap.release()


def calibration_key(meta: dict) -> Optional[str]:
    """보정을 저장하는 열쇠: 폰 기기 id(사람마다 자기 폰) + 워치 기종."""
    dev = meta.get("devices") or {}
    pid = (dev.get("phone") or {}).get("id")
    return None if not pid else f"{pid}__{(dev.get('watch') or {}).get('model', 'watch')}".replace("/", "_").replace(" ", "_")


def run_calibration(sd: Path, out: Path, store: Path) -> dict:
    """쥐는 힘 보정 세션: 폰 화면 버튼으로 적은 저울 누르기(폰 시계) + 워치 진동 반응 → 이 사람·이 워치의 보정 저장."""
    from .session import load_session
    from .sync import align_session
    from .touch.vibration import calibrate, presses_to_watch, pulse_responses

    s = load_session(sd)
    if not s.has_watch or s.watch_vib is None:
        return {"error": "워치 기록이 없음 (워치 앱을 켜 두고 다시)"}
    rs = pulse_responses(s.watch_imu.t_ns, s.watch_imu.values, s.watch_vib.t_ns)
    al = align_session(s, rs)
    m = al.watch_to_phone
    if m is None:
        return {"error": "폰↔워치 시계를 맞추지 못함"}
    inv = lambda p_ns: (m.w_ref + (np.asarray(p_ns, np.float64) - m.b) / m.a).astype(np.int64)  # noqa: E731
    cal = calibrate(rs, presses_to_watch((s.meta.get("calibration") or {}).get("presses", []), inv))
    if cal is None:
        return {"error": "누르기 구간에 진동 반응이 부족함 (각 단계를 2초 이상 눌러 주세요)"}
    key = calibration_key(s.meta) or sd.name
    store.mkdir(parents=True, exist_ok=True)
    (store / f"{key}.json").write_text(json.dumps({**cal.to_dict(), "session": sd.name, "sync": al.to_dict()["chosen"]}, indent=2))
    rep = {"session_id": sd.name, "mission": "grip_calibration", "calibration": cal.to_dict(), "key": key, "reward": 0,
           "level": "calibration", "minutes": 0, "sync": al.to_dict()["chosen"]}
    (out / "job.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    return rep


def run_session(session_dir, out_dir, use_api: bool = True, level: Optional[str] = None, log=print,
                calibration_store: Optional[Path] = None, people=None) -> dict:
    """people: 사람 알아보기 (people.People). 이 폰에 등록된 본인 얼굴이 참여자를, 동의한 가족이 가리지 않을 사람을 정한다."""
    sd, out = Path(session_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((sd / "meta.json").read_text())
    store = Path(calibration_store or out.parent.parent / "grip_calibration")
    if (meta.get("mission") or {}).get("id") == "grip_calibration":
        return run_calibration(sd, out, store)
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

            pts = [[h[FINGERTIPS, :2] for h in f.hands if h is not None] for f in labels]   # 장면마다 손별 손끝
            dev = ((meta.get("devices") or {}).get("phone") or {}).get("id")
            fc_emb = face_check_embedding(sd, meta)
            from .people import matcher_from

            match = people.matcher(dev, fc_emb) if people is not None else matcher_from(None, fc_emb)
            r = mask_video(video, out / "masked.mp4", manipulated=mission.get("objects", [])[:3], hand_pts=pts, match=match)
            masked, objects, privacy = r.masked_video, r.objects, r.report
            log(f"[{sd.name}] 가리기: {json.dumps(r.report, ensure_ascii=False)[:300]}")
        except BudgetExceeded as e:
            report["skipped"].append(f"privacy: {e}")
        except Exception as e:
            report["skipped"].append(f"privacy: {type(e).__name__}: {str(e)[:200]}")
    # 2-1. 참여자 얼굴 → 실존하지 않는 가상 인물 (LivePortrait + MediaPipe 자르기, 재식별 검사)
    if masked is not None and not synthetic and os.environ.get("DATACACHE_FACESWAP", "1") == "1":
        try:
            from .privacy.faceswap import rank_donors, swap_video

            donors = rank_donors(video)
            dev = ((meta.get("devices") or {}).get("phone") or {}).get("id")
            mine = people.donor_for(dev, "owner") if people is not None else None      # 본인이 고른 가상 인물
            fam = {oid: people.donor_for(dev, pid) for oid, pid in getattr(r, "consented_who", {}).items()} if people is not None else {}
            if donors:
                sr = swap_video(video, masked, out / "masked_swap.mp4", mine or donors[0],
                                participant_boxes=getattr(r, "participant_boxes", None), log=log,
                                consented_boxes=getattr(r, "consented_boxes", None), extra_donors=donors[1:],
                                family_donors={k: v for k, v in fam.items() if v})
                masked = out / "masked_swap.mp4"
                privacy = {**(privacy or {}), "face_swap": sr.to_dict()}
        except Exception as e:
            report["skipped"].append(f"face_swap: {type(e).__name__}: {str(e)[:200]}")
    if masked is None and not synthetic:
        report["error"] = "가린 영상이 없어 판매용 데이터셋을 만들지 않음"
        (out / "job.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    # 2-2. 말소리 개인정보 끄기 (Voice Transcribe + Spark, 받아쓴 글자는 남기지 않는다)
    if use_api and (sd / "audio.wav").exists() and not synthetic:
        try:
            from .privacy.speech import clean_audio

            sp = clean_audio(sd / "audio.wav", sd / "audio_clean.wav")
            privacy = {**(privacy or {}), "speech": sp}
        except BudgetExceeded as e:
            report["skipped"].append(f"speech: {e}")
        except Exception as e:
            report["skipped"].append(f"speech: {type(e).__name__}: {str(e)[:200]}")

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
            from .pipeline import face_check_end_s

            t0 = 0.0
            if (sd / "video_frames.csv").exists():   # 1단계 얼굴 확인 구간은 빼고 보낸다
                ft = np.loadtxt(sd / "video_frames.csv", delimiter=",", skiprows=1, ndmin=2)[:, 1]
                t0 = min(face_check_end_s(meta, int(ft[0])), max(0.0, dur - 2)) if len(ft) else 0.0
            clip = clip_mp4(src, out / "understand_clip.mp4", t0, min(dur - t0, 90), max_side=640, audio=False)
            u = understand_clip(clip, f"{mission.get('title_en')} ({mission.get('title')})", min(dur - t0, 90),
                               primitives=mission.get("primitives"))
            for st_ in u["steps"]:            # 자른 구간만큼 녹화 시각으로 되돌린다
                st_["start_s"] += t0
                st_["end_s"] += t0
            for m_ in u["mistakes"]:
                m_["t_s"] += t0
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
    key = calibration_key(meta)
    stored = json.loads((store / f"{key}.json").read_text()) if key and (store / f"{key}.json").exists() else None
    cert = process(sd, out, level=lvl, masked_video=masked, labels=labels, objects=objects, steps=steps,
                   mistakes=mistakes, scene=scene, privacy=privacy, extract=False, grip_calibration=stored, **kw)
    minutes = cert["dataset"]["frames"] / cert["dataset"]["fps"] / 60
    today = False
    try:   # 녹화한 날(한국 날짜)의 오늘 미션이면 +50%
        import datetime

        ids = [m["id"] for m in json.loads(Path(__file__).with_name("missions.json").read_text())["missions"]]
        rec = datetime.datetime.fromisoformat(meta.get("recorded_at", "")).astimezone(datetime.timezone(datetime.timedelta(hours=9)))
        today = pricing.today_mission(ids, rec.date()) == mission.get("id")
    except (ValueError, KeyError):
        pass
    report.update(level=lvl, minutes=round(minutes, 2), reward=pricing.reward(minutes, lvl, today=today), today_bonus=today,
                  sync=cert["sync"]["chosen"], certificate=str(out / "certificate.md"))
    (out / "job.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
