"""녹화 세션 하나 → 4감 LeRobot 데이터셋 + 품질 인증서.

  1. 4감 자동 동기화 (워치 시계 → 폰 시계)
  2. 촉각: 능동 진동 반응 → 사람별 보정(GripSense) → 쥐는 세기 하·중·상 + 0~100 / 영상 접촉 단계 / 접촉음 + 손목 충격
  3. 몸 감각: 심박·HRR·피부온도 변화 (생체 동의만)
  4. 손·몸 관절 (주어지지 않으면 MediaPipe 로 뽑는다)
  5. 장면 격자에 모아 내보내기 (등급별 채널) + 인증서

가리기(privacy)·단계 라벨(understand)은 앞 단계에서 만든 결과를 받는다 (masked_video, objects, steps).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .export import certificate
from .export.channels import (EpisodeChannels, NAMES, audio_rows, body_rows, event_rows, hand_rows, interp, make_grid,
                              object_rows, touch_rows)
from .export.lerobot import export
from .physio import physio_series
from .session import load_session
from .sync import align_session
from .touch.contact import contact_phases, phase_events
from .touch.sounds import audio_onsets, confirmed_contacts, wrist_impacts
from .touch.gripsense import GRADES, LEVELS, GripModel, labels_for, train, window_features
from .touch.vibration import pulse_responses

FINGERTIPS = [4, 8, 12, 16, 20]


def _fingertips(labels) -> List[Optional[np.ndarray]]:
    out = []
    for f in labels or []:
        pts = [np.asarray(h)[FINGERTIPS, :2] for h in f.hands if h is not None]
        out.append(np.concatenate(pts) if pts else None)
    return out


def grip_segments_to_watch(meta: dict, m) -> list:
    """보정 구간 (폰 시계로 적은 '가볍게·평소처럼·꽉·손 놓기') → 워치 시계. 시계를 못 맞췄으면 빈 목록."""
    cal = meta.get("calibration") or {}
    if cal.get("method") != "gripsense" or not cal.get("segments") or m is None:
        return []
    inv = lambda p_ns: (m.w_ref + (np.asarray(p_ns, np.float64) - m.b) / m.a).astype(np.int64)  # noqa: E731
    out = []
    for g in cal["segments"]:
        a, b = inv(np.array([g["start_ns"], g["end_ns"]])) if cal.get("clock", "phone") == "phone" else (g["start_ns"], g["end_ns"])
        out.append({"start_ns": int(a), "end_ns": int(b), "level": int(g["level"])})
    return out


def face_check_end_s(meta: dict, first_frame_ns: int, video_s: Optional[float] = None) -> float:
    """녹화 1단계(얼굴 확인)가 끝난 시각(첫 장면 기준 초, 0.3초 여유). 없으면 0.
    얼굴 확인이 끝나지 않은 채 멈췄으면(통과 못 함 · 건너뛰기도 안 누름 · 남은 동작이 3초 미만) 1단계 동안 동작을 한 것으로 보고
    처음부터(0). video_s(영상 길이)를 줄 때만 본다."""
    fc = meta.get("face_check") or {}
    end = fc.get("end_ns")
    t = max(0.0, (int(end) - first_frame_ns) / 1e9 + 0.3) if end else 0.0
    if video_s is not None and not fc.get("passed") and not fc.get("no_person") and video_s - t < 3.0:
        return 0.0
    return t


def process(session_dir, out_dir, level: str = "bio", fps: float = 30.0, masked_video=None, labels=None,
            objects: Optional[Sequence[dict]] = None, hand_points=None, steps: Optional[list] = None,
            verified_steps: Optional[list] = None, episodes_s: Optional[list] = None, privacy: Optional[dict] = None,
            extract: bool = True, mistakes: Optional[list] = None, scene: Optional[dict] = None,
            grip_calibration: Optional[dict] = None) -> dict:
    s = load_session(session_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(masked_video) if masked_video else s.video_path
    cap = cv2.VideoCapture(str(video))
    size = (int(cap.get(3)), int(cap.get(4)))
    cap.release()

    # 1. 동기화
    responses = pulse_responses(s.watch_imu.t_ns, s.watch_imu.values, s.watch_vib.t_ns) if s.watch_vib else []
    al = align_session(s, responses)
    m = al.watch_to_phone
    wrist_ok = m is not None and al.ok.get("wrist", False)

    # 2. 촉각 — GripSense 방식: 창(0.5초, 0.25초 간격)마다 자이로·가속도 특징 → 사람마다 학습한 결정 트리로 쥐는 세기 단계
    win = window_features(s.watch_imu.t_ns, s.watch_imu.values, s.watch_vib.t_ns) if s.watch_vib is not None and s.watch_imu is not None else None
    model = None
    segs = grip_segments_to_watch(s.meta, m)
    if win is not None and segs:              # 이 녹화 안에 보정 구간이 있으면 (보정 세션·합성) 바로 학습
        model = train(win, labels_for(win, segs))
    if model is None and grip_calibration and grip_calibration.get("method") == "gripsense":
        model = GripModel.from_dict(grip_calibration)        # 이 사람·이 워치로 미리 해 둔 보정
    if labels is None and extract and video is not None:
        from .labels import extract_labels

        labels = extract_labels(s.video_path or video)
    pts = hand_points if hand_points is not None else _fingertips(labels)
    cframes = contact_phases(pts, objects, fps) if objects is not None and pts else None
    grip, vib = [], []
    if wrist_ok and win is not None and len(win.t_ns):
        tp = m(win.t_ns)
        ok = win.x[:, 8] > 0.5
        if model is not None:
            held = None
            n = min(len(cframes), len(objects)) if cframes and objects is not None else 0
            if n and s.frame_ns is not None and abs(len(s.frame_ns) - n) <= max(3, n // 100):   # 영상: 창 시각에 손이 물건에 닿았나
                fi = np.clip(np.searchsorted(s.frame_ns, tp), 0, n - 1)                 # (영상 장면 수와 시각 기록이 1~2장 다를 수 있다)
                held = np.array([(1 if cframes[i].contact >= 1.0 else 0) if objects[i] else -1 for i in fi])   # 물체가 안 보인 장면은 모름
            lv, score = model.predict(win.x, win.t_ns, held)
            grip = [(int(t), int(l), float(sc)) for t, l, sc, o in zip(tp, lv, score, ok) if o]
        vib = [(int(t), *[float(v) for v in x[:6]]) for t, x, o in zip(tp, win.x, ok) if o]      # 보내는 특징 6개
    onsets = audio_onsets(s.audio.x, s.audio.rate, s.audio.start_ns) if s.audio is not None else []
    impacts = []
    if wrist_ok:
        wi = wrist_impacts(s.watch_imu.t_ns, s.watch_imu.values, s.watch_vib.t_ns if s.watch_vib else None)
        impacts = [int(v) for v in m(np.array([e["t_ns"] for e in wi]))] if wi else []
    confirmed = confirmed_contacts(onsets, impacts)

    # 3. 몸 감각
    cons = (s.meta.get("participant") or {}).get("consent") or {}
    phys = None
    if s.has_physio and cons.get("bio") and m is not None and al.ok.get("physio", False):
        hr_ns = m(s.watch_hr.t_ns) if s.watch_hr else None
        sk_ns = m(s.watch_skin.t_ns) if s.watch_skin else None
        phys = physio_series(hr_ns, s.watch_hr.values[:, 0] if s.watch_hr else None, sk_ns,
                             s.watch_skin.values[:, 0] if s.watch_skin else None,
                             (s.meta.get("participant") or {}).get("age_band"))

    # 단계 라벨 검증 (Muse Spark 라벨을 촉각·소리·손목 움직임으로)
    frame_ns = s.frame_ns
    if steps:
        from .understand.verify import check_mistakes, verify_steps

        g_all, src_all = make_grid(frame_ns, int(frame_ns[0]), int(frame_ns[-1]), fps)
        t_all = (g_all - frame_ns[0]) / 1e9
        motion = None
        if wrist_ok:
            acc, _ = interp(g_all, m(s.watch_imu.t_ns), s.watch_imu.values[:, :3])
            motion = np.abs(np.linalg.norm(acc, axis=1) - np.median(np.linalg.norm(acc, axis=1)))
        T_all, _ = touch_rows(cframes, src_all, g_all, grip, vib)
        ph = phase_events(cframes, (frame_ns - frame_ns[0]) / 1e9) if cframes else []
        _, hp = hand_rows(labels, src_all, size) if labels is not None else (None, None)
        steps = verify_steps(steps, t_all, motion=motion, contact=T_all[:, 0] if cframes else None,
                             hands=hp.max(1) if hp is not None else None,
                             grip=T_all[:, 1] if grip else None,
                             sound_t_s=[(e["t_ns"] - frame_ns[0]) / 1e9 for e in onsets],
                             contact_change_t_s=[e["t_s"] for e in ph])
        verified_steps = [x["verified"] is True for x in steps]
        if mistakes:
            mistakes = check_mistakes(mistakes, [(t - frame_ns[0]) / 1e9 for t in (s.meta.get("self_marks_ns") or [])])

    # 4~5. 에피소드마다 격자 + 채널
    spans = episodes_s or [(0.0, (frame_ns[-1] - frame_ns[0]) / 1e9)]
    cut = face_check_end_s(s.meta, int(frame_ns[0]), (frame_ns[-1] - frame_ns[0]) / 1e9)   # 1단계 얼굴 확인 구간은 판매 데이터에서 뺀다
    if cut > 0:
        spans = [(max(a, cut), b) for a, b in spans if b - max(a, cut) >= 1.0]
    eps, vids = [], []
    marks = s.meta.get("self_marks_ns") or []
    for a_s, b_s in spans:
        grid, src = make_grid(frame_ns, int(frame_ns[0] + a_s * 1e9), int(frame_ns[0] + b_s * 1e9), fps)
        ep = EpisodeChannels(task=(s.meta.get("mission") or {}).get("title", "task"), grid_ns=grid, src_frame=src)
        d = ep.data
        d["observation.hands"], d["observation.hands_present"] = hand_rows(labels, src, size)
        d["observation.body"], d["observation.body_world"] = body_rows(labels, src, size)
        if s.phone_imu is not None:
            d["observation.imu.phone"], _ = interp(grid, s.phone_imu.t_ns, s.phone_imu.values)
        else:
            d["observation.imu.phone"] = np.zeros((len(grid), 6), np.float32)
        if wrist_ok:
            d["observation.imu.wrist"], _ = interp(grid, m(s.watch_imu.t_ns), s.watch_imu.values)
        d["observation.audio"] = audio_rows(s.audio, grid, onsets, confirmed, fps)
        d["observation.touch"], tv = touch_rows(cframes, src, grid, grip, vib)
        if wrist_ok:
            d["observation.touch.vibration"] = tv
        if phys is not None:
            P = np.zeros((len(grid), 6), np.float32)
            if phys["hr"] is not None:
                P[:, 0:1], ok = interp(grid, phys["hr_ns"], phys["hr"][:, None], max_gap_s=2.0)
                P[:, 1] = ok
                if phys["hrr"] is not None:
                    P[:, 2:3], ok = interp(grid, phys["hr_ns"], phys["hrr"][:, None], max_gap_s=2.0)
                    P[:, 3] = ok
            if phys["skin_delta"] is not None:
                P[:, 4:5], ok = interp(grid, phys["skin_ns"], phys["skin_delta"][:, None], max_gap_s=5.0)
                P[:, 5] = ok
            d["observation.physio"] = P
        d["observation.objects"] = object_rows(objects, src, size)
        ep_steps = [x for x in (steps or []) if x["end_s"] > a_s and x["start_s"] < b_s]
        d["annotation.events"], d["annotation.step"], d["annotation.primitive"] = event_rows(grid, fps, marks, confirmed, ep_steps, verified_steps)
        err = al.error_ms
        d["sync.error_ms"] = np.tile([err.get("video", 0), err.get("audio", 0), err.get("touch", err.get("wrist", 0)),
                                      err.get("physio", 0)], (len(grid), 1)).astype(np.float32)
        t0 = (grid[0] - frame_ns[0]) / 1e9
        ep.info = {
            "steps": ep_steps or None,
            "scene": scene,
            "mistakes": [x for x in (mistakes or []) if a_s <= x["t_s"] <= b_s] or None,
            "outcome": {**(s.meta.get("outcome") or {}),
                        "self_marks_s": [round(float((t - grid[0]) / 1e9), 2) for t in marks if grid[0] <= t <= grid[-1]]},
            "sync": al.to_dict(),
            "touch": {"grip_model": model.to_dict() if model else None, "grip_levels": GRADES, "grip_score": "쥔 동안 0~100 (하 0~33 · 중 34~66 · 상 67~100)",
                      "contact_events": phase_events(cframes, (frame_ns - frame_ns[0]) / 1e9 - t0) if cframes else [],
                      "confirmed_sounds": [round((e["t_ns"] - grid[0]) / 1e9, 3) for e in confirmed if grid[0] <= e["t_ns"] <= grid[-1]]},
            "human_state": {"physio": phys["summary"], "self_report": s.meta.get("self_report")} if phys else None,
            "privacy": privacy,
            "source": {"session_id": s.meta.get("session_id"), "start_s": a_s, "end_s": b_s},
        }
        eps.append(ep)
        vids.append(video)
    ds = export(out / "lerobot", eps, vids, fps, level=level)
    info = json.loads((ds / "meta" / "info.json").read_text())
    cert = {
        "session_id": s.meta.get("session_id", Path(session_dir).name), "level": level,
        "mission": s.meta.get("mission"), "devices": s.meta.get("devices"),
        "consent": {**cons, "age_band": (s.meta.get("participant") or {}).get("age_band")},
        "dataset": {"episodes": len(eps), "frames": info["total_frames"], "fps": fps,
                    "features": sorted(k for k in info["features"] if k not in ("timestamp", "frame_index", "episode_index", "index", "task_index"))},
        "sync": al.to_dict(),
        "touch": {"pulses": len(responses), "pulses_valid": sum(r.valid for r in responses),
                  "grip_model": {"method": "gripsense", "accuracy_cv": round(model.accuracy_cv, 3), "accuracy_held_cv": model.to_dict()["accuracy_held_cv"], "n_windows": model.n} if model else None,
                  "contact_events": eps[0].info["touch"]["contact_events"] if eps else [],
                  "confirmed_sounds": eps[0].info["touch"]["confirmed_sounds"] if eps else []},
        "human_state": eps[0].info["human_state"] if eps else None,
        "steps": steps, "mistakes": mistakes, "scene": scene,
        "outcome": eps[0].info["outcome"] if eps else None, "privacy": privacy,
    }
    certificate.write(cert, out)
    return cert
