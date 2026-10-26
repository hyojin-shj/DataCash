"""녹화 세션 하나 → 4감 LeRobot 데이터셋 + 품질 인증서.

  1. 4감 자동 동기화 (워치 시계 → 폰 시계)
  2. 촉각: 능동 진동 반응 → 저울 보정 → 쥐는 힘 / 영상 접촉 단계 / 접촉음 + 손목 충격
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
from .touch.vibration import calibrate, pulse_responses

FINGERTIPS = [4, 8, 12, 16, 20]


def _fingertips(labels) -> List[Optional[np.ndarray]]:
    out = []
    for f in labels or []:
        pts = [np.asarray(h)[FINGERTIPS, :2] for h in f.hands if h is not None]
        out.append(np.concatenate(pts) if pts else None)
    return out


def process(session_dir, out_dir, level: str = "bio", fps: float = 30.0, masked_video=None, labels=None,
            objects: Optional[Sequence[dict]] = None, hand_points=None, steps: Optional[list] = None,
            verified_steps: Optional[list] = None, episodes_s: Optional[list] = None, privacy: Optional[dict] = None,
            extract: bool = True) -> dict:
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

    # 2. 촉각
    cal = calibrate(responses, (s.meta.get("calibration") or {}).get("presses", [])) if responses else None
    grip, vib = [], []
    if wrist_ok:
        for r in responses:
            if r.valid and r.onset_ns:
                t = int(m(np.array([r.onset_ns]))[0])
                vib.append((t, r.amp, r.decay_ms, r.energy))
                if cal is not None:
                    grip.append((t, cal.force(r)))
    if labels is None and extract and video is not None:
        from .labels import extract_labels

        labels = extract_labels(s.video_path or video)
    pts = hand_points if hand_points is not None else _fingertips(labels)
    cframes = contact_phases(pts, objects, fps) if objects is not None and pts else None
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

    # 4~5. 에피소드마다 격자 + 채널
    frame_ns = s.frame_ns
    spans = episodes_s or [(0.0, (frame_ns[-1] - frame_ns[0]) / 1e9)]
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
        d["annotation.events"], d["annotation.step"] = event_rows(grid, fps, marks, confirmed, ep_steps, verified_steps)
        err = al.error_ms
        d["sync.error_ms"] = np.tile([err.get("video", 0), err.get("audio", 0), err.get("touch", err.get("wrist", 0)),
                                      err.get("physio", 0)], (len(grid), 1)).astype(np.float32)
        t0 = (grid[0] - frame_ns[0]) / 1e9
        ep.info = {
            "steps": ep_steps or None,
            "outcome": {**(s.meta.get("outcome") or {}),
                        "self_marks_s": [round(float((t - grid[0]) / 1e9), 2) for t in marks if grid[0] <= t <= grid[-1]]},
            "sync": al.to_dict(),
            "touch": {"calibration": cal.to_dict() if cal else None,
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
                  "calibration": cal.to_dict() if cal else None,
                  "contact_events": eps[0].info["touch"]["contact_events"] if eps else [],
                  "confirmed_sounds": eps[0].info["touch"]["confirmed_sounds"] if eps else []},
        "human_state": eps[0].info["human_state"] if eps else None,
        "steps": steps, "outcome": eps[0].info["outcome"] if eps else None, "privacy": privacy,
    }
    certificate.write(cert, out)
    return cert
