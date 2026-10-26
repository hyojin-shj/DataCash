"""합성 4감 세션 — 정답(시계 차이·쥐는 힘·접촉 시각)을 알고 있는 녹화를 만들어 동기화·촉각·내보내기를 검증한다.

장면 (기본 20초): 폰은 거치, 손(살색 원)이 컵(사각형)으로 다가가 6초에 잡고 10초에 내려놓는다.
  - 1~4초: 쥐는 힘 보정 — 주방 저울을 2·5·10N 으로 누른다 (meta.calibration 에 저울 값)
  - 6~10초: 컵을 8N 으로 쥐고 옮긴다, 10초에 '탁' 내려놓음 (소리 + 손목 충격), 8초에 '실수했어요'
  - 심박: 평소 70 → 컵을 드는 동안 95 까지
시계: 폰 시계 P0 기준. 워치 시계 = W0 + (1 + drift) × (폰 경과) — 동기화가 찾아야 할 정답.
폰 스피커 음향 비콘: 2초마다 18→20kHz 0.05초 처프, 스피커 지연(모름) 35ms.
워치 능동 진동: 0.5초마다 0.05초, 모터 지연 15ms. 쥘수록 손목이 단단해져 울림이 작아지고 빨리 사그라든다.
폰↔워치 시각 주고받기는 왕복이 비대칭(가는 길이 더 느림)이라 그것만 쓰면 수 ms 치우친다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

P0 = 1_000_000_000_000          # 폰 시계 시작 (ns)
W0 = 7_300_000_000_000          # 워치 시계 시작 (ns)
NS = 1_000_000_000


def _chirp(rate, f0, f1, dur):
    t = np.arange(int(rate * dur)) / rate
    k = (f1 - f0) / dur
    w = np.hanning(len(t))
    return (np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t)) * w).astype(np.float32)


def generate(out, duration: float = 20.0, fps: float = 30.0, drift_ppm: float = 40.0, seed: int = 0,
             phone_audio_rate: int = 48000, watch_audio_rate: int = 48000, watch_imu_rate: float = 400.0,
             physio: bool = True, size=(320, 240), grip_n: float = 8.0, hr_per_n: float = 3.1) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    drift = drift_ppm * 1e-6

    def to_watch(tau):  # 폰 경과 초 → 워치 시계 ns
        return (W0 + (np.asarray(tau) * (1 + drift)) * NS).astype(np.int64)

    # ---- 장면 대본 (폰 경과 초) ----
    grasp0, grasp1, mark = 6.0, 10.0, 8.0
    presses = [(1.0, 1.8, 2.0), (2.0, 2.8, 5.0), (3.0, 3.8, 10.0)]   # 저울 누르기 (시작, 끝, N)

    def force_at(tau):
        f = np.zeros_like(np.asarray(tau, float))
        for a, b, n in presses:
            f = np.where((tau >= a) & (tau < b), n, f)
        return np.where((tau >= grasp0) & (tau < grasp1), grip_n, f)

    W, H = size
    cup0 = np.array([210.0, 150.0])

    def hand_xy(tau):
        if tau < 5.0:
            return np.array([60.0 + 4 * tau, 120.0])
        if tau < grasp0:
            a = (tau - 5.0) / 1.0
            return (1 - a) * np.array([80.0, 120.0]) + a * (cup0 + [0, -5])
        if tau < grasp1:
            a = (tau - grasp0) / (grasp1 - grasp0)
            return cup0 + [0, -5] + a * np.array([-110.0, 0.0]) + [0, -25 * np.sin(np.pi * a)]
        return cup0 + [-110, -5] + (tau - grasp1) * np.array([-10.0, -15.0])

    def cup_xy(tau):
        if tau < grasp0:
            return cup0.copy()
        if tau < grasp1:
            return hand_xy(tau) + [0, 5]
        return cup0 + [-110, 0]

    # ---- 영상 + 프레임 시각 ----
    n_frames = int(duration * fps)
    frame_tau = np.arange(n_frames) / fps + rng.normal(0, 0.0007, n_frames)
    frame_tau[0] = 0.0
    frame_tau = np.maximum.accumulate(frame_tau)
    vw = cv2.VideoWriter(str(out / "video_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for tau in frame_tau:
        img = np.full((H, W, 3), (210, 215, 220), np.uint8)
        cv2.rectangle(img, (0, 175), (W, H), (120, 140, 160), -1)                        # 조리대
        cx, cy = cup_xy(tau)
        cv2.rectangle(img, (int(cx - 14), int(cy - 22)), (int(cx + 14), int(cy + 22)), (60, 90, 200), -1)  # 컵
        hx, hy = hand_xy(tau)
        cv2.circle(img, (int(hx), int(hy)), 16, (120, 160, 220), -1)                     # 손
        vw.write(img)
    vw.release()
    from ..media import run
    run(["-i", out / "video_raw.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         out / "video.mp4"])
    (out / "video_raw.mp4").unlink()
    frame_ns = (P0 + frame_tau * NS).astype(np.int64)
    (out / "video_frames.csv").write_text("frame,t_ns\n" + "".join(f"{i},{t}\n" for i, t in enumerate(frame_ns)))

    # ---- 소리 (폰 마이크·워치 마이크) ----
    def audio_track(rate):
        return (rng.normal(0, 0.002, int(rate * duration))).astype(np.float32)

    pa, wa = audio_track(phone_audio_rate), audio_track(watch_audio_rate)

    def add(track, rate, tau, sig, gain):
        i = int(round(tau * rate))
        if 0 <= i < len(track):
            j = min(len(track), i + len(sig))
            track[i:j] += gain * sig[: j - i]

    beacon_rows, spk_lat = [], 0.035
    for tau in np.arange(0.5, duration - 0.3, 2.0):
        c_p, c_w = _chirp(phone_audio_rate, 18000, 20000, 0.05), _chirp(watch_audio_rate, 18000, 20000, 0.05)
        add(pa, phone_audio_rate, tau + spk_lat, c_p, 0.3)                 # 폰 마이크가 자기 소리를 듣는다
        add(wa, watch_audio_rate, (tau + spk_lat + 0.0012) * (1 + drift), c_w, 0.08)  # 워치까지 약 40cm, 워치 시계로
        beacon_rows.append((int(P0 + tau * NS), 18000, 20000, 50))
    (out / "beacon.csv").write_text("t_ns,freq0_hz,freq1_hz,dur_ms\n" + "".join(f"{a},{b},{c},{d}\n" for a, b, c, d in beacon_rows))
    t_clink = np.arange(int(0.03 * phone_audio_rate)) / phone_audio_rate
    clink = (np.sin(2 * np.pi * 3200 * t_clink) * np.exp(-t_clink / 0.006)).astype(np.float32)
    add(pa, phone_audio_rate, grasp1, clink, 0.5)                                 # 내려놓는 '탁'
    add(wa, watch_audio_rate, (grasp1 + 0.0012) * (1 + drift), clink[: int(0.03 * watch_audio_rate)] if watch_audio_rate == phone_audio_rate
        else (np.sin(2 * np.pi * 3200 * np.arange(int(0.03 * watch_audio_rate)) / watch_audio_rate)
              * np.exp(-np.arange(int(0.03 * watch_audio_rate)) / watch_audio_rate / 0.006)).astype(np.float32), 0.6)

    # ---- 워치 능동 진동 + 손목 IMU ----
    n_w = int(duration * watch_imu_rate)
    w_tau = np.arange(n_w) / watch_imu_rate
    acc = np.zeros((n_w, 3))
    hx = np.array([hand_xy(t) for t in w_tau])
    from scipy.ndimage import gaussian_filter1d
    pos_m = gaussian_filter1d(hx * 0.002, sigma=watch_imu_rate * 0.05, axis=0)   # 화면 1px = 2mm
    accel_motion = np.gradient(np.gradient(pos_m, axis=0), axis=0) * watch_imu_rate ** 2
    acc[:, 0], acc[:, 1] = accel_motion[:, 0], accel_motion[:, 1]
    acc[:, 2] = 9.81
    acc += rng.normal(0, 0.02, acc.shape)
    vib_rows, motor_lat, fv = [], 0.015, 170.0
    for tau in np.arange(0.25, duration - 0.1, 0.5):
        f = float(force_at(np.array([tau + 0.03]))[0])
        amp = 0.6 * (1.0 - 0.045 * f)                 # 쥘수록 울림이 작다
        dec = 0.040 * (1.0 - 0.05 * f)                # 쥘수록 빨리 사그라든다 (초)
        i0 = int(round((tau + motor_lat) * watch_imu_rate))
        k = np.arange(int(0.15 * watch_imu_rate))
        tt = k / watch_imu_rate
        env = np.where(tt < 0.05, 1.0, np.exp(-(tt - 0.05) / max(dec, 1e-3)))
        sig = amp * env * np.sin(2 * np.pi * fv * tt)
        j = min(n_w, i0 + len(k))
        acc[i0:j, 2] += sig[: j - i0]
        acc[i0:j, 0] += 0.4 * sig[: j - i0]
        vib_rows.append((int(to_watch(tau)), 50, 1.0))
        # 폰 마이크가 진동음을 작게 듣는다 (워치 시계와 폰 시계를 잇는 또 하나의 사건)
        tv = np.arange(int(0.05 * phone_audio_rate)) / phone_audio_rate
        add(pa, phone_audio_rate, tau + motor_lat + 0.0012, (np.sin(2 * np.pi * fv * tv) * np.hanning(len(tv))).astype(np.float32), 0.01)
    i_put = int(grasp1 * watch_imu_rate)
    if i_put + 8 <= n_w:
        acc[i_put:i_put + 8, 1] += np.linspace(3.0, 0.0, 8)    # 내려놓는 충격
    gyro = rng.normal(0, 0.01, (n_w, 3))
    w_ns = to_watch(w_tau)
    (out / "watch_imu.csv").write_text("t_ns,ax,ay,az,gx,gy,gz\n" + "".join(
        f"{t},{a[0]:.5f},{a[1]:.5f},{a[2]:.5f},{g[0]:.5f},{g[1]:.5f},{g[2]:.5f}\n" for t, a, g in zip(w_ns, acc, gyro)))
    (out / "watch_vib.csv").write_text("t_ns,dur_ms,amp\n" + "".join(f"{a},{b},{c}\n" for a, b, c in vib_rows))

    # ---- 폰 IMU (거치, 잡음만) ----
    p_tau = np.arange(int(duration * 200)) / 200
    pi = np.column_stack([rng.normal(0, 0.01, len(p_tau)), rng.normal(0, 0.01, len(p_tau)), 9.81 + rng.normal(0, 0.01, len(p_tau)),
                          rng.normal(0, 0.002, (len(p_tau), 3))])
    (out / "phone_imu.csv").write_text("t_ns,ax,ay,az,gx,gy,gz\n" + "".join(
        f"{int(P0 + t * NS)}," + ",".join(f"{v:.5f}" for v in r) + "\n" for t, r in zip(p_tau, pi)))

    # ---- 소리 파일 ----
    import soundfile as sf

    sf.write(str(out / "audio.wav"), np.clip(pa, -1, 1), phone_audio_rate, subtype="PCM_16")
    (out / "audio.json").write_text(json.dumps({"start_ns": P0, "rate": phone_audio_rate}))
    sf.write(str(out / "watch_audio.wav"), np.clip(wa, -1, 1), watch_audio_rate, subtype="PCM_16")
    (out / "watch_audio.json").write_text(json.dumps({"start_ns": int(to_watch(0.0)), "rate": watch_audio_rate}))

    # ---- 생체 ----
    if physio:
        hr_tau = np.arange(0, duration, 1.0)
        # 무거울수록(쥐는 힘이 클수록) 심박이 더 오른다 — 들기 시작하면 오르고 내려놓은 뒤 천천히 돌아온다
        rise = hr_per_n * grip_n + rng.normal(0, 1.5)
        up = np.clip((hr_tau - grasp0) / 3.0, 0, 1) * (hr_tau < grasp1)
        down = np.where(hr_tau >= grasp1, np.exp(-(hr_tau - grasp1) / 4.0), 0.0)
        hr = 70 + rise * np.maximum(up, down * (hr_tau >= grasp1)) + rng.normal(0, 0.8, len(hr_tau))
        (out / "watch_hr.csv").write_text("t_ns,bpm\n" + "".join(f"{int(to_watch(t))},{b:.1f}\n" for t, b in zip(hr_tau, hr)))
        (out / "watch_skin.csv").write_text("t_ns,temp_c\n" + "".join(
            f"{int(to_watch(t))},{33.0 + 0.02 * t:.3f}\n" for t in hr_tau))

    # ---- 폰↔워치 시각 주고받기 (가는 길 6±2ms, 오는 길 2±1ms → 왕복 가운데로 보면 2ms 치우침) ----
    rows = []
    for tau in np.arange(0.0, duration, 0.5):
        up, down = abs(rng.normal(0.006, 0.002)), abs(rng.normal(0.002, 0.001))
        rows.append((int(P0 + tau * NS), int(to_watch(tau + up)), int(P0 + (tau + up + down) * NS)))
    (out / "clock_sync.csv").write_text("phone_send_ns,watch_ns,phone_recv_ns\n" + "".join(f"{a},{b},{c}\n" for a, b, c in rows))

    meta = {
        "session_id": out.name, "synthetic": True, "mission": {"id": "carry_cup", "title": "컵 옮기기", "situation": "합성"},
        "devices": {"phone": {"model": "synthetic-phone"}, "watch": {"model": "synthetic-watch"}},
        "participant": {"consent": {"video": True, "bio": physio}, "age_band": "30대"},
        "self_marks_ns": [int(P0 + mark * NS)],
        "outcome": {"outcome": "success", "retries": 1},
        "calibration": {"kind": "scale_press", "presses": [
            {"start_ns": int(to_watch(a)), "end_ns": int(to_watch(b)), "force_n": n} for a, b, n in presses]},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    tracks = [{"t_s": round(float(t), 4), "hand": [round(float(v), 2) for v in hand_xy(t)],
               "cup": [round(float(v), 2) for v in (cup_xy(t)[0] - 14, cup_xy(t)[1] - 22, cup_xy(t)[0] + 14, cup_xy(t)[1] + 22)]}
              for t in frame_tau]
    (out / "tracks.json").write_text(json.dumps(tracks))
    truth = {"P0": P0, "W0": W0, "drift": drift, "speaker_latency_s": spk_lat, "motor_latency_s": motor_lat,
             "grasp": [grasp0, grasp1], "putdown_s": grasp1, "mark_s": mark, "grip_force_n": grip_n,
             "presses": presses, "fps": fps, "duration": duration}
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def watch_to_phone_truth(truth: dict, watch_ns):
    """정답 사상: 워치 시계 ns → 폰 시계 ns."""
    return (truth["P0"] + (np.asarray(watch_ns, np.float64) - truth["W0"]) / (1 + truth["drift"])).astype(np.int64)
