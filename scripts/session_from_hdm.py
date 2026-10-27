"""홈데이터미션 앱으로 실제 폰에서 찍은 녹화 → 데이터캐시 세션 형식 (실험용).

  python scripts/session_from_hdm.py <홈데이터미션 raw 녹화 폴더> <세션 폴더> [--dur 20] [--fps 10] [--side 640]
옮기는 것
  영상        video.mp4(원본)가 있으면 그것, 없으면 video_masked.mp4(폰에서 이미 가린 것). 1단계 얼굴 확인이 끝난 뒤부터
              --dur 초만, --fps 로 장면을 솎고(원본 장면 번호 n 을 그대로 지켜 장면 시각을 잇는다), 긴 변 --side px
              — SAM 3.1 은 장면당 값이 붙어서 (명사구마다 $0.0002) 실험 비용을 줄인다
  장면 시각   video_frames.csv (BOOTTIME, 카메라가 장면마다 적은 값) 에서 남긴 장면만
  소리        audio.wav + 시작 시각 (AudioTimestamp BOOTTIME)
  폰 IMU      imu_accel.csv + imu_gyro.csv → phone_imu.csv (가속도 시각에 자이로를 맞춘다)
  손목        wrist/ (갤럭시 워치를 '손목 기기'로 쓴 녹화) → watch_imu.csv, clock_sync.csv (폰↔워치 UDP 왕복)
  meta        미션·얼굴 확인 구간·기기
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from datacache.media import run

MISSION_OBJECTS = {"dishwashing": ["plate", "cup", "sponge"], "cooking": ["frying pan", "spatula", "bowl"],
                   "game": ["game controller"], "squat": [], "vacuum": ["vacuum cleaner"], "makeup": ["brush"]}


def imu(acc_csv: Path, gyro_csv: Path) -> np.ndarray:
    a = np.loadtxt(acc_csv, delimiter=",", skiprows=1)
    g = np.loadtxt(gyro_csv, delimiter=",", skiprows=1)
    gi = np.stack([np.interp(a[:, 0], g[:, 0], g[:, k]) for k in (1, 2, 3)], 1)
    return np.column_stack([a[:, 0].astype(np.int64), a[:, 1:4], gi])


def write_imu(path: Path, rows: np.ndarray, t0: int, t1: int) -> int:
    sel = rows[(rows[:, 0] >= t0) & (rows[:, 0] <= t1)]
    path.write_text("t_ns,ax,ay,az,gx,gy,gz\n" + "".join(
        f"{int(r[0])},{r[1]:.5f},{r[2]:.5f},{r[3]:.5f},{r[4]:.5f},{r[5]:.5f},{r[6]:.5f}\n" for r in sel))
    return len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--side", type=int, default=640)
    ap.add_argument("--from-start", action="store_true", help="얼굴 확인 구간부터 (참여자 알아보기 시험)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads((src / "session.json").read_text())
    fr = np.loadtxt(src / "video_frames.csv", delimiter=",", skiprows=1, dtype=np.int64)
    fc = s.get("face_check") or {}
    t_begin = int(fr[0, 1]) if a.from_start or not fc.get("end_ns") else int(fc["end_ns"]) + 500_000_000
    i0 = int(np.searchsorted(fr[:, 1], t_begin))
    i1 = int(np.searchsorted(fr[:, 1], t_begin + a.dur * 1e9))
    step = max(1, round((len(fr) / max(1e-9, (fr[-1, 1] - fr[0, 1]) / 1e9)) / a.fps))
    keep = fr[i0:i1:step]
    video = src / "video.mp4" if (src / "video.mp4").exists() else src / "video_masked.mp4"
    run(["-i", video, "-vf", f"select='between(n\\,{i0}\\,{i1 - 1})*not(mod(n-{i0}\\,{step}))',"
         f"scale='if(gt(iw,ih),{a.side},-2)':'if(gt(iw,ih),-2,{a.side})'", "-vsync", "0", "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", out / "video.mp4"])
    import cv2

    n_out = int(cv2.VideoCapture(str(out / "video.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    keep = keep[:n_out]
    (out / "video_frames.csv").write_text("frame,t_ns\n" + "".join(f"{k},{int(t)}\n" for k, t in enumerate(keep[:, 1])))
    t0, t1 = int(keep[0, 1]) - 1_000_000_000, int(keep[-1, 1]) + 1_000_000_000
    if (src / "audio.wav").exists():
        shutil.copy(src / "audio.wav", out / "audio.wav")
        aj = json.loads((src / "audio.json").read_text())
        (out / "audio.json").write_text(json.dumps({"start_ns": int(aj["t_start_phone_ns"]), "rate": aj["rate"], "clock": "BOOTTIME"}))
    n_phone = write_imu(out / "phone_imu.csv", imu(src / "imu_accel.csv", src / "imu_gyro.csv"), t0, t1)
    watch = None
    if (src / "wrist" / "imu_accel.csv").exists() and (src / "clock_sync.csv").exists():
        cs = np.loadtxt(src / "clock_sync.csv", delimiter=",", skiprows=1, dtype=np.int64)
        (out / "clock_sync.csv").write_text("phone_send_ns,watch_ns,phone_recv_ns\n" + "".join(f"{x},{y},{z}\n" for x, y, z in cs))
        mid = (cs[:, 0] + cs[:, 2]) // 2
        off = int(np.median(cs[:, 1] - mid))                   # 워치 시계 ≈ 폰 시계 + off (자르기 범위용 대략값)
        rows = imu(src / "wrist" / "imu_accel.csv", src / "wrist" / "imu_gyro.csv")
        n_w = write_imu(out / "watch_imu.csv", rows, t0 + off, t1 + off)
        ws = json.loads((src / "wrist" / "session.json").read_text())
        watch = {"model": ws.get("device"), "android": ws.get("android"), "imu_rate_hz": (ws.get("imu") or {}).get("accel_rate_hz"),
                 "imu_rows": n_w, "role": "hdm wrist (vibration pulses 없음)"}
    m = s.get("mission") or {}
    mid_ = m.get("id", "unknown")
    meta = {"session_id": out.name, "format": "datacache/1", "clock": "BOOTTIME", "source": f"home-data-mission {src.name}",
            "mission": {"id": f"hdm_{mid_}", "title": m.get("title", ""), "situation": m.get("situation", ""),
                        "objects": MISSION_OBJECTS.get(mid_, [])},
            "devices": {"phone": {"model": s.get("device"), "android": s.get("android"), "id": "hdmphone01"},
                        **({"watch": watch} if watch else {})},
            "participant": {"consent": {"video": True, "bio": False}, "age_band": ""},
            "video": {**{k: v for k, v in (s.get("video") or {}).items() if k in ("width", "height", "lens_facing", "orientation_hint")},
                      "source_file": video.name, "kept_every": step, "fps_out": round(len(keep) / max(1e-9, (keep[-1, 1] - keep[0, 1]) / 1e9), 2)},
            "phone_imu_rows": n_phone, "recorded_at": "2026-10-2xT00:00:00+0900"}
    if a.from_start and fc:
        meta["face_check"] = {k: fc[k] for k in ("passed", "start_ns", "end_ns", "clock", "identity") if k in fc}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps({"frames": len(keep), "every": step, "phone_imu": n_phone, "watch": watch, "video": video.name}, ensure_ascii=False))


if __name__ == "__main__":
    main()
