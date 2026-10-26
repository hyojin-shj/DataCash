"""공개 영상 한 구간을 '폰만 있는 녹화 세션' 형식으로 만든다 (워치·IMU 없음) — 실제 영상으로 서버 처리 전체를 시험할 때.

  python scripts/session_from_video.py <영상> <세션 폴더> --start 40 --dur 8 --mission cooking
장면 시각은 fps 로 만든 값이다 (실제 폰 녹화는 카메라가 장면마다 시각을 적는다).
"""

import argparse
import json
from pathlib import Path

import cv2

from datacache.media import clip_mp4, wav

P0 = 2_000_000_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=8)
    ap.add_argument("--mission", default="cooking")
    ap.add_argument("--title", default="요리")
    ap.add_argument("--source", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clip_mp4(a.video, out / "video.mp4", a.start, a.dur, max_side=960, fps=24, audio=False)
    cap = cv2.VideoCapture(str(out / "video.mp4"))
    n, fps = int(cap.get(7)), cap.get(5) or 24
    (out / "video_frames.csv").write_text("frame,t_ns\n" + "".join(f"{i},{int(P0 + i * 1e9 / fps)}\n" for i in range(n)))
    try:
        wav(a.video, out / "audio.wav", rate=48000, start_s=a.start, dur_s=a.dur)
        (out / "audio.json").write_text(json.dumps({"start_ns": P0, "rate": 48000}))
    except Exception:
        pass
    meta = {"session_id": out.name, "format": "datacache/1", "public_video": True, "source": a.source or str(a.video),
            "mission": {"id": a.mission, "title": a.title, "situation": "공개 영상"},
            "devices": {"phone": {"model": "public-video"}},
            "participant": {"consent": {"video": True, "bio": False}, "age_band": ""}, "self_marks_ns": [],
            "outcome": {"outcome": None, "retries": None}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(out, n, "frames")


if __name__ == "__main__":
    main()
