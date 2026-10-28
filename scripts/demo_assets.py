"""데모 영상 재료 (대본 장면 3) — 처리된 녹화 하나로 세 가지 영상을 만든다. 출력은 data/demo/<녹화>/ (원본 얼굴이 들어가 git 밖).

  python scripts/demo_assets.py 20261027_131032
  pose.mp4   판매용 영상(얼굴 치환·가림) 위에 손 21점 × 2 · 몸 관절선 — "precise pose recognition"
  wipe.mp4   원본 → 치환으로 세로선이 쓸고 지나가는 비교 — "Your face? Swapped"
  gaze.mp4   눈 주변 확대, 원본 | 치환 나란히 — "Same gaze. Same expression."
원본 영상이 서버에 남아 있어야 한다 (시연 서버는 원본을 지우지 않는다). 본인 동의 영상만 쓴다.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacash.media import run  # noqa: E402

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
PINK, ORANGE, WHITE = (95, 95, 255), (95, 175, 255), (255, 255, 255)


def frames(path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def writer(path, fps, size):
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def finish(raw: Path, out: Path):
    run(["-i", raw, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out])
    raw.unlink()


def pose_video(swap: Path, df: pd.DataFrame, out: Path, fps: float):
    hands = np.stack(df["observation.hands"].values).reshape(len(df), 2, 21, 3)
    body = np.stack(df["observation.body"].values).reshape(len(df), 33, 3)
    skip = int(cv2.VideoCapture(str(swap)).get(cv2.CAP_PROP_FRAME_COUNT)) - len(df)   # 데이터셋은 앞의 얼굴 확인 구간을 뺐다
    vw = None
    for j, f in enumerate(frames(swap)):
        i = j - skip
        h, w = f.shape[:2]
        if vw is None:
            vw = writer(out.with_suffix(".raw.mp4"), fps, (w, h))
        if 0 <= i < len(df):
            glow = f.copy()
            b = body[i]
            for a, c in BODY_EDGES:
                if b[a, 2] > 0.5 and b[c, 2] > 0.5:
                    cv2.line(glow, (int(b[a, 0] * w), int(b[a, 1] * h)), (int(b[c, 0] * w), int(b[c, 1] * h)), WHITE, 3, cv2.LINE_AA)
            for k, col in ((0, ORANGE), (1, PINK)):
                p = hands[i, k]
                if not np.isfinite(p).all() or not p.any():
                    continue
                pts = [(int(x * w), int(y * h)) for x, y, _ in p]
                for a, c in HAND_EDGES:
                    cv2.line(glow, pts[a], pts[c], col, 3, cv2.LINE_AA)
                for q in pts:
                    cv2.circle(glow, q, 5, WHITE, -1, cv2.LINE_AA)
            f = cv2.addWeighted(glow, 0.85, f, 0.15, 0)
        vw.write(f)
    vw.release()
    finish(out.with_suffix(".raw.mp4"), out)


def wipe_video(orig: Path, swap: Path, out: Path, fps: float):
    o, s = list(frames(orig)), list(frames(swap))
    n = min(len(o), len(s))
    h, w = s[0].shape[:2]
    vw = writer(out.with_suffix(".raw.mp4"), fps, (w, h))
    for i in range(n):
        a = cv2.resize(o[i], (w, h))
        ph = (i / n) * 2.4 - 0.7                       # 세로선이 왼쪽 밖에서 오른쪽 밖으로 한 번 쓸고 간다
        x = int(np.clip(ph, 0, 1) * w)
        f = s[i].copy()
        f[:, x:] = a[:, x:]
        if 0 < x < w:
            cv2.line(f, (x, 0), (x, h), WHITE, 4)
        cv2.putText(f, "ORIGINAL", (w - 210, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2, cv2.LINE_AA) if x < w - 230 else None
        cv2.putText(f, "SWAPPED", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2, cv2.LINE_AA) if x > 220 else None
        vw.write(f)
    vw.release()
    finish(out.with_suffix(".raw.mp4"), out)


def gaze_video(orig: Path, swap: Path, out: Path, fps: float):
    from datacash.privacy.faceswap import FaceLandmarks

    fl = FaceLandmarks()
    o, s = list(frames(orig)), list(frames(swap))
    n = min(len(o), len(s))
    W, H = 640, 260
    vw = writer(out.with_suffix(".raw.mp4"), fps, (W * 2 + 8, H))
    box = None
    for i in range(n):
        a = cv2.resize(o[i], (s[i].shape[1], s[i].shape[0]))
        found = fl(a)
        if found:
            lm = found[0]
            eyes = lm[[33, 133, 362, 263, 159, 145, 386, 374]]
            cx, cy = eyes.mean(0)
            span = np.ptp(eyes[:, 0]) * 1.6
            nb = np.array([cx - span / 2, cy - span * 0.2, cx + span / 2, cy + span * 0.2])
            box = nb if box is None else box * 0.8 + nb * 0.2   # 흔들림 줄이기
        if box is None:
            continue
        x0, y0, x1, y1 = box.astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        crop = lambda img: cv2.resize(img[y0:max(y0 + 2, y1), x0:max(x0 + 2, x1)], (W, H))  # noqa: E731
        f = np.zeros((H, W * 2 + 8, 3), np.uint8)
        f[:, :W], f[:, W + 8:] = crop(a), crop(s[i])
        cv2.putText(f, "ORIGINAL", (16, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
        cv2.putText(f, "SWAPPED - same gaze", (W + 24, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
        vw.write(f)
    vw.release()
    finish(out.with_suffix(".raw.mp4"), out)


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20261027_131032"
    root = Path("data/server")
    orig, out_dir = root / "sessions" / sid / "video.mp4", root / "out" / sid
    swap = out_dir / "masked_swap.mp4"
    dst = Path("data/demo") / sid
    dst.mkdir(parents=True, exist_ok=True)
    fps = cv2.VideoCapture(str(swap)).get(cv2.CAP_PROP_FPS) or 30
    df = pd.concat([pd.read_parquet(p) for p in sorted((out_dir / "lerobot" / "data").rglob("*.parquet"))], ignore_index=True)
    pose_video(swap, df, dst / "pose.mp4", fps)
    print("pose.mp4")
    if orig.exists():
        wipe_video(orig, swap, dst / "wipe.mp4", fps)
        print("wipe.mp4")
        gaze_video(orig, swap, dst / "gaze.mp4", fps)
        print("gaze.mp4")
    print(json.dumps({"out": str(dst)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
