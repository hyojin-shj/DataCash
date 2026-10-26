"""세션 폴더 읽기. 없는 파일은 None — 기기 조합(폰만 / 폰+워치 / +생체)에 따라 채널이 달라진다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


def read_csv(path: Path) -> Optional[np.ndarray]:
    """머리줄 한 줄 + 숫자. 시각(ns)이 커서 float64 로는 잘리므로 첫 열은 int64 로 따로 읽는다."""
    if not path.exists():
        return None
    rows = [ln.split(",") for ln in path.read_text().splitlines()[1:] if ln.strip()]
    if not rows:
        return np.zeros((0, 0))
    return np.array([[float(v) for v in r] for r in rows], dtype=np.float64)


def read_ns(path: Path, col: int = 0) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.array([int(ln.split(",")[col]) for ln in path.read_text().splitlines()[1:] if ln.strip()], dtype=np.int64)


def read_wav(path: Path):
    if not path.exists():
        return None, None
    import soundfile as sf

    x, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, rate


@dataclass
class Stream:
    t_ns: np.ndarray                 # 그 기기 시계
    values: np.ndarray               # (n, k)
    clock: str                       # "phone" | "watch"


@dataclass
class Audio:
    x: np.ndarray
    rate: int
    start_ns: int
    clock: str

    def t_ns(self) -> np.ndarray:
        return self.start_ns + (np.arange(len(self.x)) * 1e9 / self.rate).astype(np.int64)


@dataclass
class Session:
    root: Path
    meta: dict
    video_path: Optional[Path]
    frame_ns: Optional[np.ndarray]
    audio: Optional[Audio] = None
    phone_imu: Optional[Stream] = None
    beacon_ns: Optional[np.ndarray] = None
    beacon: Optional[np.ndarray] = None          # (n, 3) freq0, freq1, dur_ms
    watch_imu: Optional[Stream] = None
    watch_audio: Optional[Audio] = None
    watch_vib: Optional[Stream] = None           # values: dur_ms, amp
    watch_hr: Optional[Stream] = None
    watch_skin: Optional[Stream] = None
    clock_sync: Optional[np.ndarray] = None      # (n, 3) int64
    extra: dict = field(default_factory=dict)

    @property
    def has_watch(self) -> bool:
        return self.watch_imu is not None

    @property
    def has_physio(self) -> bool:
        return self.watch_hr is not None or self.watch_skin is not None


def _stream(root: Path, name: str, clock: str) -> Optional[Stream]:
    p = root / name
    t = read_ns(p)
    if t is None or not len(t):
        return None
    v = read_csv(p)[:, 1:]
    return Stream(t, v, clock)


def _audio(root: Path, stem: str, clock: str) -> Optional[Audio]:
    x, rate = read_wav(root / f"{stem}.wav")
    if x is None:
        return None
    info = json.loads((root / f"{stem}.json").read_text())
    return Audio(x, int(rate), int(info["start_ns"]), clock)


def load_session(root) -> Session:
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text()) if (root / "meta.json").exists() else {}
    cs = None
    if (root / "clock_sync.csv").exists():
        cs = np.array([[int(v) for v in ln.split(",")] for ln in (root / "clock_sync.csv").read_text().splitlines()[1:]
                       if ln.strip()], dtype=np.int64)
    beacon = read_csv(root / "beacon.csv")
    return Session(
        root=root, meta=meta,
        video_path=(root / "video.mp4") if (root / "video.mp4").exists() else None,
        frame_ns=read_ns(root / "video_frames.csv", 1),
        audio=_audio(root, "audio", "phone"),
        phone_imu=_stream(root, "phone_imu.csv", "phone"),
        beacon_ns=read_ns(root / "beacon.csv"), beacon=None if beacon is None else beacon[:, 1:],
        watch_imu=_stream(root, "watch_imu.csv", "watch"),
        watch_audio=_audio(root, "watch_audio", "watch"),
        watch_vib=_stream(root, "watch_vib.csv", "watch"),
        watch_hr=_stream(root, "watch_hr.csv", "watch"),
        watch_skin=_stream(root, "watch_skin.csv", "watch"),
        clock_sync=cs,
    )
