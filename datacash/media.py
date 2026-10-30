"""영상·소리 변환 도우미 (ffmpeg 는 imageio-ffmpeg 가 넣어 준 실행 파일을 쓴다)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args: list) -> None:
    r = subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[-500:])


def clip_mp4(src, dst, start_s: float = 0.0, dur_s: Optional[float] = None, max_side: int = 960,
             fps: Optional[float] = None, audio: bool = True, gain: float = 1.0, slow: float = 1.0) -> Path:
    """H.264 MP4 (faststart) 로 자른다 — SAM 3.1·Muse Spark 가 받는 형식. 긴 쪽을 max_side 로 줄인다.
    gain > 1 이면 어두운 영상을 그만큼 곱해 밝힌다 (light.gain_for).
    slow > 1 이면 그만큼 느리게 늘린다 (영상만 — 소리는 뺀다). 결과 영상 길이는 dur_s × slow."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    if slow > 1.001:
        vf += f",setpts={slow:g}*PTS"
        audio = False
    if fps:
        vf += f",fps={fps}"
    if gain > 1.01:
        vf += f",lutrgb=r=val*{gain:.2f}:g=val*{gain:.2f}:b=val*{gain:.2f}"
    args = ["-ss", start_s, "-i", src] + (["-t", dur_s * slow] if dur_s else []) + \
           ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"] + (["-c:a", "aac", "-b:a", "96k"] if audio else ["-an"]) + [dst]
    run(args)
    return dst


def wav(src, dst, rate: int = 16000, start_s: float = 0.0, dur_s: Optional[float] = None) -> Path:
    """모노 16비트 PCM WAV (Muse Voice Transcribe 는 16k·24k 모노만 받는다)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["-ss", start_s, "-i", src] + (["-t", dur_s] if dur_s else []) +
        ["-vn", "-ac", 1, "-ar", rate, "-c:a", "pcm_s16le", "-map_metadata", "-1", dst])
    return dst
