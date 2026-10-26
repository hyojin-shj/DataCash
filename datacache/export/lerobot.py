"""LeRobot v3.0 데이터셋 쓰기 — 4감 채널 + 가린 영상.

  meta/info.json                       형식·fps·feature 정의(dtype·shape·names)
  meta/stats.json                      feature 별 min·max·mean·std·count
  meta/tasks.parquet                   과제 문장 ↔ task_index
  meta/episodes/chunk-000/file-000.parquet   에피소드: 길이·위치·영상 구간·에피소드별 통계 + 우리 정보(JSON 문자열)
  data/chunk-000/file-000.parquet      장면마다 모든 채널 + timestamp·frame_index·episode_index·index·task_index
  videos/observation.images.ego/chunk-000/file-000.mp4  에피소드를 이어 붙인 H.264 영상

등급(level)에 따라 채널을 뺀다 (channels.LEVEL_DROP). 에피소드 정보(info)는 사람이 읽는 JSON 문자열로 넣는다:
steps(단계 문장·시각·검증), outcome(본인 결과·실수), sync(감각별 오차·방법), touch(보정·접촉 사건), human_state(생체 요약·본인 응답).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..media import run
from ..understand.spark import PRIMITIVES
from .channels import LEVEL_DROP, NAMES, EpisodeChannels

VIDEO_KEY = "observation.images.ego"
INFO_KEYS = ("steps", "mistakes", "scene", "outcome", "sync", "touch", "human_state", "privacy", "source")


def _stats(a: np.ndarray) -> dict:
    a = np.asarray(a, np.float64)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
            "std": a.std(0).tolist(), "count": [int(len(a))]}


def _write_video(frames_iter, out: Path, fps: float, size) -> int:
    tmp = out.with_suffix(".raw.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = 0
    for f in frames_iter:
        if (f.shape[1], f.shape[0]) != size:
            f = cv2.resize(f, size)
        vw.write(f)
        n += 1
    vw.release()
    run(["-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", out])
    tmp.unlink()
    return n


def _frames(video_path, src_idx):
    """src_idx 순서대로 장면을 낸다 (같은 장면이 되풀이되거나 뒤로 가면 다시 찾는다)."""
    cap = cv2.VideoCapture(str(video_path))
    cur, frame = -1, None
    for i in (int(v) for v in src_idx):
        if i < cur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            cur, frame = i - 1, None
        while cur < i:
            ok, f = cap.read()
            if not ok:
                break
            frame, cur = f, cur + 1
        yield frame
    cap.release()


def export(out_dir, episodes: List[EpisodeChannels], videos: List[Path], fps: float, level: str = "bio",
           robot_type: str = "human_phone_watch", max_side: int = 640) -> Path:
    """episodes[i] 의 장면 격자(src_frame)대로 videos[i](가린 영상)에서 장면을 뽑아 이어 붙인다."""
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    drop = LEVEL_DROP[level]
    keys = [k for k in NAMES if k not in drop and all(k in ep.data for ep in episodes)]
    cap = cv2.VideoCapture(str(videos[0]))
    w0, h0 = int(cap.get(3)), int(cap.get(4))
    cap.release()
    k = min(1.0, max_side / max(w0, h0))
    size = (int(round(w0 * k / 2) * 2), int(round(h0 * k / 2) * 2))

    def all_frames():
        for ep, v in zip(episodes, videos):
            yield from _frames(v, ep.src_frame)

    vpath = out / "videos" / VIDEO_KEY / "chunk-000" / "file-000.mp4"
    n_written = _write_video(all_frames(), vpath, fps, size)

    tasks = sorted({ep.task for ep in episodes})
    task_idx = {t: i for i, t in enumerate(tasks)}
    cols: Dict[str, list] = {k: [] for k in keys}
    meta_cols = {"timestamp": [], "frame_index": [], "episode_index": [], "index": [], "task_index": []}
    ep_rows, total = [], 0
    for e, ep in enumerate(episodes):
        n = len(ep.grid_ns)
        for k in keys:
            cols[k].append(np.asarray(ep.data[k], np.float32).reshape(n, -1))
        meta_cols["timestamp"].append((np.arange(n) / fps).astype(np.float32))
        meta_cols["frame_index"].append(np.arange(n, dtype=np.int64))
        meta_cols["episode_index"].append(np.full(n, e, np.int64))
        meta_cols["index"].append(np.arange(total, total + n, dtype=np.int64))
        meta_cols["task_index"].append(np.full(n, task_idx[ep.task], np.int64))
        row = {"episode_index": e, "tasks": [ep.task], "length": n,
               "data/chunk_index": 0, "data/file_index": 0, "dataset_from_index": total, "dataset_to_index": total + n,
               f"videos/{VIDEO_KEY}/chunk_index": 0, f"videos/{VIDEO_KEY}/file_index": 0,
               f"videos/{VIDEO_KEY}/from_timestamp": total / fps, f"videos/{VIDEO_KEY}/to_timestamp": (total + n) / fps,
               "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
        for k in keys:
            for s, v in _stats(cols[k][-1]).items():
                row[f"stats/{k}/{s}"] = v
        for ik in INFO_KEYS:
            row[ik] = json.dumps(ep.info.get(ik), ensure_ascii=False, default=float) if ep.info.get(ik) is not None else ""
        ep_rows.append(row)
        total += n
    if n_written != total:
        raise RuntimeError(f"영상 장면 수 {n_written} ≠ 데이터 장면 수 {total}")

    arrays, fields = [], []
    for k in keys:
        a = np.concatenate(cols[k])
        if a.shape[1] == 1:   # LeRobot 은 길이 1 feature 를 목록이 아닌 숫자 하나로 읽는다
            arrays.append(pa.array(a[:, 0], pa.float32()))
        else:
            arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(a.ravel(), pa.float32()), a.shape[1]))
        fields.append(k)
    for k, v in meta_cols.items():
        arrays.append(pa.array(np.concatenate(v)))
        fields.append(k)
    dpath = out / "data" / "chunk-000" / "file-000.parquet"
    dpath.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, names=fields), dpath)

    epath = out / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    epath.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ep_rows), epath)
    import pandas as pd   # LeRobot 은 tasks.parquet 의 (pandas) 색인을 과제 문장으로 읽는다

    pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task")).to_parquet(
        out / "meta" / "tasks.parquet")

    stats = {k: _stats(np.concatenate(cols[k])) for k in keys}
    stats["timestamp"] = _stats(np.concatenate(meta_cols["timestamp"])[:, None])
    (out / "meta" / "stats.json").write_text(json.dumps(stats))
    features = {VIDEO_KEY: {"dtype": "video", "shape": [size[1], size[0], 3], "names": ["height", "width", "channels"],
                            "info": {"video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                     "video.height": size[1], "video.width": size[0], "has_audio": False}}}
    for k in keys:
        features[k] = {"dtype": "float32", "shape": [len(NAMES[k])], "names": NAMES[k]}
    for k, dt in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"),
                  ("index", "int64"), ("task_index", "int64")):
        features[k] = {"dtype": dt, "shape": [1], "names": None}
    info = {"codebase_version": "v3.0", "robot_type": robot_type, "total_episodes": len(episodes),
            "total_frames": total, "total_tasks": len(tasks), "chunks_size": 1000, "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 500, "fps": fps, "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features, "datacache": {"level": level, "channels_dropped": sorted(drop),
                                                  "primitives": PRIMITIVES}}   # annotation.primitive 번호 → 이름
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return out


def read_episode_info(dataset: Path) -> List[dict]:
    t = pq.read_table(Path(dataset) / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
    return [{k: (json.loads(r[k]) if r.get(k) else None) for k in INFO_KEYS} | {"episode_index": r["episode_index"]} for r in t]
