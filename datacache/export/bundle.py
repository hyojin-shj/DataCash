"""구매 묶음 — 여러 녹화의 LeRobot 데이터셋을 하나로 합치고, 산 등급에 없는 채널을 뺀다.

영상은 다시 압축하지 않고 이어 붙인다 (크기가 다르면 첫 데이터셋 크기로 맞춰 다시 압축). 에피소드 번호·index 를 새로 매기고,
녹화별 품질 인증서를 certificates/ 에 함께 넣는다.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..media import run
from .channels import LEVEL_DROP
from .lerobot import VIDEO_KEY, _stats

META_COLS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def merge(sources: List[Path], out_dir, level: str) -> Path:
    """sources: 녹화별 out 폴더 (안에 lerobot/·certificate.*)."""
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    drop = LEVEL_DROP[level]
    infos = [json.loads((Path(s) / "lerobot" / "meta" / "info.json").read_text()) for s in sources]
    common = set.intersection(*[set(i["features"]) for i in infos]) - drop
    fps = infos[0]["fps"]
    tasks: List[str] = []
    data_tables, ep_rows, videos = [], [], []
    total, ep_i = 0, 0
    for s, info in zip(sources, infos):
        root = Path(s) / "lerobot"
        t = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
        src_tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
        tmap = {}
        for task, row in src_tasks.iterrows():
            if task not in tasks:
                tasks.append(task)
            tmap[int(row["task_index"])] = tasks.index(task)
        eps = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
        keep = [c for c in t.column_names if c in common or c in META_COLS]
        t = t.select(keep)
        n = t.num_rows
        ep_old = np.array(t.column("episode_index").to_pylist())
        remap = {e["episode_index"]: ep_i + k for k, e in enumerate(eps)}
        cols = {c: t.column(c) for c in keep}
        cols["episode_index"] = pa.array([remap[int(e)] for e in ep_old], pa.int64())
        cols["index"] = pa.array(np.arange(total, total + n), pa.int64())
        cols["task_index"] = pa.array([tmap[int(v)] for v in t.column("task_index").to_pylist()], pa.int64())
        data_tables.append(pa.table(cols))
        for k, e in enumerate(eps):
            r = {kk: v for kk, v in e.items() if not (kk.startswith("stats/") and kk.split("/")[1] not in common)}
            if level != "bio":
                r["human_state"] = ""
            off = total - e["dataset_from_index"]
            r.update({"episode_index": ep_i + k, "dataset_from_index": e["dataset_from_index"] + off,
                      "dataset_to_index": e["dataset_to_index"] + off,
                      f"videos/{VIDEO_KEY}/from_timestamp": (e["dataset_from_index"] + off) / fps,
                      f"videos/{VIDEO_KEY}/to_timestamp": (e["dataset_to_index"] + off) / fps,
                      "source_session": Path(s).name})
            ep_rows.append(r)
        videos.append(root / "videos" / VIDEO_KEY / "chunk-000" / "file-000.mp4")
        total += n
        ep_i += len(eps)
        cert = Path(s) / "certificate.md"
        if cert.exists():
            (out / "certificates").mkdir(parents=True, exist_ok=True)
            shutil.copy(cert, out / "certificates" / f"{Path(s).name}.md")
    table = pa.concat_tables(data_tables, promote_options="default")
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "data" / "chunk-000" / "file-000.parquet")
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ep_rows), out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    import pandas as pd

    pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task")).to_parquet(out / "meta" / "tasks.parquet")
    vdir = out / "videos" / VIDEO_KEY / "chunk-000"
    vdir.mkdir(parents=True, exist_ok=True)
    lst = out / "concat.txt"
    lst.write_text("".join(f"file '{Path(v).resolve()}'\n" for v in videos))
    run(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", vdir / "file-000.mp4"])
    lst.unlink()
    stats = {}
    for c in table.column_names:
        if c in common:
            a = np.array(table.column(c).to_pylist(), np.float64)
            stats[c] = _stats(a.reshape(len(a), -1))
    (out / "meta" / "stats.json").write_text(json.dumps(stats))
    info = dict(infos[0])
    info["features"] = {k: v for k, v in infos[0]["features"].items() if k in common or k in META_COLS}
    info.update(total_episodes=ep_i, total_frames=total, total_tasks=len(tasks), splits={"train": f"0:{ep_i}"},
                datacache={"level": level, "channels_dropped": sorted(drop), "sources": [Path(s).name for s in sources]})
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return out
