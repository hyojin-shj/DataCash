"""LeRobot v3 4감 내보내기: 채널·등급별 빼기·장면 수 일치·에피소드 정보·인증서."""

import json

import numpy as np
import pyarrow.parquet as pq

from datacache.export.lerobot import read_episode_info
from datacache.pipeline import process


def _run(synth_session, synth_tracks, tmp_path, level):
    d, _ = synth_session
    hand, objs, _ = synth_tracks
    steps = [{"start_s": 6.0, "end_s": 10.0, "instruction": "Carry the cup"}]
    return process(d, tmp_path / level, level=level, hand_points=hand, objects=objs, steps=steps, extract=False,
                   episodes_s=[(4.5, 12.0), (12.0, 19.0)])


def test_bio_dataset_has_all_senses(synth_session, synth_tracks, tmp_path):
    c = _run(synth_session, synth_tracks, tmp_path, "bio")
    ds = tmp_path / "bio" / "lerobot"
    info = json.loads((ds / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["total_episodes"] == 2
    for k in ("observation.touch", "observation.touch.vibration", "observation.physio", "observation.imu.wrist",
              "observation.audio", "observation.objects", "sync.error_ms", "observation.images.ego"):
        assert k in info["features"], k
    t = pq.read_table(ds / "data" / "chunk-000" / "file-000.parquet")
    assert t.num_rows == info["total_frames"]
    touch = np.array(t.column("observation.touch").to_pylist())
    ts = np.array(t.column("timestamp").to_pylist())
    ep = np.array(t.column("episode_index").to_pylist())
    held = (ep == 0) & (ts > 2.0) & (ts < 5.0)                      # 에피소드 0 은 4.5초부터 → 6.5~9.5초
    assert np.median(touch[held, 1]) > 7.0 and touch[held, 2].all()  # 쥐는 힘 8N 근처, 있음
    phys = np.array(t.column("observation.physio").to_pylist())
    assert phys[:, 1].all() and phys[:, 0].max() >= 90
    infos = read_episode_info(ds)
    assert infos[0]["steps"][0]["instruction"] == "Carry the cup" and infos[1]["steps"] is None
    assert infos[0]["sync"]["chosen"] == "beacon"
    assert c["touch"]["calibration"]["resid_n"] < 1.0
    assert (tmp_path / "bio" / "certificate.md").read_text().count("4감 동기화") == 1
    import cv2

    cap = cv2.VideoCapture(str(ds / "videos" / "observation.images.ego" / "chunk-000" / "file-000.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == info["total_frames"]


def test_levels_drop_channels(synth_session, synth_tracks, tmp_path):
    _run(synth_session, synth_tracks, tmp_path, "basic")
    f = json.loads((tmp_path / "basic" / "lerobot" / "meta" / "info.json").read_text())["features"]
    assert "observation.physio" not in f and "observation.imu.wrist" not in f and "observation.touch.vibration" not in f
    assert "observation.touch" in f                                   # 영상 접촉은 기본 등급에도
