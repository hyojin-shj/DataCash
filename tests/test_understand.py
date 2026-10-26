"""Muse Spark 라벨 검증: 맞는 단계는 감각이 받쳐 주고, 지어낸 단계는 걸러진다 (API 호출 없음)."""

from datacache.pipeline import process
from datacache.understand.verify import check_mistakes


def test_true_steps_verified_fake_step_rejected(synth_session, synth_tracks, tmp_path):
    d, _ = synth_session
    hand, objs, _ = synth_tracks
    steps = [
        {"start_s": 6.0, "end_s": 7.0, "instruction": "Grasp the cup", "hand": "right", "objects": ["cup"], "action": "grasp"},
        {"start_s": 10.0, "end_s": 11.0, "instruction": "Place the cup", "hand": "right", "objects": ["cup"], "action": "place"},
        # 지어낸 단계: 16~17초에는 아무것도 안 잡고 아무 소리도 없다
        {"start_s": 16.0, "end_s": 17.0, "instruction": "Place a plate", "hand": "right", "objects": ["plate"], "action": "place"},
    ]
    c = process(d, tmp_path / "o", hand_points=hand, objects=objs, steps=steps, extract=False)
    v = {x["instruction"]: x for x in c["steps"]}
    assert v["Grasp the cup"]["verified"] is True and v["Grasp the cup"]["evidence"]["contact"]
    assert v["Place the cup"]["verified"] is True and v["Place the cup"]["evidence"]["sound"]
    assert v["Place a plate"]["verified"] is False


def test_mistake_confirmed_by_self_mark():
    m = check_mistakes([{"t_s": 8.6, "description": "cup slips", "recovered": True},
                        {"t_s": 15.0, "description": "spill", "recovered": False}], [8.0])
    assert m[0]["confirmed_by_participant"] and not m[1]["confirmed_by_participant"]


def test_mute_only_personal_spans():
    import numpy as np

    from datacache.privacy.speech import mute

    rate = 16000
    x = np.ones(rate * 4, np.float32)
    y = mute(x, rate, [(1000, 1500)])
    assert y[int(0.5 * rate)] == 1.0 and y[int(1.2 * rate)] == 0.0 and y[int(3 * rate)] == 1.0
    assert 0.0 <= y[int(0.86 * rate)] <= 1.0                      # 페이드


def test_primitive_channel_in_lerobot(synth_session, synth_tracks, tmp_path):
    """Spark 단계의 기본 동작이 장면마다 annotation.primitive 로 들어가고, 번호표가 info.json 에 있다."""
    import json

    import numpy as np
    import pyarrow.parquet as pq

    from datacache.understand.spark import PRIMITIVES

    d, _ = synth_session
    hand, objs, _ = synth_tracks
    steps = [{"start_s": 5.0, "end_s": 6.0, "instruction": "Reach for the cup", "hand": "right", "objects": ["cup"], "action": "reach"},
             {"start_s": 6.0, "end_s": 10.0, "instruction": "Grasp the cup", "hand": "right", "objects": ["cup"], "action": "grasp"}]
    process(d, tmp_path / "o", hand_points=hand, objects=objs, steps=steps, extract=False)
    t = pq.read_table(tmp_path / "o" / "lerobot" / "data" / "chunk-000" / "file-000.parquet")
    p = np.array(t.column("annotation.primitive").to_pylist(), dtype=float).ravel()
    ts = np.array(t.column("timestamp").to_pylist())
    names = json.loads((tmp_path / "o" / "lerobot" / "meta" / "info.json").read_text())["datacache"]["primitives"]
    assert names == PRIMITIVES
    assert names[int(p[np.argmin(abs(ts - 5.5))])] == "reach" and names[int(p[np.argmin(abs(ts - 8.0))])] == "grasp"
    assert p[np.argmin(abs(ts - 2.0))] == -1


def test_missions_use_known_primitives():
    """미션 12개의 예상 순서가 Spark 라벨 단위 안에 있고, 앱 안의 미션 목록이 서버와 같다."""
    import json
    from pathlib import Path

    from datacache.understand.spark import PRIMITIVES

    root = Path(__file__).parents[1]
    srv = json.loads((root / "datacache" / "missions.json").read_text())
    app = json.loads((root / "app" / "app" / "src" / "main" / "assets" / "missions.json").read_text())
    assert srv == app and len(srv["missions"]) == 12
    assert set(srv["primitives_ko"]) == set(PRIMITIVES)
    for m in srv["missions"]:
        assert m["primitives"][0] == "reach" and m["primitives"][-1] == "release" and set(m["primitives"]) <= set(PRIMITIVES)
