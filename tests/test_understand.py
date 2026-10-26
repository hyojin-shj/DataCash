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
