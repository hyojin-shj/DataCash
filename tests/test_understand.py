"""Muse Spark 라벨 검증: 맞는 단계는 감각이 받쳐 주고, 지어낸 단계는 걸러진다 (API 호출 없음)."""

from datacash.pipeline import process
from datacash.understand.verify import check_mistakes


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

    from datacash.privacy.speech import mute

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

    from datacash.understand.spark import PRIMITIVES

    d, _ = synth_session
    hand, objs, _ = synth_tracks
    steps = [{"start_s": 5.0, "end_s": 6.0, "instruction": "Reach for the cup", "hand": "right", "objects": ["cup"], "action": "reach"},
             {"start_s": 6.0, "end_s": 10.0, "instruction": "Grasp the cup", "hand": "right", "objects": ["cup"], "action": "grasp"}]
    process(d, tmp_path / "o", hand_points=hand, objects=objs, steps=steps, extract=False)
    t = pq.read_table(tmp_path / "o" / "lerobot" / "data" / "chunk-000" / "file-000.parquet")
    p = np.array(t.column("annotation.primitive").to_pylist(), dtype=float).ravel()
    ts = np.array(t.column("timestamp").to_pylist())
    names = json.loads((tmp_path / "o" / "lerobot" / "meta" / "info.json").read_text())["datacash"]["primitives"]
    assert names == PRIMITIVES
    assert names[int(p[np.argmin(abs(ts - 5.5))])] == "reach" and names[int(p[np.argmin(abs(ts - 8.0))])] == "grasp"
    assert p[np.argmin(abs(ts - 2.0))] == -1


def test_missions_use_known_primitives():
    """미션 22개(회사용 10개 포함)의 예상 순서가 Spark 라벨 단위 안에 있고, 앱 안의 미션 목록이 서버와 같다."""
    import json
    from pathlib import Path

    from datacash.understand.spark import PRIMITIVES

    root = Path(__file__).parents[1]
    srv = json.loads((root / "datacash" / "missions.json").read_text())
    app = json.loads((root / "app" / "app" / "src" / "main" / "assets" / "missions.json").read_text())
    assert srv == app and len(srv["missions"]) == 22
    assert set(srv["primitives_ko"]) == set(PRIMITIVES)
    for m in srv["missions"]:
        assert m["primitives"][0] == "reach" and m["primitives"][-1] == "release" and set(m["primitives"]) <= set(PRIMITIVES)


def test_review_answer_counts_as_evidence():
    """2차 검증(Spark Open)이 같은 동작이라고 본 것이 단계 검증의 증거 하나가 된다. 바로잡은 단계는 증거에서 뺀다."""
    import numpy as np

    from datacash.understand.verify import verify_steps

    t = np.arange(0, 10, 0.1)
    steps = [{"start_s": 1, "end_s": 3, "action": "other", "spark_open": {"agree": True}},
             {"start_s": 4, "end_s": 6, "action": "other", "spark_open": {"agree": False}},
             {"start_s": 7, "end_s": 9, "action": "other", "spark_open": {"agree": False}, "relabeled_from": "grasp"}]
    v = verify_steps(steps, t)
    assert v[0]["evidence"] == {"review": True} and v[0]["verified"] is True
    assert v[1]["verified"] is False and v[2]["verified"] == "unverifiable"


def test_second_opinion_relabels_and_feeds_examples(tmp_path, monkeypatch):
    """2차 검증: 다르게 본 단계는 Spark Open 동작으로 바로잡고(바로잡음 = 어려운 예시), 같게 본 순서는 다음 질문의 정답 예시가 된다."""
    from datacash import evolve
    from datacash.understand import spark_open

    monkeypatch.setattr(evolve, "ROOT", tmp_path)
    monkeypatch.setattr(spark_open, "available", lambda: True)
    so = [{"start_s": 0.0, "end_s": 1.0, "action": "reach"}, {"start_s": 1.0, "end_s": 2.0, "action": "grasp"},
          {"start_s": 2.0, "end_s": 3.0, "action": "tear"}, {"start_s": 3.0, "end_s": 4.0, "action": "release"}]
    monkeypatch.setattr(spark_open, "label_frames", lambda *a, **k: {"steps": so, "frames": [0.1 * i for i in range(40)],
                                                                    "model": "muse-spark-open (test)"})
    steps = [{"start_s": 0.0, "end_s": 1.0, "action": "reach", "verified": True},
             {"start_s": 1.0, "end_s": 2.0, "action": "grasp", "verified": True},
             {"start_s": 2.0, "end_s": 3.0, "action": "pull", "verified": True},
             {"start_s": 3.0, "end_s": 4.0, "action": "release", "verified": True}]
    asked = []
    monkeypatch.setattr(spark_open, "adjudicate",                  # 모든 단계를 촘촘히 다시 봄: 셋째 단계는 tear
                        lambda v, d, *a, **k: asked.extend(d) or ["reach", "grasp", "tear", "release"])
    r = evolve.second_opinion("20261029_120000", "v.mp4", 0.0, "Pull a tissue", "pull_tissue", steps, log=lambda *_: None)
    assert len(asked) == 4 and asked[2]["options"] == ["pull", "tear"] and asked[0]["options"] == ["reach", "reach"]
    assert r["agree"] == 3 and r["disputed"] == 1 and r["relabeled"] == 1 and r["gold"] == 3
    assert steps[2]["action"] == "tear" and steps[2]["relabeled_from"] == "pull"
    assert evolve.corrections("20261029_120000")[0]["source"] == "review"
    assert "reach 1.0s -> grasp 1.0s" in evolve.gold_examples("pull_tissue") and evolve.gold_examples("twist_cap") == ""
    assert "was actually \"tear\"" in evolve.correction_hints()
    assert evolve._registry()["agreement"][-1]["rate"] == 0.75
    monkeypatch.setattr(spark_open, "adjudicate", lambda *a, **k: ["reach", "place", "pull", "release"])   # 다시 보니 1.3 이 맞으면 유지
    steps[2].update(action="pull", relabeled_from=None)
    r = evolve.second_opinion("20261029_120001", "v.mp4", 0.0, "Pull a tissue", "pull_tissue", steps, log=lambda *_: None)
    assert r["upheld"] == 1 and r["relabeled"] == 0 and steps[2]["action"] == "pull" and steps[2]["spark_open"]["agree"]
    assert not steps[1]["spark_open"]["agree"] and r["gold"] == 3      # 두 모델은 같았지만 촘촘히 보니 달라 정답에서 뺌


def test_spark_open_packs_frames_for_image_limit():
    """Baseten 은 이미지 8장까지 — 장면 16장은 2장씩 나란히 붙여 8장으로, 위치와 시각을 글로 알려 준다."""
    import numpy as np

    from datacash.understand.spark_open import _frame_parts

    frames = [(0.5 * k, np.zeros((60, 80, 3), np.uint8)) for k in range(16)]
    full, packed = _frame_parts(frames, None), _frame_parts(frames, 8)
    imgs = lambda p: [x for x in p if x["type"] == "image_url"]  # noqa: E731
    assert len(imgs(full)) == 16 and len(imgs(packed)) == 8
    assert "left t=0.00s, right t=0.50s" in packed[0]["text"]
