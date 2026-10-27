"""이의 신청 → 더 높은 설정(Spark 1.3 최대 추론)으로 다시 보기 → 보상·다시 처리·바로잡은 라벨 (모델 호출은 가짜로)."""

import json

from datacash import evolve, judge
from datacash.appeals import Appeals

OLD = [{"action": "reach", "start_s": 0, "end_s": 1}, {"action": "lift", "start_s": 1, "end_s": 2},
       {"action": "release", "start_s": 2, "end_s": 3}]


def _dirs(tmp_path):
    sd, out = tmp_path / "s", tmp_path / "o"
    sd.mkdir()
    out.mkdir()
    (sd / "meta.json").write_text(json.dumps({"mission": {"title": "컵 옮기기"}}))
    (out / "masked_swap.mp4").write_bytes(b"x")
    (out / "certificate.json").write_text(json.dumps({"steps": OLD}))
    return sd, out


def _relabel(steps):
    return lambda video, t0, t1, mission, detail: steps


def test_rejected_recording_reprocessed_when_actions_seen(tmp_path):
    sd, out = _dirs(tmp_path)
    ap = Appeals(tmp_path / "appeals")
    st = {"state": "rejected", "error": "손이 거의 안 보여요"}
    a = ap.file("dev1", "s1", "failed", "손 다 보였어요", st, requeue=lambda s: None, withhold=lambda s: None, review_model=True)
    assert a["state"] == "judging"
    res = judge.judge(a, st, sd, out, relabel=_relabel(OLD), log=lambda *x: None)
    assert res["decision"] == "reprocess"
    assert ap.judged(a["id"], res)["state"] == "reprocessing"
    assert judge.judge(a, st, sd, out, relabel=_relabel([]), log=lambda *x: None)["decision"] == "review"


def test_label_appeal_fixed_rewarded_and_learned(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve, "ROOT", tmp_path / "evolve")
    sd, out = _dirs(tmp_path)
    ap = Appeals(tmp_path / "appeals")
    st = {"state": "done", "mission": "pick_place_cup"}
    a = ap.file("dev1", "s1", "label", "들기가 아니라 당기기예요", st, requeue=lambda s: None, withhold=lambda s: None, review_model=True)
    new = [dict(OLD[0]), {"action": "pull", "start_s": 1.1, "end_s": 2.0, "instruction": "pull the cup"}, dict(OLD[2])]
    res = judge.judge(a, st, sd, out, relabel=_relabel(new), log=lambda *x: None)
    assert res["decision"] == "accept" and res["extra"] == judge.LABEL_FIX_BONUS
    assert res["corrections"][0]["was"] == "lift" and res["corrections"][0]["action"] == "pull"
    done = ap.judged(a["id"], res)
    assert done["state"] == "accepted" and done["extra_reward"] == judge.LABEL_FIX_BONUS
    evolve.add_correction("s1", res["corrections"], "pick_place_cup")
    assert evolve.corrections("s1")[0]["action"] == "pull"
    assert '"lift" was actually "pull"' in evolve.correction_hints()
    assert judge.judge(a, st, sd, out, relabel=_relabel(OLD), log=lambda *x: None)["decision"] == "review"
