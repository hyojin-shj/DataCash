"""단계 라벨 채점: 같으면 만점, 경계가 밀리면 경계 오차, 동작이 틀리면 F1·검증 정밀도가 떨어진다."""

from datacash.understand.score import edit_score, pooled, score

GOLD = [{"start_s": 0.0, "end_s": 1.0, "action": "reach"}, {"start_s": 1.0, "end_s": 1.5, "action": "grasp"},
        {"start_s": 1.5, "end_s": 3.0, "action": "lift"}, {"start_s": 3.0, "end_s": 4.0, "action": "place"}]


def test_identical_is_perfect():
    r = score(GOLD, [{**s, "verified": True} for s in GOLD])
    assert r["frame_acc"] == 1.0 and r["f1@50"] == 1.0 and r["edit"] == 100.0
    assert r["boundary_ms_median"] == 0 and r["verified_precision"] == 1.0


def test_shifted_boundaries():
    pred = [{**s, "start_s": s["start_s"] + 0.2, "end_s": s["end_s"] + 0.2} for s in GOLD]
    r = score(GOLD, pred)
    assert r["boundary_ms_median"] == 200 and r["edit"] == 100.0 and 0.75 < r["frame_acc"] < 0.95


def test_wrong_action_caught_even_if_verified():
    pred = [dict(s) for s in GOLD]
    pred[3].update(action="release", verified=True)     # 검증 규칙은 통과했지만 동작이 틀림
    pred[0]["verified"] = True
    r = score(GOLD, pred)
    assert r["f1@50"] == 0.75 and r["verified_precision"] == 0.5
    assert edit_score(GOLD, pred) == 75.0


def test_merge_coarse_gold():
    coarse = [{"start_s": 0.0, "end_s": 3.0, "action": "take"}, {"start_s": 3.0, "end_s": 4.0, "action": "put"}]
    r = score(coarse, GOLD, merge={"reach": "take", "grasp": "take", "lift": "take", "place": "put"})
    assert r["frame_acc"] == 1.0 and r["edit"] == 100.0
    assert pooled([r, score(GOLD, GOLD)])["gold_steps"] == 6
