from datacash.understand.fuse import fuse


def test_fuse_majority_and_switch():
    a = [{"start_s": 0, "end_s": 1, "action": "grasp"}, {"start_s": 1, "end_s": 2, "action": "place"}]
    b = [{"start_s": 0, "end_s": 1.2, "action": "grasp"}, {"start_s": 1.2, "end_s": 2, "action": "place"}]
    c = [{"start_s": 0, "end_s": 0.4, "action": "grasp"}, {"start_s": 0.4, "end_s": 0.5, "action": "push"},
         {"start_s": 0.5, "end_s": 1.1, "action": "grasp"}, {"start_s": 1.1, "end_s": 2, "action": "place"}]
    out = fuse([a, b, c], 0, 2, switch=2.0)
    assert [s["action"] for s in out] == ["grasp", "place"]          # 한 벌에만 있는 짧은 push 는 사라진다
    assert abs(out[0]["end_s"] - 1.1) < 0.11                          # 경계는 가운데쯤
    assert out[0]["agree"] > 0.9
