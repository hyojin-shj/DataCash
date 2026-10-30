"""경계 다듬기: 손이 멈춘 순간을 찾고, 가까운 경계만 그 순간으로 옮긴다 (동작 이름·순서는 그대로)."""

import numpy as np

from datacash.understand.refine import hand_events, snap

STEPS = [{"start_s": 0.0, "end_s": 1.0, "action": "reach"}, {"start_s": 1.0, "end_s": 2.0, "action": "grasp"},
         {"start_s": 2.0, "end_s": 3.0, "action": "lift"}]


def test_snap_moves_only_near_boundaries():
    out = snap(STEPS, [1.3, 2.9], tol=0.4)
    assert out[1]["start_s"] == 1.3 and out[0]["end_s"] == 1.3      # 0.3초 안 → 옮김 (앞 단계 끝도 같이)
    assert out[2]["start_s"] == 2.0                                  # 0.9초 떨어짐 → 그대로
    assert [s["action"] for s in out] == ["reach", "grasp", "lift"]


def test_snap_keeps_order():
    out = snap(STEPS, [2.05], tol=0.4)                               # 1.0 경계의 가장 가까운 사건은 2.05 → 뒤 단계 끝을 넘어 무시
    assert out[1]["start_s"] == 1.0 and out[2]["start_s"] == 2.05


def test_hand_pause_found():
    fps, n = 30.0, 90
    hands = []
    for i in range(n):
        t = i / fps
        x = 100 + 200 * t if t < 1.0 else (300 if t < 1.5 else 300 + 200 * (t - 1.5))   # 1.0~1.5초에 멈춘다
        pts = np.zeros((21, 3), np.float32)
        pts[:, 0], pts[:, 1] = x, 200
        hands.append([None, pts])
    ev = hand_events(hands, fps, 640)
    assert ev and all(0.9 <= t <= 1.6 for t in ev)
