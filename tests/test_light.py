"""어두운 영상 밝히기 · 얼굴 확인이 안 끝난 녹화 · 동작 구간이 없을 때 '어두워요' 로 잘못 반려하지 않기."""

import numpy as np


def test_lift_brightens_only_dark():
    from datacash import light

    dark = np.full((90, 160, 3), 30, np.uint8)
    bright = np.full((90, 160, 3), 140, np.uint8)
    assert light.gain_for(light.luma(bright)) == 1.0 and light.lift(bright) is bright
    assert 100 < light.luma(light.lift(dark)) < 140
    assert light.gain_for(2.0) == light.MAX_GAIN


def test_face_check_never_finished_starts_from_zero():
    from datacash.pipeline import face_check_end_s

    t0 = 1_000_000_000
    stuck = {"face_check": {"passed": False, "end_ns": t0 + int(23.9e9)}}
    assert face_check_end_s(stuck, t0, 23.5) == 0.0                       # 1단계에서 멈춤 → 처음부터 동작으로
    assert face_check_end_s(stuck, t0) > 23                              # 영상 길이를 안 주면 예전처럼
    ok = {"face_check": {"passed": True, "end_ns": t0 + int(2e9)}}
    assert abs(face_check_end_s(ok, t0, 20.0) - 2.3) < 1e-6


def test_precheck_no_action_frames_is_not_dark(tmp_path):
    import cv2

    from datacash.precheck import precheck

    p = str(tmp_path / "v.mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 90))
    for _ in range(30):
        w.write(np.full((90, 160, 3), 120, np.uint8))
    w.release()
    r = precheck(p, None, action_start_s=5.0)
    assert not r["ok"] and r["luma"] is None and not any("어두" in x for x in r["reasons"])
