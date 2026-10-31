"""가리기: 참여자의 두 손 (API 없이, SAM 결과를 흉내 낸 물체로)."""

import cv2
import numpy as np

from datacash.meta.sam import SamObject
from datacash.privacy.mask import participant_hands


def _obj(oid, box, shape=(240, 320)):
    m = np.zeros(shape, bool)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = True
    return SamObject(oid, box, 0.9, m)


def _tips(cx, cy):
    return np.array([[cx + dx, cy + dy] for dx, dy in ((-8, 0), (-4, -6), (0, -8), (4, -6), (8, -2))], np.float32)


def test_both_hands_belong_to_participant():
    """SAM 이 한쪽 팔만 참여자로 잡아도, 그 옆의 다른 손도 참여자 손이고 멀리 있는 남의 손은 아니다."""
    right_arm = _obj("0", (180, 120, 260, 240))
    mine_in = _tips(220, 140)            # 참여자 영역 안
    mine_near = _tips(140, 150)          # 다른 팔: 참여자 상자 바로 옆
    stranger = _tips(20, 20)             # 화면 반대쪽 다른 사람 손
    hs = participant_hands(right_arm, [mine_in, mine_near, stranger])
    assert len(hs) == 2 and any(np.allclose(h, mine_near) for h in hs)
    assert not any(np.allclose(h, stranger) for h in hs)


def test_first_person_without_tracked_body_keeps_all_hands():
    assert len(participant_hands(None, [_tips(50, 200), _tips(250, 200)])) == 2


def test_small_face_found_for_swap():
    """먼 장면의 작은 얼굴(MediaPipe 가 놓치는 크기)도 YuNet + 키워서 다시 찾기로 478점을 얻는다."""
    import cv2
    import pytest

    from datacash import config

    if not (config.MODELS / "yunet.onnx").exists() or not (config.MODELS / "face_landmarker.task").exists():
        pytest.skip("얼굴 모델이 없음 (data/models)")
    from datacash.privacy.faceswap import FaceLandmarks, Reid, find_small_faces

    donor = cv2.imread(str(config.ROOT / "datacash" / "donors" / "donor0.png"))
    fl, reid = FaceLandmarks(), Reid()
    big = fl(donor)[0]
    k = 40 / np.ptp(big[:, 1])                          # 얼굴 높이 40px 로 줄인다
    small = cv2.resize(donor, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    frame = np.full((540, 960, 3), 235, np.uint8)
    y0, x0 = 120, 600
    frame[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    assert not fl(frame)                                # 원래 방식으로는 못 찾는다
    found = find_small_faces(frame, fl, reid)
    assert len(found) == 1
    expect = big * k + np.array([x0, y0])
    assert np.abs(found[0].mean(0) - expect.mean(0)).max() < 4    # 원래 좌표로 맞게 되돌린다


def test_mps_conv3d_as_2d_same():
    """맥 GPU 빠른 길: 7×7×7 Conv3d 를 2D 합성곱 7번으로 — 값이 같아야 한다 (CPU 에서 확인).
    torch 는 따로 떨어진 프로세스에서 — 같은 프로세스의 OpenCV·MediaPipe 스레드와 엉켜 멈춘 적이 있다."""
    import subprocess
    import sys

    code = ("import torch; from datacash.privacy.mps_speed import conv3d_as_2d; torch.manual_seed(0); "
            "c = torch.nn.Conv3d(6, 4, 7, padding=3); x = torch.randn(1, 6, 5, 12, 12); "
            "torch.no_grad().__enter__(); assert torch.allclose(c(x), conv3d_as_2d(x, c.weight, c.bias, 3), atol=1e-5)")
    assert subprocess.run([sys.executable, "-c", code], timeout=120).returncode == 0


def test_track_crop_follows_face_and_flips_count():
    """놓친 장면에 붙일 자리: 앞 장면의 얼굴 점을 따라가 자르기를 같이 옮긴다. 끊김은 반 넘게 바뀐 장면 사이만 센다."""
    from datacash.privacy.faceswap import count_flips, face_points, track_crop

    rng = np.random.default_rng(0)
    g0 = cv2.GaussianBlur((rng.random((360, 480)) * 255).astype(np.uint8), (5, 5), 0)
    g1 = np.roll(g0, (6, 9), axis=(0, 1))                       # 얼굴이 오른쪽 9px · 아래 6px 움직임
    M = np.array([[0.4, 0, 140], [0, 0.4, 80]], np.float32)     # 512 자르기 → 장면 (205px 얼굴)
    M1, pts = track_crop(g0, g1, face_points(g0, M), M)
    assert np.abs(M1[:, 2] - M[:, 2] - [9, 6]).max() < 1 and np.allclose(M1[:, :2], M[:, :2], atol=0.02) and len(pts) >= 8
    assert track_crop(g0, np.zeros_like(g0), face_points(g0, M), M) is None   # 따라갈 게 없으면 잇지 않는다
    seq = [(1, False), (1, False), (0, False), (1, False), None, None, (1, False), (0.8, False), (0.4, False), (0, False), (0, True), (0, 0.9), (0, 0.45), (0, 0.0)]
    assert count_flips(seq) == 3                                 # 뚝 바뀜 2번 + 흐림 시작 1번 (서서히 뺀 것·서서히 끈 흐림·얼굴 없음은 세지 않음)


def test_face_values_gaze_and_head():
    """표정·시선 값: 눈동자가 안쪽 꼬리로 가면 x>0, 아래로 가면 y>0 — 고개를 기울여도 같은 값. 고개 방향은 변환 행렬에서."""
    from datacash.export.channels import NAMES, face_rows
    from datacash.labels.pose import FACE_VALUES, FrameLabels, gaze, head_angles

    def eyes(shift, roll=0.0):
        p = np.zeros((478, 2), np.float32)
        for c, outer, inner, up, low, x0 in ((468, 33, 133, 159, 145, 100), (473, 263, 362, 386, 374, 300)):
            d = 1 if x0 == 100 else -1                       # 오른눈: 바깥 꼬리가 왼쪽, 왼눈: 바깥 꼬리가 오른쪽
            p[outer], p[inner] = [x0 - 30 * d, 200], [x0 + 30 * d, 200]
            p[up], p[low] = [x0, 190], [x0, 210]
            p[c] = [x0 + shift[0] * 30 * d, 200 + shift[1] * 10]
        c, s = np.cos(roll), np.sin(roll)
        return (p - 200) @ np.array([[c, -s], [s, c]]).T + 200

    g = gaze(eyes((0.5, 0.4)))
    assert np.allclose(g, [0.5, 0.4, 0.5, 0.4], atol=1e-3)
    assert np.allclose(gaze(eyes((0.5, 0.4), roll=0.4)), g, atol=1e-3)
    yaw = np.radians(30)
    M = np.eye(4)
    M[:3, :3] = [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    assert np.allclose(head_angles(M), (30, 0, 0), atol=1e-6)
    f = FrameLabels(face=np.arange(len(FACE_VALUES), dtype=np.float32))
    F = face_rows([f, FrameLabels()], np.array([0, 1, 1]))
    assert F.shape == (3, len(NAMES["observation.face"])) and F[0, -1] == 1 and F[1, -1] == 0 and F[0, 25] == 25


def test_match_texture_blurs_and_adds_grain():
    """매끈한 만든 얼굴을 흐리고 거친 장면에 맞춘다 — 선명도·입자가 장면 쪽으로 가까워진다."""
    from datacash.privacy.faceswap import _lap_var, _noise, match_texture

    rng = np.random.default_rng(0)
    base = cv2.resize((rng.random((24, 24)) * 255).astype(np.uint8), (200, 200), interpolation=cv2.INTER_CUBIC)
    face = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)                                  # 선명하고 깨끗한 얼굴
    scene = cv2.GaussianBlur(face, (0, 0), 2.0).astype(np.float32) + rng.normal(0, 6, (200, 200, 1))
    scene = np.clip(scene, 0, 255).astype(np.uint8)                               # 번지고 거친 장면
    mw = np.zeros((200, 200), np.float32)
    mw[40:160, 40:160] = 1
    sel = mw > 0.5
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    out = match_texture(face, scene, mw, rng)
    assert abs(_noise(g(out), sel) - _noise(g(scene), sel)) < abs(_noise(g(face), sel) - _noise(g(scene), sel))
    assert abs(np.log(_lap_var(g(out).astype(np.float32), sel) / _lap_var(g(scene).astype(np.float32), sel))) < \
        abs(np.log(_lap_var(g(face).astype(np.float32), sel) / _lap_var(g(scene).astype(np.float32), sel)))


def test_view_pick_nearest_with_margin():
    """장면 고개 각도에 가까운 각도 사진을 원본으로 — 조금 더 가까운 정도로는 바꾸지 않는다 (깜빡임 방지)."""
    from datacash.privacy.faceswap import VIEW_MARGIN, LivePortraitSwap

    lp = object.__new__(LivePortraitSwap)
    lp.s_info, lp.s_feat, lp.s_kp = {"yaw": 0.0, "pitch": 0.0}, "front", "kp0"
    lp.views = [(40.0, 0.0, {"yaw": 40.0}, "v40", "kp40"), (60.0, 20.0, {"yaw": 60.0}, "v60", "kp60")]
    lp.cur = None
    d0 = {"yaw": 0.0, "pitch": 0.0}
    assert lp._pick({"yaw": 2.0, "pitch": 0.0}, d0)[1] == "front"
    assert lp._pick({"yaw": 38.0, "pitch": 0.0}, d0)[1] == "v40"
    assert lp._pick({"yaw": 50.0, "pitch": 10.0}, d0)[1] == "v40"      # 60 쪽이 조금 더 가깝지만 VIEW_MARGIN 안
    assert lp._pick({"yaw": 62.0, "pitch": 22.0}, d0)[1] == "v60"
    assert VIEW_MARGIN > 0


def test_open_run_filter_drops_short_profile_swaps():
    """'원래대로' 사이에 짧게 낀 옆모습 치환은 되돌리고, 충분히 이어진 것은 둔다. 장면 순서·수는 그대로."""
    from datacash.privacy.faceswap import OPEN_MIN_RUN, OpenRunFilter, SwapReport

    rep = SwapReport()
    out = []
    f = OpenRunFilter(lambda fr, tv: out.append((fr, tv)), rep)
    kinds = ["orig"] * 3 + ["open"] * 2 + ["orig"] * 3 + ["open"] * (OPEN_MIN_RUN + 1) + ["orig"] * 2
    rep.swapped = rep.profile_swapped = kinds.count("open")
    for k, kind in enumerate(kinds):
        f.push(f"s{k}" if kind == "open" else f"o{k}", f"o{k}" if kind == "open" else None, kind, (1.0, 0.0) if kind == "open" else (0.0, 0.0))
    f.flush()
    assert [fr for fr, _ in out][:8] == [f"o{k}" for k in range(8)]          # 짧은 2장면은 되돌림
    assert all(fr.startswith("s") for fr, _ in out[8:8 + OPEN_MIN_RUN + 1])   # 긴 것은 그대로
    assert len(out) == len(kinds) and rep.kept_why == {"short": 2} and rep.swapped == OPEN_MIN_RUN + 1
