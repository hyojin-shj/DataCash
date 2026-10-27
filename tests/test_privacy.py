"""가리기: 참여자의 두 손 (API 없이, SAM 결과를 흉내 낸 물체로)."""

import numpy as np

from datacache.meta.sam import SamObject
from datacache.privacy.mask import participant_hands


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

    from datacache import config

    if not (config.MODELS / "yunet.onnx").exists() or not (config.MODELS / "face_landmarker.task").exists():
        pytest.skip("얼굴 모델이 없음 (data/models)")
    from datacache.privacy.faceswap import FaceLandmarks, Reid, find_small_faces

    donor = cv2.imread(str(config.ROOT / "datacache" / "donors" / "donor0.png"))
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
    """맥 GPU 빠른 길: 7×7×7 Conv3d 를 2D 합성곱 7번으로 — 값이 같아야 한다 (CPU 에서 확인)."""
    import torch

    from datacache.privacy.mps_speed import conv3d_as_2d

    torch.manual_seed(0)
    conv = torch.nn.Conv3d(6, 4, 7, padding=3)
    x = torch.randn(1, 6, 5, 12, 12)
    with torch.no_grad():
        assert torch.allclose(conv(x), conv3d_as_2d(x, conv.weight, conv.bias, 3), atol=1e-5)
