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
