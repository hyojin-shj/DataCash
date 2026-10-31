# 가상 인물 (치환용)

해커톤 중(2026-10-27) **Muse Image (`muse-image-1.0`)** 로 만든 실존하지 않는 인물 정면 사진 6장 (768px).
프롬프트: 정면 상반신, 무표정, 눈 뜨고 입 다묾, 안경 없음, 고른 조명, 밝은 회색 바탕 — `donor0` 한국 여성 30대, `donor1` 한국 남성 30대,
`donor2` 동아시아 여성 20대, `donor3` 동아시아 남성 50대, `donor4` 여성 40대, `donor5` 남성 20대.
치환할 때 본인과 같은 성별 가운데, 몇 장면을 시험으로 바꿔 봤을 때 알아볼 수 없으면서 가장 닮은 사람을 자동으로 고른다
(`datacash.privacy.faceswap.rank_donors`). 원본 1024px 은 `data/donors/` (git 밖).

**각도 사진 (`views/<이름>/*.jpg`, 2026-11-01):** 사람마다 정면 사진을 Muse Image 편집(`images.edit`)에 넣어 '같은 사람, 고개만 돌림'으로 만든
24장 (좌우 20~90°, 위·아래, 숙임 × 30~80°, 512px JPEG, 34명 약 40MB, $8.2 — `scripts/make_donor_views.py`). 참여자로 알아보는 옆모습을 만들 때
장면 각도에 가장 가까운 사진을 원본으로 쓴다 (`LivePortraitSwap.add_views`). 자세한 평가는 `docs/privacy_eval.md`.
