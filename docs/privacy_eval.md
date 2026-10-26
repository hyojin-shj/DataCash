# SAM 3.1 가리기 평가 (2026-10-27, 공개 영상)

라이선스가 확인된 Wikimedia Commons 영상만 썼다 (아래 출처). 영상마다 8초를 640px·15fps 로 잘라 명사구별로 SAM 3.1 영상 요청을 보냈다.
재현: `python scripts/eval_privacy.py` (결과 data/eval/privacy/report.json, 나란히 그림 *_sheet.jpg — 저장소 밖).

| 영상 | 사람 (참여자 외) | 민감 물체 | 가린 넓이 평균 | 손 유지 | 가린 뒤 다시 찾은 다른 사람 | 작은 검출 (<40px) |
|---|---|---|---|---|---|---|
| home_kitchen_wide | 0명 | - | 0.0% | 0.0% | 0/6 | - |
| gloves_dishwashing | 0명 | - | 0.0% | 0.0% (고무장갑이라 손 검출 안 됨) | 0/6 | 2/6 |
| food_shop | 4명 | - | 2.27% | 100.0% | 0/6 | 0/6 |
| game_lounge_screens | 0명 | television screen 3, computer monitor 3 | 23.25% | 0.0% | 0/6 | - |
| mirror_bathroom | 9명 | mirror 4 | 30.59% | 87.5% | 0/6 | - |

읽은 것

- 화면(TV 3·모니터 3, 게임 라운지)과 거울(욕실 거울 4, 거울 속 사람 반사)은 장면 100% 에서 가려졌다. 참여자 손·몸은 남는다.
- 음식점: 5명 중 참여자 1명만 남기고 가렸고, 원본에서 찾은 손은 가린 뒤에도 100% 다시 찾아진다.
- '가린 뒤 다시 찾은 다른 사람'은 표본 6장면에 SAM 'person' 을 다시 돌려, 참여자가 아닌 사람이 보이는 장면 수다.
  - (10-27 오전 수정) 아래 검출은 참여자로 본다.
    - 참여자 상자와 겹치는 검출 (IoU≥0.3).
    - 대부분 참여자 상자 안에 든 검출 (≥60%, 참여자 팔·손).
    - 참여자 상자를 절반 이상 품은 검출 (손만 추적된 참여자의 몸 전체).
  - 절반 이상 회색인 검출은 이미 가린 실루엣으로 본다.
  - 키 40px 미만 검출은 얼굴이 몇 픽셀이라 알아볼 수 없어서 '작은 검출'로 따로 센다.
  - 판정마다 상자·신뢰도는 report.json 의 `redetect_detail` 에 남긴다.
- 가리기 쪽 개선 (`datacache/privacy/mask.py`): 장면마다 참여자의 손(손끝이 참여자 영역 안이거나 상자 가까이)을 찾는다.
  - 참여자 상자를 두 손 모두로 넓힌다.
  - SAM 이 참여자의 다른 팔을 별개 '사람'으로 잡아도 가리지 않는다 (`participant_parts_kept`: 음식점 3, 거울 2).
  - 1인칭이라 참여자 몸이 추적되지 않으면 모든 손을 참여자 손으로 본다.
  - 테스트: `tests/test_privacy.py`.
- 지표 바꾸기 전 남았던 검출을 하나씩 확인했다.
  - 거울 2/6: 참여자 자신의 반사 조각과 가린 실루엣이었다.
  - 음식점 1/6: 참여자(빨간 옷 요리사)는 손만 추적됐고, 재검사가 같은 사람의 몸 전체를 잡은 것이었다. 가린 영상에서도 요리사는 그대로 보인다 — 맞는 동작.
  - 고무장갑 2/6: 머그컵에 인쇄된 사람 그림(15×30px, 신뢰도 0.54~0.57)이었다 → '작은 검출'로 옮김.
  - 실제로 노출된 다른 사람은 없었다.
- 한계
  - 고무장갑을 끼면 MediaPipe 가 손을 못 찾는다 → 손 기준 판단은 SAM 참여자 추적에만 기댄다.
  - '품은 검출' 규칙은 참여자 손 바로 뒤에 선 다른 사람을 참여자로 볼 수 있다 → 실제 거실 녹화로 다시 잴 것.
- 아이·액자·서류가 나오는 공개 영상은 아직 없다 → 해커톤 중 직접 찍은 거실 녹화로 잰다.
- 앱 에뮬레이터 녹화(가상 거실): TV 화면이 656장면 모두 가려짐 (가린 넓이 11%).

이 평가까지 누적 API 비용 $5.64 (10-27 오전 재평가 포함)

## 출처

- home_kitchen_wide: [File:(Yum-Cast) Our Daily Life! Let's clean and do the dishes!.webm](https://commons.wikimedia.org/wiki/File:(Yum-Cast)_Our_Daily_Life!_Let%27s_clean_and_do_the_dishes!.webm) · 윰댕(yum-cast) · CC BY 3.0
- gloves_dishwashing: [File:(Yum-Cast) Our Daily Life! Let's clean and do the dishes!.webm](https://commons.wikimedia.org/wiki/File:(Yum-Cast)_Our_Daily_Life!_Let%27s_clean_and_do_the_dishes!.webm) · 윰댕(yum-cast) · CC BY 3.0
- food_shop: [File:Dishes you can't miss in Xi'an.webm](https://commons.wikimedia.org/wiki/File:Dishes_you_can%27t_miss_in_Xi%27an.webm) · China News Service · CC BY 3.0
- game_lounge_screens: [File:2023-03-31 Retro Games Lounge.webm](https://commons.wikimedia.org/wiki/File:2023-03-31_Retro_Games_Lounge.webm) · Computerarchiv-Muenchen.de · CC BY-SA 4.0
- mirror_bathroom: [File:Man in hotel bathroom with mirrors all around, turning and making video thereof.webm](https://commons.wikimedia.org/wiki/File:Man_in_hotel_bathroom_with_mirrors_all_around,_turning_and_making_video_thereof.webm) · Pittigrilli · CC BY-SA 4.0

## 얼굴 치환 감도 (10-27 오후)

공개 요리 영상(960×540, 192장면, 먼 장면과 가까운 장면이 섞임)으로 쟀다.

| | 치환 | 작은 얼굴 치환 | 모자이크 | 얼굴 없음 | 재식별 막음 |
|---|---|---|---|---|---|
| 전 (MediaPipe 만) | 106 | 0 | 86 | 86 | 0 |
| 후 (YuNet → 둘레 키워 478점) | **183** | 81 | 4 | 4 | 5 |

- **원인:** 먼 장면의 얼굴(높이 약 40px)을 MediaPipe 얼굴 478점이 못 찾아서, 치환 대신 모자이크로 덮고 있었다.
- **고침:**
  - YuNet 으로 작은 얼굴 상자를 먼저 찾는다.
  - 상자 3배 둘레를 360px 로 키워 478점을 찾고, 원래 좌표로 되돌려 같은 방식으로 치환한다 (`find_small_faces`).
  - 테스트 `test_small_face_found_for_swap`: 40px 얼굴을 원래 방식으로는 못 찾고 새 방식으로 찾는다.
- **재식별 막음 5장면:** 크게 돌린 옆얼굴이다.
  - 가상 얼굴이 원래 사람과 너무 닮게 나와 검사가 막았다 (최대 0.675).
  - 예전엔 이런 장면을 회색 네모로 덮었는데, 지금은 얼굴 둘레를 굵은 모자이크(가로 7칸)로 덮어 덜 튄다.
- **동의한 가족도 치환한다:**
  - 본인·가족 알아보기는 SFace 0.45 이상이고, 다음으로 닮은 등록자보다 0.05 높아야 한다.
  - OpenCV 권장 0.363 보다 엄격하다. 저장소의 가상 인물 6명끼리도 0.33~0.60 이 나와서다.
  - 가족마다 서로 다른 가상 인물을 쓴다.
