# SAM 3.1 가리기 평가 (2026-10-27, 공개 영상)

라이선스가 확인된 Wikimedia Commons 영상만 썼다 (아래 출처). 영상마다 8초를 640px·15fps 로 잘라 명사구별로 SAM 3.1 영상 요청을 보냈다.
재현: `python scripts/eval_privacy.py` (결과 data/eval/privacy/report.json, 나란히 그림 *_sheet.jpg — 저장소 밖).

| 영상 | 사람 (참여자 외) | 민감 물체 | 가린 넓이 평균 | 손 유지 | 가린 뒤 다시 찾은 다른 사람 |
|---|---|---|---|---|---|
| home_kitchen_wide | 0명 | - | 0.0% | 0.0% | 0/6 |
| gloves_dishwashing | 0명 | - | 0.0% | 0.0% | 2/6 |
| food_shop | 4명 | - | 2.45% | 100.0% | 1/6 |
| game_lounge_screens | 0명 | television screen 3, computer monitor 3 | 23.27% | 0.0% | 0/6 |
| mirror_bathroom | 9명 | mirror 4 | 31.03% | 75.0% | 2/6 |

읽은 것

- 화면(TV 3·모니터 3, 게임 라운지)과 거울(욕실 거울 4, 거울 속 사람 반사)은 장면 100% 에서 가려졌다. 참여자 손·몸은 남는다.
- 음식점: 5명 중 참여자 1명만 남기고 가렸고, 원본에서 찾은 손은 가린 뒤에도 100% 다시 찾아진다.
- '가린 뒤 다시 찾은 다른 사람'은 표본 6장면에 SAM 'person' 을 다시 돌려 참여자와 겹치지 않는 사람이 나온 장면 수. 거울 영상의 2/6 은 참여자 자신이 거울 가장자리에 비친 조각이고, 고무장갑 설거지(등장인물 1명)의 2/6 은 참여자의 팔이 따로 '사람'으로 잡힌 것이다 → 재검사 지표를 '참여자 상자에 대부분 들어가면 참여자'(겹침 비율)로 바꿀 것.
- 아이·액자·서류가 나오는 공개 영상은 아직 없다 → 해커톤 중 직접 찍은 거실 녹화로 잰다.
- 앱 에뮬레이터 녹화(가상 거실): TV 화면이 656장면 모두 가려짐 (가린 넓이 11%).

이 평가까지 누적 API 비용 $2.049809

## 출처

- home_kitchen_wide: [File:(Yum-Cast) Our Daily Life! Let's clean and do the dishes!.webm](https://commons.wikimedia.org/wiki/File:(Yum-Cast)_Our_Daily_Life!_Let%27s_clean_and_do_the_dishes!.webm) · 윰댕(yum-cast) · CC BY 3.0
- gloves_dishwashing: [File:(Yum-Cast) Our Daily Life! Let's clean and do the dishes!.webm](https://commons.wikimedia.org/wiki/File:(Yum-Cast)_Our_Daily_Life!_Let%27s_clean_and_do_the_dishes!.webm) · 윰댕(yum-cast) · CC BY 3.0
- food_shop: [File:Dishes you can't miss in Xi'an.webm](https://commons.wikimedia.org/wiki/File:Dishes_you_can%27t_miss_in_Xi%27an.webm) · China News Service · CC BY 3.0
- game_lounge_screens: [File:2023-03-31 Retro Games Lounge.webm](https://commons.wikimedia.org/wiki/File:2023-03-31_Retro_Games_Lounge.webm) · Computerarchiv-Muenchen.de · CC BY-SA 4.0
- mirror_bathroom: [File:Man in hotel bathroom with mirrors all around, turning and making video thereof.webm](https://commons.wikimedia.org/wiki/File:Man_in_hotel_bathroom_with_mirrors_all_around,_turning_and_making_video_thereof.webm) · Pittigrilli · CC BY-SA 4.0
