# 데이터캐시 3분 데모 영상 — 대본·촬영 지시서

- 길이: **2:58** (제출 한도 3:00 안, 2초 여유)
- 언어: 내레이션은 영어(심사위원용), 화면 자막은 영어 + 한국어 작게
- 만드는 법: 대부분 AI 영상 생성(Veo · Sora · Kling · Runway 등) + **실제 화면 녹화 4컷**
  - 실제 녹화 컷(🎥 표시)은 꼭 넣는다. 심사는 "실제로 돌아가는가"를 본다 — 전부 AI 영상이면 콘셉트 영상으로 보인다.
  - AI 컷(🤖 표시)의 프롬프트는 영어로 그대로 붙여 넣는다. 한 컷 5~8초로 만들어 잘라 쓴다.
- 내레이션 분량: 약 290단어 (분당 120~130단어 속도로 읽으면 2:40 안팎, 나머지는 음악·효과음 여백)

## 원래 대본에서 바꾼 것과 이유

| 원래 | 바꾼 것 | 이유 |
|---|---|---|
| "완벽한 물리적 촉감", 손끝 압력이 팡 터지는 효과 | **"워치의 진동으로 잰 쥐는 세기"** | 손끝 압력 센서는 없다. 실제로 하는 건 워치가 스스로 떨리고 그 떨림이 손에 흡수되는 정도로 쥐는 세기를 재는 것(GripSense 방식). 과장하면 심사 질문에서 깨진다 |
| "얼굴은 모자이크로 숨겨진다" | **"얼굴은 실존하지 않는 사람으로 바뀐다"** | 실제 기능이 모자이크가 아니라 Muse Image 로 만든 가상 인물로 치환이다. 이게 더 강한 장면이다 |
| "you'll get the cache" | **"you get paid"** / 화면에는 **Cash** | cache(저장소)와 cash(돈) 혼동. 이름이 DataCache 라 말장난으로 쓸 수 있다: "Your data. Your cash." |
| 길거리를 걷는 사람들 | **집 안의 일상 동작**(컵 옮기기, 티슈 뽑기, 상자 들기) | 미션이 집에서 하는 로봇용 동작이다 |
| 4장면 | **5장면** (의뢰 → 미션 장면을 따로) | 우리만의 차별점(기업이 말로 의뢰하면 AI가 미션을 거꾸로 설계)이 묻히지 않게 |
| — | **Meta 모델 이름을 화면에** | Meta 해커톤이므로 SAM 3.1 · Muse Spark · Muse Image 를 쓴 장면에서 자막으로 보여 준다 |

## 전체 흐름

| 장면 | 시간 | 핵심 한 줄 |
|---|---|---|
| 1. 훅 | 0:00–0:22 | 로봇 데이터는 비싼 장비가 필요했다 → 잊어라 |
| 2. 센서 | 0:22–0:58 | 폰 + 워치로 보고·듣고·심박·시간 맞추기, 그리고 쥐는 세기 |
| 3. 의뢰 → 미션 | 0:58–1:28 | 기업이 말로 의뢰 → AI가 미션을 만들어 전 세계로 |
| 4. 처리·프라이버시 | 1:28–2:15 | 관절 인식, 얼굴은 가상 인물로, 원본은 지운다 |
| 5. 보상·로봇 | 2:15–2:58 | 띠링 보상 → 데이터가 로봇 손이 된다 → 로고 |

---

## 장면 1 · 훅 (0:00–0:22)

**음악:** 0:00–0:12 낮고 무거운 오케스트라 드론(심장 박동 같은 저음). 0:12 에 완전히 끊긴다(무음 0.5초). 이어서 밝고 미니멀한 비트(애플 광고풍 피아노 + 클랩).

| # | 시간 | 종류 | 화면 | AI 프롬프트 |
|---|---|---|---|---|
| 1-1 | 0:00–0:04 | 🤖 | 어두운 연구실, 모션캡처 슈트를 입은 사람. 몸에 반사 마커 수십 개 | `Cinematic close-up, dark robotics lab, a person wearing a full black motion-capture suit covered with dozens of glowing reflective markers, cold blue rim light, slow dolly-in, shallow depth of field, heavy atmosphere, 24fps film look` |
| 1-2 | 0:04–0:08 | 🤖 | 두꺼운 햅틱 장갑과 케이블 다발 클로즈업 | `Extreme close-up of a bulky haptic data glove with exposed wires and sensors on a hand, cables trailing to a rack of equipment, dramatic side lighting, slow motion, dust particles in the air` |
| 1-3 | 0:08–0:12 | 🤖 | 무거운 VR 헤드셋을 쓴 연구원, 뒤로 서버 랙 | `A researcher wearing a heavy VR headset in a cluttered lab full of server racks and tripods with cameras, wide shot, moody teal and orange grade, camera slowly pulls back to reveal how much equipment surrounds them` |
| 1-4 | 0:12–0:13 | 편집 | **검은 화면 0.5초 + 장비가 한꺼번에 화면 밖으로 휙 빠지는 모션 그래픽** | (편집에서: 1-3 장면을 그대로 두고 장비 이미지만 바깥으로 날려 보낸다) |
| 1-5 | 0:13–0:22 | 🤖 | 밝은 흰 배경. 평범한 사람이 주머니에서 스마트폰을 꺼내고, 손목에 스마트워치. 미니멀 | `Bright minimal white studio, an ordinary person in a casual sweater pulls a smartphone out of their jeans pocket, a plain smartwatch on their wrist, clean product-commercial lighting, slow motion, centered composition, no logos on any device` |

**자막 (1-1~1-3):** `$100,000+ in motion capture gear` → `Weeks of lab time` → `Still not enough data`

**내레이션:**
> (무겁게) "Robots are learning to use their hands. But they're starving — for real human data."
> "Until now, collecting it meant labs, suits, gloves… and a fortune."
> (0:12 무음 뒤, 가볍고 자신 있게) "Forget all of that."
> "What if everything you needed… was already in your pocket?"

(한국어: 로봇은 손을 쓰는 법을 배우고 있습니다. 그런데 진짜 사람 데이터가 너무 부족합니다. 지금까지는 연구실, 슈트, 장갑, 그리고 막대한 비용이 필요했죠. 그 모든 건 잊으세요. 필요한 모든 게 이미 주머니 속에 있다면요?)

---

## 장면 2 · 센서 (0:22–0:58)

**음악:** 비트가 들어온다. 감각이 하나씩 켜질 때마다 짧은 효과음('틱').

| # | 시간 | 종류 | 화면 | AI 프롬프트 · 만드는 법 |
|---|---|---|---|---|
| 2-1 | 0:22–0:30 | 🤖 + 그래픽 | 부엌 식탁, 폰을 거치대에 세우고 컵을 집는 손. 폰 화면에서 **카메라 → 소리 파형 → 심박 그래프 → 시계 동기화** 레이어가 하나씩 떠오른다 | `Cozy home kitchen table in soft morning light, a smartphone standing on a small tripod facing a person, the person wearing a smartwatch reaches for a paper cup and lifts it, natural and candid, medium shot, warm tones, no logos` — 레이어는 편집에서 얹는다(아래 '그래픽' 참고) |
| 2-2 | 0:30–0:36 | 🎥 | **실제 앱 녹화 화면**: 미션 고르기 → 녹화 시작 → 얼굴 확인 초록 동그라미 → 동작 | 폰 화면 녹화. 9:16 을 가운데에 두고 양옆은 흐리게 |
| 2-3 | 0:36–0:46 | 🤖 + 그래픽 | 손목 클로즈업. 워치가 아주 짧게 떨리고(진동 물결 그래픽), 손이 컵을 꽉 쥐면 물결이 손에 흡수되듯 작아진다. 옆에 **쥐는 세기 막대 0 → 1 → 2 → 3** | `Macro close-up of a wrist wearing a plain smartwatch, the hand slowly tightens its grip around a paper cup, knuckles subtly whitening, soft studio light, slow motion, shallow depth of field, no logos` — 진동 물결과 막대는 그래픽 |
| 2-4 | 0:46–0:58 | 그래픽 | 네 감각이 한 줄 타임라인에 나란히 맞춰지는 화면: 영상 · 소리 · 손목 움직임 · 쥐는 세기. 마지막에 `±1 ms` 숫자가 딱 맞춰지며 정지 | 실제 웹 화면(구매 미리보기의 쥐는 세기 타임라인)을 녹화해 써도 된다 🎥 |

**그래픽 레이어 (2-1):** 화면 오른쪽에 얇은 카드 4장이 차례로 — `👁 Video` `🔊 Sound` `❤ Heart rate` `⏱ Time sync`

**자막 (2-3):** `Grip force — measured by the watch's own vibration` / 작게 `Inspired by GripSense (UIST 2012)`

**내레이션:**
> "Just your phone. And the watch on your wrist."
> "Sight. Sound. Heart rate. All locked to the same clock — down to the millisecond."
> "And the part robots need most: how hard you grip."
> "Your watch gently vibrates — and your hand absorbs it. The tighter you hold, the more it fades. No glove. No sensors. Just physics."

(한국어: 당신의 폰, 그리고 손목의 워치. 시각, 소리, 심박이 1천분의 1초까지 같은 시계로 맞춰집니다. 그리고 로봇에게 가장 필요한 것 — 얼마나 세게 쥐는가. 워치가 살짝 떨리고, 손이 그 떨림을 흡수합니다. 꽉 쥘수록 떨림은 사라지죠. 장갑도, 센서도 없이. 물리만으로.)

> ⚠ 워치 쥐는 세기는 실제 워치로 확인한 뒤 촬영한다. 확인 전이면 2-3 은 그래픽으로만, 내레이션은 그대로 둔다(방식 설명이므로).

---

## 장면 3 · 의뢰 → 전 세계 미션 (0:58–1:28)

**음악:** 비트가 한 단계 빨라진다.

| # | 시간 | 종류 | 화면 | AI 프롬프트 · 만드는 법 |
|---|---|---|---|---|
| 3-1 | 0:58–1:10 | 🎥 | **실제 웹 화면**: 기업 담당자가 채팅창에 입력 — "We build warehouse robots. We need box lifting data, ages 20–50, budget $3,000." → Muse Spark 가 미션 2개·참여 조건·센서·견적을 카드로 만들어 낸다 | 웹 `#ask` 채팅 녹화. 타이핑은 빠르게(2배속), 카드가 뜨는 순간은 정상 속도 |
| 3-2 | 1:10–1:16 | 그래픽 | 어두운 지구본 위로 미션 카드가 빛줄기처럼 여러 도시로 날아간다 | `Stylized dark globe at night with thin glowing light trails shooting from one point to dozens of cities across every continent, elegant motion graphics, cinematic` |
| 3-3 | 1:16–1:28 | 🤖 몽타주 | 2초씩 6컷: 서로 다른 나라·나이의 사람들이 집에서 폰을 세우고 컵 옮기기, 티슈 뽑기, 상자 들기, 서랍 열기 | 아래 6개 프롬프트 |

**3-3 몽타주 프롬프트** (모두 끝에 `, phone on a small tripod in the foreground, natural home light, candid documentary style, no logos` 를 붙인다)
1. `A young Nigerian woman in her twenties lifting a cardboard box onto a shelf in a bright apartment`
2. `An elderly Japanese man in his seventies carefully moving a teacup across a wooden table`
3. `A Brazilian father in his forties pulling a tissue from a box on a kitchen counter`
4. `A South Asian teenage-looking college student opening a drawer and taking out a spoon`
5. `A middle-aged white European woman folding a towel on a sofa`
6. `A Middle Eastern man in his thirties pouring water from a bottle into a glass`

**자막 (3-1):** `Powered by Muse Spark` (카드가 뜰 때 오른쪽 아래)
**자막 (3-3):** `Every age. Every country. Every home.`

**내레이션:**
> "Need data for your robot? Just ask — in plain words."
> "Our AI turns your request into missions: the actions, the people, the sensors, the budget."
> "Then it goes live — to everyone, everywhere."
> "Every age. Every country. Every kitchen becomes a robotics lab."

(한국어: 로봇에 필요한 데이터가 있나요? 그냥 말로 요청하세요. AI가 요청을 미션으로 바꿉니다 — 동작, 참여자, 센서, 예산까지. 그리고 전 세계 모두에게 열립니다. 모든 나이, 모든 나라. 모든 부엌이 로봇 연구실이 됩니다.)

---

## 장면 4 · 처리와 프라이버시 (1:28–2:15) — 가장 중요한 장면

**음악:** 0.5초 낮아졌다가(긴장), 얼굴이 바뀌는 순간 화려하게 올라온다.

| # | 시간 | 종류 | 화면 | 만드는 법 |
|---|---|---|---|---|
| 4-1 | 1:28–1:40 | 🎥 | **실제 처리 결과**: 점심 녹화(티슈 뽑기)에 손 21점 관절선과 몸 관절선이 그려지고, 티슈 상자에 SAM 테두리가 붙는다. 배경의 모니터는 회색으로 가려진다 | 처리 결과 영상 + 관절 겹친 영상을 만든다(요청하면 만들어 줌). 자막 `Pose · Hands` `SAM 3.1` |
| 4-2 | 1:40–1:56 | 🎥 | **원본 ↔ 치환 좌우 와이프**: 가운데 세로선이 왼쪽→오른쪽으로 쓸고 지나가면 얼굴만 다른 사람으로 바뀐다. 손·옷·동작은 그대로 | 동의한 본인 영상만 쓴다. 원본은 촬영용으로만 보관 |
| 4-3 | 1:56–2:05 | 그래픽 | 가상 인물 34명의 얼굴 격자(동아시아·아프리카계·유럽계·라틴계·중동·남아시아, 20대~60대)가 빠르게 지나가고, 하나가 골라져 빛난다 | `datacache/donors/` 의 사진을 그대로 쓴다. 자막 `34 people who don't exist — made with Muse Image` |
| 4-4 | 2:05–2:15 | 그래픽 | 원본 영상 파일 아이콘이 휴지통으로 사라지고 `Original deleted ✓`, 판매용 파일만 남는다 | 미니멀 아이콘 애니메이션 |

**내레이션:**
> "Every frame becomes training data — hands, joints, objects, every grip and every release."
> "But what about you?"
> (와이프 순간, 한 박자 쉬고) "Your face is replaced — by someone who doesn't exist."
> "Same motion. Same hands. A different person. Generated by AI, for every skin tone and every age."
> "And your original video? Deleted the moment processing ends."

(한국어: 모든 장면이 학습 데이터가 됩니다 — 손, 관절, 물체, 쥐고 놓는 순간까지. 그런데 당신은요? 당신의 얼굴은 바뀝니다 — 존재하지 않는 사람으로. 같은 움직임, 같은 손, 다른 사람. 모든 피부색과 나이에 맞춰 AI가 만든 얼굴입니다. 원본 영상은요? 처리가 끝나는 순간 지워집니다.)

> ⚠ 4-2 는 원본 얼굴이 나오므로 **본인 동의 영상만**, 영상 설명란에 "출연자 본인 동의" 표기.

---

## 장면 5 · 보상 → 로봇 → 마무리 (2:15–2:58)

| # | 시간 | 종류 | 화면 | AI 프롬프트 · 만드는 법 |
|---|---|---|---|---|
| 5-1 | 2:15–2:25 | 🎥 | **실제 앱 화면**: 홈에 '띠링!' 알림이 연달아 뜨고 적립금 숫자가 올라간다 | 앱 홈 녹화. 띠링 소리를 2~3번 겹쳐 리듬처럼 |
| 5-2 | 2:25–2:40 | 🤖 | 사람 손이 컵을 쥐는 장면 위로 관절 점·선이 겹치고, 같은 선이 그대로 로봇 손으로 옮겨가 로봇이 같은 컵을 부드럽게 쥔다 | `Split-screen transition: on the left, a human hand gently grasps a paper cup with a glowing skeletal overlay of 21 hand keypoints; the glowing skeleton flows to the right and becomes a sleek white robotic hand that grasps an identical cup with the same gentle motion, clean bright lab, cinematic, slow motion` |
| 5-3 | 2:40–2:50 | 🤖 | 집 부엌에서 로봇 팔이 티슈를 뽑아 사람에게 건네는 따뜻한 장면 | `A friendly white robot arm in a warm home kitchen pulls a tissue from a box and hands it to a smiling elderly woman, soft golden hour light, heartwarming, cinematic, shallow depth of field` |
| 5-4 | 2:50–2:58 | 그래픽 | 흰 바탕에 로고 **DataCache**, 아래 한 줄 + Meta 모델 크레딧 | 로고 애니메이션 |

**자막 (5-1):** `Your data. Your cash.`
**마지막 화면 (5-4):**
```
DataCache
Your everyday moves, teaching tomorrow's robots.
Built with SAM 3.1 · Muse Spark · Muse Image · Muse Glimmer
```

**내레이션:**
> "And yes — you get paid. Every mission, straight to your account."
> "Your everyday moves… become a robot's first skills."
> (조용히) "The future of AI is in your hands."
> (로고에서) "DataCache."

(한국어: 그리고 물론, 보상을 받습니다. 미션마다 바로 계좌로. 당신의 평범한 일상 동작이 로봇의 첫 기술이 됩니다. AI의 미래는 당신의 손안에 있습니다. 데이터캐시.)

---

## 만들기 체크리스트

### 실제 화면 녹화 (🎥 4컷 — 이것부터)
- [ ] 2-2 앱: 미션 고르기 → 얼굴 확인(초록 동그라미) → 녹화 (폰 화면 녹화)
- [ ] 3-1 웹: 채팅 의뢰 → 미션 카드·견적 (맥 화면 녹화, 영어 문장으로)
- [ ] 4-1·4-2 처리 결과: 관절 겹친 영상, 원본↔치환 와이프 영상 (데이터캐시에서 만들어 줄 수 있음)
- [ ] 5-1 앱 홈: 띠링 보상 알림

### AI 영상 생성 팁
- 모든 프롬프트 끝에 `no logos, no text` 를 붙인다 — 실제 브랜드(갤럭시·애플 등) 로고가 나오면 쓰지 않는다
- 한 컷을 3~4번 만들어 손가락이 이상하지 않은 것을 고른다 (손이 주인공이라 손가락 수·모양을 꼭 확인)
- 실존 인물·유명인을 닮은 얼굴이 나오면 쓰지 않는다
- 색: 장면 1 은 차갑고 어둡게(청록·남색), 장면 2 부터 밝고 따뜻하게 — 바뀌는 대비가 '반전'을 만든다

### 소리
- 내레이션: 장면 1 앞부분만 무겁게, 0:12 이후는 가볍고 자신 있게 (같은 성우가 톤만 바꿔도 된다)
- 효과음: 감각이 켜질 때 '틱', 미션이 날아갈 때 '휙', 얼굴이 바뀔 때 '샥', 보상 '띠링'
- 음악 저작권: 무료 라이선스 음악만 (YouTube 오디오 보관함 등)

### 길이 맞추기
- 내레이션을 먼저 녹음해 길이를 잰다 → 2:45 를 넘으면 장면 2 의 "Just physics." 와 장면 3 의 마지막 줄을 먼저 뺀다
- 장면 4 는 줄이지 않는다 (차별점: 가상 인물 치환 + 원본 삭제)
