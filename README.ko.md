[English](README.md) · **한국어** · [Español](README.es.md) · [中文](README.zh-CN.md) · [日本語](README.ja.md)

<a href="https://youtu.be/gw3gqoHlFvA"><img src="docs/demo_video.jpg" width="49%" alt="DataCash 영상 (2:35)"></a> <a href="https://youtu.be/vClUIqrZjBY"><img src="docs/demo_3min.ko.jpg" width="49%" alt="DataCash 3분 데모"></a>

[소개 영상 (2:35)](https://youtu.be/gw3gqoHlFvA) · 3분 데모: [English](https://youtu.be/o1sGUME0GHw) · [한국어](https://youtu.be/vClUIqrZjBY)

# DataCash (데이터캐시)

**Pysical AI Data 프로젝트**
**집에 이미 있는 폰과 워치만으로, 로봇이 배울 사람의 손 데이터를 모은다.**

사람들은 집에서 로봇 기본 동작 미션을 폰과 워치로 찍고 보상을 받습니다.
휴머노이드·피지컬 AI 기업들은 **5감 데이터 — 시각·청각·촉각·움직임·생체 ** — 를 한 시간축에 맞춘
**LeRobot v3.0** 형식 그대로 구매할 수 있습니다.

Meta Global AI Developer Hackathon (Indie Developer)

![DataCash](docs/1026_datacash_brand.png)

---

## 핵심 기술

### 1. 가상인물 치환
- **얼굴 치환:** 블러 대신, 참여자 얼굴을 Muse Image 로 만든 가상 인물로 바꿉니다(LivePortrait + MediaPipe). 같은 성별·비슷한 나이에서 가장 닮은 사람을 고르되, 20장면을 시험으로 바꿔 보고 얼굴 인식(SFace)이 '같은 사람'으로 보는 장면이 절반을 넘으면 다음 사람으로 치환합니다.
- **연속성:** 바꾼 얼굴이 너무 닮은 장면은 모자이크 대신 표정을 줄여 다시 만듭니다.
- **가리기:** SAM 3.1 로 데이터 수집에 동의하지 않은 사람·아이·화면·거울·액자·서류는 모자이크 처리합니다.
- **원본 삭제:** 서비스에서는 처리가 끝나면 원본 영상·소리를 바로 지웁니다. 해커톤 시연 서버는 원본↔치환 비교를 보여 주려고 이 호출만 꺼 두었습니다.

#### 얼굴 치환이 발전해 온 과정

| 날짜 | 바꾼 것 | 결과 |
|---|---|---|
| 10-27 | 블러 대신 가상 인물로 치환 (LivePortrait, Muse Image 가상 인물 34명), 장면마다 재식별 검사 | 얼굴은 남고 사람만 바뀜 · 387장면 중 모자이크 3 |
| 10-27 | 속도: 맥 GPU 에서 느린 Conv3d 를 2D 7번으로 나눔 + fp16 | 녹화 하나 636 → 84초 |
| 11-01 | 옆모습: 같은 가상 인물의 각도 사진 24장 (Muse Image 편집) → 장면 각도에 맞는 사진을 원본으로 | 숙인 고개·옆얼굴이 덜 일그러짐 |
| 11-01 | 가상 인물 고르기: 같은 성별·나이대·무리 안에서 후보마다 시험 장면을 실제로 바꿔 보고, 참여자와 가장 덜 닮은 사람 (동아시아 가상 인물 36명 추가, $9) | 엄격한 재식별로 알아보는 장면 3.3 / 11.1 / 17.3 / 30.9% → 2.4 / 6.4 / 1.7 / 5.0% |

![가상 인물 4명의 여러 각도 사진 (정면 + 8장)](docs/faceswap_views.jpg)
![정면 사진 한 장으로 만든 옆모습·숙인 고개 vs 각도가 가까운 사진으로 만든 같은 장면](docs/faceswap_angle_source.ko.png)
![전: 사진이 가장 닮은 사람 / 후: 같은 성별·나이대·무리에서 시험해 가장 덜 닮은 사람, 알아보는 장면 비율](docs/faceswap_donor_pick.ko.png)
![새 가상 인물과 알아보는 장면 전·후](docs/faceswap_more_donors.ko.png)

### 2. 장갑 없는 촉각 — GripSense 방식
Goel·Wobbrock·Patel, *GripSense* (UIST 2012)를 촉각 측정 기술로 이용하였습니다. 
워치가 짧게 떨리면, 세게 쥘수록 손이 떨림을 흡수해 자이로 고주파가 줄고 손가락이 버티는 4~12Hz 떨림이 커지는 원리를 이용했습니다.
<img width="560" height="369" alt="스크린샷 2026-11-02 오전 9 32 36" src="https://github.com/user-attachments/assets/cf02f9f1-a91b-472a-8073-cc072cd7790a" />

Galaxy Watch4 실측 (IMU 최대 100Hz, 개인 보정 녹화 3편, 1회차로 학습 → 2회차로 평가, 반대로도 교차 검증):
- 분류: 랜덤 포레스트(트리 25그루) + 2초 평활화로 4단계(안 쥠·가볍게·보통·꽉) 정확도 59% (무작위 25%, '꽉' 93%)
- 회귀: 영상이 손-물체 접촉을 감지한 구간에서는 회귀 포레스트(트리 50그루)가 쥐는 세기를 0~100 연속 점수로 추정 (예: "중 (57)")
  → 하·중·상 3구간으로 나누면 83%, 평균 절대 오차(MAE) 15점
워치 심박·박동 간격·피부온도는 Samsung Health Sensor SDK 로 (`docs/samsung_health_sdk.md`).

![처음에 한 번 세기 맞추기 → 나만의 모델 → 모든 녹화에 쥐는 세기 0~100; 배운 적 없는 회차에서 가볍게 97% 하, 평소 56% 중, 꽉 95% 상으로 맞힘](docs/grip_quality.ko.png)


### 3. 5감 데이터 동기화
18~20kHz 음향 비콘(고주파음)으로 폰↔워치 시계 왕복 시간을 측정하여 기기간 데이터의 싱크를 맞췄습니다.
실제 폰+워치 녹화 37편 중 24편을 음향 비콘으로 맞추고(짝 잔차 0.1~1.9ms), 
비콘보다 24.7ms(±3.5) 치우쳐 있는 경우, 기기의 서버 경유 시계로 데이터의 싱크를 맞췄습니다. 

![워치와 영상의 시간 어긋남: 보정 전 24.7ms, 손뼉 동기화 최대 16.7ms, 지금 3.7ms (판매 기준 10ms)](docs/sync_quality.ko.png)

### 4. Spark 1.3 라벨링 정확도 향상

구간별, 동작별 라벨링 정확도 향상을 위해 
1. 5번 라벨링 후 비교 검증,
2. 2.Spark Open Weight816B로 재검증,
3. 3.사람 라벨링 검증
4. Spark를 이용한 미션별 라벨 자체 생성을 통해 라벨링 정확도를 36% → 54%(+18%p)로 향상시켰습니다.
Muse Wonderful을 이용한 정확도 재검증을 고려하였으나, 지역(한국 지원 불가)로 인하여 오픈웨이트 모델 사용했습니다.

### 5. Glimmer를 활용한 API 절감
라벨링을 매번 Muse Spark API에 맡기지 않도록, 오픈웨이트 Muse Glimmer 30B를 우리 데이터로 학습시켰습니다(Fireworks LoRA).
Spark가 라벨 초안을 만들고, 사람이 확인·수정한 단계를 정답으로 삼아 Glimmer가 배웁니다.
학습 전 41.8% → 확인된 정답으로 학습 54.1% → 미션별 동작 순서를 이용한 풀이 추가 64.1%.
향후 데이터가 쌓일수록 Glimmer가 판단을 넘겨받아 Spark의 API 호출을 줄여 사업 이익률을 높일 수 있습니다.

### 6. SAM 물체 판단 비용 절감
- 쓸 수 없는 녹화(너무 짧음·손이 안 보임·어두움)는 **API 를 부르기 전에** 이 맥에서 반려시킵니다.
- 이 맥의 물체 찾기로 먼저 훑고, SAM 에는 필요한 것만 (혼자 찍으면 '사람'은 안 물음, 화면은 보일 때만, 정적인 물건은 초당 1장면) 판단을 시킵니다.
- 실제 녹화 4편: SAM 비용 **$2.92 → $0.27 (−91%)**, 영상 1시간에 약 $5~7 (판매가 60,000원/시간)
- SAM 마스크 해독을 Numba 로 — 처리 시간 593초 → 8초

## LeRobot 판매 데이터 (LeRobot v3.0)

<img width="892" height="786" alt="스크린샷 2026-11-02 오전 9 59 35" src="https://github.com/user-attachments/assets/8131397c-a152-4891-8100-d7066f90d2fa" />

`observation.images.ego` (가린·치환 영상) · `observation.hands` · `observation.body` · `observation.face` · `observation.imu.phone` · `observation.imu.wrist` ·
`observation.audio` · **`observation.touch`** (접촉·쥐는 세기 하·중·상 + 0~100 점수) · `observation.touch.vibration` · **`observation.physio`** (심박·피부온도, 생체 동의만) ·
`observation.objects` · `annotation.step` · `annotation.primitive` (reach·grasp·…·release) · `sync.error_ms` + 품질 인증서.
실제 `lerobot` 0.6.1 로 읽힙니다 (`scripts/validate_lerobot.py`).

센서는 묶음으로 골라 삽니다 — 영상(필수) 35,000원 · 관절 20,000 · 소리 5,000 · 손목 IMU 15,000 · 촉각 25,000 · 생체 50,000 (시간당, `datacash/pricing.py`).

---

## 구조

**전체 흐름** — 기업이 말로 의뢰하면 미션이 열리고, 참여자가 폰·워치로 찍어 올리면 서버가 처리해 데이터로 납품하고 보상합니다.

![전체 구조](docs/architecture.ko.svg)

**녹화 한 편의 처리 순서** (`datacash/jobs.py`) — 돈이 들지 않는 단계를 먼저 해서, 쓸 수 없는 녹화는 Meta 모델을 부르기 전에 반려합니다.

![처리 순서](docs/pipeline.ko.svg)

**기술 스택과 배포** — 지금은 Docker·클라우드 없이 맥 한 대가 서버와 처리를 모두 맡고, 앱·워치는 같은 와이파이에서 mDNS 로 서버를 찾아 직접 올립니다.

![기술 스택](docs/tech_stack.ko.svg)

## 실행

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" torch pyyaml rich
echo "MODEL_API_KEY=..." > .env                       # 저장소에 올리지 않는다
# data/models 에 MediaPipe hand/pose/face landmarker .task, efficientdet_lite0.tflite, OpenCV yunet.onnx·sface.onnx
.venv/bin/python -m pytest                            # 정답을 아는 합성 5감 세션으로 검증
DATACASH_TOKEN=... .venv/bin/python -m datacash.server.app --port 8800   # 서버 + 구매 기업 웹
cd app && ./gradlew assembleDebug                     # 안드로이드 앱 + 워치 앱
```

## 폴더

| 폴더 | 하는 일 |
|---|---|
| `app/` | 안드로이드 앱(동의·미션·얼굴 확인·녹화·실수 표시·질문·업로드·보상·이의 신청) + 워치 앱 |
| `web/` | 구매 기업 웹(말로 의뢰·5감 타임라인·견적·주문·내려받기), 운영자 화면 |
| `datacash/privacy` | SAM 가리기 · 얼굴 치환 · 소리 개인정보 · 미리 훑기 |
| `datacash/sync` · `touch` · `labels` | 5감 동기화 · 촉각 · 손·몸 관절 |
| `datacash/understand` | Muse Spark 이해 · Spark Open 2차 검증 · 감각으로 검증 |
| `datacash/export` · `jobs.py` | LeRobot 내보내기 · 인증서 · 처리 순서 |
| `datacash/server` | 수집 서버 + 구매 기업 API |


## Meta 모델 이용

| 모델 | 쓰임 | 코드 |
|---|---|---|
| **SAM 3.1** | 동의 안 한 사람·아이·화면·거울·액자·서류 가리기, 다루는 물체 및 동작 인식|
| **Muse Spark 1.3** | 가린 영상 → 단계·지시문·쓰는 손·물체·실수 라벨링, 기업의 말 의뢰 → 미션 설계 →미션별 라벨 생성|
| **Muse Image** | 실존하지 않는 가상 인물 34명 마스크 생성, 얼굴 치환용, 사람마다 같은 사람의 각도 사진 24장 | 
| **Muse Voice Transcribe** | 녹화 소리에서 개인정보가 들린 구간 끄기 |
| **Muse Spark Open** |녹화의 2차 검증과 이의 신청을 다시 보는 '더 높은 모델'. 두 모델이 다르게 본 단계는 바로잡고, 같게 보고 센서로 검증된 단계는 정답 예시가 되어 다음 녹화의 Spark 1.3 질문에 붙어요 | 
| **Glimmer** | Spark 1.3, 오픈웨이트 모델, 사람 검증 라벨을 통해 영상 인식 학습을 진행해요. 데이터가 모일수록 Glimmer의 성능이 개선되어 API 의존도를 낮출 수 있어요.|


## 측정한 성능

| 무엇 | 처음 → 지금 | 잰 곳 |
|---|---|---|
| 단계 라벨 장면 정확도 (Muse Spark 1.3) | **35.6% → 54.3%** | EPIC-KITCHENS-100 새 8구간, 사람 정답 271단계 |
| 단계 경계 오차 (중앙값) | **430 → 182ms** | 같음 |
| Muse Glimmer 30B 단계 정확도 (LoRA, Fireworks) | **41.8% → 54.1% → 64.1%** (v80 사람 정답으로 학습 · v82 + 미션별 순서 디코딩, 범위 60.7–67.2% · Spark 라벨로 학습한 v79 는 32.8%) | 학습에 안 쓴 녹화 8편의 사람 정답 122단계 |
| 워치 ↔ 영상 시간 오차 | **24.7 → 3.7ms** (기준 10ms) | 폰 + 워치 녹화 37편 |
| 녹화 하나 얼굴 치환 시간 | **636 → 84초** (7.5배) | `docs/privacy_eval.md` |
| SAM 3.1 비용, 녹화 4편 | **$2.92 → $0.27** (−91%) | 실제 녹화 4편 |
| 쥐는 세기 하·중·상 | **83%** (우연 33%) | 보정 녹화 3편, 갤럭시 워치4 |

![단계 라벨 정확도](docs/perf/label_accuracy.ko.png)
![혼동 행렬](docs/perf/confusion.ko.png)
![Glimmer 학습](docs/perf/glimmer.ko.png)
![서비스 모니터링](docs/perf/monitoring.ko.png)

서비스 모니터링 (운영자 화면과 같은 값, `datacash/perf.py`): 녹화마다 Spark 1.3 ↔ Spark Open 일치율과 최근 5편 평균 (50% 아래면 경고), 녹화 하나 처리 시간 (중앙값 2배 넘으면 경고), 얼굴 치환 뒤에도 같은 사람으로 보이는 장면 (SFace 0.363, 5% 넘으면 경고 — 그 녹화는 검토 대상).

---

## 해커톤 전 / 해커톤 중

| | 내용 |
|---|---|
| 해커톤 전 (10-26 이전) | 계획 문서와 브랜드 이미지만. 코드 없음 |
| 해커톤 중 (한국 10-27 01:00 부터) | 이 저장소의 모든 코드 · 앱 · 웹 · Muse Image 가상 인물 · 문서 |

## 라이선스·출처

MediaPipe (Apache 2.0) · OpenCV YuNet·SFace (Apache 2.0) · LivePortrait (MIT) · LeRobot (Apache 2.0) · Numba (BSD).
시험 영상은 Wikimedia Commons (CC BY·CC BY-SA·퍼블릭 도메인만, 출처는 `docs/privacy_eval.md`)
Samsung Health Sensor SDK
