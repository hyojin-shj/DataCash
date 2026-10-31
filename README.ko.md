[English](README.md) · **한국어** · [Español](README.es.md) · [中文](README.zh-CN.md) · [日本語](README.ja.md)

# DataCash (데이터캐시)

**집에 이미 있는 폰과 워치만으로, 로봇이 배울 사람의 손 데이터를 모은다.**

사람들은 집에서 로봇 기본 동작 미션(티슈 뽑기·컵 옮기기·뚜껑 돌리기 등 12가지)을 폰과 워치로 찍고 보상을 받습니다.
휴머노이드·피지컬 AI 회사는 개인정보가 지워진 **5감 데이터 — 시각·청각·촉각(쥐는 세기)·움직임·생체(심박)** — 를 한 시간축에 맞춘
**LeRobot v3.0** 형식 그대로 삽니다.

Meta Global AI Developer Hackathon (Indie Developer · Push the Frontier) 출품작.

![DataCash](docs/1026_datacash_brand.png)

## 한눈에

| | |
|---|---|
| 찍는 사람 | 안드로이드 앱 + 워치 앱. 미션 고르기 → 얼굴 확인 → 녹화 → 질문 1~2개 → 보상 '띠링' |
| 사는 기업 | 웹에서 **말로 의뢰** ("창고 로봇용 상자 들기 데이터, 20~50대, 예산 300만 원") → AI가 미션을 설계해 전 세계에 연다 |
| 개인정보 | 참여자 얼굴은 **실존하지 않는 사람**으로 바꾸고, 동의 안 한 사람·화면·서류는 가리고, 원본은 처리가 끝나면 지운다 |
| 촉각 | 장갑 없이 — **워치가 스스로 떨리고, 손이 그 떨림을 흡수하는 정도**로 쥐는 세기를 잰다 (GripSense, UIST 2012) |

## Meta 모델을 어디에

| 모델 | 쓰임 | 코드 |
|---|---|---|
| **SAM 3.1** | 동의 안 한 사람·아이·화면·거울·액자·서류 가리기(영상 추적), 다루는 물체 영역 → `observation.objects` | `datacash/meta/sam.py`, `datacash/privacy/mask.py` |
| **Muse Spark 1.3** | 가린 영상 → 단계·지시문·쓰는 손·물체·실수 (구조화 JSON), 기업의 말 의뢰 → 미션 설계 | `datacash/understand/`, `datacash/requests.py` |
| **Muse Image** | 실존하지 않는 가상 인물 34명 (7개 지역 × 남녀 × 20대~60대) — 얼굴 치환용 | `scripts/make_donors.py`, `datacash/donors/` |
| **Muse Voice Transcribe** | 녹화 소리에서 개인정보가 들린 구간 끄기 | `datacash/privacy/speech.py` |
| **Muse Spark Open** (오픈 웨이트, 816B) | 모든 녹화의 2차 검증과 이의 신청을 다시 보는 '더 높은 모델' (Baseten, 안 되면 Fireworks). 두 모델이 다르게 본 단계는 바로잡고, 같게 보고 센서로 검증된 단계는 정답 예시가 되어 다음 녹화의 Spark 1.3 질문에 붙어요 — 가중치를 바꾸지 않고 쓸수록 정확해져요 | `datacash/understand/spark_open.py`, `datacash/evolve.py` ([docs/model_evolution.md](docs/model_evolution.md)) |

API 호출마다 비용을 장부(`data/api_ledger.jsonl`)에 적고, 한도를 넘으면 그 단계만 건너뜁니다.

## 핵심 기술

### 1. 가치를 지키는 개인정보 처리
- **얼굴 치환:** 블러 대신, 참여자 얼굴을 Muse Image 로 만든 가상 인물로 바꿉니다(LivePortrait + MediaPipe). 같은 성별·비슷한 나이에서 가장 닮은 사람을 고르되, 20장면을 시험으로 바꿔 보고 얼굴 인식(SFace)이 '같은 사람'으로 보는 장면이 절반을 넘으면 다음 사람으로.
- **연속성:** 바꾼 얼굴이 너무 닮은 장면은 모자이크 대신 표정을 줄여 다시 만듭니다 — 영상이 끊기지 않습니다.
- **가리기:** SAM 3.1 로 동의 안 한 사람·아이·화면·거울·액자·서류만. 동의한 가족은 가리지 않고 사람마다 다른 가상 얼굴로.
- **원본 삭제:** 서비스에서는 처리가 끝나면 원본 영상·소리를 바로 지웁니다 (`purge_originals`). 해커톤 시연 서버는 원본↔치환 비교를 보여 주려고 이 호출만 꺼 두었습니다.

### 2. 5감 자동 동기화
사용자가 손뼉 칠 필요 없이: 폰↔워치 시계 왕복, 18~20kHz 음향 비콘, 워치 진동 펄스. 실제 폰+워치 녹화 37편 중 24편을 음향 비콘으로 맞추고(짝 잔차 0.1~1.9ms), 나머지는 서버 경유 시계로 — 이 시계는 비콘보다 24.7ms(±3.5) 치우쳐 있어 기종 쌍마다 재서 뺍니다(한 편씩 빼고 검증해 최대 9.2ms). 측정은 `docs/sync_eval.md`, 코드는 `datacash/sync/`.

### 3. 장갑 없는 촉각 — GripSense 방식
Goel·Wobbrock·Patel, *GripSense* (UIST 2012)를 손목에 옮겼습니다. 워치가 짧게 떨리면, 세게 쥘수록 손이 떨림을 흡수해 자이로 고주파가 줄고 손가락이 버티는 4~12Hz 떨림이 커집니다. 사람마다 저울 없이 스스로 구분한 세기(놓기·가볍게·평소·꽉)로 학습한 결정 트리가 쥐는 세기 단계 0~3을 냅니다 (`datacash/touch/`). 갤럭시 워치4 실측(IMU 최대 100Hz, 보정 1회, 1회차로 배워 2회차를 맞히고 반대로도): 트리 25그루 + 2초 묶기로 4단계 59%(찍기 25%, 꽉 100%), 영상이 손-물건 닿음을 본 동안은 회귀 트리 50그루가 세기를 0~100 점수로 내고(예: "중 (57)"), 하·중·상으로 끊으면 88%, 점수 평균 오차 15점입니다.
워치 심박·박동 간격·피부온도는 Samsung Health Sensor SDK 로 (`docs/samsung_health_sdk.md`).

### 4. 말로 의뢰 → 미션
구매 기업이 채팅으로 쓰면 Muse Spark 가 구조화 출력으로 미션(기본 동작·물체·바꿀 조건)·참여 조건(나이·성별)·센서 묶음을 설계하고, 서버가 예산에 맞춰 견적을 냅니다. 확정하면 조건에 맞는 사람에게만 앱에 미션이 열립니다 — 의뢰 조건이 곧 수집 분포입니다.

### 5. 돈이 남는 처리 비용
- 쓸 수 없는 녹화(너무 짧음·손이 안 보임·어두움)는 **API 를 부르기 전에** 이 맥에서 반려 — 비용 0
- 이 맥의 물체 찾기로 먼저 훑고, SAM 에는 필요한 것만 (혼자 찍으면 '사람'은 안 물음, 화면은 보일 때만, 정적인 물건은 초당 1장면)
- 실제 녹화 4편: SAM 비용 **$2.92 → $0.27 (−91%)**, 영상 1시간에 약 $5~7 (판매가 60,000원/시간)
- SAM 마스크 해독을 Numba 로 — 처리 시간 593초 → 8초 (공식 파서와 비트 단위로 같음)

## 판매 데이터 (LeRobot v3.0)

`observation.images.ego` (가린·치환 영상) · `observation.hands` · `observation.body` · `observation.imu.phone` · `observation.imu.wrist` ·
`observation.audio` · **`observation.touch`** (접촉·쥐는 세기 하·중·상 + 0~100 점수) · `observation.touch.vibration` · **`observation.physio`** (심박·피부온도, 생체 동의만) ·
`observation.objects` · `annotation.step` · `annotation.primitive` (reach·grasp·…·release) · `sync.error_ms` + 품질 인증서.
실제 `lerobot` 0.6.1 로 읽힙니다 (`scripts/validate_lerobot.py`).

센서는 묶음으로 골라 삽니다 — 영상(필수) 35,000원 · 관절 20,000 · 소리 5,000 · 손목 IMU 15,000 · 촉각 25,000 · 생체 50,000 (시간당, `datacash/pricing.py`).

## 실행

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" torch pyyaml rich
echo "MODEL_API_KEY=..." > .env                       # 저장소에 올리지 않는다
# data/models 에 MediaPipe hand/pose/face landmarker .task, efficientdet_lite0.tflite, OpenCV yunet.onnx·sface.onnx
.venv/bin/python -m pytest                            # 정답을 아는 합성 5감 세션으로 검증
DATACASH_TOKEN=... .venv/bin/python -m datacash.server.app --port 8800   # 서버 + 구매 기업 웹
cd app && ./gradlew assembleDebug                     # 안드로이드 앱 + 워치 앱
```

LivePortrait 는 저장소에 넣지 않습니다 — [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) (MIT) 를 받아 `LP_ROOT` 로 지정.

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

## 문서

- 해커톤 계획: [docs/1026_meta_hackathon.md](docs/1026_meta_hackathon.md) · 첫날 기록: [docs/DAY1.md](docs/DAY1.md)
- 가리기·치환 평가: [docs/privacy_eval.md](docs/privacy_eval.md) · Meta 모델 확인: [docs/model_check.md](docs/model_check.md)
- 워치 SDK: [docs/samsung_health_sdk.md](docs/samsung_health_sdk.md) · 데모 영상 대본: [docs/demo_video_script.md](docs/demo_video_script.md)

## 해커톤 전 / 해커톤 중

| | 내용 |
|---|---|
| 해커톤 전 (10-26 이전) | 계획 문서와 브랜드 이미지만. 코드 없음 |
| 해커톤 중 (한국 10-27 01:00 부터) | 이 저장소의 모든 코드 · 앱 · 웹 · Muse Image 가상 인물 · 문서 |

## 라이선스·출처

MediaPipe (Apache 2.0) · OpenCV YuNet·SFace (Apache 2.0) · LivePortrait (MIT) · LeRobot (Apache 2.0) · Numba (BSD).
시험 영상은 Wikimedia Commons (CC BY·CC BY-SA·퍼블릭 도메인만, 출처는 `docs/privacy_eval.md`) — 저장소에 넣지 않습니다.
Samsung Health Sensor SDK 는 약관상 저장소에 넣지 않습니다.
