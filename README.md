# DataCache (데이터캐시)

집에 이미 있는 기기(폰·워치)만으로 **촉각·시각·청각·몸 감각**을 한 시간축에 모으는 생활밀착형 고품질 로봇 데이터 양산소.
사용자는 집에서 미션을 찍어 보상을 받고, 휴머노이드·피지컬 AI 회사는 개인정보가 지워진 4감 데이터를 LeRobot v3.0 그대로 산다.
Meta Global AI Developer Hackathon (Indie Developer · Push the Frontier) 출품작.

![DataCache](docs/1026_datacache_brand.png)

- 해커톤 계획: [docs/1026_meta_hackathon.md](docs/1026_meta_hackathon.md)
- 첫날 작업 기록: [docs/DAY1.md](docs/DAY1.md)
- Meta 모델 확인: [docs/model_check.md](docs/model_check.md) · 가리기 평가: [docs/privacy_eval.md](docs/privacy_eval.md)

## 해커톤 전 / 해커톤 중

| | 내용 |
|---|---|
| 해커톤 전 (10-26 이전) | 계획 문서와 브랜드 이미지만. 코드는 없다. 가져와 쓰는 자산은 계획 문서 1-1 표에 출처와 함께 밝힌다 |
| 해커톤 중 (한국 10-27 01:00 = 10-26 정오 ET 부터) | 이 저장소의 모든 코드 · 앱 · 웹 · Muse Image 로 만든 가상 인물(`datacache/donors`) · 문서 |

## Meta 모델을 어디에

| 모델 | 쓰임 | 코드 |
|---|---|---|
| **SAM 3.1** | 동의 안 한 사람·아이·화면·거울·액자·서류 가리기 (영상 추적), 조작 물체 영역 → `observation.objects`, 접촉 판정 | `datacache/meta/sam.py`, `datacache/privacy/mask.py` |
| **Muse Spark 1.3** | 가린 영상 → 구조화 JSON (과제·단계 지시문·쓰는 손·물체·실수·장소·도구) → 촉각·소리·손목으로 검증 | `datacache/understand/` |
| **Muse Image** | 실존하지 않는 가상 인물 (참여자 얼굴 치환용) | `datacache/meta/image.py`, `datacache/donors/` |
| **Muse Voice Transcribe** | 녹화 소리 받아쓰기 (개인정보 발화 구간 끄기용) | `datacache/meta/voice.py` |
| **Muse Glimmer** (오픈 웨이트) | Spark 라벨을 배워 우리 서버에서 라벨링 (진행 중) | — |

Muse Wonder 는 한국에서 쓸 수 없어 영상 이해는 Spark 1.3 으로 한다. API 호출마다 비용을 `data/api_ledger.jsonl` 에 적고 한도를 넘으면 그 단계만 건너뛴다.

## 핵심 기술

1. **가치 보존 개인정보 처리** — 블러 대신: SAM 3.1 로 동의 안 한 것만 가리고, 참여자 얼굴은 실사 가상 인물로 치환
   (LivePortrait(MIT) + MediaPipe 자르기 — 비상업 InsightFace 를 쓰지 않는다), 장면마다 SFace 재식별 검사
2. **4감 자동 동기화** — 사용자 동작 없이: 폰↔워치 시계 왕복, 18~20kHz 음향 비콘(폰 마이크·워치 마이크가 같이 들음),
   워치 진동 펄스(촉각 펄스가 동기 신호를 겸함), 영상 움직임↔손목 상관으로 검증 (`datacache/sync/`)
3. **주변 기기로 만드는 촉각** — 워치 능동 진동 감지(울림 크기·사그라드는 속도 → 저울 보정 쥐는 힘), 영상 접촉 단계,
   접촉음 + 손목 충격 (`datacache/touch/`)
4. **장면 이해와 검증된 라벨** — Muse Spark 라벨을 감각으로 확인, 지어낸 단계는 걸러진다
5. **스스로 좋아지는 수집** — 부족한 미션 보너스, 구매자 의뢰 → 미션, (진행 중) Glimmer 증류

## 판매 데이터 (LeRobot v3.0)

`observation.images.ego` (가린·치환 영상) · `observation.hands` · `observation.body` · `observation.body_world` ·
`observation.imu.phone` · `observation.imu.wrist` · `observation.audio` · **`observation.touch`** (접촉·쥐는 힘 N·접촉 단계·물체) ·
`observation.touch.vibration` · **`observation.physio`** (심박·HRR·피부온도 변화, 생체 동의만) · `observation.objects` ·
`annotation.events` · `annotation.step` · `sync.error_ms` + 에피소드 정보(단계·실수·장면·성공·동기화·촉각 보정) + 품질 인증서.
실제 `lerobot` 0.6.1 로 읽힌다 (`scripts/validate_lerobot.py`).

등급: 기본 60,000원/시간 · 고품질(+손목·쥐는 힘) 100,000원 · 생체(+심박·체온) 150,000원 (시장 사례 기준, `datacache/pricing.py`).

## 실행

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" torch pyyaml rich
echo "MODEL_API_KEY=LLM_..." > .env                     # 저장소에 올리지 않는다
# data/models 에 MediaPipe hand/pose/face landmarker .task, OpenCV yunet.onnx·sface.onnx
.venv/bin/python -m pytest                              # 합성 4감 세션으로 동기화·촉각·내보내기 검증
.venv/bin/python scripts/model_check.py                 # Meta 모델 확인
DATACACHE_TOKEN=... .venv/bin/python -m datacache.server.app --port 8800   # 서버 + 구매 기업 웹(http://localhost:8800)
cd app && ./gradlew assembleDebug                       # 일반 사용자 앱 (안드로이드)
```

LivePortrait 는 저장소에 넣지 않는다 — [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) (MIT) 를 받아 `LP_ROOT` 로 지정.

## 폴더

| 폴더 | 하는 일 |
|---|---|
| `app/` | 일반 사용자 안드로이드 앱 (동의·미션·녹화·실수 표시·미션 후 질문·업로드·적립) |
| `web/` | 구매 기업 데스크톱 웹 (4감 타임라인·카탈로그·견적·주문·내려받기), 운영자 화면 |
| `datacache/meta` | Meta Model API 클라이언트 (비용 장부·한도) |
| `datacache/session` | 녹화 세션 형식, 정답을 아는 합성 4감 세션 |
| `datacache/sync` · `touch` · `labels` | 4감 동기화 · 촉각 · 손·몸 관절 |
| `datacache/privacy` · `understand` | SAM 3.1 가리기·얼굴 치환 · Muse Spark 이해·검증 |
| `datacache/export` · `pipeline.py` · `jobs.py` | LeRobot 내보내기·인증서·묶음, 처리 순서 |
| `datacache/server` | 수집 서버 + 구매 기업 API |

## 라이선스·출처

- 시험 영상: Wikimedia Commons (CC BY·CC BY-SA·퍼블릭 도메인만, 출처는 `docs/privacy_eval.md`). 영상은 저장소에 넣지 않는다
- MediaPipe (Apache 2.0) · OpenCV YuNet·SFace (Apache 2.0) · LivePortrait (MIT) · LeRobot (Apache 2.0)
