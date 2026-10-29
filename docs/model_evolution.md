# 모델 진화 — 쓸수록 정답 예시가 쌓여 Spark 1.3 라벨이 정확해진다

코드: `datacash/evolve.py` (`second_opinion` · `gold_examples` · `correction_hints` · `record_agreement`) · 운영자 화면 '모델 진화'
운영자 API: `GET /api/admin/evolve` · 기록: `data/evolve/crosscheck/`, `data/evolve/feedback/`, `data/evolve/registry.json` (agreement)

## 10-30 방향 바꿈: Glimmer 를 빼고 Spark 두 모델로

Glimmer 가중치 학습은 막혀 있다 — Fireworks 계정 데이터셋 업로드 금지(403, 지원팀도 시점 모름), Baseten 키는 Spark Open 추론 전용,
맥은 30B 학습 불가. 그래서 **학습 없이 진화**한다. Spark 크레딧(Baseten $100 · Fireworks $100)은 넉넉하다.

```
모든 녹화 ─▶ ① Muse Spark 1.3 (Meta API, 영상)  + 같은 미션 정답 예시 · 지난 바로잡음을 질문에 붙여서
                │
                ▼
            ② Muse Spark Open 816B (Baseten → Fireworks)가 같은 동작 구간 장면 16장을 따로 라벨
                │  같은 동작?  (구간 ± 장면 간격 안에 같은 동작이 있으면 같음)
        ┌───────┴────────┐
       같음              다름 ─▶ Spark Open 동작으로 바로잡음 → feedback/ (source=review, '어려운 예시')
        │
   + 센서 검증 ─▶ 정답 (crosscheck/<녹화>.json)
                │
                ▼
            ③ 다음 녹화의 ① 질문에 붙는다 (gold_examples: 같은 미션 최근 3편의 동작·길이 순서, correction_hints: 바로잡음 6개)
            ④ 녹화마다 일치율 기록 (registry.json agreement) → 운영자 화면 '일치율 변화' — 오르면 바로잡을 단계·검증 비용이 준다
이의 신청 ─▶ Spark Open 추론 high 가 다시 본다 (judge.py) ─▶ 바로잡음 → ③ 과 같은 길
```

- 단계 검증 증거에 `review`(Spark Open 이 같은 동작으로 봄)가 더해진다. 바로잡은 단계는 바로잡은 동작 기준이라 증거에서 뺀다.
- 비용: 녹화 하나 Meta API 약 $0.13 + Spark Open 약 $0.03 (Baseten 크레딧). 시작 일치율은 점심 녹화 4편 76% (아래 교차 검증).
- 아래 Glimmer 흐름(데이터셋 만들기·평가·Fireworks 학습 스크립트)은 지우지 않고 남긴다 — 업로드가 풀리면 같은 정답으로 학습할 수 있다.

---

# (이전 계획) Meta 오픈 웨이트 Muse Glimmer 를 우리 라벨로 키운다

코드: `scripts/evolve.py` · 학습: `train/glimmer_qlora.yaml`, `scripts/train_glimmer.sh`

## 흐름

```
녹화 처리 ─▶ Muse Spark 단계 라벨 ─▶ 감각으로 검증 ─────────────┐
이의 신청 ─▶ Muse Spark 1.3 최대 추론으로 다시 보기 ─▶ 바로잡은 라벨 ┤ (바로잡은 것이 우선)
                                                                ▼
 ① 데이터 쌓기 (이 맥)  단계마다 장면 3장 + 질문 → 정답 동작 (Axolotl 학습 형식)   ← 처리가 끝날 때마다 서버가 자동으로
 ② 실력 재기 (이 맥)    llama.cpp 의 Glimmer 가 평가용 단계를 몇 % 맞히나 (학습 전 / 어댑터별)
 ③ 학습 (클라우드 GPU)  QLoRA, H100 1장 → 어댑터(작은 추가 가중치)를 GGUF 로
 ④ 쓰기 (이 맥)        어댑터를 붙여 띄우고 ②로 다시 잰다
                       정확도 ≥ 0.8 · 학습 단계 ≥ 200 → 새 녹화의 단계 라벨을 Spark 대신 Glimmer 가 (API 비용 0)
```

- **③만 CUDA 가 필요하다.** 공식 학습 도구(Axolotl·TRL)가 NVIDIA GPU 전용이고, 문서에 확인된 최소 환경이 H100 1장 QLoRA 다.
  나머지(데이터·평가·어댑터를 붙여 쓰기)는 모두 이 맥에서 한다.
- **Spark(API)는 가중치를 바꿀 수 없다** → 이의 신청에서 바로잡은 라벨을 프롬프트에 '지난 바로잡음' 예시로 붙인다.
- **이의 신청의 '더 높은 모델': Muse Spark Open** (Meta 오픈 웨이트 816B, Fireworks 서버리스, `datacash/understand/spark_open.py`).
  Fireworks 의 Spark Open 은 영상을 받지 않아 동작 구간 장면 16장 + 참여자의 이의 내용을 보낸다. 녹화 하나에 약 90초 · $0.02~0.03(가격 추정).
  Fireworks 키가 없으면 Muse Spark 1.3 추론 'max' · 1280px 영상. (Muse Wonder 는 한국 불가, contributor 모드는 쓰지 않는다.)
- **라벨 교차 검증** (`scripts/crosscheck.py`): 평소 라벨(Spark 1.3)을 Spark Open 으로 한 번 더 붙여, 단계마다 그 구간 ± 장면 간격(약 0.7초)
  안에 Spark Open 도 같은 동작을 봤는지 본다. **센서로 검증 + 두 모델이 같은 동작**인 단계만 Glimmer 학습 정답으로 쓴다.
  점심 녹화 4편: 일치 26/34 (76%), 정답 24단계. 가장 많이 갈린 곳은 물티슈 뚜껑 열기(당기기 ↔ 찢기·누르기)와 앞카메라의 손 바꿔 잡기.
  (처음엔 '가장 많이 겹치는 단계 하나'로 비교해 44% 였다 — Spark Open 은 장면 16장이라 경계가 0.5~1초 거칠어, 같은 순서도 어긋나 보였다.)
- 학습용·평가용은 **녹화 단위**로 나눈다 (같은 녹화의 장면끼리는 너무 닮아 섞으면 점수가 부풀려진다). 녹화 4개 중 1개가 평가용.

## 실행

```bash
python scripts/evolve.py build                 # ① 학습 데이터 (서버가 처리 끝날 때마다 자동으로도 한다)
python scripts/evolve.py eval                  # ② 학습 전 Glimmer (llama-server 를 켜고, 끝나면 끈다)
bash scripts/train_glimmer.sh v1               # ③ 클라우드 GPU 에서
python scripts/evolve.py eval --adapter data/evolve/adapters/glimmer-lora-v1.gguf   # ④
```

## 지금 상태 (10-27 밤)

- 학습 데이터 v1: 점심 녹화 4편 → 학습 19단계 · 평가 12단계 (장면 93장)
- 학습 데이터 v2 (교차 검증 반영, 6배로 늘림): 학습 72 · 평가 12
- **학습 전 Glimmer 평가 (평가용 컵 ② 12단계):** 정확도 **33%** (기준선 17% — 가장 많은 동작만 찍었을 때), macro F1 0.26
  - 틀린 8개는 모두 **이웃한 동작과 헷갈림**: 쥐기↔뻗기, 들기↔쥐기, 내려놓기↔옮기기, 놓기↔뻗기.
    장면 3장만 보고는 동작의 시작·끝 경계를 가르기 어렵다 — 우리 데이터로 학습(③)하면 가장 먼저 좋아질 부분이다.
- **학습 없이 질문만 바꿔 보기 (10-29 밤, 평가 12단계, 이 맥의 Glimmer):**

  | 질문 방식 | 추론 | 정확도 |
  |---|---|---|
  | 기본 (장면 3장 + 16가지 중 고르기) | low | 25% |
  | + 동작 정의 | low | 25% |
  | 기본 | medium | 25% |
  | + 동작 정의 | medium | 25% |
  | **+ 미션 동작 순서 힌트** (Spark 에게도 주는 정보) | low | **33%** |
  | + 힌트 + 정의 | low | 25% |

  평가가 12단계뿐이라 한 단계(8%p) 차이는 우연일 수 있다. 틀린 것은 대부분 '놓기→뻗기', '쥐기→뻗기'처럼 장면 3장으로는
  손이 다가가는지 떠나는지 모르는 경우 — 질문으로는 안 풀리고 우리 데이터로 학습해야 한다. 기본 질문은 힌트 방식으로 정했다
  (`evolve.STYLE = "hint"`, 학습 데이터·평가·쓰기 모두 같은 질문).
- **학습 전 Glimmer, 정답 단계 전체로 (힌트 방식):** 학습 전 모델은 어떤 녹화도 보지 않았으니 정답 24단계 전부로 쟀다
  (`python scripts/evolve.py eval --split all`) → **정확도 42%** (기준선 25%, macro F1 0.42). 이것이 학습(③) 전 출발점이다.
- ③ 학습: **Fireworks 에서** 한다 (`scripts/fireworks_train.py`). Fireworks 의 `muse-glimmer-30b` 는 LoRA 지도 학습 가능·이미지 입력 지원,
  11/30 까지 학습 무료. 예상 비용이 $5 를 넘으면 스크립트가 바로 취소한다. 학습한 모델을 띄워 평가하는 것은 전용 GPU(유료)라 따로 한다.
  **10-28: Fireworks 계정의 데이터셋 업로드가 막혀 있다** ("dataset uploads are temporarily disabled, please contact Fireworks AI") → 권한을 받으면 바로 돌린다.
  (H100 을 직접 빌릴 때를 위한 Axolotl 설정 `train/glimmer_qlora.yaml` 도 남겨 둔다.)
- 학습 데이터는 학습용 단계 하나에서 장면 뽑는 구간을 조금씩 옮겨 6벌로 늘린다 (v1: 학습 114 · 평가 12).
  이의 신청에서 바로잡은 단계는 모델이 틀리기 쉬운 '어려운 문제'라 3배 더 넣는다.
- 녹화 4편으로는 학습해도 효과가 작다 — 녹화가 쌓일수록 데이터 버전이 자동으로 늘어난다.
