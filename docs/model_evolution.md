# 모델 진화 — Meta 오픈 웨이트 Muse Glimmer 를 우리 라벨로 키운다

코드: `datacash/evolve.py` · 명령: `scripts/evolve.py` · 학습: `train/glimmer_qlora.yaml`, `scripts/train_glimmer.sh`
운영자 API: `GET /api/admin/evolve` · 기록: `data/evolve/registry.json`

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
- **이의 신청의 '더 높은 모델':** 한국에서 쓸 수 있는 가장 높은 설정은 Muse Spark 1.3 의 추론 'max' 다 (Muse Wonder 는 한국 불가).
  평소 라벨은 640px 짧은 영상 + 기본 추론, 이의 신청은 1280px 동작 구간 전체 + 참여자의 이의 내용 + 'max'.
  사용자 영상이 Meta 학습에 쓰이는 contributor 모드는 쓰지 않는다.
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
- **학습 전 Glimmer 평가 (평가용 컵 ② 12단계):** 정확도 **33%** (기준선 17% — 가장 많은 동작만 찍었을 때), macro F1 0.26
  - 틀린 8개는 모두 **이웃한 동작과 헷갈림**: 쥐기↔뻗기, 들기↔쥐기, 내려놓기↔옮기기, 놓기↔뻗기.
    장면 3장만 보고는 동작의 시작·끝 경계를 가르기 어렵다 — 우리 데이터로 학습(③)하면 가장 먼저 좋아질 부분이다.
- ③ 학습은 아직 안 했다 (GPU 크레딧을 기다리는 중). 설정은 공식 문서를 따랐지만 실제 GPU 에서 돌려 보지 않았다.
- 녹화 4편으로는 학습해도 효과가 작다 — 녹화가 쌓일수록 데이터 버전이 자동으로 늘어난다.
