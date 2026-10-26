# Meta 모델 확인 (2026-10-27 01:00~01:05 KST, 한국 계정)

`GET /v1/models` 로 열린 모델: `sam-3.1`, `muse-spark-1.3`, `muse-spark-1.3-contributor`, `muse-spark-1.2`,
`muse-spark-1.2-contributor`, `muse-spark-1.1`, `muse-image-1.0`, `muse-voice-transcribe-1.0`.
**Muse Wonder 는 없다** (문서대로 한국 불가) → 영상 이해·에이전트는 Muse Spark 1.3.

첫 호출 (공개 영상만, 재현: `python scripts/model_check.py`):

| 모델 | 입력 | 결과 | 비용 |
|---|---|---|---|
| `sam-3.1` (이미지) | Commons 설거지 영상 한 장면 (Washing Dishes, CC BY-SA 4.0) | `person` 1 (신뢰도 0.91), `hand` 1 (0.81), `plate` 4 (0.55~0.84), `sponge` 0 | 이미지당 $0.0025 |
| `muse-spark-1.3` (영상 8초 → 구조화 JSON) | 같은 영상 8초, 640px | 과제 "Hand-washing dishes in a kitchen sink", 단계 3개와 시각 (헹구기 0~3.2초 → 세제 묻히기 3.2~5.5초 → 문질러 헹구기 5.5~8초). 47초 걸림 | 토큰 과금 |
| `muse-voice-transcribe-1.0` | Commons 요리 영상 소리 40초 (라틴어 낭독) | 턴 단위 시각과 받아쓰기 | 시간당 $0.18 |
| `muse-image-1.0` | 가상 인물 정면 사진 문장 | PNG 1장 | 장당 $0.01 |

**문서와 다른 점:** Voice Transcribe 파일 엔드포인트는 `request` JSON 에 `audioEncoding` 이 필수다 (`"WAV"` 로 됨).
문서의 `LINEAR16` 류 값은 거절된다.

이 시점까지 누적 비용 약 $0.03.
