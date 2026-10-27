**English** · [한국어](README.ko.md) · [Español](README.es.md) · [中文](README.zh-CN.md) · [日本語](README.ja.md)

# DataCash

**Robot hand data from everyday people — using only the phone and watch they already own.**

People record short robot-skill missions at home (pull a tissue, move a cup, twist a lid — 12 in total) with their phone and smartwatch, and get paid.
Humanoid and physical-AI companies buy the result: privacy-scrubbed video, sound, wrist motion, grip force and heart rate on one shared clock,
delivered as a ready-to-train **LeRobot v3.0** dataset.

Built for the Meta Global AI Developer Hackathon (Indie Developer · Push the Frontier).

![DataCash](docs/1026_datacash_brand.png)

## At a glance

| | |
|---|---|
| Contributors | Android app + watch app. Pick a mission → face check → record → 1–2 quick questions → *ding*, you're paid |
| Buyers | **Ask in plain words** on the web ("box-lifting data for warehouse robots, ages 20–50, budget $3,000") → AI designs the missions and opens them worldwide |
| Privacy | The contributor's face is replaced by **a person who doesn't exist**; bystanders, screens and documents are masked; originals are deleted after processing |
| Touch | No glove — the **watch vibrates itself and measures how much the hand absorbs** to estimate grip force (GripSense, UIST 2012) |

## Where the Meta models are used

| Model | Role | Code |
|---|---|---|
| **SAM 3.1** | Masks non-consenting people, children, screens, mirrors, picture frames and documents (video tracking); segments the handled object → `observation.objects` | `datacash/meta/sam.py`, `datacash/privacy/mask.py` |
| **Muse Spark 1.3** | Masked video → steps, instructions, active hand, objects, mistakes (structured JSON); a buyer's plain-language request → mission design | `datacash/understand/`, `datacash/requests.py` |
| **Muse Image** | 34 people who don't exist (7 regions × women/men × 20s–60s) used for face replacement | `scripts/make_donors.py`, `datacash/donors/` |
| **Muse Voice Transcribe** | Finds spoken personal information in the recording and mutes it | `datacash/privacy/speech.py` |
| **Muse Glimmer** (open weights) | 30B model running locally on this Mac with llama.cpp — re-checks Spark's steps frame by frame (no API cost) | `datacash/understand/glimmer.py` |

Every API call is logged with its cost in `data/api_ledger.jsonl`; if the budget is hit, only that step is skipped.

## Key ideas

### 1. Privacy that keeps the data valuable
- **Face replacement, not blur:** the contributor's face is swapped for a Muse Image person (LivePortrait + MediaPipe). We pick the closest match of the same gender and similar age, test-swap 20 frames, and move to the next person if face recognition (SFace) would call more than half of them "the same person".
- **Continuity:** when a swapped frame comes out too similar, we regenerate it with a softer expression instead of pixelating — the video never breaks.
- **Masking:** SAM 3.1 hides only what wasn't consented: other people, children, screens, mirrors, picture frames, documents. Consenting family members stay visible, each with their own virtual face.
- **Originals deleted:** in production, raw video and audio are deleted as soon as processing ends (`purge_originals`). On the hackathon demo server only that call is switched off, so we can show the original-vs-replaced comparison.

### 2. Automatic multi-sensor sync
No clapping required: phone↔watch clock round-trips, an 18–20 kHz acoustic beacon, and watch vibration pulses. Residual 0.5–2.3 ms on real phone + watch recordings (`datacash/sync/`).

### 3. Touch without a glove — GripSense on the wrist
We moved Goel, Wobbrock & Patel's *GripSense* (UIST 2012) to the wrist. The watch buzzes briefly; the tighter you grip, the more the hand absorbs it — high-frequency gyro energy drops while 4–12 Hz finger tremor rises. A per-person decision tree, trained on self-rated levels (release · light · normal · tight, no scale needed), outputs grip level 0–3 (`datacash/touch/`).
Heart rate, beat-to-beat intervals and skin temperature come from the Samsung Health Sensor SDK (`docs/samsung_health_sdk.md`).

### 4. Plain-language requests → missions
A buyer types what they need; Muse Spark (structured output) designs the missions (actions, objects, conditions to vary), eligibility (age, gender) and sensor packs, and the server fits a quote to the budget. Once confirmed, the missions appear only for eligible contributors — the request *is* the collection distribution.

### 5. Processing costs that leave a margin
- Unusable recordings (too short, hands not visible, too dark) are rejected **on-device before any API call** — $0
- A local object detector scans first; SAM only gets what's needed (no "person" query for solo recordings, screens only when present, static objects at 1 fps)
- On 4 real recordings: SAM cost **$2.92 → $0.27 (−91%)**, about $5–7 per hour of video (sold at ~$43/hour)
- SAM mask decoding rewritten with Numba: 593 s → 8 s per recording, bit-identical to the official parser

## The dataset (LeRobot v3.0)

`observation.images.ego` (masked + face-replaced video) · `observation.hands` · `observation.body` · `observation.imu.phone` · `observation.imu.wrist` ·
`observation.audio` · **`observation.touch`** (contact, grip level 0–3) · `observation.touch.vibration` · **`observation.physio`** (heart rate, skin temperature; biometric consent only) ·
`observation.objects` · `annotation.step` · `annotation.primitive` (reach · grasp · … · release) · `sync.error_ms` + a quality certificate.
Loads with the real `lerobot` 0.6.1 (`scripts/validate_lerobot.py`).

Buyers pick sensor packs per hour of data — video (required) ₩35,000 · pose ₩20,000 · audio ₩5,000 · wrist IMU ₩15,000 · touch ₩25,000 · biometrics ₩50,000 (`datacash/pricing.py`).

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" torch pyyaml rich
echo "MODEL_API_KEY=..." > .env                       # never committed
# data/models: MediaPipe hand/pose/face landmarker .task, efficientdet_lite0.tflite, OpenCV yunet.onnx + sface.onnx
.venv/bin/python -m pytest                            # verified on synthetic sessions with known ground truth
DATACASH_TOKEN=... .venv/bin/python -m datacash.server.app --port 8800   # server + buyer web
cd app && ./gradlew assembleDebug                     # Android app + watch app
```

LivePortrait is not vendored — get [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) (MIT) and point `LP_ROOT` to it.

## Layout

| Folder | What it does |
|---|---|
| `app/` | Android app (consent, missions, face check, recording, mistake marks, questions, upload, rewards, appeals) + watch app |
| `web/` | Buyer web (plain-language requests, multi-sensor timeline, quotes, orders, downloads) and operator console |
| `datacash/privacy` | SAM masking · face replacement · speech privacy · pre-scan |
| `datacash/sync` · `touch` · `labels` | Sensor sync · touch · hand and body pose |
| `datacash/understand` | Muse Spark understanding · Glimmer checks · sensor-based verification |
| `datacash/export` · `jobs.py` | LeRobot export · certificate · processing pipeline |
| `datacash/server` | Collection server + buyer API |

## Docs (in Korean)

- Hackathon plan: [docs/1026_meta_hackathon.md](docs/1026_meta_hackathon.md) · Day 1 log: [docs/DAY1.md](docs/DAY1.md)
- Masking & face-replacement evaluation: [docs/privacy_eval.md](docs/privacy_eval.md) · Meta model check: [docs/model_check.md](docs/model_check.md)
- Watch SDK: [docs/samsung_health_sdk.md](docs/samsung_health_sdk.md) · Demo video script: [docs/demo_video_script.md](docs/demo_video_script.md)

## Before vs. during the hackathon

| | Contents |
|---|---|
| Before (until Oct 26) | Planning document and brand image only. No code |
| During (from Oct 27 01:00 KST) | All code, app, web, Muse Image people and docs in this repository |

## Licenses and sources

MediaPipe (Apache 2.0) · OpenCV YuNet & SFace (Apache 2.0) · LivePortrait (MIT) · LeRobot (Apache 2.0) · Numba (BSD).
Test videos are from Wikimedia Commons (CC BY, CC BY-SA or public domain only; sources in `docs/privacy_eval.md`) and are not committed.
The Samsung Health Sensor SDK is not committed, per its terms.
