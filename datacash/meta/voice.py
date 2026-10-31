"""Muse Voice Transcribe (파일 엔드포인트 POST /v1/asr/transcribe). 발화(턴) 단위 시각만 준다 — 단어 시각 없음."""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path
from typing import List, Optional

import httpx

from .. import config
from .client import PRICE, check_budget, record

MODEL = "muse-voice-transcribe-1.0"


def transcribe(wav_path, mode: str = "ENDPOINTING", language_bias: Optional[List[str]] = None,
               keywords: Optional[List[str]] = None) -> dict:
    """{"transcript", "turns": [{"startMs","endMs","transcript","speaker"?}], "audioDurationMs"}.
    wav: 모노 16비트 PCM 16k/24k, 10분·32MB 이하. mode: PUSH_TO_TALK(턴 없음)·ENDPOINTING·DIARIZATION."""
    with wave.open(str(wav_path)) as w:
        dur = w.getnframes() / w.getframerate()
    est = dur * PRICE[MODEL]
    check_budget(est)
    req = {"model": MODEL, "mode": mode, "audioEncoding": "WAV"}  # audioEncoding 은 필수 (문서에 없음, 10-27 확인)
    if language_bias:
        req["languageBias"] = language_bias
    if keywords:
        req["keywords"] = keywords
    t0 = time.time()
    with open(wav_path, "rb") as f:
        r = httpx.post(f"{config.API_BASE}/asr/transcribe", headers={"Authorization": f"Bearer {config.api_key()}"},
                       files={"request": (None, json.dumps(req), "application/json"),
                              "audio": (Path(wav_path).name, f, "audio/wav")}, timeout=300)
    r.raise_for_status()
    out = r.json()
    record(MODEL, int(out.get("audioDurationMs", dur * 1000) / 1000) * PRICE[MODEL], f"transcribe {Path(wav_path).name}",
           secs=time.time() - t0)
    return out
