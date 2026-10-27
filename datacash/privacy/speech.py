"""말소리 개인정보 끄기 — Muse Voice Transcribe 로 받아쓰고(발화 단위 시각), Muse Spark 가 이름·주소·전화번호·계좌 등이
들어간 발화를 고르면 그 구간만 소리를 끈다. 접촉음·물소리 같은 비언어 소리는 남는다 (청각 데이터·동기화에 필요).

Voice Transcribe 는 단어 시각이 없고 발화(턴) 시각만 주므로, 고른 발화 전체를 끈다 (앞뒤 0.15초 여유, 0.03초 페이드).
받아쓴 글자는 판매하지 않는다 (판정에만 쓰고 버린다).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..meta.client import MetaClient

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["turns"], "properties": {
    "turns": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                         "required": ["index", "personal", "kinds"],
                                         "properties": {"index": {"type": "integer"}, "personal": {"type": "boolean"},
                                                        "kinds": {"type": "array", "items": {"type": "string", "enum": [
                                                            "name", "address", "phone", "account", "id_number", "email",
                                                            "health", "other"]}}}}}}}

PROMPT = """For each numbered utterance from a home video, decide if it reveals personal information about a real person
(full or given names of people, home address or location, phone number, bank/card account, ID numbers, email, health details).
Generic words (e.g. "mom", "the plate") are not personal. Utterances:
{lines}"""


def classify_turns(turns: List[dict], client: Optional[MetaClient] = None) -> List[dict]:
    if not turns:
        return []
    lines = "\n".join(f"{i}: {t.get('transcript', '')}" for i, t in enumerate(turns))
    out = (client or MetaClient()).chat_json("muse-spark-1.3", [{"role": "user", "content": PROMPT.format(lines=lines)}],
                                             SCHEMA, "pii_turns", "speech pii", estimate_usd=0.01)
    flags = {t["index"]: t for t in out["turns"]}
    return [{**t, "personal": bool(flags.get(i, {}).get("personal")), "kinds": flags.get(i, {}).get("kinds", [])}
            for i, t in enumerate(turns)]


def mute(x: np.ndarray, rate: int, spans_ms: List[tuple], pad_s: float = 0.15, fade_s: float = 0.03) -> np.ndarray:
    y = x.copy()
    g = np.ones(len(y), np.float32)
    f = max(1, int(fade_s * rate))
    for a, b in spans_ms:
        i0, i1 = max(0, int((a / 1000 - pad_s) * rate)), min(len(y), int((b / 1000 + pad_s) * rate))
        if i1 <= i0:
            continue
        g[i0:i1] = 0.0
        g[max(0, i0 - f):i0] = np.minimum(g[max(0, i0 - f):i0], np.linspace(1, 0, i0 - max(0, i0 - f)))
        g[i1:min(len(y), i1 + f)] = np.minimum(g[i1:min(len(y), i1 + f)], np.linspace(0, 1, min(len(y), i1 + f) - i1))
    return (y * g).astype(y.dtype)


def clean_audio(wav_path, out_path, client: Optional[MetaClient] = None) -> dict:
    """wav → 개인정보 발화만 끈 wav. 보고: 발화 수, 끈 발화 수·종류·길이 (글자는 남기지 않는다)."""
    import soundfile as sf

    from ..media import wav as to_wav
    from ..meta.voice import transcribe

    x, rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    mono16 = Path(out_path).with_suffix(".16k.wav")
    to_wav(wav_path, mono16, rate=16000)
    tr = transcribe(mono16)
    mono16.unlink()
    turns = classify_turns(tr.get("turns") or [], client)
    spans = [(t["startMs"], t["endMs"]) for t in turns if t["personal"]]
    sf.write(str(out_path), mute(x, rate, spans), rate, subtype="PCM_16")
    return {"turns": len(turns), "muted": len(spans), "muted_s": round(sum(b - a for a, b in spans) / 1000, 2),
            "kinds": sorted({k for t in turns if t["personal"] for k in t["kinds"]})}
