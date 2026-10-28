"""Muse Spark Open (Meta 오픈 웨이트 816B, Fireworks 서버리스) — 이의 신청 다시 보기에 쓰는 '더 높은 모델'.

Fireworks 의 Spark Open 은 영상 대신 이미지를 받는다 → 동작 구간에서 장면 N 장을 고르게 뽑아 시각과 함께 보낸다.
평소 라벨(Muse Spark 1.3, 영상 640px)과 같은 형식(spark.SCHEMA)의 단계를 돌려준다.
비용은 Meta API 한도와 섞지 않고 data/fireworks_ledger.jsonl 에 따로 적는다 (Fireworks 크레딧).
키: .env 의 FIREWORKS_API_KEY.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import cv2

from .. import config
from .spark import PRIMITIVES, PROMPT, SCHEMA

MODEL = "accounts/fireworks/models/muse-spark-open"
URL = "https://api.fireworks.ai/inference/v1"
FRAMES = 16
SIDE = 768
REASONING = "high"
PRICE = (1.25e-6, 4.25e-6)       # 달러/토큰 (입력·출력) — Fireworks 가격표 확인 전 추정치, 장부에 '추정'으로 남긴다
LEDGER = config.DATA / "fireworks_ledger.jsonl"


def available() -> bool:
    config.load_env()
    return bool(os.environ.get("FIREWORKS_API_KEY"))


def spent_usd() -> float:
    if not LEDGER.exists():
        return 0.0
    return round(sum(json.loads(l).get("usd", 0) for l in LEDGER.read_text().splitlines() if l.strip()), 4)


def _record(usage, what: str) -> None:
    u = usage.model_dump() if usage else {}
    usd = (u.get("prompt_tokens") or 0) * PRICE[0] + (u.get("completion_tokens") or 0) * PRICE[1]
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": "muse-spark-open (fireworks)", "usd": round(usd, 6),
                            "estimated_price": True, "what": what, "usage": u}, ensure_ascii=False) + "\n")


def label_frames(video, t0: float, t1: Optional[float], mission: str, extra: str = "", primitives: Optional[list] = None,
                 n: int = FRAMES, client=None) -> dict:
    """동작 구간 [t0, t1] 의 장면 n 장 → spark.SCHEMA 형식 {task, completed, steps, mistakes, environment} (시각은 영상 기준 초)."""
    from openai import OpenAI

    cap = cv2.VideoCapture(str(video))
    dur = cap.get(7) / (cap.get(5) or 30)
    t1 = min(dur, t1 if t1 is not None else dur)
    t0 = max(0.0, min(t0, t1 - 0.5))
    parts, times = [], []
    for k in range(n):
        t = t0 + (t1 - t0) * (k + 0.5) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        s = SIDE / max(f.shape[:2])
        f = cv2.resize(f, None, fx=s, fy=s) if s < 1 else f
        times.append(round(t, 2))
        parts.append({"type": "text", "text": f"frame at t={t:.2f}s"})
        parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                      base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()}})
    cap.release()
    prompt = (PROMPT.format(mission=mission, dur=t1, primitives=" -> ".join(primitives) if primitives else "not given")
              .replace("The clip is", "You see frames sampled from a video; times below are seconds from the video start. The video is")
              + f"\nFrames cover {t0:.1f}s to {t1:.1f}s. Give step times in the same seconds."
              + ("\n" + extra if extra else "")
              + "\nReturn ONLY a JSON object matching this JSON schema:\n" + json.dumps(SCHEMA))
    config.load_env()
    c = client or OpenAI(base_url=URL, api_key=os.environ["FIREWORKS_API_KEY"], timeout=600)
    r = c.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] + parts}],
                                  response_format={"type": "json_object"}, reasoning_effort=REASONING, max_tokens=16000)
    _record(getattr(r, "usage", None), f"spark-open label {Path(video).name} {len(times)} frames")
    text = r.choices[0].message.content or "{}"
    out = json.loads(text[text.find("{"): text.rfind("}") + 1])
    steps = []
    for s in out.get("steps") or []:
        try:
            a, b = float(s["start_s"]), float(s["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if s.get("action") in PRIMITIVES and b > a:
            steps.append({**s, "start_s": max(t0, a), "end_s": min(t1, b)})
    out["steps"] = sorted(steps, key=lambda s: s["start_s"])
    out["model"] = "muse-spark-open (fireworks)"
    out["frames"] = times
    return out
