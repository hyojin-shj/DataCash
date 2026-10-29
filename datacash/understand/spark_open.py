"""Muse Spark Open (Meta 오픈 웨이트 816B) — 이의 신청 다시 보기·라벨 교차 검증에 쓰는 '더 높은 모델'.

어디서 부르나: Baseten(해커톤 크레딧 $100, 키는 이 모델 전용) 먼저, 안 되면 Fireworks 서버리스.
  Fireworks 크레딧은 Glimmer 학습(scripts/fireworks_train.py)에 아껴 둔다.
영상 대신 이미지를 받는다 → 동작 구간에서 장면 N 장을 고르게 뽑아 시각과 함께 보낸다.
평소 라벨(Muse Spark 1.3, 영상 640px)과 같은 형식(spark.SCHEMA)의 단계를 돌려준다.
비용은 Meta API 한도와 섞지 않고 data/fireworks_ledger.jsonl 에 따로 적는다 (어디서 불렀는지 provider 와 함께).
키: .env 의 BASETEN_API_KEY · FIREWORKS_API_KEY.
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

PROVIDERS = [   # (이름, 키 이름, 주소, 모델) — 앞에서부터 쓴다
    ("baseten", "BASETEN_API_KEY", "https://inference.baseten.co/v1", "meta-models/muse-spark-open-1026"),
    ("fireworks", "FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1", "accounts/fireworks/models/muse-spark-open"),
]
FRAMES = 16
SIDE = 768
REASONING = "high"
PRICE = (1.25e-6, 4.25e-6)       # 달러/토큰 (입력·출력) — 가격표 확인 전 추정치 (두 곳 같게), 장부에 '추정'으로 남긴다
LEDGER = config.DATA / "fireworks_ledger.jsonl"


ALIASES = {"BASETEN_API_KEY": ["baseten_api", "BASETEN_API"]}    # .env 에 다른 이름으로 넣었을 때


def _key(name: str) -> str:
    config.load_env()
    for k in [name] + ALIASES.get(name, []):
        if os.environ.get(k):
            return os.environ[k]
    return ""


def providers() -> list:
    """키가 있는 곳들 (쓰는 순서)."""
    return [p for p in PROVIDERS if _key(p[1])]


def available() -> bool:
    return bool(providers())


def provider_name() -> str:
    ps = providers()
    return ps[0][0].capitalize() if ps else ""


def spent_usd(provider: Optional[str] = None) -> float:
    if not LEDGER.exists():
        return 0.0
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    return round(sum(r.get("usd", 0) for r in rows if provider is None or r.get("provider", "fireworks") == provider), 4)


def _ask(prompt_parts: list):
    """Baseten → Fireworks 순서로. 한 곳이 실패하면 다음 곳. (provider, 응답)."""
    from openai import OpenAI

    err = None
    for name, key, url, model in providers():
        c = OpenAI(base_url=url, api_key=_key(key), timeout=600)
        kw = {"response_format": {"type": "json_object"}, "reasoning_effort": REASONING}
        for _ in range(3):              # 받지 않는 선택지가 있으면 빼고 다시
            try:
                return name, c.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt_parts}],
                                                       max_tokens=16000, **kw)
            except Exception as e:      # noqa: BLE001
                err = e
                msg = str(e).lower()
                drop = [k for k in kw if k.split("_")[0] in msg]
                if not drop or getattr(e, "status_code", 400) not in (400, 422):
                    break
                for k in drop:
                    kw.pop(k)
    raise RuntimeError(f"Spark Open 을 부르지 못함: {err}")


def _record(usage, what: str, provider: str = "fireworks") -> None:
    u = usage.model_dump() if usage else {}
    usd = (u.get("prompt_tokens") or 0) * PRICE[0] + (u.get("completion_tokens") or 0) * PRICE[1]
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": f"muse-spark-open ({provider})", "provider": provider,
                            "usd": round(usd, 6), "estimated_price": True, "what": what, "usage": u}, ensure_ascii=False) + "\n")


def label_frames(video, t0: float, t1: Optional[float], mission: str, extra: str = "", primitives: Optional[list] = None,
                 n: int = FRAMES, client=None) -> dict:
    """동작 구간 [t0, t1] 의 장면 n 장 → spark.SCHEMA 형식 {task, completed, steps, mistakes, environment} (시각은 영상 기준 초).
    client 를 주면 그것으로 부른다 (시험용)."""

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
    content = [{"type": "text", "text": prompt}] + parts
    if client is not None:
        prov, r = "test", client.chat.completions.create(model=PROVIDERS[0][3], messages=[{"role": "user", "content": content}],
                                                        response_format={"type": "json_object"}, reasoning_effort=REASONING,
                                                        max_tokens=16000)
    else:
        prov, r = _ask(content)
    _record(getattr(r, "usage", None), f"spark-open label {Path(video).name} {len(times)} frames", prov)
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
    out["model"] = f"muse-spark-open ({prov})"
    out["frames"] = times
    return out
