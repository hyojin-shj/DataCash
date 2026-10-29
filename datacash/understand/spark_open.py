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
import numpy as np

from .. import config
from .spark import PRIMITIVES, PROMPT, SCHEMA

PROVIDERS = [   # (이름, 키 이름, 주소, 모델) — 앞에서부터 쓴다
    ("baseten", "BASETEN_API_KEY", "https://inference.baseten.co/v1", "meta-models/muse-spark-open-1026"),
    ("fireworks", "FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1", "accounts/fireworks/models/muse-spark-open"),
]
MAX_IMAGES = {"baseten": 8}     # 한 번에 받는 이미지 수 (Baseten: 8장) — 넘으면 장면을 나란히 붙여 이미지 수를 줄인다
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


def _ask(build):
    """Baseten → Fireworks 순서로. 한 곳이 실패하면 다음 곳. build(최대 이미지 수 또는 None) → 보낼 내용. (provider, 응답)."""
    from openai import OpenAI

    err = None
    for name, key, url, model in providers():
        c = OpenAI(base_url=url, api_key=_key(key), timeout=600)
        prompt_parts = build(MAX_IMAGES.get(name))
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


def _grab(video, times: List[float]) -> List[tuple]:
    """[(t, 장면)] — 긴 쪽 SIDE."""
    cap = cv2.VideoCapture(str(video))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if ok:
            s = SIDE / max(f.shape[:2])
            out.append((round(t, 2), cv2.resize(f, None, fx=s, fy=s) if s < 1 else f))
    cap.release()
    return out


def _jpg(img) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
            base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()}}


POS = {2: ["left", "right"], 4: ["top-left", "top-right", "bottom-left", "bottom-right"]}


def _tile(imgs: list):
    """장면 2장 → 나란히, 4장 → 2×2 (같은 크기로)."""
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(x, (w, h)) for x in imgs] + [np.zeros((h, w, 3), np.uint8)] * (len(imgs) % 2)
    rows = [np.hstack(imgs[k:k + 2]) for k in range(0, len(imgs), 2)]
    return np.vstack(rows)


def _frame_parts(frames: List[tuple], max_images: Optional[int]) -> list:
    """장면들을 내용으로. 이미지 수 한도가 있으면 2장(또는 4장)씩 붙여 한 이미지로, 위치와 시각을 글로 알려 준다."""
    k = 1 if not max_images or len(frames) <= max_images else (2 if len(frames) <= 2 * max_images else 4)
    parts = []
    for g in range(0, len(frames), k):
        grp = frames[g:g + k]
        if len(grp) == 1:
            parts += [{"type": "text", "text": f"frame at t={grp[0][0]:.2f}s"}, _jpg(grp[0][1])]
        else:
            where = ", ".join(f"{POS[k][j]} t={t:.2f}s" for j, (t, _) in enumerate(grp))
            parts += [{"type": "text", "text": f"one image with {len(grp)} frames: {where}"}, _jpg(_tile([f for _, f in grp]))]
    return parts


def label_frames(video, t0: float, t1: Optional[float], mission: str, extra: str = "", primitives: Optional[list] = None,
                 n: int = FRAMES, client=None) -> dict:
    """동작 구간 [t0, t1] 의 장면 n 장 → spark.SCHEMA 형식 {task, completed, steps, mistakes, environment} (시각은 영상 기준 초).
    client 를 주면 그것으로 부른다 (시험용)."""

    cap = cv2.VideoCapture(str(video))
    dur = cap.get(7) / (cap.get(5) or 30)
    cap.release()
    t1 = min(dur, t1 if t1 is not None else dur)
    t0 = max(0.0, min(t0, t1 - 0.5))
    frames = _grab(video, [t0 + (t1 - t0) * (k + 0.5) / n for k in range(n)])
    times = [t for t, _ in frames]
    prompt = (PROMPT.format(mission=mission, dur=t1, primitives=" -> ".join(primitives) if primitives else "not given")
              .replace("The clip is", "You see frames sampled from a video; times below are seconds from the video start. The video is")
              + f"\nFrames cover {t0:.1f}s to {t1:.1f}s. Give step times in the same seconds."
              + ("\n" + extra if extra else "")
              + "\nReturn ONLY a JSON object matching this JSON schema:\n" + json.dumps(SCHEMA))
    build = lambda mx: [{"type": "text", "text": prompt}] + _frame_parts(frames, mx)  # noqa: E731
    if client is not None:
        prov, r = "test", client.chat.completions.create(model=PROVIDERS[0][3], messages=[{"role": "user", "content": build(None)}],
                                                        response_format={"type": "json_object"}, reasoning_effort=REASONING,
                                                        max_tokens=16000)
    else:
        prov, r = _ask(build)
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


JUDGE = """You are reviewing step labels of a robot-learning video, mission: "{mission}". Gray areas are privacy masks; ignore them.
Two labelers disagreed on the steps below. For each step you get one image with 4 frames from inside that step in time order
(top-left, top-right, bottom-left, bottom-right). Decide which manipulation primitive the hand is doing during that step.
Prefer one of the two options; pick another primitive only if both are clearly wrong. Primitives: {prims}.
Expected order in this mission (a hint): {order}.
Return ONLY JSON {{"steps": [{{"k": <step number>, "action": <primitive>, "why": <short reason>}}]}}."""


def adjudicate(video, disputes: List[dict], mission: str, primitives: Optional[list] = None) -> List[Optional[str]]:
    """두 모델이 다르게 본 단계 [{start_s, end_s, options: [a, b]}] → 단계마다 Spark Open 의 판단 (그 구간 안 장면 4장을 촘촘히).
    이미지 한도(Baseten 8장) 안에 들게 8단계씩 나눠 묻는다. 못 정한 단계는 None."""
    out: List[Optional[str]] = [None] * len(disputes)
    for g in range(0, len(disputes), 8):
        chunk = disputes[g:g + 8]
        tiles = []
        for d in chunk:
            a, b = d["start_s"], d["end_s"]
            fr = _grab(video, [a + (b - a) * (k + 0.5) / 4 for k in range(4)])
            tiles.append(_tile([f for _, f in fr]) if len(fr) == 4 else None)
        parts = [{"type": "text", "text": JUDGE.format(mission=mission, prims=", ".join(PRIMITIVES),
                                                        order=" -> ".join(primitives) if primitives else "not given")}]
        for j, (d, tl) in enumerate(zip(chunk, tiles)):
            if tl is None:
                continue
            parts += [{"type": "text", "text": f"Step {j + 1}: {d['start_s']:.2f}s to {d['end_s']:.2f}s, options: "
                                               f"{d['options'][0]} or {d['options'][1]}"}, _jpg(tl)]
        prov, r = _ask(lambda mx, p=parts: p)
        _record(getattr(r, "usage", None), f"spark-open adjudicate {Path(video).name} {len(chunk)} steps", prov)
        text = r.choices[0].message.content or "{}"
        try:
            ans = json.loads(text[text.find("{"): text.rfind("}") + 1]).get("steps") or []
        except ValueError:
            ans = []
        for x in ans:
            try:
                k = int(x.get("k")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= k < len(chunk) and x.get("action") in PRIMITIVES:
                out[g + k] = x["action"]
    return out
