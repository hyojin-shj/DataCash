"""Muse Glimmer (Meta 오픈 웨이트, 이 맥에서 llama.cpp 로) — Muse Spark 가 붙인 단계 라벨을 장면으로 다시 확인한다.

Spark 단계마다 가운데 장면(가린 영상) 한 장을 보여 주고 "이 손이 지금 <동작>을 <물체>에 하고 있나?"를 묻는다.
대답(맞다/아니다 + 보이는 기본 동작)이 단계 검증의 증거 하나가 된다 (verify.py 의 glimmer).
모델 파일: data/glimmer/ (30B Q4_K_M 17GB + 그림 투영 1.4GB). 서버: llama-server --port 8090 (DATACASH_GLIMMER).
API 비용 없음 — 내 하드웨어에서 돈다.
"""

from __future__ import annotations

import base64
import json
import os
from typing import List, Optional

import cv2

from .spark import PRIMITIVES

URL = os.environ.get("DATACASH_GLIMMER", "http://127.0.0.1:8090/v1")
MAX_STEPS = 8          # 한 장에 30초 안팎 — 녹화 하나에 몇 분 안으로

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["agree", "seen", "note"],
          "properties": {"agree": {"type": "boolean"}, "seen": {"type": "string", "enum": PRIMITIVES},
                         "note": {"type": "string"}}}

PROMPT = """You check labels for robot-learning videos. Gray areas are privacy masks; ignore them.
Claimed step: action "{action}" with {hand} hand on {objects}. Instruction: "{instruction}".
Look at this frame from the middle of that step. Does the frame plausibly show that action? Answer JSON:
agree (true/false), seen (the primitive you actually see, one of: {prims}), note (max 12 words)."""


MODEL_FILE = "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf"
_proc = None


def ensure(wait_s: float = 240, log=print, lora=None) -> bool:
    """Glimmer 서버가 없으면 켠다 (메모리 17GB 를 쓰므로 평소엔 꺼 두고 필요할 때만). 켜졌으면 True.
    lora: 학습한 어댑터(GGUF) — 모델 진화로 만든 버전을 붙여 켠다 (이미 켜져 있으면 그대로 쓴다)."""
    import shutil
    import subprocess
    import time

    global _proc
    if available():
        return True
    from .. import config

    d = config.DATA / "glimmer"
    exe = shutil.which("llama-server")
    if exe is None or not (d / MODEL_FILE).exists() or not (d / MMPROJ_FILE).exists():
        log("Glimmer 를 켤 수 없음 (llama-server 또는 모델 파일 없음)")
        return False
    port = URL.rsplit(":", 1)[1].split("/")[0]
    _proc = subprocess.Popen([exe, "-m", str(d / MODEL_FILE), "--mmproj", str(d / MMPROJ_FILE), "--port", port,
                              "-ngl", "99", "--jinja", "-c", "8192"] + (["--lora", str(lora)] if lora else []),
                             stdout=open(d / "server.log", "ab"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if available():
            log(f"Glimmer 켬 ({time.time() - t0:.0f}초)")
            return True
        if _proc.poll() is not None:
            break
        time.sleep(2)
    stop()
    return False


def stop() -> None:
    """ensure() 가 켠 Glimmer 만 끈다 (사용자가 따로 켠 것은 두기)."""
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(20)
        except Exception:
            _proc.kill()
    _proc = None


def available(timeout: float = 1.0) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(URL.rsplit("/v1", 1)[0] + "/health", timeout=timeout) as r:
            return b"ok" in r.read()
    except Exception:
        return False


def frame_at(video, t_s: float, side: int = 640):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    k = side / max(f.shape[:2])
    return cv2.resize(f, None, fx=k, fy=k) if k < 1 else f


def check_steps(video, steps: List[dict], client=None, log=print) -> List[Optional[dict]]:
    """단계마다 {agree, seen, note, model} (못 물어본 단계는 None). 단계가 많으면 고르게 MAX_STEPS 개만."""
    from openai import OpenAI

    c = client or OpenAI(base_url=URL, api_key="local", timeout=180)
    pick = set(range(len(steps))) if len(steps) <= MAX_STEPS else \
        {round(i * (len(steps) - 1) / (MAX_STEPS - 1)) for i in range(MAX_STEPS)}
    out: List[Optional[dict]] = []
    for i, st in enumerate(steps):
        if i not in pick:
            out.append(None)
            continue
        f = frame_at(video, (st["start_s"] + st["end_s"]) / 2)
        if f is None:
            out.append(None)
            continue
        img = base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
        text = PROMPT.format(action=st.get("action"), hand=st.get("hand", "a"), objects=", ".join(st.get("objects") or []) or "an object",
                             instruction=st.get("instruction", ""), prims=", ".join(PRIMITIVES))
        try:
            r = c.chat.completions.create(model="muse-glimmer", max_tokens=400, temperature=0,
                                          extra_body={"reasoning_effort": "low"},
                                          response_format={"type": "json_schema", "json_schema": {"name": "check", "schema": SCHEMA}},
                                          messages=[{"role": "user", "content": [{"type": "text", "text": text},
                                                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}]}])
            ans = json.loads(r.choices[0].message.content or "{}")
            out.append({"agree": bool(ans.get("agree")), "seen": ans.get("seen"), "note": str(ans.get("note", ""))[:120],
                        "model": "muse-glimmer-30b-q4 (local)"})
        except Exception as e:
            log(f"Glimmer 확인 실패 {i}: {type(e).__name__}")
            out.append(None)
    return out

