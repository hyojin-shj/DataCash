"""Muse Spark 1.3 로 가린 영상 구간 이해 → 구조화 출력 (Chat Completions + json_schema, strict).

가리기가 끝난 영상만 보낸다 (원본·얼굴은 보내지 않는다). 긴 녹화는 30초 안팎 구간으로 나눠 보내고 시각을 이어 붙인다.
Muse Wonder 는 한국에서 쓸 수 없어 Spark 1.3 을 쓴다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..meta.client import MetaClient, data_url

MODEL = "muse-spark-1.3"

# 로봇 조작 기본 동작 (라벨 단위). 미션마다 예상 순서는 missions.json 의 primitives.
PRIMITIVES = ["reach", "grasp", "lift", "pull", "push", "twist", "press", "squeeze", "tear", "wipe", "insert", "transport",
              "place", "handover", "release", "pour", "fold", "other"]
GRASP_TYPES = ["power", "precision", "pinch", "lateral", "two_hand", "none", "unclear"]

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["task", "completed", "steps", "mistakes", "environment"],
    "properties": {
        "task": {"type": "string", "description": "One-line description of the manipulation task"},
        "completed": {"type": "string", "enum": ["yes", "partial", "no", "unclear"]},
        "steps": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["start_s", "end_s", "action", "instruction", "hand", "grasp_type", "objects", "success"],
            "properties": {
                "start_s": {"type": "number"}, "end_s": {"type": "number"},
                "action": {"type": "string", "enum": PRIMITIVES, "description": "One manipulation primitive per step"},
                "instruction": {"type": "string", "description": "Imperative robot instruction, e.g. 'Pinch the top tissue and pull it up'"},
                "hand": {"type": "string", "enum": ["left", "right", "both", "none", "unclear"]},
                "grasp_type": {"type": "string", "enum": GRASP_TYPES,
                               "description": "power=whole hand, precision=fingertips, pinch=thumb+index, lateral=thumb on side"},
                "objects": {"type": "array", "items": {"type": "string"}},
                "success": {"type": "string", "enum": ["yes", "no", "unclear"]}}}},
        "mistakes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["t_s", "description", "recovered"],
            "properties": {"t_s": {"type": "number"}, "description": {"type": "string"}, "recovered": {"type": "boolean"}}}},
        "environment": {"type": "object", "additionalProperties": False, "required": ["place", "tools", "materials", "lighting"],
                        "properties": {"place": {"type": "string"}, "tools": {"type": "array", "items": {"type": "string"}},
                                       "materials": {"type": "array", "items": {"type": "string"}},
                                       "lighting": {"type": "string"}}},
    },
}

PROMPT = """You label hand-manipulation videos for training robot hands and humanoids. Gray regions are privacy masks (ignore them).
Mission: {mission}. Expected primitive order (a hint, not a rule; repetitions are normal): {primitives}.
The clip is {dur:.1f} seconds long.
Split the active part of the clip into consecutive steps, ONE manipulation primitive per step, from this set:
reach (hand moves toward the object), grasp (fingers close on it), lift, pull, push, twist, press (button/trigger), squeeze,
tear, wipe, insert (snap/fit into place), transport (carry while holding), place (set it down), handover (give to/take from
another person), release (fingers open), other. Repeat the sequence for every repetition you see.
For each step give start/end seconds (relative to clip start, within 0..{dur:.1f}), an imperative instruction a robot could
follow, which hand, the grasp type, the objects touched, and whether the step succeeded.
Also list mistakes (drops, slips, misses, re-grasps, tearing too much) with time and whether they were recovered, and the
environment (place type, tools, materials, lighting). Only describe what is visible. Do not guess identities."""


def _hints(mission_id: str = "") -> str:
    """모델 진화: 같은 미션의 정답 예시(evolve.gold_examples) + 지난 바로잡음(evolve.correction_hints)을 붙인다."""
    try:
        from ..evolve import correction_hints, gold_examples

        h = [x for x in (gold_examples(mission_id), correction_hints()) if x]
        return ("\n" + "\n".join(h)) if h else ""
    except Exception:
        return ""


def understand_clip(clip, mission: str, duration_s: float, client: Optional[MetaClient] = None,
                    model: str = MODEL, primitives: Optional[list] = None, reasoning: Optional[str] = None,
                    extra: str = "", mission_id: str = "") -> dict:
    """reasoning: 추론 세기 ('max' 는 이의 신청 다시 보기용 — 느리고 비싸지만 가장 꼼꼼하다). extra: 프롬프트 끝에 붙일 말.
    mission_id: 같은 미션의 정답 예시를 붙인다 (모델 진화)."""
    c = client or MetaClient()
    kw = {"reasoning_effort": reasoning} if reasoning else {}
    out = c.chat_json(model, [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.format(mission=mission, dur=duration_s,
                                               primitives=" -> ".join(primitives) if primitives else "not given")
                 + _hints(mission_id) + (("\n" + extra) if extra else "")},
        {"type": "video_url", "video_url": {"url": data_url(clip)}}]}],
        SCHEMA, "manipulation_labels", f"spark understand{' ' + reasoning if reasoning else ''} {Path(clip).name}",
        estimate_usd=0.3 if reasoning == "max" else 0.05, **kw)
    for st in out["steps"]:   # 구간 밖 시각은 잘라 둔다
        st["start_s"] = float(max(0.0, min(duration_s, st["start_s"])))
        st["end_s"] = float(max(st["start_s"], min(duration_s, st["end_s"])))
    out["steps"] = sorted([s for s in out["steps"] if s["end_s"] > s["start_s"]], key=lambda s: s["start_s"])
    out["model"] = model
    return out
