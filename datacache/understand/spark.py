"""Muse Spark 1.3 로 가린 영상 구간 이해 → 구조화 출력 (Chat Completions + json_schema, strict).

가리기가 끝난 영상만 보낸다 (원본·얼굴은 보내지 않는다). 긴 녹화는 30초 안팎 구간으로 나눠 보내고 시각을 이어 붙인다.
Muse Wonder 는 한국에서 쓸 수 없어 Spark 1.3 을 쓴다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..meta.client import MetaClient, data_url

MODEL = "muse-spark-1.3"

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["task", "completed", "steps", "mistakes", "environment"],
    "properties": {
        "task": {"type": "string", "description": "One-line description of the household task"},
        "completed": {"type": "string", "enum": ["yes", "partial", "no", "unclear"]},
        "steps": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["start_s", "end_s", "instruction", "hand", "objects", "action"],
            "properties": {
                "start_s": {"type": "number"}, "end_s": {"type": "number"},
                "instruction": {"type": "string", "description": "Imperative robot instruction, e.g. 'Pick up the plate with the right hand'"},
                "hand": {"type": "string", "enum": ["left", "right", "both", "none", "unclear"]},
                "objects": {"type": "array", "items": {"type": "string"}},
                "action": {"type": "string", "enum": ["reach", "grasp", "lift", "carry", "place", "release", "wipe", "scrub",
                                                      "rinse", "pour", "cut", "stir", "push", "open", "close", "other"]}}}},
        "mistakes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["t_s", "description", "recovered"],
            "properties": {"t_s": {"type": "number"}, "description": {"type": "string"}, "recovered": {"type": "boolean"}}}},
        "environment": {"type": "object", "additionalProperties": False, "required": ["place", "tools", "materials", "lighting"],
                        "properties": {"place": {"type": "string"}, "tools": {"type": "array", "items": {"type": "string"}},
                                       "materials": {"type": "array", "items": {"type": "string"}},
                                       "lighting": {"type": "string"}}},
    },
}

PROMPT = """You label household-task videos for training humanoid robots. Gray regions are privacy masks (ignore them).
Mission: {mission}. The clip is {dur:.1f} seconds long.
Return: the task, whether it was completed, consecutive steps covering the active part of the clip with start/end seconds
(relative to clip start, within 0..{dur:.1f}), an imperative instruction a robot could follow, which hand, objects touched,
and the action type; any mistakes (drops, misses, re-grasps, spills) with time and whether they were recovered; and the
environment (place type, tools, materials, lighting). Only describe what is visible. Do not guess identities."""


def understand_clip(clip, mission: str, duration_s: float, client: Optional[MetaClient] = None,
                    model: str = MODEL) -> dict:
    c = client or MetaClient()
    out = c.chat_json(model, [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.format(mission=mission, dur=duration_s)},
        {"type": "video_url", "video_url": {"url": data_url(clip)}}]}],
        SCHEMA, "household_labels", f"spark understand {Path(clip).name}", estimate_usd=0.05)
    for st in out["steps"]:   # 구간 밖 시각은 잘라 둔다
        st["start_s"] = float(max(0.0, min(duration_s, st["start_s"])))
        st["end_s"] = float(max(st["start_s"], min(duration_s, st["end_s"])))
    out["steps"] = sorted([s for s in out["steps"] if s["end_s"] > s["start_s"]], key=lambda s: s["start_s"])
    out["model"] = model
    return out
