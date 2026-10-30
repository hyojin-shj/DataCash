"""Muse Spark 1.3 로 가린 영상 구간 이해 → 구조화 출력 (Chat Completions + json_schema, strict).

가리기가 끝난 영상만 보낸다 (원본·얼굴은 보내지 않는다). 동작 구간을 20초씩 잘라 0.5배속으로 보내고 시각을 이어 붙인다 (understand_chunked).
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


CHUNK_S = 20.0     # 동작 구간을 이만큼씩 잘라 보낸다 — 60초를 한 번에 보내면 단계를 고른 간격으로 거칠게 나눈다
SLOW = 2.0         # 그리고 이만큼 느리게 늘려 보낸다 (모델이 실제 1초에서 보는 장면이 는다) — 시각은 다시 나눠 되돌린다
# EPIC-KITCHENS 사람 정답으로 잰 값 (docs/label_eval.md, 새 구간 8개): 60초 한 번에 장면 정확도 36%·경계 430ms →
# 20초 × 0.5배속 47%·230ms. 비용은 1분에 약 $0.045 → $0.09.


def understand_chunked(src, t0: float, t1: float, mission: str, primitives: Optional[list] = None, mission_id: str = "",
                       gain: float = 1.0, work_dir=None, chunk_s: float = CHUNK_S, slow: float = SLOW, side: int = 640,
                       client: Optional[MetaClient] = None, label=None) -> dict:
    """src 의 [t0, t1] 를 chunk_s 씩 잘라 slow 배 느리게 보내고, 단계·실수 시각을 녹화 시각(초)으로 이어 붙인다.
    단계는 가운데가 그 조각에 든 것만 쓰고, 자른 자리에 맞닿은 같은 동작은 하나로 잇는다.
    돌려주는 형식은 understand_clip 과 같다 (task·environment 는 첫 조각, completed 는 마지막 조각).
    label(clip, mission, dur) → understand_clip 형식 (시험용으로 바꿔 끼운다)."""
    from concurrent.futures import ThreadPoolExecutor

    from ..media import clip_mp4

    work = Path(work_dir or Path(src).parent)
    cuts = []
    a = t0
    while a < t1 - 0.5:
        b = min(t1, a + chunk_s)
        if t1 - b < 0.5 * chunk_s and t1 - b > 0:     # 끝에 짧게 남는 조각은 앞 조각에 붙인다
            b = t1
        cuts.append((a, b))
        a = b
    ask = label or (lambda clip, m, dur: understand_clip(clip, m, dur, client=client, primitives=primitives, mission_id=mission_id))

    def one(k):
        a, b = cuts[k]
        clip = clip_mp4(src, work / f"understand_clip_{k}.mp4", a, b - a, max_side=side, audio=False, gain=gain, slow=slow)
        try:
            return ask(clip, mission, (b - a) * slow)
        finally:
            Path(clip).unlink(missing_ok=True)

    with ThreadPoolExecutor(3) as ex:
        outs = list(ex.map(one, range(len(cuts))))
    steps, mistakes = [], []
    for (a, b), u in zip(cuts, outs):
        for s in u["steps"]:
            x = {**s, "start_s": a + s["start_s"] / slow, "end_s": a + s["end_s"] / slow}
            if a <= (x["start_s"] + x["end_s"]) / 2 < b:
                prev = steps[-1] if steps else None
                if prev and prev["action"] == x["action"] and abs(prev["end_s"] - a) < 0.15 and abs(x["start_s"] - a) < 0.15:
                    prev["end_s"] = x["end_s"]          # 자른 자리에서 둘로 갈린 같은 단계
                else:
                    steps.append(x)
        mistakes += [{**m, "t_s": a + m["t_s"] / slow} for m in u["mistakes"]]
    return {"task": outs[0]["task"] if outs else "", "completed": outs[-1]["completed"] if outs else "unclear",
            "environment": outs[0]["environment"] if outs else {"place": "", "tools": [], "materials": [], "lighting": ""},
            "steps": sorted(steps, key=lambda s: s["start_s"]), "mistakes": mistakes,
            "model": f"{MODEL} ({len(cuts)} chunks of {chunk_s:g}s, {slow:g}x slow)"}
