"""자연어 의뢰 → 미션 — 구매 기업이 채팅으로 필요한 데이터를 말하면, Muse Spark 가 거꾸로 미션을 설계한다.

예: "제조업 회사야. 택배 상하차 로봇을 학습시킬 데이터가 필요해. 성별 무관 50대 이하, 예산 300만 원, 손 관절이랑 쥐는 세기 꼭"
→ Spark (구조화 출력): 필요한 로봇 동작(기본 동작), 물체, 찍을 상황(조건을 하나씩 바꿔), 참여 조건(연령대·성별),
  센서 묶음, 예산·시간, 모르면 되물을 질문
→ 서버가 견적(맞춤 수집)을 계산해 예산에 맞는 시간을 정하고, 승인하면 미션을 앱에 연다.
  참여 조건에 맞는 사람에게만 보이므로 '다양성·분포 관리'를 따로 하지 않고 의뢰 조건이 곧 수집 조건이 된다.
저장: data/server/requests/<id>.json
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from . import pricing
from .understand.spark import PRIMITIVES

MODEL = "muse-spark-1.3"
AGE_BANDS = ["19~29세", "30대", "40대", "50대", "60대 이상"]
GENDERS = ["any", "female", "male"]

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["reply", "ready", "title", "domain", "robot_use", "sensors", "eligibility", "budget_krw", "hours", "missions", "questions"],
    "properties": {
        "reply": {"type": "string", "description": "Korean reply to the buyer: what you understood and what you propose (2-4 sentences)"},
        "ready": {"type": "boolean", "description": "true when task, sensors and budget-or-hours are known well enough to open missions"},
        "title": {"type": "string", "description": "Short Korean title of the data request"},
        "domain": {"type": "string"}, "robot_use": {"type": "string"},
        "sensors": {"type": "array", "items": {"type": "string", "enum": list(pricing.SENSORS)}},
        "eligibility": {"type": "object", "additionalProperties": False, "required": ["age_bands", "gender"],
                        "properties": {"age_bands": {"type": "array", "items": {"type": "string", "enum": AGE_BANDS}},
                                       "gender": {"type": "string", "enum": GENDERS}}},
        "budget_krw": {"type": "number", "description": "Total budget in KRW, 0 if not given"},
        "hours": {"type": "number", "description": "Requested accepted hours, 0 if not given"},
        "missions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["title", "title_en", "guide", "purpose", "objects", "primitives", "situations"],
            "properties": {
                "title": {"type": "string", "description": "Korean mission title a person at home can do, e.g. '상자 들어 선반에 올리기'"},
                "title_en": {"type": "string"},
                "guide": {"type": "string", "description": "Korean one-line how-to incl. duration and repetitions, e.g. '... · 30초~1분 · 5번 반복'"},
                "purpose": {"type": "string", "description": "Korean: what the robot learns from it"},
                "objects": {"type": "array", "items": {"type": "string"}, "description": "English nouns SAM 3.1 can find, max 3"},
                "primitives": {"type": "array", "items": {"type": "string", "enum": PRIMITIVES}},
                "situations": {"type": "array", "items": {"type": "string"}, "description": "6-10 Korean conditions, change one thing at a time"}}}},
        "questions": {"type": "array", "items": {"type": "string"}, "description": "Korean questions for missing info"},
    },
}

SYSTEM = """You design data-collection missions for DataCache. Ordinary people record short tasks at home with a phone
(+ optional smartwatch). Buyers are robotics/physical-AI companies. Turn the buyer's request into 1-3 missions people can
safely do at home with household objects (no heavy machinery, no dangerous loads: boxes up to ~10 kg, stay realistic).
Each mission starts with reach and ends with release, using only these primitives: {prims}.
Sensor packs (always include video): {sensors}.
If the buyer names data (e.g. hand joints -> pose, grip/touch -> touch + wrist, heart rate -> physio, sound -> audio), include it.
Participants are adults only; age bands: {ages}; gender any/female/male.
Price per accepted hour = sum of chosen packs (custom collection x1.5, min 10 h, setup 500,000 KRW). If a budget is given,
you may leave hours 0; the server fits hours to the budget. Never invent personal data. Reply in Korean, warm and concise."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def system_prompt() -> str:
    return SYSTEM.format(prims=", ".join(PRIMITIVES), ages=", ".join(AGE_BANDS),
                         sensors="; ".join(f"{k}={v[0]} ({v[1]}) {v[2]:,} KRW/h" for k, v in pricing.SENSORS.items()))


def fit_quote(plan: dict) -> dict:
    """의뢰 계획 → 맞춤 수집 견적. 예산이 있으면 그 안에서 가장 많은 시간 (최소 10시간)."""
    sensors = pricing.sensors_for(None, plan.get("sensors") or ["video"])
    hours = float(plan.get("hours") or 0)
    budget = float(plan.get("budget_krw") or 0)
    if budget > 0 and hours <= 0:
        lo, hi = pricing.CUSTOM_MIN_HOURS, 5000.0
        if pricing.quote(lo, custom=True, sensors=sensors)["total"] > budget:
            hours = lo
        else:
            for _ in range(40):                      # 할인 구간이 있어 이분 탐색
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if pricing.quote(mid, custom=True, sensors=sensors)["total"] <= budget else (lo, mid)
            hours = int(lo)
    q = pricing.quote(max(hours, pricing.CUSTOM_MIN_HOURS), custom=True, sensors=sensors)
    return {**q, "budget_krw": budget, "within_budget": budget <= 0 or q["total"] <= budget,
            "reward_per_hour": pricing.REWARD_PER_HOUR[q["level"]]}


def chat(messages: List[dict], client=None) -> dict:
    """messages: [{role: user|assistant, content}] → {reply, plan, quote}."""
    from .meta.client import MetaClient

    c = client or MetaClient()
    msgs = [{"role": "system", "content": system_prompt()}] + [
        {"role": m["role"], "content": str(m["content"])[:4000]} for m in messages[-12:] if m.get("role") in ("user", "assistant")]
    plan = c.chat_json(MODEL, msgs, SCHEMA, "data_request", "spark request", estimate_usd=0.01)
    plan["sensors"] = pricing.sensors_for(None, plan.get("sensors") or ["video"])
    plan["missions"] = [m for m in plan.get("missions", [])[:3] if m.get("primitives")]
    for m in plan["missions"]:
        m["primitives"] = [p for p in m["primitives"] if p in PRIMITIVES] or ["reach", "grasp", "release"]
        m["objects"] = m.get("objects", [])[:3]
        m["situations"] = m.get("situations", [])[:12]
    elig = plan.get("eligibility") or {}
    elig["age_bands"] = [a for a in elig.get("age_bands", []) if a in AGE_BANDS] or list(AGE_BANDS)
    plan["eligibility"] = elig
    return {"reply": plan.pop("reply", ""), "plan": plan, "quote": fit_quote(plan)}


class Requests:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def all(self) -> List[dict]:
        return sorted((json.loads(p.read_text()) for p in self.root.glob("*.json")), key=lambda r: r["created"], reverse=True)

    def get(self, rid: str) -> dict:
        p = self.root / f"{rid}.json"
        if not rid.isalnum() or not p.exists():
            raise KeyError(rid)
        return json.loads(p.read_text())

    def save(self, r: dict) -> dict:
        (self.root / f"{r['id']}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        return r

    def submit(self, plan: dict, org: str, email: str, conversation: Optional[list] = None) -> dict:
        """의뢰 확정 → 미션을 앱에 연다 (참여 조건에 맞는 사람에게만). 데이터 대금은 운영자가 계약 뒤 주문으로 처리."""
        if not plan.get("missions"):
            raise ValueError("미션이 없는 의뢰예요 — 무엇을 찍어야 하는지 더 알려 주세요")
        q = fit_quote(plan)
        rid = secrets.token_hex(5)
        missions = []
        for k, m in enumerate(plan["missions"]):
            missions.append({"id": f"req_{rid}_{k}", "title": m["title"], "title_en": m.get("title_en", m["title"]),
                             "guide": m["guide"], "purpose": m["purpose"], "objects": m["objects"], "primitives": m["primitives"],
                             "situations": m["situations"], "sponsor": org.strip()[:60], "request": rid,
                             "eligibility": plan["eligibility"], "sensors": q["sensors"], "level": q["level"],
                             "target_hours": round(q["hours"] / max(1, len(plan["missions"])), 2)})
        r = {"id": rid, "org": org.strip()[:120], "email": email.strip()[:200], "title": plan.get("title", ""),
             "domain": plan.get("domain", ""), "robot_use": plan.get("robot_use", ""), "plan": plan, "quote": q,
             "missions": missions, "status": "collecting", "created": _now(), "conversation": (conversation or [])[-12:]}
        return self.save(r)

    def live_missions(self) -> List[dict]:
        return [m for r in self.all() if r["status"] == "collecting" for m in r["missions"]]

    def close(self, rid: str) -> dict:
        r = self.get(rid)
        r["status"] = "closed"
        return self.save(r)


def eligible(m: dict, age_band: Optional[str], gender: Optional[str]) -> bool:
    """의뢰 미션의 참여 조건. 연령대·성별을 모르면 조건이 걸린 미션은 보이지 않는다."""
    e = m.get("eligibility") or {}
    bands = e.get("age_bands") or AGE_BANDS
    if set(bands) != set(AGE_BANDS) and age_band not in bands:
        return False
    g = e.get("gender", "any")
    return g == "any" or g == gender
