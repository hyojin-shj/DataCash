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
    "required": ["reply", "ready", "title", "title_en", "domain", "robot_use", "sensors", "eligibility", "budget_krw", "hours", "missions", "questions",
                 "rarity", "rare_reason"],
    "properties": {
        "reply": {"type": "string", "description": "Reply to the buyer in the requested language: what you understood and what you propose (2-3 sentences)"},
        "rarity": {"type": "string", "enum": list(pricing.RARITY), "description": "How hard the participant conditions are to recruit (e.g. 90+ years old = very_rare, 70-80s = rare, seniors only or one narrow age band = uncommon)"},
        "rare_reason": {"type": "string"},
        "ready": {"type": "boolean", "description": "true when task, sensors and budget-or-hours are known well enough to open missions"},
        "title": {"type": "string", "description": "Short Korean title of the data request"},
        "title_en": {"type": "string", "description": "The same title in English"},
        "domain": {"type": "string"}, "robot_use": {"type": "string"},
        "sensors": {"type": "array", "items": {"type": "string", "enum": list(pricing.SENSORS)}},
        "eligibility": {"type": "object", "additionalProperties": False, "required": ["age_bands", "gender"],
                        "properties": {"age_bands": {"type": "array", "items": {"type": "string", "enum": AGE_BANDS}},
                                       "gender": {"type": "string", "enum": GENDERS}}},
        "budget_krw": {"type": "number", "description": "Total budget in KRW, 0 if not given"},
        "hours": {"type": "number", "description": "Requested accepted hours, 0 if not given"},
        "missions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "title", "title_en", "guide", "guide_en", "purpose", "purpose_en", "objects", "primitives", "situations"],
            "properties": {
                "id": {"type": "string", "description": "id of an existing mission from the list if it fits, else empty string"},
                "title": {"type": "string", "description": "Korean mission title a person at home can do, e.g. '상자 들어 선반에 올리기'"},
                "title_en": {"type": "string"},
                "guide": {"type": "string", "description": "Korean one-line how-to incl. duration and repetitions, e.g. '... · 30초~1분 · 5번 반복'"},
                "guide_en": {"type": "string", "description": "the same guide in English"},
                "purpose": {"type": "string", "description": "Korean: what the robot learns from it"},
                "purpose_en": {"type": "string", "description": "the same purpose in English"},
                "objects": {"type": "array", "items": {"type": "string"}, "description": "English nouns SAM 3.1 can find, max 3"},
                "primitives": {"type": "array", "items": {"type": "string", "enum": PRIMITIVES}},
                "situations": {"type": "array", "items": {"type": "string"}, "description": "4-6 short Korean conditions, change one thing at a time"}}}},
        "questions": {"type": "array", "items": {"type": "string"}, "description": "Korean questions for missing info"},
    },
}

SYSTEM = """You design data-collection missions for DataCash. Ordinary people record short tasks at home with a phone
(+ optional smartwatch). Buyers are robotics/physical-AI companies. Turn the buyer's request into 1-3 missions people can
safely do at home with household objects (no heavy machinery, no dangerous loads: boxes up to ~10 kg, stay realistic).
Each mission starts with reach and ends with release, using only these primitives: {prims}.
Sensor packs (always include video): {sensors}.
If the buyer names data (e.g. hand joints -> pose, grip/touch -> touch + wrist, heart rate -> physio, sound -> audio), include it.
Participants are adults only; age bands: {ages}; gender any/female/male.
Price per accepted hour = sum of chosen packs (custom collection x1.5, min 10 h, setup 500,000 KRW). If a budget is given,
you may leave hours 0; the server fits hours to the budget. Never invent personal data.
Existing missions we already collect (reuse their id when one fits, the buyer may then buy stored data): {existing}
Write "reply" and "questions" in {lang}; titles/guides/purposes stay Korean (title_en, guide_en, purpose_en in English). Be concise."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


LANG_NAME = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh-CN": "Simplified Chinese", "es": "Spanish"}


def system_prompt(existing: str = "", lang: str = "ko") -> str:
    return SYSTEM.format(prims=", ".join(PRIMITIVES), ages=", ".join(AGE_BANDS), existing=existing or "none", lang=LANG_NAME.get(lang, "English"),
                         sensors="; ".join(f"{k}={v[0]} ({v[1]}) {v[2]:,} KRW/h" for k, v in pricing.SENSORS.items()))


def fit_quote(plan: dict) -> dict:
    """의뢰 계획 → 맞춤(신규) 수집 견적. 예산이 있으면 그 안에서 가장 많은 시간 (최소 10시간). 모으기 어려운 조건은 배율."""
    sensors = pricing.sensors_for(None, plan.get("sensors") or ["video"])
    rar = pricing.RARITY.get(plan.get("rarity") or "normal", 1.0)
    hours = float(plan.get("hours") or 0)
    budget = float(plan.get("budget_krw") or 0)
    if budget > 0 and hours <= 0:
        hours = _fit(sensors, budget, True, rar)
    q = pricing.quote(max(hours, pricing.CUSTOM_MIN_HOURS), custom=True, sensors=sensors, rarity=rar)
    return {**q, "budget_krw": budget, "within_budget": budget <= 0 or q["total"] <= budget,
            "reward_per_hour": pricing.REWARD_PER_HOUR[q["level"]]}


def chat(messages: List[dict], client=None, catalog: Optional[dict] = None, lang: str = "ko") -> dict:
    """messages: [{role: user|assistant, content}] → {reply, plan, quote, choice(기존·신규·섞기)}."""
    from .meta.client import MetaClient

    c = client or MetaClient()
    existing = "; ".join(f'{m["id"]}={m.get("title_en") or m["title"]}' for m in (catalog or {}).get("missions", []) if not m["id"].startswith("req_"))
    msgs = [{"role": "system", "content": system_prompt(existing, lang)}] + [
        {"role": m["role"], "content": str(m["content"])[:4000]} for m in messages[-12:] if m.get("role") in ("user", "assistant")]
    plan = c.chat_json(MODEL, msgs, SCHEMA, "data_request", "spark request", estimate_usd=0.01,
                       reasoning_effort="low")     # 20초 → 10초 (바로 나오는 초안이 먼저 보이고 이걸로 다듬는다)
    plan["sensors"] = pricing.sensors_for(None, plan.get("sensors") or ["video"])
    plan["missions"] = [m for m in plan.get("missions", [])[:3] if m.get("primitives")]
    for m in plan["missions"]:
        m["primitives"] = [p for p in m["primitives"] if p in PRIMITIVES] or ["reach", "grasp", "release"]
        m["objects"] = m.get("objects", [])[:3]
        m["situations"] = m.get("situations", [])[:12]
    elig = plan.get("eligibility") or {}
    elig["age_bands"] = [a for a in elig.get("age_bands", []) if a in AGE_BANDS] or list(AGE_BANDS)
    plan["eligibility"] = elig
    plan["source"] = "spark"
    if plan.get("rarity") not in pricing.RARITY:
        plan["rarity"] = "normal"
    out = {"reply": plan.pop("reply", ""), "plan": plan, "quote": fit_quote(plan)}
    if catalog is not None:
        out["choice"] = options(plan, catalog)
    return out


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
                             "guide": m["guide"], "guide_en": m.get("guide_en", ""), "purpose": m["purpose"],
                             "purpose_en": m.get("purpose_en", ""), "objects": m["objects"], "primitives": m["primitives"],
                             "situations": m["situations"], "sponsor": org.strip()[:60], "request": rid,
                             "eligibility": plan["eligibility"], "sensors": q["sensors"], "level": q["level"],
                             "target_hours": round(q["hours"] / max(1, len(plan["missions"])), 2)})
        r = {"id": rid, "org": org.strip()[:120], "email": email.strip()[:200], "title": plan.get("title", ""),
             "title_en": plan.get("title_en", ""), "domain": plan.get("domain", ""), "robot_use": plan.get("robot_use", ""), "plan": plan, "quote": q,
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


# ---------- 바로 나오는 초안 (모델 없이, 0.1초) + 기존·신규·섞기 선택 ----------
# 말로 의뢰는 Spark 가 미션을 설계하는 데 16초쯤 걸린다 → 먼저 문장에서 예산·시간·연령·성별·센서를 읽고
# 보관 중인 미션과 짝지어 바로 보여 주고, Spark 결과가 오면 덮어쓴다.

SYN = {   # 말 → 미션 (보관 데이터와 짝짓기)
    "lift_box": ["택배", "상하차", "박스", "상자", "box", "parcel", "carry", "lift", "물류", "logistics", "무거운", "heavy"],
    "pack_box": ["포장", "packing", "pack", "택배", "물류", "logistics", "e-commerce", "이커머스"],
    "tape_box": ["테이프", "tape", "봉하", "seal", "포장", "packing"],
    "restock_shelf": ["선반", "진열", "shelf", "restock", "retail", "리테일", "편의점", "마트", "store", "물류"],
    "pick_from_bin": ["피킹", "picking", "bin", "통에서", "물류", "warehouse", "창고", "logistics"],
    "fold_cloth": ["빨래", "개기", "옷", "수건", "laundry", "fold", "cloth", "towel", "가사", "household"],
    "hang_clothes": ["옷걸이", "걸기", "hang", "hanger", "옷", "laundry", "가사"],
    "pour_water": ["따르", "물", "pour", "water", "주방", "kitchen", "요리", "cook", "카페", "cafe"],
    "plug_cable": ["케이블", "충전", "cable", "plug", "조립", "assembly", "전자", "electronics"],
    "screw_bolt": ["나사", "드라이버", "screw", "bolt", "조립", "assembly", "제조", "manufactur", "공장", "factory"],
    "open_drawer": ["서랍", "drawer", "가구", "furniture", "가사", "household"],
    "pick_place_cup": ["컵", "cup", "집어", "pick", "place", "옮기", "주방", "kitchen"],
    "pull_tissue": ["티슈", "휴지", "tissue", "뽑"],
    "spray_bottle": ["분무", "스프레이", "spray", "청소", "cleaning"],
    "twist_cap": ["뚜껑", "병", "cap", "bottle", "twist", "열고", "제조", "assembly"],
    "wipe_surface": ["닦", "wipe", "청소", "cleaning", "주방"],
    "handover": ["건네", "전달", "handover", "hand over", "협업", "collaborat", "서비스 로봇", "service robot"],
    "stack_blocks": ["블록", "쌓", "stack", "block", "조립", "assembly"],
    "tear_pouch": ["파우치", "봉지", "뜯", "pouch", "tear", "식품", "food"],
    "squeeze_bottle": ["짜기", "소스", "튜브", "squeeze", "sauce", "tube"],
    "pull_stacked_cup": ["겹친", "stacked", "컵"],
    "pick_small_fruit": ["과일", "포도", "fruit", "grape", "농업", "agri", "수확", "harvest"],
}
SENSOR_WORDS = {"touch": ["쥐는", "쥐기", "촉각", "grip", "touch", "tactile", "힘"], "wrist": ["손목", "wrist", "워치", "watch", "쥐는", "grip", "touch", "촉각"],
                "pose": ["관절", "joint", "pose", "손 모양", "skeleton", "keypoint"], "physio": ["심박", "heart", "생체", "피로", "fatigue", "physio"],
                "audio": ["소리", "sound", "audio", "음성"]}
RARE_WORDS = {"very_rare": ["90대", "100세", "90s", "nonagenarian", "장애", "disab", "휠체어", "wheelchair"],
              "rare": ["80대", "80s", "70대", "70s", "임산부", "pregnan", "외과", "의사", "surgeon", "doctor", "간호사", "nurse", "용접", "weld"],
              "uncommon": ["노인", "어르신", "시니어", "senior", "elderly", "왼손잡이", "left-hand", "60대", "60s", "요리사", "chef"]}


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_budget(t: str) -> float:
    import re

    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(억|천만|백만|만)\s*원?", t)
    if m:
        return _num(m.group(1)) * {"억": 1e8, "천만": 1e7, "백만": 1e6, "만": 1e4}[m.group(2)]
    m = re.search(r"(\d[\d,]*)\s*원", t)
    if m and _num(m.group(1)) >= 100_000:
        return _num(m.group(1))
    m = re.search(r"(?:\$|usd\s*)(\d[\d,]*(?:\.\d+)?)\s*(k|m)?", t, re.I) or re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\s*(?:usd|dollars?)", t, re.I)
    if m:
        return _num(m.group(1)) * {"k": 1e3, "m": 1e6}.get((m.group(2) or "").lower(), 1) * 1400
    return 0.0


def parse_hours(t: str) -> float:
    import re

    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(시간|hours?|hrs?|h\b)", t, re.I)
    return _num(m.group(1)) if m else 0.0


def parse_ages(t: str) -> List[str]:
    import re

    dec = {"19~29세": 2, "30대": 3, "40대": 4, "50대": 5, "60대 이상": 6}
    m = re.search(r"(\d)0\s*대\s*(이하|미만|까지)", t) or re.search(r"(?:under|below)\s*(\d)0", t, re.I)
    if m:
        d = int(m.group(1)) - (1 if (m.lastindex or 0) >= 2 and m.group(2) == "미만" else 0)
        return [b for b, k in dec.items() if k <= d] or AGE_BANDS[:1]
    m = re.search(r"(\d)0\s*대\s*(이상|부터)", t) or re.search(r"(\d)0\s*s?\s*(?:and|or)\s*(?:over|older|above)|(?:over|above)\s*(\d)0", t, re.I)
    if m:
        d = int(next(g for g in m.groups() if g and g.isdigit()))
        return [b for b, k in dec.items() if k >= min(d, 6)]
    found = [b for b, k in dec.items() if re.search(fr"{k}0\s*대|{k}0s", t)] + (["60대 이상"] if re.search(r"[7-9]0\s*대|[7-9]0s|노인|어르신|시니어|senior|elderly", t, re.I) else [])
    return sorted(set(found), key=AGE_BANDS.index) or list(AGE_BANDS)


def parse_rarity(t: str, ages: List[str], gender: str) -> tuple:
    tl = t.lower()
    for lv in ("very_rare", "rare", "uncommon"):
        hit = [w for w in RARE_WORDS[lv] if w.lower() in tl]
        if hit:
            return lv, hit[0]
    if ages != list(AGE_BANDS) and len(ages) <= 1:
        return "uncommon", ages[0] if ages else ""
    if gender != "any" and ages != list(AGE_BANDS):
        return "uncommon", "gender+age"
    return "normal", ""


def instant(messages: List[dict], catalog: dict) -> dict:
    """모델 없이 바로 만드는 초안 plan (Spark plan 과 같은 모양 + source='instant')."""
    t = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
    tl = t.lower()
    ages = parse_ages(t)
    gender = "female" if any(w in tl for w in ("여성", "여자", "female", "women", "woman")) else \
        "male" if any(w in tl for w in ("남성", "남자", "male", " men", "man ")) else "any"
    sensors = ["video", "pose", "audio"] + [k for k, ws in SENSOR_WORDS.items() if k not in ("pose", "audio") and any(w in tl for w in ws)]
    rarity, why = parse_rarity(t, ages, gender)
    cat = {m["id"]: m for m in catalog.get("missions", [])}
    score = {mid: sum(1 for w in ws if w.lower() in tl) for mid, ws in SYN.items() if mid in cat}
    picks = [mid for mid, s in sorted(score.items(), key=lambda kv: -kv[1]) if s > 0][:3]
    missions = [{"id": mid, "title": cat[mid]["title"], "title_en": cat[mid].get("title_en", ""), "guide": cat[mid].get("guide", ""),
                 "purpose": cat[mid].get("purpose", ""), "objects": cat[mid].get("objects", []), "primitives": cat[mid].get("primitives", []),
                 "situations": [], "existing": True} for mid in picks]
    plan = {"title": missions[0]["title"] if missions else "", "title_en": missions[0]["title_en"] if missions else "", "domain": "", "robot_use": "", "sensors": pricing.sensors_for(None, sensors),
            "eligibility": {"age_bands": ages, "gender": gender}, "budget_krw": parse_budget(t), "hours": parse_hours(t),
            "missions": missions, "questions": [], "ready": bool(missions), "rarity": rarity, "rare_reason": why, "source": "instant"}
    return plan


def _fit(sensors: list, budget: float, custom: bool, rarity: float = 1.0, cap: float = 5000.0) -> float:
    """예산 안에서 가장 많은 시간 (할인 구간이 있어 이분 탐색)."""
    lo, hi = (pricing.CUSTOM_MIN_HOURS if custom else 0.0), cap
    if pricing.quote(lo, custom=custom, sensors=sensors, rarity=rarity)["total"] > budget:
        return lo
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pricing.quote(mid, custom=custom, sensors=sensors, rarity=rarity)["total"] <= budget else (lo, mid)
    return int(lo * 10) / 10       # 내림 (반올림하면 예산을 조금 넘는다)


MIN_REUSE_HOURS = 1.0     # 보관 데이터가 이보다 적으면 기존·섞기를 내놓지 않는다 (2분짜리 ₩3,000 같은 선택지는 의미가 없다)
REUSE_SHARE = 0.1         # 섞기: 보관 데이터가 전체의 10% 는 넘어야


def options(plan: dict, catalog: dict) -> dict:
    """기존 데이터 · 신규 수집 · 섞기 — 예산·보관량·참여 조건을 보고 추천.
    기존: 보관 중인 같은 미션 데이터를 판매가로 (착수비 없음, 바로 받음). 참여 조건이 특별하면 기존 데이터로는 못 맞춘다.
    신규: 맞춤 수집 × 1.5 × 조건 배율(pricing.RARITY) + 착수비. 섞기: 기존을 먼저 쓰고 모자란 만큼 신규."""
    sensors = pricing.sensors_for(None, plan.get("sensors") or ["video"])
    level = pricing.level_of(sensors)
    rar_key = plan.get("rarity") or "normal"
    rar = pricing.RARITY.get(rar_key, 1.0)
    budget = float(plan.get("budget_krw") or 0)
    want = float(plan.get("hours") or 0)
    cat = {m["id"]: m for m in catalog.get("missions", [])}
    ids = [m.get("id") for m in plan.get("missions", []) if m.get("id") in cat]
    stock = round(sum((cat[i].get("hours") or {}).get(level, 0) for i in ids), 2)
    restricted = rar_key != "normal" or (plan.get("eligibility") or {}).get("age_bands", AGE_BANDS) != list(AGE_BANDS) \
        or (plan.get("eligibility") or {}).get("gender", "any") != "any"
    out = {}
    if stock >= MIN_REUSE_HOURS and not restricted:
        h = min(stock, want) if want else (min(stock, _fit(sensors, budget, False)) if budget else stock)
        q = pricing.quote(h, sensors=sensors)
        out["existing"] = {**q, "stock_hours": stock, "missions": ids, "delivery": "now"}
    if want:
        nh = want
    elif budget:
        nh = _fit(sensors, budget, True, rar)
    else:
        nh = float(pricing.CUSTOM_MIN_HOURS)
    out["new"] = {**pricing.quote(nh, custom=True, sensors=sensors, rarity=rar), "delivery": "collect"}
    if "existing" in out and out["existing"]["hours"] < (want or 1e9) and (not want or stock >= REUSE_SHARE * want):
        ex = pricing.quote(stock, sensors=sensors)
        if want:
            rest = max(pricing.CUSTOM_MIN_HOURS, want - stock)
        elif budget and budget > ex["total"] + pricing.CUSTOM_SETUP:
            rest = _fit(sensors, budget - ex["total"], True, rar)
        else:
            rest = 0
        if rest > 0:
            nq = pricing.quote(rest, custom=True, sensors=sensors, rarity=rar)
            out["mix"] = {"hours": round(stock + nq["hours"], 2), "existing_hours": stock, "new_hours": nq["hours"],
                          "total": ex["total"] + nq["total"], "existing": ex, "new": nq, "sensors": sensors, "delivery": "partly_now"}
    if "mix" in out and budget > 0:                 # 섞기가 반올림으로 예산을 조금 넘으면 신규 시간을 줄인다
        while out["mix"]["total"] > budget and out["mix"]["new_hours"] > pricing.CUSTOM_MIN_HOURS:
            nq = pricing.quote(out["mix"]["new_hours"] - 0.5, custom=True, sensors=sensors, rarity=rar)
            ex = out["mix"]["existing"]
            out["mix"].update(hours=round(stock + nq["hours"], 2), new_hours=nq["hours"], total=ex["total"] + nq["total"], new=nq)
    within = lambda o: budget <= 0 or o["total"] <= budget * 1.001  # noqa: E731
    if restricted or "existing" not in out:
        rec, why = "new", ("conditions" if restricted else "stock_small" if stock > 0 else "no_stock")
    elif want and stock >= want:
        rec, why = "existing", "stock_enough"
    else:                                         # 예산 안에서 가장 많은 시간, 같으면 싼 쪽 (기존은 바로 받는다)
        ok = {k: o for k, o in out.items() if within(o)} or out
        rec = max(ok, key=lambda k: (round(ok[k]["hours"], 1), -ok[k]["total"]))
        why = {"existing": "stock_enough", "mix": "stock_short", "new": "stock_small"}[rec]
    return {"options": out, "recommended": rec, "why": why, "rarity": rar_key, "rarity_mult": rar,
            "rare_reason": plan.get("rare_reason", ""), "stock_hours": stock, "budget_krw": budget,
            "reward_per_hour": pricing.REWARD_PER_HOUR[level]}
