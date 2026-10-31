"""가격 — 2026-10-26 시장 사례로 잡은 값 (docs/1026_meta_hackathon.md 2-3).

판매 (받아들인 시간당): 센서 묶음을 골라 더한다 — 등급은 미리 맞춘 조합이다
  영상(필수 35,000: 가린·치환 영상·폰 IMU·물체 영역·단계·기본 동작 라벨·성공·실수·인증서) + 관절 20,000 + 소리 5,000
  + 손목 IMU 15,000 + 촉각 25,000 + 생체 50,000
  기본 = 영상·관절·소리 60,000원 · 고품질 = + 손목·촉각 100,000원 · 생체 = + 생체 150,000원
맞춤 수집: 판매가 × 1.5, 최소 10시간, 착수비 500,000원 · 모으기 어려운 조건(고령·한 성별·좁은 연령 등) × 1.3~2.5
묶음 할인: 100시간 이상 20%, 500시간 이상 35%
정제 요청: 금액의 10%, 최소 50,000원
촬영자 보상 (받아들인 시간당, 판매가의 약 1/3): 기본 20,000 · 고품질 33,000 · 생체 50,000원
  + 부족한 조건 보너스(+50%) · 오늘의 미션(+50%)
"""

from __future__ import annotations

from typing import Optional

LEVELS = ("basic", "premium", "bio")
LEVEL_KO = {"basic": "기본", "premium": "고품질", "bio": "생체"}
LEVEL_DESC = {
    "basic": "가린·치환 영상 + 폰 IMU + 손·전신 관절 + 물체 영역 + 단계 지시문 + 접촉 이벤트 + 성공·실수 + 품질 인증서",
    "premium": "기본 + 손목 IMU(ms 동기화) + 촉각: 접촉 단계·워치로 잰 쥐는 세기(GripSense)",
    "bio": "고품질 + 심박·HRR·피부온도 변화 + 미션 후 본인 응답 (생체 동의 녹화만)",
}
PER_HOUR = {"basic": 60_000, "premium": 100_000, "bio": 150_000}

# 센서 묶음: id → (이름, 설명, 시간당 원, LeRobot 채널, 녹화에 꼭 있어야 하는 채널)
SENSORS = {
    "video": ("영상 (필수)", "가린·치환 1인칭 영상 · 폰 IMU · 물체 영역(SAM 3.1) · 단계·기본 동작 라벨 · 성공·실수 · 동기화 오차", 35_000,
              ["observation.imu.phone", "observation.objects", "annotation.events", "annotation.step", "annotation.primitive",
               "sync.error_ms"], []),
    "pose": ("손·몸 관절", "손 21점 × 2 · 몸 33점 (화면·미터 좌표) · 얼굴 표정 52값·시선·고개 방향", 20_000,
             ["observation.hands", "observation.hands_present", "observation.body", "observation.body_world", "observation.face"],
             ["observation.hands"]),
    "audio": ("소리", "들리는 소리 세기 · 소리 사건 · 확인된 접촉음", 5_000, ["observation.audio"], ["observation.audio"]),
    "wrist": ("손목 IMU", "워치 가속도·자이로 (ms 동기화)", 15_000, ["observation.imu.wrist"], ["observation.imu.wrist"]),
    "touch": ("촉각", "접촉 단계 · 쥐는 세기 하·중·상 + 0~100 점수 · GripSense 특징 (워치 능동 진동)", 25_000,
              ["observation.touch", "observation.touch.vibration"], ["observation.touch.vibration"]),
    "physio": ("생체", "심박 · HRR · 피부온도 변화 + 미션 후 본인 응답 (생체 동의 녹화만)", 50_000,
               ["observation.physio"], ["observation.physio"]),
}
PRESETS = {"basic": ["video", "pose", "audio"], "premium": ["video", "pose", "audio", "wrist", "touch"],
           "bio": ["video", "pose", "audio", "wrist", "touch", "physio"]}


def sensors_for(level: Optional[str] = None, sensors: Optional[list] = None) -> list:
    """고른 센서 묶음 (영상은 늘 포함, 정해진 순서). 등급만 주면 그 조합."""
    pick = set(sensors) if sensors else set(PRESETS[level or "basic"])
    bad = pick - set(SENSORS)
    if bad:
        raise ValueError(f"없는 센서 묶음: {sorted(bad)}")
    return [k for k in SENSORS if k in pick or k == "video"]


def per_hour(sensors: list) -> int:
    return sum(SENSORS[k][2] for k in sensors)


def channels_for(sensors: list) -> set:
    return {c for k in sensors for c in SENSORS[k][3]}


def required_channels(sensors: list) -> set:
    return {c for k in sensors for c in SENSORS[k][4]}


def level_of(sensors: list) -> str:
    """조합이 어느 등급까지 필요하나 (녹화 등급 거르기·보상용)."""
    s = set(sensors)
    return "bio" if "physio" in s else "premium" if s & {"wrist", "touch"} else "basic"
CUSTOM_MULT, CUSTOM_MIN_HOURS, CUSTOM_SETUP = 1.5, 10, 500_000
DISCOUNTS = [(500, 0.35), (100, 0.20)]
REFINE_RATE, REFINE_MIN = 0.10, 50_000
REWARD_PER_HOUR = {"basic": 20_000, "premium": 33_000, "bio": 50_000}
BONUS_SCARCE, BONUS_TODAY = 0.5, 0.5


def _round100(x: float) -> int:
    return int(round(x / 100.0)) * 100


RARITY = {"normal": 1.0, "uncommon": 1.3, "rare": 1.8, "very_rare": 2.5}   # 모으기 어려운 참여 조건 (신규 수집에만 붙는다)


def quote(hours: float, level: Optional[str] = "basic", custom: bool = False, refine: bool = False,
          sensors: Optional[list] = None, rarity: float = 1.0) -> dict:
    if level not in LEVELS and not sensors:
        raise ValueError(f"등급은 {LEVELS} 중 하나")
    sel = sensors_for(level if level in LEVELS else None, sensors)
    level = level_of(sel)
    preset = next((k for k, v in PRESETS.items() if v == sel), None)
    hours = max(0.0, float(hours))
    per = per_hour(sel) * (CUSTOM_MULT if custom else 1.0) * (rarity if custom else 1.0)
    if custom:
        hours = max(hours, CUSTOM_MIN_HOURS)
    base = per * hours
    rate = next((r for h, r in DISCOUNTS if hours >= h), 0.0)
    data = base * (1 - rate)
    setup = CUSTOM_SETUP if custom else 0
    ref = max(REFINE_MIN, data * REFINE_RATE) if refine else 0
    total = data + setup + ref
    return {"level": level, "level_ko": LEVEL_KO[preset] if preset else "맞춤 조합", "sensors": sel,
            "sensors_ko": [SENSORS[k][0] for k in sel], "hours": round(hours, 2), "per_hour": _round100(per),
            "discount_pct": int(rate * 100), "data": _round100(data), "setup": setup, "refine": _round100(ref),
            "total": _round100(total), "custom": custom, "rarity": rarity if custom else 1.0}


def reward(accepted_minutes: float, level: str = "basic", scarce: bool = False, today: bool = False) -> int:
    """촬영자 보상 (원). 받아들인(검수 통과) 분만."""
    base = REWARD_PER_HOUR[level] * accepted_minutes / 60.0
    mult = 1.0 + (BONUS_SCARCE if scarce else 0.0) + (BONUS_TODAY if today else 0.0)
    return max(0, int(round(base * mult / 10.0)) * 10)


def today_mission(mission_ids: list, day: "datetime.date | None" = None) -> str:
    """오늘의 보너스 미션 (한국 날짜로 하루에 하나, 미션 목록을 차례로 돈다). 앱도 같은 식으로 고른다."""
    import datetime

    d = day or datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date()
    return mission_ids[d.toordinal() % len(mission_ids)] if mission_ids else ""
