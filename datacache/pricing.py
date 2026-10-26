"""가격 — 2026-10-26 시장 사례로 잡은 값 (docs/1026_meta_hackathon.md 2-3).

판매 (받아들인 시간당): 기본 60,000원 · 고품질 100,000원 · 생체 150,000원
맞춤 수집: 판매가 × 1.5, 최소 10시간, 착수비 500,000원
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
    "premium": "기본 + 손목 IMU(ms 동기화) + 촉각: 워치로 잰 쥐는 힘·진동 반응",
    "bio": "고품질 + 심박·HRR·피부온도 변화 + 미션 후 본인 응답 (생체 동의 녹화만)",
}
PER_HOUR = {"basic": 60_000, "premium": 100_000, "bio": 150_000}
CUSTOM_MULT, CUSTOM_MIN_HOURS, CUSTOM_SETUP = 1.5, 10, 500_000
DISCOUNTS = [(500, 0.35), (100, 0.20)]
REFINE_RATE, REFINE_MIN = 0.10, 50_000
REWARD_PER_HOUR = {"basic": 20_000, "premium": 33_000, "bio": 50_000}
BONUS_SCARCE, BONUS_TODAY = 0.5, 0.5


def _round100(x: float) -> int:
    return int(round(x / 100.0)) * 100


def quote(hours: float, level: str = "basic", custom: bool = False, refine: bool = False) -> dict:
    if level not in LEVELS:
        raise ValueError(f"등급은 {LEVELS} 중 하나")
    hours = max(0.0, float(hours))
    per = PER_HOUR[level] * (CUSTOM_MULT if custom else 1.0)
    if custom:
        hours = max(hours, CUSTOM_MIN_HOURS)
    base = per * hours
    rate = next((r for h, r in DISCOUNTS if hours >= h), 0.0)
    data = base * (1 - rate)
    setup = CUSTOM_SETUP if custom else 0
    ref = max(REFINE_MIN, data * REFINE_RATE) if refine else 0
    total = data + setup + ref
    return {"level": level, "level_ko": LEVEL_KO[level], "hours": round(hours, 2), "per_hour": _round100(per),
            "discount_pct": int(rate * 100), "data": _round100(data), "setup": setup, "refine": _round100(ref),
            "total": _round100(total), "custom": custom}


def reward(accepted_minutes: float, level: str = "basic", scarce: bool = False, today: bool = False) -> int:
    """촬영자 보상 (원). 받아들인(검수 통과) 분만."""
    base = REWARD_PER_HOUR[level] * accepted_minutes / 60.0
    mult = 1.0 + (BONUS_SCARCE if scarce else 0.0) + (BONUS_TODAY if today else 0.0)
    return max(0, int(round(base * mult / 10.0)) * 10)
