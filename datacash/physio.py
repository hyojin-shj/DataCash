"""몸 감각 — 심박·HRR·피부온도 변화 (생체 동의한 녹화만). 판정·조언 라벨은 붙이지 않는다: 원값과 본인 응답만.

HRR(심박 예비량 비율) = (심박 − 안정 심박) / (최대 심박 − 안정 심박) × 100
  최대 심박: Tanaka 식 208 − 0.7 × 나이 (나이는 동의서 연령대의 가운데 값, 개인 식별 정보는 쓰지 않는다)
  안정 심박: 이 녹화 앞 20초(또는 처음 1/5) 중 낮은 쪽 10% 의 평균
피부온도 변화: 녹화 시작 값 대비 °C
"""

from __future__ import annotations

from typing import Optional

import numpy as np

AGE_MID = {"19~29세": 24, "20대": 25, "30대": 35, "40대": 45, "50대": 55, "60대 이상": 65}


def hr_max(age_band: Optional[str]) -> Optional[float]:
    age = AGE_MID.get(age_band or "")
    return 208 - 0.7 * age if age else None


def physio_series(hr_ns, hr_bpm, skin_ns, skin_c, age_band: Optional[str]) -> dict:
    """심박·피부온도 시계열(이미 폰 시계로 옮긴 것) → HRR·피부온도 변화와 요약."""
    out = {"hr_ns": hr_ns, "hr": hr_bpm, "hrr": None, "skin_ns": skin_ns, "skin_delta": None, "summary": {}}
    if hr_ns is not None and len(hr_ns):
        n0 = max(3, min(len(hr_bpm) // 5, int(np.sum(hr_ns <= hr_ns[0] + 20e9))))
        rest = float(np.mean(np.sort(hr_bpm[:n0])[:max(1, n0 // 10 + 1)]))
        mx = hr_max(age_band)
        if mx and mx > rest:
            out["hrr"] = np.clip((hr_bpm - rest) / (mx - rest) * 100, -50, 150)
        out["summary"].update(hr_rest_bpm=round(rest, 1), hr_mean_bpm=round(float(np.mean(hr_bpm)), 1),
                              hr_max_bpm=round(float(np.max(hr_bpm)), 1), hr_max_formula="Tanaka 208-0.7*age",
                              age_band=age_band, hrr_peak_pct=None if out["hrr"] is None else round(float(np.max(out["hrr"])), 1))
    if skin_ns is not None and len(skin_ns):
        out["skin_delta"] = skin_c - skin_c[0]
        out["summary"].update(skin_start_c=round(float(skin_c[0]), 2), skin_delta_end_c=round(float(skin_c[-1] - skin_c[0]), 3))
    return out
