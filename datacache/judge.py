"""이의 신청은 평소보다 높은 설정으로 다시 본다 — Muse Spark 1.3 최대 추론('max').

평소 라벨: 동작 구간을 640px 로 줄인 영상 + 기본 추론.
이의 신청: 동작 구간 전체를 원래 해상도(긴 쪽 1280px)로, 참여자가 쓴 이의 내용과 함께, 추론 'max' 로.
(Muse Wonder 가 가장 높지만 한국에서 쓸 수 없다. 사용자 영상이 Meta 학습에 쓰이는 contributor 모드는 쓰지 않는다.)

  반려된 녹화 ('처리가 실패했어요', 상태 rejected)
      다시 본 결과 미션 동작 단계가 나오면 → 공짜 검수를 건너뛰고 다시 처리 (다시 처리한 결과대로 보상)
      동작이 안 보이면 → 검토자에게
  라벨이 틀렸어요 ('label')
      다시 붙인 단계와 원래 단계를 시간으로 맞대어, 동작이 달라진 단계를 바로잡는다
        바로잡은 단계가 있으면 → 인정, 하나에 LABEL_FIX_BONUS 원 (최대 LABEL_FIX_MAX)
                                바로잡은 라벨은 모델 진화(Glimmer 학습 데이터)의 정답으로 (evolve.add_correction)
        모두 같으면 → 검토자에게
판단 결과·근거는 이의 신청 단계 기록에 남는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

LABEL_FIX_BONUS = 100
LABEL_FIX_MAX = 500
MODEL = "muse-spark-1.3"
REASONING = "max"
SIDE = 1280


def needs_review_model(reason: str, status: dict) -> bool:
    return (reason == "failed" and status.get("state") == "rejected") or (reason == "label" and status.get("state") == "done")


def _video(out: Path, sd: Path) -> Optional[Path]:
    return next((p for p in (out / "masked_swap.mp4", out / "masked.mp4") if p.exists()), None)   # 원본은 보내지 않는다


def _overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))


def corrections(old: List[dict], new: List[dict]) -> List[dict]:
    """원래 단계마다 시간이 가장 많이 겹치는 새 단계의 동작이 다르면 바로잡음."""
    fixes = []
    for o in old:
        best = max(new, key=lambda n: _overlap(o, n), default=None)
        if best is None or _overlap(o, best) < 0.5 * max(1e-6, o["end_s"] - o["start_s"]):
            continue
        if best.get("action") and best["action"] != o.get("action") and best["action"] != "other":
            fixes.append({"start_s": o["start_s"], "end_s": o["end_s"], "was": o.get("action"), "action": best["action"],
                          "note": best.get("instruction", "")[:120], "by": f"{MODEL} reasoning={REASONING}"})
    return fixes


def judge(appeal: dict, status: dict, session_dir, out_dir, relabel=None, log=print) -> dict:
    """{'decision': 'reprocess' | 'accept' | 'review', 'note', 'extra', 'corrections'}.
    relabel(video, t0, t1, mission, detail) → 단계 목록 (바꿔 끼우기용, 기본은 Spark 1.3 max)."""
    sd, out = Path(session_dir), Path(out_dir)
    video = _video(out, sd)
    if video is None:
        return {"decision": "review", "note": "가린 영상이 없어 검토자에게 넘겼어요", "extra": 0, "corrections": []}
    meta = json.loads((sd / "meta.json").read_text()) if (sd / "meta.json").exists() else {}
    m = meta.get("mission") or {}
    mission = f"{m.get('title_en') or m.get('title') or status.get('mission') or 'mission'}"
    t0 = float((status.get("precheck") or {}).get("action_start_s") or 0.0)
    try:
        steps = (relabel or _spark_relabel)(video, t0, None, mission, appeal.get("detail", ""))
    except Exception as e:
        return {"decision": "review", "note": f"다시 보기 실패 ({type(e).__name__}) → 검토자 확인", "extra": 0, "corrections": []}

    if appeal["reason"] == "failed":
        acts = [s for s in steps if s.get("action") not in (None, "other")]
        if acts:
            return {"decision": "reprocess", "extra": 0, "corrections": [],
                    "note": f"Spark 1.3(최대 추론)이 미션 동작 {len(acts)}단계를 봤어요 → 검수를 건너뛰고 다시 처리"}
        return {"decision": "review", "note": "Spark 1.3(최대 추론)도 미션 동작을 찾지 못했어요 → 검토자 확인", "extra": 0, "corrections": []}

    cert = json.loads((out / "certificate.json").read_text()) if (out / "certificate.json").exists() else {}
    old = cert.get("steps") or []
    if not old:
        return {"decision": "review", "note": "원래 단계 라벨이 없어 검토자에게 넘겼어요", "extra": 0, "corrections": []}
    fixes = corrections(old, steps)
    if not fixes:
        return {"decision": "review", "note": "Spark 1.3(최대 추론)도 원래 라벨과 같다고 봤어요 → 검토자 확인", "extra": 0, "corrections": []}
    extra = min(LABEL_FIX_MAX, LABEL_FIX_BONUS * len(fixes))
    listed = ", ".join(f"{f['start_s']:.1f}초 {f['was']}→{f['action']}" for f in fixes[:4])
    return {"decision": "accept", "extra": extra, "corrections": fixes,
            "note": f"Spark 1.3(최대 추론)으로 다시 보니 단계 {len(fixes)}개가 달랐어요 ({listed}) — 바로잡고 +{extra:,}원"}


def _spark_relabel(video, t0: float, t1, mission: str, detail: str) -> List[dict]:
    import cv2

    from .media import clip_mp4
    from .understand.spark import understand_clip

    cap = cv2.VideoCapture(str(video))
    dur = cap.get(7) / (cap.get(5) or 30)
    cap.release()
    t0 = min(max(0.0, t0), max(0.0, dur - 2))
    clip = clip_mp4(video, Path(video).with_name("appeal_clip.mp4"), t0, min(dur - t0, 120), max_side=SIDE, audio=False)
    try:
        u = understand_clip(clip, mission, min(dur - t0, 120), model=MODEL, reasoning=REASONING,
                            extra=f'The participant appealed the earlier labels and says: "{detail[:300]}". '
                                  "Re-check carefully, frame by frame; do not simply agree with them.")
    finally:
        Path(clip).unlink(missing_ok=True)
    for s in u["steps"]:
        s["start_s"] += t0
        s["end_s"] += t0
    return u["steps"]
