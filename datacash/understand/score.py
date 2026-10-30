"""단계 라벨 채점 — 정답 단계(사람이 붙였거나 저울·지시 녹화로 기계가 찍은)와 모델 단계를 맞댄다.

정답·모델 모두 [{"start_s", "end_s", "action", ("verified")}]. 영상 동작 나누기 연구에서 흔히 쓰는 지표:
  frame_acc     정답 구간을 0.1초마다 보며 같은 동작인 비율 (정답에 단계가 없는 틈은 'none')
  f1@10/25/50   모델 단계가 같은 동작의 정답 단계와 IoU ≥ 0.1/0.25/0.5 로 겹치면 맞힘 (정답 하나에 하나만)
  edit          단계 순서(같은 동작이 이어지면 하나로)의 편집 거리 점수 100 × (1 − 거리 / 긴 쪽 길이)
  boundary_ms   정답 단계 시작마다 가장 가까운 모델 단계 시작까지 (중앙값·90번째)
  verified_precision  모델이 verified 로 판정한 단계 중 그 구간 정답 동작(가장 많이 겹친)과 같은 비율 — 검증 규칙이 믿을 만한가
merge: 동작 이름 바꾸기 (예: 정답이 거칠면 {"reach": "take", "grasp": "take"}) — 정답·모델에 똑같이 쓴다.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

STEP_S = 0.1


def _norm(steps: List[dict], merge: Optional[Dict[str, str]]) -> List[dict]:
    m = merge or {}
    return sorted(({**s, "action": m.get(s["action"], s["action"])} for s in steps if s["end_s"] > s["start_s"]),
                  key=lambda s: s["start_s"])


def _at(steps: List[dict], t: np.ndarray) -> List[str]:
    out = ["none"] * len(t)
    for s in steps:
        for i in np.nonzero((t >= s["start_s"]) & (t < s["end_s"]))[0]:
            out[i] = s["action"]
    return out


def _iou(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
    return inter / (max(a["end_s"], b["end_s"]) - min(a["start_s"], b["start_s"]))


def f1_at(gold: List[dict], pred: List[dict], k: float) -> float:
    used, tp = set(), 0
    for p in pred:
        best, bi = 0.0, None
        for i, g in enumerate(gold):
            if i not in used and g["action"] == p["action"]:
                v = _iou(p, g)
                if v > best:
                    best, bi = v, i
        if bi is not None and best >= k:
            used.add(bi)
            tp += 1
    prec, rec = tp / max(1, len(pred)), tp / max(1, len(gold))
    return 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)


def _seq(steps: List[dict]) -> List[str]:
    out = []
    for s in steps:
        if not out or out[-1] != s["action"]:
            out.append(s["action"])
    return out


def edit_score(gold: List[dict], pred: List[dict]) -> float:
    a, b = _seq(gold), _seq(pred)
    if not a and not b:
        return 100.0
    d = np.arange(len(b) + 1, dtype=float)
    for i in range(1, len(a) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(b) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
    return 100.0 * (1 - d[-1] / max(len(a), len(b)))


def score(gold: List[dict], pred: List[dict], merge: Optional[Dict[str, str]] = None, gold_only: bool = False) -> dict:
    """gold_only: 정답에 단계가 없는 틈은 장면 정확도에서 뺀다 (공개 데이터셋처럼 정답이 동작 사이를 비워 둘 때)."""
    g, p = _norm(gold, merge), _norm(pred, merge)
    if not g:
        return {"gold_steps": 0}
    t = np.arange(g[0]["start_s"], g[-1]["end_s"], STEP_S)
    pairs = [(x, y) for x, y in zip(_at(g, t), _at(p, t)) if not (gold_only and x == "none")]
    fa = float(np.mean([x == y for x, y in pairs])) if pairs else 0.0
    starts = np.array([s["start_s"] for s in p])
    bd = [float(np.min(np.abs(starts - s["start_s"]))) * 1000 for s in g] if len(starts) else []
    ver = [s for s in p if s.get("verified") is True]
    ok = 0
    for s in ver:
        best = max(g, key=lambda x: max(0.0, min(x["end_s"], s["end_s"]) - max(x["start_s"], s["start_s"])))
        ok += best["action"] == s["action"] and _iou(s, best) > 0
    return {"gold_steps": len(g), "pred_steps": len(p), "frame_acc": round(fa, 3),
            **{f"f1@{int(k * 100)}": round(f1_at(g, p, k), 3) for k in (0.1, 0.25, 0.5)},
            "edit": round(edit_score(g, p), 1),
            "boundary_ms_median": round(float(np.median(bd)), 0) if bd else None,
            "boundary_ms_p90": round(float(np.percentile(bd, 90)), 0) if bd else None,
            "verified_steps": len(ver), "verified_precision": round(ok / len(ver), 3) if ver else None}


def pooled(results: List[dict]) -> dict:
    """녹화 여러 편: 정답 단계 수로 가중 평균 (경계는 녹화별 중앙값의 중앙값)."""
    r = [x for x in results if x.get("gold_steps")]
    if not r:
        return {}
    w = np.array([x["gold_steps"] for x in r], float)
    avg = {k: round(float(np.average([x[k] for x in r], weights=w)), 3) for k in ("frame_acc", "f1@10", "f1@25", "f1@50", "edit")}
    bd = [x["boundary_ms_median"] for x in r if x.get("boundary_ms_median") is not None]
    vs = sum(x["verified_steps"] for x in r)
    vp = sum((x["verified_precision"] or 0) * x["verified_steps"] for x in r)
    return {"recordings": len(r), "gold_steps": int(w.sum()), **avg,
            "boundary_ms_median": round(float(np.median(bd)), 0) if bd else None,
            "verified_precision": round(vp / vs, 3) if vs else None}
