"""성능 모니터링 — 녹화가 처리될 때마다 쌓이는 기록으로 품질이 갑자기 떨어지는지 본다.

운영자 화면 '성능' (GET /api/admin/perf) 과 scripts/perf_report.py 가 같은 계산을 쓴다.
- 라벨: 녹화마다 Spark 1.3 ↔ Spark Open 일치율 (evolve registry agreement), 최근 5편 평균
- 처리: 녹화 하나 처리 시간 (단계별), 실패
- 개인정보: 얼굴 치환 뒤에도 '같은 사람'으로 보이는 장면 비율 (certificate privacy.face_swap)
- Glimmer: 학습 데이터 버전 · 학습 작업 · 평가 정확도
- 고정 기준(벤치마크): docs/perf/benchmarks.json (사람 정답과 비교한 값, perf_report.py 가 만든다)
기준을 넘으면 alerts 에 남는다.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

from . import evolve

BENCH = Path(__file__).resolve().parents[1] / "docs" / "perf" / "benchmarks.json"
LIMITS = {"agreement_last5": 0.5,   # 최근 5편 평균이 이보다 낮으면 라벨이 흔들리는 것
          "time_x_median": 2.0,     # 처리 시간이 중앙값의 두 배를 넘으면
          "reid": 0.05}             # 치환 뒤에도 같은 사람으로 보이는 장면이 5% 를 넘으면


def _sessions(root: Path) -> list:
    rows = []
    for p in (root / "status").glob("*.json"):
        s = json.loads(p.read_text())
        if s.get("state") not in ("done", "rejected", "failed") or s.get("level") == "calibration":
            continue
        cert = root / "out" / p.stem / "certificate.json"
        fs = ((json.loads(cert.read_text()).get("privacy") or {}).get("face_swap") or {}) if cert.exists() else {}
        sec = s.get("seconds") or {}
        rows.append({"sid": p.stem, "mission": s.get("mission"), "state": s.get("state"), "updated": s.get("updated"),
                     "seconds": sec, "total_s": round(sum(sec.values()), 1),
                     "minutes": s.get("minutes"),
                     "reid": round(fs["over_reid"] / fs["frames"], 4) if fs.get("frames") and fs.get("over_reid") is not None else None})
    return sorted(rows, key=lambda r: r["sid"])


def collect(root: Path) -> dict:
    reg = evolve._registry()
    ag, alerts = [], []
    for i, a in enumerate(reg.get("agreement", [])):
        w = reg["agreement"][max(0, i - 4):i + 1]
        ag.append({"sid": a["sid"], "mission": a.get("mission"), "rate": a["rate"], "at": a.get("at"),
                   "last5": round(sum(x["rate"] for x in w) / len(w), 3)})
    if ag and ag[-1]["last5"] < LIMITS["agreement_last5"]:
        alerts.append({"kind": "agreement", "sid": ag[-1]["sid"], "value": ag[-1]["last5"], "limit": LIMITS["agreement_last5"]})

    ss = _sessions(root)
    done = [s["total_s"] for s in ss if s["state"] == "done" and s["total_s"]]
    med = median(done) if done else 0
    for s in ss:
        s["alert"] = []
        if s["state"] == "failed":
            s["alert"].append("failed")
        if med and s["total_s"] > LIMITS["time_x_median"] * med:
            s["alert"].append("slow")
        if s["reid"] is not None and s["reid"] > LIMITS["reid"]:
            s["alert"].append("reid")
        for k in s["alert"]:
            alerts.append({"kind": k, "sid": s["sid"],
                           "value": {"slow": s["total_s"], "reid": s["reid"]}.get(k), "limit": {"slow": round(LIMITS["time_x_median"] * med, 1), "reid": LIMITS["reid"]}.get(k)})

    stage = {}
    for s in ss:
        for k, v in s["seconds"].items():
            stage.setdefault(k, []).append(v)
    reids = [s["reid"] for s in ss if s["reid"] is not None]
    glim = {"datasets": [{k: d.get(k) for k in ("version", "built_at", "recordings", "train_steps", "eval_steps")} for d in reg.get("datasets", [])],
            "trainings": [{k: t.get(k) for k in ("output_model", "dataset", "epochs", "state", "started", "finished")} for t in reg.get("trainings", [])],
            "evals": [{k: e.get(k) for k in ("at", "model", "dataset", "split", "steps", "accuracy", "macro_f1", "baseline_accuracy")} for e in reg.get("evals", [])]}
    summary = {"recordings": len(ss), "failed": sum(s["state"] == "failed" for s in ss), "rejected": sum(s["state"] == "rejected" for s in ss),
               "time_median_s": round(med, 1), "stage_median_s": {k: round(median(v), 1) for k, v in stage.items()},
               "reid_median": round(median(reids), 4) if reids else None,
               "agreement_first5": round(sum(a["rate"] for a in ag[:5]) / len(ag[:5]), 3) if ag else None,
               "agreement_last5": ag[-1]["last5"] if ag else None, "agreement_all": round(sum(a["rate"] for a in ag) / len(ag), 3) if ag else None}
    bench = json.loads(BENCH.read_text()) if BENCH.exists() else None
    if bench:
        bench.pop("live_summary", None)
    return {"limits": LIMITS, "summary": summary, "agreement": ag, "sessions": ss, "glimmer": glim, "alerts": alerts, "benchmarks": bench}
