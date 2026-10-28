"""라벨 교차 검증 — 평소 라벨(Muse Spark 1.3)을 더 큰 Muse Spark Open(Fireworks)으로 다시 붙여 단계마다 맞대 본다.

  python scripts/crosscheck.py [--root data/server]
녹화마다 data/evolve/crosscheck/<녹화>.json: Spark Open 단계, Spark 1.3 단계마다 '같은 동작인가'(시간이 가장 많이 겹치는 단계로).
모델 진화(datacash/evolve.py)는 이 결과가 있으면 '감각으로 검증 + 두 모델이 같은 동작' 단계만 Glimmer 학습 정답으로 쓴다.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacash import evolve  # noqa: E402
from datacash.understand import spark_open  # noqa: E402


def overlap(a, b):
    return max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))


def compare(steps, so_steps, tol: float) -> list:
    """Spark 1.3 단계마다: 그 구간 ± tol(Spark Open 장면 간격) 안에 Spark Open 도 같은 동작을 봤는가.
    (Spark Open 은 장면 16장만 봐 경계가 0.5~1초 거칠다 — 가장 많이 겹치는 단계 하나만 보면 같은 동작도 어긋난다.)"""
    rows = []
    for s in steps:
        win = {"start_s": s["start_s"] - tol, "end_s": s["end_s"] + tol}
        near = [n for n in so_steps if overlap(win, n) > 0]
        best = max(near, key=lambda n: overlap(s, n), default=None)
        same = any(n["action"] == s["action"] for n in near)
        rows.append({"start_s": s["start_s"], "end_s": s["end_s"], "spark13": s["action"],
                     "spark_open": s["action"] if same else (best["action"] if best else None), "agree": same,
                     "verified": bool(s.get("verified"))})
    return rows


def summarize(name: str, mission: str, rows: list, so_steps: list) -> dict:
    return {"recording": name, "mission": mission, "steps": len(rows), "agree": sum(x["agree"] for x in rows),
            "gold": sum(x["agree"] and x["verified"] for x in rows), "rows": rows, "spark_open_steps": so_steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/server")
    ap.add_argument("--redo", action="store_true", help="Spark Open 을 다시 부른다")
    ap.add_argument("--rescore", action="store_true", help="저장된 Spark Open 답으로 비교만 다시 (비용 없음)")
    a = ap.parse_args()
    out_dir = evolve.ROOT / "crosscheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = agree = 0
    for out in sorted((Path(a.root) / "out").iterdir()):
        cert, video = out / "certificate.json", out / "masked_swap.mp4"
        dst = out_dir / f"{out.name}.json"
        if not cert.exists() or not video.exists():
            continue
        if dst.exists() and not a.redo:
            r = json.loads(dst.read_text())
            if a.rescore:
                steps = json.loads(cert.read_text()).get("steps") or []
                span = max((x["end_s"] for x in r["spark_open_steps"]), default=1) - min((x["start_s"] for x in r["spark_open_steps"]), default=0)
                r = summarize(out.name, r["mission"], compare(steps, r["spark_open_steps"], span / spark_open.FRAMES), r["spark_open_steps"])
                dst.write_text(json.dumps(r, ensure_ascii=False, indent=1))
                print(f"{out.name} {r['mission']}: 같은 동작 {r['agree']}/{r['steps']} · 정답(검증+일치) {r['gold']}", flush=True)
            total += r["steps"]
            agree += r["agree"]
            continue
        c = json.loads(cert.read_text())
        steps = c.get("steps") or []
        if not steps:
            continue
        status = Path(a.root) / "status" / f"{out.name}.json"
        st = json.loads(status.read_text()) if status.exists() else {}
        m = c.get("mission") or {}
        mission = (m.get("title_en") or m.get("title") or m.get("id")) if isinstance(m, dict) else str(m)
        t0 = float((st.get("precheck") or {}).get("action_start_s") or 0.0)
        so = spark_open.label_frames(video, t0, None, mission)
        tol = (so["frames"][-1] - so["frames"][0]) / max(1, len(so["frames"]) - 1) if len(so.get("frames") or []) > 1 else 0.7
        r = summarize(out.name, mission, compare(steps, so["steps"], tol), so["steps"])
        dst.write_text(json.dumps(r, ensure_ascii=False, indent=1))
        total += r["steps"]
        agree += r["agree"]
        print(f"{out.name} {mission}: 같은 동작 {r['agree']}/{r['steps']} · 정답(검증+일치) {r['gold']}", flush=True)
    if total:
        print(f"전체 일치 {agree}/{total} ({agree / total:.0%}) · Fireworks 누적 ${spark_open.spent_usd():.2f}")


if __name__ == "__main__":
    main()
