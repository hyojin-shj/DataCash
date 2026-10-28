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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/server")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    out_dir = evolve.ROOT / "crosscheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = agree = 0
    for out in sorted((Path(a.root) / "out").iterdir()):
        cert, video = out / "certificate.json", out / "masked_swap.mp4"
        dst = out_dir / f"{out.name}.json"
        if not cert.exists() or not video.exists() or (dst.exists() and not a.redo):
            if dst.exists():
                r = json.loads(dst.read_text())
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
        rows = []
        for s in steps:
            best = max(so["steps"], key=lambda n: overlap(s, n), default=None)
            same = bool(best and overlap(s, best) > 0 and best["action"] == s["action"])
            rows.append({"start_s": s["start_s"], "end_s": s["end_s"], "spark13": s["action"],
                         "spark_open": best["action"] if best else None, "agree": same, "verified": bool(s.get("verified"))})
        r = {"recording": out.name, "mission": mission, "steps": len(rows), "agree": sum(x["agree"] for x in rows),
             "gold": sum(x["agree"] and x["verified"] for x in rows), "rows": rows, "spark_open_steps": so["steps"]}
        dst.write_text(json.dumps(r, ensure_ascii=False, indent=1))
        total += r["steps"]
        agree += r["agree"]
        print(f"{out.name} {mission}: 같은 동작 {r['agree']}/{r['steps']} · 정답(검증+일치) {r['gold']}", flush=True)
    if total:
        print(f"전체 일치 {agree}/{total} ({agree / total:.0%}) · Fireworks 누적 ${spark_open.spent_usd():.2f}")


if __name__ == "__main__":
    main()
