"""활용 예시 영상 만들기: 생체 등급 데이터셋으로 '몇 초 뒤 심박 여유' 예측 모델을 배우고, 같이 나르는 휴머노이드 장면을 그린다.

  python scripts/demo_collab.py --synthetic            # 합성 녹화 10편으로 원리 시연 (정답을 아는 데이터)
  python scripts/demo_collab.py --train out/a/lerobot out/b/lerobot --test out/c/lerobot --show out/c/lerobot
"""

import argparse
import json
from pathlib import Path

from datacache.demo.collab import render
from datacache.demo.effort import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--show")
    ap.add_argument("--out", default="out/demo/collab.mp4")
    a = ap.parse_args()
    if a.synthetic:
        from datacache.jobs import run_session
        from datacache.session.synth import generate

        sets = []
        for k, g in enumerate([3, 5, 7, 9, 11, 13, 4, 8, 12, 6]):
            d = Path(f"data/synth/effort_{k}")
            generate(d, seed=100 + k, grip_n=float(g))
            m = json.loads((d / "meta.json").read_text())
            m["mission"] = {"id": "carry_laundry", "title": "빨래 바구니 나르기"}
            (d / "meta.json").write_text(json.dumps(m, ensure_ascii=False))
            run_session(d, f"out/synth/effort_{k}", use_api=False)
            sets.append(Path(f"out/synth/effort_{k}/lerobot"))
        a.train, a.test, a.show = sets[:7], sets[7:], str(sets[5])
    em = train([Path(p) for p in a.train], [Path(p) for p in a.test])
    mj = Path(a.out).with_suffix(".model.json")
    mj.parent.mkdir(parents=True, exist_ok=True)
    mj.write_text(json.dumps(em.to_dict(), indent=1))
    print(json.dumps(em.to_dict(), indent=1))
    print(render(a.show, mj, a.out))


if __name__ == "__main__":
    main()
