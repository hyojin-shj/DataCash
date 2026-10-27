"""모델 진화 명령 — ① 학습 데이터 만들기, ② Glimmer 실력 재기 (학습 전 또는 어댑터를 붙여서).

  python scripts/evolve.py build [--root data/server]
  python scripts/evolve.py eval [--adapter data/evolve/adapters/glimmer-lora-v1.gguf] [--split eval]
  python scripts/evolve.py show
③ 학습은 scripts/train_glimmer.sh (클라우드 GPU).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacash import evolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval", "show"])
    ap.add_argument("--root", default="data/server")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="eval")
    a = ap.parse_args()
    if a.cmd == "build":
        out = Path(a.root) / "out"
        print(json.dumps(evolve.build_dataset([p for p in out.iterdir() if p.is_dir()]), ensure_ascii=False, indent=1))
    elif a.cmd == "eval":
        print(json.dumps(evolve.evaluate(a.adapter, a.split), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(evolve._registry(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
