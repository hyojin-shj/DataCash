"""우리 녹화의 진짜 정확도 — 운영자가 /label 에서 단 사람 정답(data/evolve/human/<녹화>.json)과 처리 결과 라벨(인증서 steps)을 비교.

  python scripts/eval_human.py            # 녹화마다 + 합친 값
EPIC 채점(scripts/eval_epic.py)과 같은 채점 함수(datacash/understand/score.py), 정답 단계 사이 틈은 뺀다.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacash import config  # noqa: E402
from datacash.understand.score import pooled, score  # noqa: E402

HUMAN = config.DATA / "evolve" / "human"
OUT = config.DATA / "server" / "out"


def main():
    rs = []
    for p in sorted(HUMAN.glob("*.json")):
        h = json.loads(p.read_text())
        cert = OUT / h["sid"] / "certificate.json"
        if not cert.exists() or not h["steps"]:
            continue
        r = score(h["steps"], json.loads(cert.read_text()).get("steps") or [], gold_only=True)
        rs.append(r)
        print(f'{h["sid"]}: 장면 {r["frame_acc"]:.1%} · 경계 {r["boundary_ms_median"]:.0f}ms · F1@25 {r["f1@25"]:.2f} '
              f'(정답 {len(h["steps"])}단계)')
    if not rs:
        print("사람 정답이 아직 없어요 — /label 에서 달아 주세요")
        return
    p = pooled(rs)
    print(f'합쳐서 ({len(rs)}편): 장면 {p["frame_acc"]:.1%} · 경계 {p["boundary_ms_median"]:.0f}ms · F1@25 {p["f1@25"]:.2f}')


if __name__ == "__main__":
    main()
