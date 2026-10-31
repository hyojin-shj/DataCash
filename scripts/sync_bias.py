"""서버 경유 시계의 치우침 재기 — 음향 비콘이 잡힌 녹화에서 '서버 경유 시계 − 비콘' 을 폰·워치 기종 쌍마다 모은다.

  python scripts/sync_bias.py [녹화 폴더 ...]      (없으면 data/server/sessions/* 전부)

폰·워치가 각자 서버와 시각을 주고받을 때 가는 길·오는 길 지연이 다르면 그 차이의 절반만큼 치우치는데, 이 치우침은
왕복 잔차에 드러나지 않는다. 비콘(폰 스피커 → 두 마이크)은 이 비대칭이 없으니 정답 삼아 잰다.
결과: data/sync_bias.json {"폰 기종|워치 기종": {bias_ms, sd_ms, n}} — sync.align 이 비콘을 못 잡은 녹화의 서버 경유 시계에서 뺀다.
비콘이 잡힌 녹화가 MIN_N 편 이상인 기종 쌍만 적는다.
"""

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacash import config  # noqa: E402
from datacash.session import load_session  # noqa: E402
from datacash.sync import align as A  # noqa: E402
from datacash.sync.clock import via_server  # noqa: E402

MIN_N = 5
warnings.filterwarnings("ignore")
A._motion_check = lambda s, m, **k: None   # 여기서는 시계만 본다


def main(dirs):
    by_pair = defaultdict(list)
    for d in dirs:
        if not (d / "watch_imu.csv").exists():
            continue
        s = load_session(d)
        if s.phone_server_sync is None or s.watch_server_sync is None:
            continue
        srv = via_server(s.phone_server_sync, s.watch_server_sync)
        al = A.align_session(s)
        if srv is None or al.watch_to_phone is None or al.watch_to_phone.method != "beacon":
            continue
        mid = np.array([int(np.median(s.watch_imu.t_ns))])
        diff = float((srv(mid)[0] - al.watch_to_phone(mid)[0]) / 1e6)
        pair = "|".join(str(m) for m in A._models(s))
        by_pair[pair].append(diff)
        print(f"{d.name}  {pair}  서버 − 비콘 {diff:+.1f}ms")
    p = config.DATA / "sync_bias.json"
    table = json.loads(p.read_text()) if p.exists() else {}
    for pair, v in by_pair.items():
        v = np.array(v)
        print(f"{pair}: {len(v)}편  평균 {v.mean():+.1f}ms  표준편차 {v.std(ddof=1) if len(v) > 1 else 0:.1f}ms")
        if len(v) >= MIN_N:
            table[pair] = {"bias_ms": round(float(np.median(v)), 2), "sd_ms": round(float(v.std(ddof=1)), 2), "n": int(len(v))}
    p.write_text(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"→ {p}")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]] or sorted(Path(config.DATA / "server" / "sessions").glob("2*"))
    main(args)
