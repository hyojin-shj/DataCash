"""모델 진화: 검증된 라벨로 동작 모델을 배우고, 버전을 남기고, 준비 전에는 Spark 를 대신하지 않는다."""

import json

import numpy as np
import pandas as pd

from datacache import evolve
from datacache.understand.spark import PRIMITIVES


def _recording(out, seed: int, n: int = 240):
    """손이 물체로 다가가면 reach, 물체 위에서 오르내리면 lift, 다시 멀어지면 release — 규칙을 아는 가짜 녹화."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 30.0
    third = n // 3
    y = np.array([PRIMITIVES.index(a) for a in ["reach"] * third + ["lift"] * third + ["release"] * (n - 2 * third)])
    hands = np.zeros((n, 2, 21, 3))
    for i in range(n):
        ph = i / n
        x = 0.2 + 0.5 * min(ph * 3, 1) - (0.5 * (ph - 2 / 3) * 3 if ph > 2 / 3 else 0)
        h = 0.6 - (0.2 * np.sin((ph - 1 / 3) * 3 * np.pi) if 1 / 3 <= ph <= 2 / 3 else 0)
        pts = np.array([[x + 0.03 * np.cos(k), h + 0.03 * np.sin(k), 0] for k in range(21)])
        pts[0] = [x, h + 0.05, 0]
        hands[i, 1] = pts + rng.normal(0, 0.003, pts.shape)
    objs = np.tile([0.65, 0.5, 0.75, 0.65, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (n, 1))
    df = pd.DataFrame({"observation.hands": list(hands.reshape(n, -1)), "observation.objects": list(objs),
                       "observation.imu.phone": list(rng.normal(0, 0.1, (n, 6))), "observation.audio": list(np.zeros((n, 3))),
                       "annotation.primitive": y.astype(float), "timestamp": t})
    (out / "lerobot" / "data").mkdir(parents=True)
    df.to_parquet(out / "lerobot" / "data" / "e.parquet")
    steps = [{"action": a, "start_s": float(t[i]), "end_s": float(t[min(j, n - 1)]) + 1 / 30, "verified": True}
             for a, i, j in (("reach", 0, third), ("lift", third, 2 * third), ("release", 2 * third, n))]
    (out / "certificate.json").write_text(json.dumps({"steps": steps, "mission": {"id": "toy"}}))


def test_learns_versions_and_waits_until_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve, "ROOT", tmp_path / "evolve")
    dirs = []
    for k in range(3):
        d = tmp_path / f"rec{k}"
        _recording(d, k)
        dirs.append(d)
    e = evolve.train(dirs, log=lambda *a: None)
    pr = e["eval"]["primitive"]
    assert e["version"] == 1 and e["frames"] > 600
    assert pr["accuracy"] > pr["baseline_accuracy"] + 0.3          # 기준선(많은 쪽만 찍기)보다 확실히 낫다
    assert e["curve"]["phase"] and not e["ready"]                    # 녹화 3개로는 아직 Spark 를 대신하지 않는다
    assert evolve.train(dirs, log=lambda *a: None)["version"] == 1   # 같은 녹화로는 새 버전을 만들지 않는다
    assert evolve.try_label(dirs[0]) is None

    monkeypatch.setattr(evolve, "READY_RECORDINGS", 2)               # 준비 기준을 낮추면 우리 모델이 단계를 쓴다
    monkeypatch.setattr(evolve, "READY_F1", 0.5)
    (tmp_path / "rec3").mkdir()
    _recording(tmp_path / "rec3", 7)
    e2 = evolve.train(dirs + [tmp_path / "rec3"], log=lambda *a: None)
    assert e2["version"] == 2 and e2["ready"]
    r = evolve.try_label(dirs[1])
    assert r and [s["action"] for s in r["steps"]][:3] == ["reach", "lift", "release"]


def test_phase_map():
    y = np.array([PRIMITIVES.index(p) for p in ("reach", "grasp", "pull", "place", "release", "other")] + [-1])
    assert evolve._level(y, "phase").tolist() == [0, 1, 1, 2, 2, 3, -1]
