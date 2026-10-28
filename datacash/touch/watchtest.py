"""워치 시험 (web/watchtest.html) 결과 — 브라우저로 모은 손목 센서 + 진동 시각으로 GripSense 가 되는지.

녹화(세그먼트)마다: '놓기·가볍게·평소·꽉' 중 하나를 8초, 0.5초마다 0.05초 진동. 1회차로 배우고 2회차를 맞힌다
(같은 회차 안의 창끼리는 너무 닮아 섞어 나누면 점수가 부풀려진다).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gripsense import LEVELS, Windows, labels_for, train, window_features


def _segment(d: dict) -> Windows:
    imu = np.asarray(d.get("imu") or [], float)
    if len(imu) < 64:
        return Windows(np.zeros(0, np.int64), np.zeros((0, 9), np.float32))
    t = (imu[:, 0] * 1e6).astype(np.int64)
    keep = np.r_[True, np.diff(t) > 0]
    return window_features(t[keep], imu[keep, 1:7], (np.asarray(d.get("motor") or [], float) * 1e6).astype(np.int64),
                           motor_ms=float(d.get("pulse_ms", 50)))


def evaluate_run(folder) -> dict:
    folder = Path(folder)
    check = json.loads((folder / "check.json").read_text()) if (folder / "check.json").exists() else {}
    wins = {0: [], 1: []}
    hz = []
    for p in sorted(folder.glob("seg_r*_l*.json")):
        d = json.loads(p.read_text())
        w = _segment(d)
        if not len(w.t_ns):
            continue
        imu = np.asarray(d["imu"], float)
        hz.append(1000 * (len(imu) - 1) / max(1e-6, imu[-1, 0] - imu[0, 0]))
        y = labels_for(w, [{"start_ns": int(w.t_ns[0] - 0.25e9), "end_ns": int(w.t_ns[-1] + 0.25e9), "level": int(d["level"])}])
        wins[int(d["round"])].append((w, y))

    def stack(lst):
        if not lst:
            return None, None
        return Windows(np.concatenate([w.t_ns for w, _ in lst]), np.concatenate([w.x for w, _ in lst])), np.concatenate([y for _, y in lst])

    out = {"hz": round(float(np.median(hz)), 1) if hz else check.get("hz"), "vibrate": check.get("vibrate"), "gyro": check.get("gyro"),
           "segments": sum(len(v) for v in wins.values()), "windows": 0, "round_accuracy": None, "cv_accuracy": None}
    w0, y0 = stack(wins[0])
    w1, y1 = stack(wins[1])
    if w0 is None:
        return out
    out["windows"] = int((y0 >= 0).sum() + ((y1 >= 0).sum() if y1 is not None else 0))
    m = train(w0, y0)
    if m is None:
        out["note"] = "1회차 창이 부족하거나 단계가 하나뿐"
        return out
    out["cv_accuracy"] = None if np.isnan(m.accuracy_cv) else round(float(m.accuracy_cv), 3)
    if w1 is not None and (y1 >= 0).any():
        k = y1 >= 0
        pred, _ = m.predict(w1.x[k])
        yt = y1[k]
        out["round_accuracy"] = round(float(np.mean(pred == yt)), 3)
        cm = np.zeros((4, 4), int)
        for a, b in zip(yt, pred):
            cm[a, b] += 1
        out["confusion"] = {LEVELS[i]: cm[i].tolist() for i in range(4)}
    means = {}
    for lvl in range(4):
        sel = [w.x[y == lvl] for w, y in wins[0] + wins[1]]
        x = np.concatenate(sel) if sel else np.zeros((0, 9))
        if len(x):
            means[LEVELS[lvl]] = {"hf_ratio_gyro": round(float(np.mean(x[:, 2])), 3), "hf_p90_gyro": round(float(np.mean(x[:, 0])), 4),
                                  "lf_var_gyro": round(float(np.mean(x[:, 4])), 6)}
    out["feature_means"] = means
    (folder / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out
