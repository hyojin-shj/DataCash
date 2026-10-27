"""모델 진화 — 검증된 Muse Spark 라벨로 우리 동작 모델을 키운다 (녹화가 쌓일수록 다시 학습, 버전을 남긴다).

  정답      Spark 가 쓴 단계 가운데 감각으로 검증된 것만 (annotation.primitive, 인증서 steps[].verified)
  재료      이 맥에서 공짜로 뽑은 것만 — 손 21점 × 2, 몸 관절, 물체 상자(SAM), 폰 IMU, 손목 IMU(있으면), 소리
            단계 경계·검증 표시·접촉 단계처럼 라벨에서 나온 열은 쓰지 않는다 (답을 미리 보는 셈)
  모델      장면마다 기본 동작(reach·grasp·lift…release)을 맞히는 그래디언트 부스팅 + 앞뒤 장면으로 다듬기
  평가      녹화 하나씩 빼고 나머지로 학습해 맞혀 보기 (같은 녹화의 장면끼리는 너무 닮아 섞으면 점수가 부풀려진다)
  성장 곡선 학습 녹화 수 1, 2, … 개일 때의 점수 — 데이터가 늘수록 오르는지
  쓰임      점수가 READY_F1 이상이고 이 녹화에서 확신이 높으면 Spark 를 부르지 않고 우리 모델로 단계를 쓴다
저장: data/evolve/registry.json (버전 기록), data/evolve/v<n>.pkl
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config
from .understand.spark import PRIMITIVES

ROOT = config.DATA / "evolve"
READY_F1 = 0.8            # 이 점수(녹화 하나씩 빼고 잰 macro F1)를 넘어야 Spark 대신 쓴다
READY_RECORDINGS = 20     # 그리고 학습 녹화가 이만큼은 있어야
CONFIDENT = 0.7           # 이 녹화 동작 구간의 평균 확신
SMOOTH = 9                # 확률을 앞뒤 장면으로 평균 (장면 수)
MIN_SEG = 6               # 이보다 짧은 동작 구간은 이웃에 붙인다
WINDOWS = (5, 15)         # 앞뒤로 볼 장면 수 (30fps 에서 약 0.17초·0.5초)

TIP, WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = (4, 8, 12, 16, 20), 0, 4, 8, 9

# 두 수준으로 배운다: 세부 동작(PRIMITIVES 16가지)과, 미션이 달라도 공통인 큰 단계 4가지
PHASES = ["approach", "handle", "put_down", "other"]
_PHASE_OF = {"reach": 0, "place": 2, "release": 2, "other": 3}
PHASE_MAP = np.array([_PHASE_OF.get(p, 1) for p in PRIMITIVES])      # 나머지(grasp·lift·pull·…)는 잡고 다루기
LEVELS = {"primitive": PRIMITIVES, "phase": PHASES}


def _level(y: np.ndarray, level: str) -> np.ndarray:
    return y if level == "primitive" else np.where(y >= 0, PHASE_MAP[np.clip(y, 0, None)], -1)


def _hand(h: np.ndarray) -> np.ndarray:
    """(21, 3) 손 한쪽 → 손목 위치·손 크기·벌림·집게 간격·높이 (없으면 0)."""
    if not np.isfinite(h).all() or not np.any(h):
        return np.zeros(6, np.float32)
    w = h[WRIST, :2]
    size = np.linalg.norm(h[MIDDLE_MCP, :2] - w) + 1e-6
    spread = np.mean(np.linalg.norm(h[list(TIP), :2] - w, axis=1)) / size          # 손을 편 정도
    pinch = np.linalg.norm(h[THUMB_TIP, :2] - h[INDEX_TIP, :2]) / size              # 엄지·검지 간격
    return np.array([1.0, w[0], w[1], size, spread, pinch], np.float32)


def frame_features(df) -> np.ndarray:
    """LeRobot 장면 표 → (장면 수, 특징 수). 라벨에서 나온 열은 쓰지 않는다."""
    n = len(df)
    hands = np.stack(df["observation.hands"].values).reshape(n, 2, 21, 3)
    base = []
    for k in range(2):
        hk = np.stack([_hand(hands[i, k]) for i in range(n)])
        base.append(hk)
    objs = np.stack(df["observation.objects"].values).reshape(n, 4, 5) if "observation.objects" in df else np.zeros((n, 4, 5))
    oc = np.stack([(objs[:, :, 0] + objs[:, :, 2]) / 2, (objs[:, :, 1] + objs[:, :, 3]) / 2], -1)    # 물체 가운데
    present = objs[:, :, 4] > 0
    near = []
    for k in range(2):                                   # 손목 ↔ 가장 가까운 물체 거리, 그 물체의 높이
        w = base[k][:, 1:3]
        d = np.linalg.norm(oc - w[:, None, :], axis=-1)
        d = np.where(present & (base[k][:, :1] > 0), d, 2.0)
        j = d.argmin(1)
        near.append(np.stack([d.min(1), oc[np.arange(n), j, 1], present.any(1).astype(float)], -1))
    imu = np.stack(df["observation.imu.phone"].values) if "observation.imu.phone" in df else np.zeros((n, 6))
    wrist = (np.stack(df["observation.imu.wrist"].values) if "observation.imu.wrist" in df else np.full((n, 6), np.nan))
    has_w = np.isfinite(wrist).all(1, keepdims=True).astype(float)
    wrist = np.nan_to_num(wrist)
    audio = np.stack(df["observation.audio"].values)[:, :2] if "observation.audio" in df else np.zeros((n, 2))
    motion = np.stack([np.linalg.norm(imu[:, :3], axis=1), np.linalg.norm(imu[:, 3:], axis=1),
                       np.linalg.norm(wrist[:, :3], axis=1), np.linalg.norm(wrist[:, 3:], axis=1)], -1)
    x = np.concatenate([base[0], base[1], near[0], near[1], motion, has_w, np.nan_to_num(audio)], 1).astype(np.float32)
    feats = [x]
    for w in WINDOWS:                                    # 앞뒤 흐름: 평균과 변화량 (속도·들어 올림·놓기)
        pad = np.pad(x, ((w, w), (0, 0)), mode="edge")
        c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), pad]), 0)
        mean = (c[2 * w + 1:] - c[:-2 * w - 1]) / (2 * w + 1)
        feats += [mean, pad[2 * w:] - pad[:-2 * w]]
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]  # 녹화 안 상대 위치 (뻗기는 앞, 놓기는 뒤)
    return np.concatenate(feats + [t], 1)


def load_recording(out_dir) -> Optional[dict]:
    """처리 결과 폴더 → {'id', 'X', 'y'(검증된 장면만, 나머지 -1), 'mission'}. 검증 단계가 없으면 None."""
    import pandas as pd

    out = Path(out_dir)
    cert = out / "certificate.json"
    parts = sorted((out / "lerobot" / "data").rglob("*.parquet"))
    if not cert.exists() or not parts:
        return None
    c = json.loads(cert.read_text())
    steps = c.get("steps") or []
    if not any(s.get("verified") for s in steps):
        return None
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    y = df["annotation.primitive"].to_numpy().astype(int).ravel()
    t = df["timestamp"].to_numpy().astype(float).ravel()
    ok = np.zeros(len(df), bool)
    for s in steps:                                      # 검증된 단계 구간의 장면만 정답으로
        if s.get("verified"):
            ok |= (t >= s["start_s"] - 1e-6) & (t < s["end_s"])
    y = np.where(ok, y, -1)
    mission = c.get("mission")
    return {"id": out.name, "X": frame_features(df), "y": y, "t": t,
            "mission": mission.get("id") if isinstance(mission, dict) else mission}


def _fit(recs: List[dict]):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.concatenate([r["X"][r["y"] >= 0] for r in recs])
    y = np.concatenate([r["y"][r["y"] >= 0] for r in recs])
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0,
                                       class_weight="balanced", random_state=0)
    m.fit(X, y)
    return m


def predict(model, X: np.ndarray):
    """장면마다 (기본 동작 번호, 확신). 확률을 앞뒤 SMOOTH 장면으로 평균해 깜빡임을 없앤다."""
    p = model.predict_proba(X)
    k = np.ones(SMOOTH) / SMOOTH
    p = np.stack([np.convolve(np.pad(p[:, j], SMOOTH // 2, mode="edge"), k, "valid") for j in range(p.shape[1])], 1)
    return model.classes_[p.argmax(1)], p.max(1)


def segments(labels: np.ndarray, t: np.ndarray) -> List[dict]:
    """장면별 동작 → 단계 구간 [{action, start_s, end_s}]. 짧은 구간은 앞 구간에 붙인다."""
    segs = []
    for i, a in enumerate(labels):
        if segs and segs[-1]["a"] == a:
            segs[-1]["j"] = i
        else:
            segs.append({"a": int(a), "i": i, "j": i})
    merged = []
    for s in segs:
        if merged and s["j"] - s["i"] + 1 < MIN_SEG:
            merged[-1]["j"] = s["j"]
        else:
            merged.append(s)
    return [{"action": PRIMITIVES[s["a"]], "start_s": float(t[s["i"]]), "end_s": float(t[min(s["j"] + 1, len(t) - 1)])}
            for s in merged]


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """정확도·macro F1 + 기준선(무조건 가장 많은 동작만 찍었을 때의 정확도 — 이보다 높아야 배운 것이다)."""
    from sklearn.metrics import accuracy_score, f1_score

    k = y_true >= 0
    yt = y_true[k]
    base = float(np.bincount(yt).max() / len(yt)) if len(yt) else 0.0
    return {"frames": int(k.sum()), "accuracy": round(float(accuracy_score(yt, y_pred[k])), 3),
            "macro_f1": round(float(f1_score(yt, y_pred[k], average="macro", zero_division=0)), 3),
            "baseline_accuracy": round(base, 3)}


def evaluate(recs: List[dict], level: str = "primitive") -> dict:
    """녹화 하나씩 빼고 학습 → 뺀 녹화를 맞혀 본다. 모든 녹화를 합친 점수."""
    recs = [{**r, "y": _level(r["y"], level)} for r in recs]
    if len(recs) < 2:
        return {"recordings": len(recs), "note": "녹화가 2개 이상 있어야 평가할 수 있다"}
    yt, yp, per = [], [], {}
    for i, r in enumerate(recs):
        m = _fit(recs[:i] + recs[i + 1:])
        pred, _ = predict(m, r["X"])
        yt.append(r["y"])
        yp.append(pred)
        per[r["id"]] = _score(r["y"], pred)
    return {"recordings": len(recs), **_score(np.concatenate(yt), np.concatenate(yp)), "per_recording": per}


def learning_curve(recs: List[dict], level: str = "primitive", repeats: int = 3, seed: int = 0) -> List[dict]:
    """학습 녹화 수 k = 1 … N-1 일 때, 남은 녹화를 맞힌 점수 (무작위로 repeats 번 골라 평균)."""
    recs = [{**r, "y": _level(r["y"], level)} for r in recs]
    rng = np.random.default_rng(seed)
    out = []
    for k in range(1, len(recs)):
        f1s, accs = [], []
        for _ in range(repeats):
            order = rng.permutation(len(recs))
            test, train = recs[order[0]], [recs[j] for j in order[1:k + 1]]
            pred, _ = predict(_fit(train), test["X"])
            s = _score(test["y"], pred)
            f1s.append(s["macro_f1"])
            accs.append(s["accuracy"])
        out.append({"train_recordings": k, "macro_f1": round(float(np.mean(f1s)), 3), "accuracy": round(float(np.mean(accs)), 3)})
    return out


def _registry() -> dict:
    p = ROOT / "registry.json"
    return json.loads(p.read_text()) if p.exists() else {"versions": []}


def train(out_dirs, log=print) -> Optional[dict]:
    """처리된 녹화들로 새 버전을 학습·평가해 남긴다. 학습 녹화가 이전 버전과 같으면 그대로 둔다."""
    recs = [r for r in (load_recording(d) for d in out_dirs) if r is not None]
    if not recs:
        return None
    reg = _registry()
    ids = sorted(r["id"] for r in recs)
    if reg["versions"] and reg["versions"][-1]["trained_on"] == ids:
        return reg["versions"][-1]
    ev = {lv: evaluate(recs, lv) for lv in LEVELS}
    curve = {lv: learning_curve(recs, lv) for lv in LEVELS} if len(recs) >= 3 else {}
    model = _fit(recs)
    v = len(reg["versions"]) + 1
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / f"v{v}.pkl").write_bytes(pickle.dumps(model))
    ready = ev["primitive"].get("macro_f1", 0) >= READY_F1 and len(recs) >= READY_RECORDINGS
    entry = {"version": v, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trained_on": ids,
             "frames": int(sum((r["y"] >= 0).sum() for r in recs)),
             "classes": [PRIMITIVES[c] for c in model.classes_], "missions": sorted({r["mission"] for r in recs if r["mission"]}),
             "eval": ev, "curve": curve, "ready": ready}
    reg["versions"].append(entry)
    (ROOT / "registry.json").write_text(json.dumps(reg, indent=1, ensure_ascii=False))
    pr, ph = ev["primitive"], ev["phase"]
    log(f"동작 모델 v{v}: 녹화 {len(recs)}개 · 장면 {entry['frames']} · 세부 동작 정확도 {pr.get('accuracy')} (기준선 {pr.get('baseline_accuracy')})"
        f" · 큰 단계 {ph.get('accuracy')} (기준선 {ph.get('baseline_accuracy')}) · Spark 대신 쓰기 {'가능' if ready else '아직'}")
    return entry


def current() -> Optional[tuple]:
    """(모델, 기록) — 가장 새 버전. 없으면 None."""
    reg = _registry()
    if not reg["versions"]:
        return None
    e = reg["versions"][-1]
    p = ROOT / f"v{e['version']}.pkl"
    return (pickle.loads(p.read_bytes()), e) if p.exists() else None


def try_label(out_dir) -> Optional[dict]:
    """준비된 모델이고 이 녹화에서 확신이 높으면 단계 구간을 낸다 (Spark 를 부르지 않는다). 아니면 None."""
    cur = current()
    if cur is None or not cur[1]["ready"]:
        return None
    import pandas as pd

    parts = sorted((Path(out_dir) / "lerobot" / "data").rglob("*.parquet"))
    if not parts:
        return None
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    labels, conf = predict(cur[0], frame_features(df))
    if float(np.mean(conf)) < CONFIDENT:
        return None
    t = df["timestamp"].to_numpy().astype(float).ravel()
    return {"model": f"datacache-v{cur[1]['version']}", "confidence": round(float(np.mean(conf)), 3), "steps": segments(labels, t)}
