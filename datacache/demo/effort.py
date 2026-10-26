"""활용 예시용 작은 모델 — 지금까지의 촉각·손목 움직임·심박으로 '몇 초 뒤 사람의 심박 여유(HRR)가 얼마나 오를지' 예측.

입력 (과거 3초 창): 쥐는 힘 평균(N) · 잡고 있는 비율 · 손목 움직임 크기 · 지금 HRR
출력: HORIZON 초 뒤 HRR − 지금 HRR (%p)
판정·조언을 하지 않는다 — 로봇이 자기 분담을 정하는 데만 쓴다 (사람을 이해하며 같이 일하는 로봇).
LeRobot 생체 등급 데이터셋에서 바로 학습한다 (사용자 녹화가 쌓이면 다시 학습).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pyarrow.parquet as pq

WINDOW_S, HORIZON_S = 3.0, 3.0
FEATURES = ["grip_mean_n", "hold_frac", "wrist_motion", "hrr_now", "bias"]


def episode_arrays(dataset: Path):
    t = pq.read_table(Path(dataset) / "data" / "chunk-000" / "file-000.parquet")
    fps = json.loads((Path(dataset) / "meta" / "info.json").read_text())["fps"]
    ep = np.array(t.column("episode_index").to_pylist())
    touch = np.array(t.column("observation.touch").to_pylist())
    phys = np.array(t.column("observation.physio").to_pylist())
    wrist = np.array(t.column("observation.imu.wrist").to_pylist()) if "observation.imu.wrist" in t.column_names else None
    for e in np.unique(ep):
        sel = ep == e
        acc = wrist[sel][:, :3] if wrist is not None else np.zeros((sel.sum(), 3))
        motion = np.abs(np.linalg.norm(acc, axis=1) - np.median(np.linalg.norm(acc, axis=1)))
        yield fps, touch[sel], phys[sel], motion


def windows(fps, touch, phys, motion, step_s=0.5):
    w, hz = int(WINDOW_S * fps), int(HORIZON_S * fps)
    X, y, idx = [], [], []
    for i in range(w, len(touch) - hz, max(1, int(step_s * fps))):
        if not (phys[i, 3] and phys[i + hz, 3]):
            continue
        g = touch[i - w:i, 1] * touch[i - w:i, 2]
        X.append([g.mean(), float(np.mean(touch[i - w:i, 3] == 3)), motion[i - w:i].mean(), phys[i, 2], 1.0])
        y.append(phys[i + hz, 2] - phys[i, 2])
        idx.append(i)
    return np.array(X), np.array(y), idx


@dataclass
class EffortModel:
    coef: np.ndarray
    mae_train: float
    mae_test: float
    n_train: int
    n_test: int

    def predict(self, x) -> float:
        return float(np.dot(self.coef, x))

    def to_dict(self):
        return {"features": FEATURES, "coef": [round(float(v), 4) for v in self.coef], "mae_train_pctp": round(self.mae_train, 2),
                "mae_test_pctp": round(self.mae_test, 2), "n_train": self.n_train, "n_test": self.n_test,
                "window_s": WINDOW_S, "horizon_s": HORIZON_S}


def train(train_sets: List[Path], test_sets: List[Path], ridge: float = 1.0) -> EffortModel:
    def stack(sets):
        X, y = [], []
        for ds in sets:
            for fps, touch, phys, motion in episode_arrays(ds):
                a, b, _ = windows(fps, touch, phys, motion)
                if len(a):
                    X.append(a)
                    y.append(b)
        return np.vstack(X), np.concatenate(y)

    Xtr, ytr = stack(train_sets)
    Xte, yte = stack(test_sets)
    mu = np.r_[Xtr[:, :-1].std(0) + 1e-6, 1.0]
    A = Xtr / mu
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ ytr) / mu
    return EffortModel(coef, float(np.mean(np.abs(Xtr @ coef - ytr))), float(np.mean(np.abs(Xte @ coef - yte))), len(ytr), len(yte))


def robot_share(pred_hrr_pct: float) -> float:
    """몇 초 뒤 사람의 심박 여유(HRR, 지금 + 예상 상승)가 높을 것 같으면 로봇이 더 든다 (기본 50%, 최대 90%)."""
    return float(np.clip(0.5 + 0.018 * pred_hrr_pct, 0.5, 0.9))
