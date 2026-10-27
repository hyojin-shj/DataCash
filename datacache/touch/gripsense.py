"""GripSense 방식의 쥐는 힘 단계 — Goel, Wobbrock, Patel, "GripSense", UIST 2012 를 워치 손목에 옮긴 것.

논문: 폰의 진동 모터를 울리면 쥐는 손이 떨림을 흡수한다 → 자이로의 고주파(모터) 성분이 줄어든다.
세게 누를수록 엄지와 나머지 손가락이 서로 버티며 생기는 낮은 떨림이 커진다 → 저주파 분산이 커진다.
1kHz 자이로를 0.5초 묶음으로 0.25초마다 보고, 고주파 90번째 백분위수 · 저주파 분산을 특징으로
사람마다 학습한 결정 트리(J48)가 약·중·강 3단계를 가린다 (3단계 95.1%, 쥐기 squeeze 가 가장 정확).
저울 같은 기준 기구 없이, 사용자가 스스로 구분한 세기로 학습한다.

여기서: 워치가 0.5초마다 0.05초 울리고(능동 진동 펄스), 워치 IMU(가속도·자이로)를 같은 방식으로 본다.
  창(0.5초, 0.25초 간격)마다 특징 8개:
    hf_p90_gyro · hf_p90_acc   모터가 울리는 동안 고역(>30Hz, IMU 속도에 맞춰 낮춤) 크기의 90번째 백분위수
    hf_ratio_gyro · hf_ratio_acc  울리는 동안 / 안 울리는 동안 고역 크기 (주변 진동·사람 차이를 줄인다)
    lf_var_gyro · lf_var_acc   4~12Hz(버티는 떨림·생리적 떨림 대역) 분산 — 팔 전체 움직임(4Hz 아래)은 뺀다
    motor_on                   창 안에서 모터가 울린 비율
    arm_motion                 0.5~4Hz 가속도 분산 — 팔 전체 움직임 (논문의 폰은 가만히 있었다)
    valid                      특징을 잴 만큼 샘플이 있었는지
  단계: 0 안 쥠 · 1 가볍게 · 2 평소처럼 · 3 꽉. 사람마다 보정(그 사람이 스스로 구분해 쥔 세 세기 + 손 놓기)으로
  학습한 결정 트리 (scikit-learn CART — C4.5/J48 의 가까운 친척). 모델은 JSON(트리 배열)으로 저장한다.
  미션 중에는 팔이 움직여 저주파 떨림 특징이 흔들리므로, 팔 움직임이 보정 때보다 크면 고주파(모터 흡수) 특징만 쓴
  두 번째 트리로 가린다.
한계: 논문은 손에 쥔 폰 자체가 울린다. 워치는 손목에서 울리므로 흡수는 손목 힘줄 긴장을 거친다 — 실측으로 다시 재야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

WIN_S, HOP_S = 0.5, 0.25
LEVELS = ["안 쥠", "가볍게", "평소처럼", "꽉"]
FEATURES = ["hf_p90_gyro", "hf_p90_acc", "hf_ratio_gyro", "hf_ratio_acc", "lf_var_gyro", "lf_var_acc", "motor_on", "arm_motion", "valid"]
MODEL_FEATURES = FEATURES[:6]
HF_ONLY = 4            # 앞 4개 = 고주파(모터 흡수) 특징
VALID, ARM = 8, 7


def _band(x: np.ndarray, rate: float, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    """축마다 거른 뒤 크기."""
    nyq = rate / 2
    if lo is not None and hi is not None:
        sos = butter(4, [lo, min(hi, nyq * 0.9)], btype="band", fs=rate, output="sos")
    elif lo is not None:
        sos = butter(4, min(lo, nyq * 0.6), btype="high", fs=rate, output="sos")
    else:
        sos = butter(4, min(hi, nyq * 0.9), btype="low", fs=rate, output="sos")
    return np.linalg.norm(sosfiltfilt(sos, x, axis=0), axis=1)


@dataclass
class Windows:
    t_ns: np.ndarray        # 창 가운데 시각 (워치 시계)
    x: np.ndarray           # (n, 8) FEATURES


def window_features(imu_ns: np.ndarray, imu: np.ndarray, motor_ns: np.ndarray, motor_ms: float = 50.0) -> Windows:
    """imu: (n, 6) ax ay az gx gy gz. motor_ns: 진동 명령 시각. 창마다 GripSense 특징."""
    if imu_ns is None or len(imu_ns) < 64:
        return Windows(np.zeros(0, np.int64), np.zeros((0, len(FEATURES)), np.float32))
    rate = 1e9 / float(np.median(np.diff(imu_ns)))
    hf_cut = min(30.0, rate * 0.2)
    hf_a, hf_g = _band(imu[:, :3], rate, hf_cut, None), _band(imu[:, 3:6], rate, hf_cut, None)
    lf_a, lf_g = _band(imu[:, :3], rate, 4.0, 12.0), _band(imu[:, 3:6], rate, 4.0, 12.0)
    arm = _band(imu[:, :3], rate, 0.5, 4.0)
    on = np.zeros(len(imu_ns), bool)
    for c in np.asarray(motor_ns, np.int64):     # 모터 지연(수십 ms)을 넉넉히 덮는다
        on[np.searchsorted(imu_ns, c):np.searchsorted(imu_ns, c + int((motor_ms + 40) * 1e6))] = True
    t0, t1 = int(imu_ns[0]), int(imu_ns[-1])
    centers = np.arange(t0 + WIN_S / 2 * 1e9, t1 - WIN_S / 2 * 1e9, HOP_S * 1e9).astype(np.int64)
    X = np.zeros((len(centers), len(FEATURES)), np.float32)
    for k, c in enumerate(centers):
        i0, i1 = np.searchsorted(imu_ns, c - WIN_S / 2 * 1e9), np.searchsorted(imu_ns, c + WIN_S / 2 * 1e9)
        sel = slice(i0, i1)
        m = on[sel]
        if i1 - i0 < 16 or m.sum() < 3 or (~m).sum() < 3:
            continue
        g_on, a_on = hf_g[sel][m], hf_a[sel][m]
        g_off, a_off = hf_g[sel][~m], hf_a[sel][~m]
        X[k] = [np.percentile(g_on, 90), np.percentile(a_on, 90),
                np.percentile(g_on, 90) / (np.median(g_off) + 1e-4), np.percentile(a_on, 90) / (np.median(a_off) + 1e-4),
                np.var(lf_g[sel]), np.var(lf_a[sel]), float(m.mean()), np.var(arm[sel]), 1.0]
    return Windows(centers, X)


def labels_for(win: Windows, segments: List[dict]) -> np.ndarray:
    """보정 구간 [{start_ns, end_ns, level}] (워치 시계) → 창마다 단계 (-1 = 구간 밖 또는 경계 0.5초 안)."""
    y = np.full(len(win.t_ns), -1, int)
    for s in segments:
        a, b = int(s["start_ns"]) + int(0.5e9), int(s["end_ns"]) - int(0.25e9)   # 쥐기 시작·끝의 흔들림은 뺀다
        y[(win.t_ns >= a) & (win.t_ns <= b)] = int(s["level"])
    y[win.x[:, VALID] < 0.5] = -1
    return y


@dataclass
class Tree:
    feature: list
    threshold: list
    left: list
    right: list
    value: list             # 마디마다 단계별 비율
    classes: list

    def proba(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), len(LEVELS)), np.float32)
        for i, x in enumerate(X):
            node = 0
            while self.left[node] != -1:
                node = self.left[node] if x[self.feature[node]] <= self.threshold[node] else self.right[node]
            v = np.asarray(self.value[node], np.float32)
            for c, p in zip(self.classes, v / max(v.sum(), 1e-9)):
                out[i, int(c)] = p
        return out

    def to_dict(self) -> dict:
        return {"feature": self.feature, "threshold": self.threshold, "left": self.left, "right": self.right,
                "value": self.value, "classes": self.classes}

    @staticmethod
    def fit(X, y, seed=0) -> "Tree":
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=seed).fit(X, y)
        t = clf.tree_
        return Tree(t.feature.tolist(), [float(v) for v in t.threshold], t.children_left.tolist(), t.children_right.tolist(),
                    [[float(c) for c in row[0]] for row in t.value], [int(c) for c in clf.classes_])


@dataclass
class GripModel:
    """사람마다 학습한 결정 트리 둘 (전체 특징 · 고주파만) + 보정 때 팔 움직임 한계. JSON 으로 저장·불러오기."""
    full: Tree
    hf: Tree
    arm_limit: float
    accuracy_cv: float
    n: int

    def proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
        moving = X[:, ARM] > self.arm_limit
        out = np.zeros((len(X), len(LEVELS)), np.float32)
        if (~moving).any():
            out[~moving] = self.full.proba(X[~moving][:, :len(MODEL_FEATURES)])
        if moving.any():                     # 팔이 움직이면 떨림 특징을 믿지 않는다
            out[moving] = self.hf.proba(X[moving][:, :HF_ONLY])
        return out

    def predict(self, X: np.ndarray):
        """(단계 0~3, 연속 점수 = 단계 기댓값)."""
        p = self.proba(X)
        return p.argmax(1), p @ np.arange(len(LEVELS), dtype=np.float32)

    def to_dict(self) -> dict:
        return {"method": "gripsense", "features": MODEL_FEATURES, "levels": LEVELS, "full": self.full.to_dict(),
                "hf_only": self.hf.to_dict(), "arm_limit": self.arm_limit, "accuracy_cv": round(self.accuracy_cv, 3),
                "n_windows": self.n}

    @staticmethod
    def from_dict(d: dict) -> "GripModel":
        return GripModel(Tree(**d["full"]), Tree(**d["hf_only"]), float(d["arm_limit"]), float(d.get("accuracy_cv", 0)),
                         int(d.get("n_windows", 0)))


def train(win: Windows, y: np.ndarray, seed: int = 0) -> Optional[GripModel]:
    """결정 트리 (깊이 ≤ 5, 잎 ≥ 3창) 둘. 5겹 교차 검증 정확도(전체 특징 트리)를 같이 남긴다."""
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    ok = y >= 0
    X, y = win.x[ok], y[ok]
    if len(X) < 12 or len(set(y.tolist())) < 2:
        return None
    counts = np.bincount(y)[np.bincount(y) > 0]
    folds = int(min(5, counts.min()))
    acc = float(np.mean(cross_val_score(DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=seed),
                                        X[:, :len(MODEL_FEATURES)], y, cv=folds))) if folds >= 2 else float("nan")
    arm_limit = float(np.percentile(X[:, ARM], 95) * 2 + 1e-6)
    return GripModel(Tree.fit(X[:, :len(MODEL_FEATURES)], y, seed), Tree.fit(X[:, :HF_ONLY], y, seed), arm_limit, acc, int(len(X)))
