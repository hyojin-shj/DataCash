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

워치4 실측 (10-29, 보정 1회 · 1회차로 배워 2회차를 맞히고 반대로도):
  워치4 IMU 는 하드웨어 최대 100Hz (논문 1kHz) → 모터 떨림은 거의 걸러지고, 진동 명령→실제 울림 지연도 20~80ms 로 흔들린다.
  결정 트리 하나·창 하나: 4단계 47%. 트리 25그루(forest) + 2초 묶기: 4단계 58% (꽉 100%, 안 쥠 32%).
  '안 쥠' 이 가장 흔들린다 (손을 편 자세가 매번 달라 진동이 퍼지는 모양이 바뀐다) → 쥐었는지는 영상의 손-물건 닿음이 정하고,
  쥔 동안만 가볍게·평소·꽉 3단계를 따로 배운 트리로 가린다: 82% (트리 하나 63%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

WIN_S, HOP_S = 0.5, 0.25
SMOOTH_S = 2.0         # 창 확률을 앞뒤 2초 평균 (쥐는 세기는 천천히 바뀐다 — 창 하나는 흔들린다)
N_TREES = 25
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
    def from_sklearn(t, classes) -> "Tree":
        return Tree(t.feature.tolist(), [float(v) for v in t.threshold], t.children_left.tolist(), t.children_right.tolist(),
                    [[float(c) for c in row[0]] for row in t.value], [int(c) for c in classes])


@dataclass
class Forest:
    """결정 트리 여러 그루 (scikit-learn RandomForest — 트리마다 창을 다시 뽑고 특징 일부만 본다). 확률은 트리 평균."""
    trees: List[Tree]

    def proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([t.proba(X) for t in self.trees], axis=0)

    def to_dict(self) -> dict:
        return {"trees": [t.to_dict() for t in self.trees]}

    @staticmethod
    def from_dict(d: dict) -> "Forest":
        return Forest([Tree(**t) for t in d["trees"]]) if "trees" in d else Forest([Tree(**d)])    # 예전 보정: 트리 하나

    @staticmethod
    def fit(X, y, seed=0) -> "Forest":
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(N_TREES, max_depth=6, min_samples_leaf=2, random_state=seed).fit(X, y)
        return Forest([Tree.from_sklearn(e.tree_, rf.classes_) for e in rf.estimators_])


def _smooth(p: np.ndarray, t_ns: Optional[np.ndarray]) -> np.ndarray:
    """창 확률을 앞뒤 SMOOTH_S/2 초 평균."""
    if t_ns is None or len(p) < 2:
        return p
    t = np.asarray(t_ns, np.int64)
    a = np.searchsorted(t, t - int(SMOOTH_S / 2 * 1e9))
    b = np.searchsorted(t, t + int(SMOOTH_S / 2 * 1e9), side="right")
    c = np.vstack([np.zeros((1, p.shape[1]), p.dtype), np.cumsum(p, 0)])
    return (c[b] - c[a]) / np.maximum(1, b - a)[:, None]


@dataclass
class GripModel:
    """사람마다 학습한 트리 묶음 (전체 특징 · 고주파만) + 보정 때 팔 움직임 한계. JSON 으로 저장·불러오기.
    held: 쥔 동안만(가볍게·평소·꽉) 따로 배운 묶음 — 영상이 '손이 물건에 닿음' 을 알려 주면 이것으로 가린다."""
    full: Forest
    hf: Forest
    arm_limit: float
    accuracy_cv: float
    n: int
    held: Optional[Forest] = None
    held_hf: Optional[Forest] = None
    accuracy_held_cv: float = float("nan")

    def _switch(self, full: Forest, hf: Forest, X: np.ndarray) -> np.ndarray:
        moving = X[:, ARM] > self.arm_limit
        out = np.zeros((len(X), len(LEVELS)), np.float32)
        if (~moving).any():
            out[~moving] = full.proba(X[~moving][:, :len(MODEL_FEATURES)])
        if moving.any():                     # 팔이 움직이면 떨림 특징을 믿지 않는다
            out[moving] = hf.proba(X[moving][:, :HF_ONLY])
        return out

    def proba(self, X: np.ndarray, t_ns: Optional[np.ndarray] = None, held: Optional[np.ndarray] = None) -> np.ndarray:
        """held: 창마다 손이 물건에 닿았는지 (영상). 주면 닿은 창은 쥔 동안 모델로, 안 닿은 창은 '안 쥠'."""
        X = np.asarray(X, np.float32)
        if held is not None and self.held is not None:
            held = np.asarray(held, bool)
            out = np.zeros((len(X), len(LEVELS)), np.float32)
            out[:, 0] = 1.0
            if held.any():
                out[held] = _smooth(self._switch(self.held, self.held_hf, X[held]), None if t_ns is None else np.asarray(t_ns)[held])
            return out
        return _smooth(self._switch(self.full, self.hf, X), t_ns)

    def predict(self, X: np.ndarray, t_ns: Optional[np.ndarray] = None, held: Optional[np.ndarray] = None):
        """(단계 0~3, 연속 점수 = 단계 기댓값). t_ns(창 시각)를 주면 2초 묶어 본다."""
        p = self.proba(X, t_ns, held)
        return p.argmax(1), p @ np.arange(len(LEVELS), dtype=np.float32)

    def to_dict(self) -> dict:
        return {"method": "gripsense", "features": MODEL_FEATURES, "levels": LEVELS, "full": self.full.to_dict(),
                "hf_only": self.hf.to_dict(), "held": self.held.to_dict() if self.held else None,
                "held_hf": self.held_hf.to_dict() if self.held_hf else None, "arm_limit": self.arm_limit,
                "accuracy_cv": round(self.accuracy_cv, 3),
                "accuracy_held_cv": None if np.isnan(self.accuracy_held_cv) else round(self.accuracy_held_cv, 3), "n_windows": self.n}

    @staticmethod
    def from_dict(d: dict) -> "GripModel":
        opt = lambda k: Forest.from_dict(d[k]) if d.get(k) else None      # noqa: E731
        return GripModel(Forest.from_dict(d["full"]), Forest.from_dict(d["hf_only"]), float(d["arm_limit"]),
                         float(d.get("accuracy_cv", 0)), int(d.get("n_windows", 0)), opt("held"), opt("held_hf"),
                         float(d["accuracy_held_cv"]) if d.get("accuracy_held_cv") is not None else float("nan"))


def _cv(X, y, seed) -> float:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    counts = np.bincount(y)[np.bincount(y) > 0]
    folds = int(min(5, counts.min()))
    if folds < 2 or len(counts) < 2:
        return float("nan")
    return float(np.mean(cross_val_score(RandomForestClassifier(N_TREES, max_depth=6, min_samples_leaf=2, random_state=seed),
                                         X, y, cv=folds)))


def train(win: Windows, y: np.ndarray, seed: int = 0) -> Optional[GripModel]:
    """트리 25그루 묶음 (깊이 ≤ 6, 잎 ≥ 2창): 4단계 · 쥔 동안 3단계 각각 전체 특징·고주파만.
    교차 검증 정확도(창 하나씩, 묶기 전)를 같이 남긴다."""
    ok = y >= 0
    X, y = win.x[ok], y[ok]
    if len(X) < 12 or len(set(y.tolist())) < 2:
        return None
    F = len(MODEL_FEATURES)
    arm_limit = float(np.percentile(X[:, ARM], 95) * 2 + 1e-6)
    h = y > 0
    held = held_hf = None
    if h.sum() >= 12 and len(set(y[h].tolist())) >= 2:
        held, held_hf = Forest.fit(X[h][:, :F], y[h], seed), Forest.fit(X[h][:, :HF_ONLY], y[h], seed)
    return GripModel(Forest.fit(X[:, :F], y, seed), Forest.fit(X[:, :HF_ONLY], y, seed), arm_limit, _cv(X[:, :F], y, seed),
                     int(len(X)), held, held_hf, _cv(X[h][:, :F], y[h], seed) if held else float("nan"))
