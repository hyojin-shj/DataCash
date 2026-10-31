"""여러 라벨 벌 합치기 — 같은 구간을 다른 눈(조각 시작점을 엇갈린 Spark 1.3, Spark Open, 다른 이미지 모델)으로 본 단계들을
0.1초마다 표로 모아 (벌마다 무게), 동작이 너무 자주 바뀌지 않게 Viterbi 로 한 줄로 고른다.

  steps = fuse([run_a, run_b, ...], weights=[1, 1, 0.5], switch=3.0)

- 같은 모델을 여러 번 돌리면 실수도 같이 반복된다 (EPIC 10-31: Spark 1.3 두 벌 합치기는 새 구간에서 +1.7%p, 흔들림 안쪽).
  그래서 성격이 다른 벌을 섞는 데 쓴다.
- switch: 동작을 바꿀 때 내는 값 (로그 확률). 0 이면 0.1초마다 가장 많이 나온 동작, 크면 짧은 단계를 지운다.
- 경계는 합친 표에서 나오므로, 벌마다 경계가 조금씩 다르면 가운데쯤으로 모인다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

STEP_S = 0.1


def table(runs: Sequence[List[dict]], t0: float, t1: float, weights: Optional[Sequence[float]] = None,
          step_s: float = STEP_S, merge: Optional[Dict[str, str]] = None):
    """(시각들, 동작 이름들, 0.1초마다 동작별 몫) — 단계가 없는 곳은 'none'."""
    t = np.arange(t0, t1, step_s)
    m = merge or {}
    labels = sorted({m.get(s["action"], s["action"]) for r in runs for s in r} | {"none"})
    idx = {a: k for k, a in enumerate(labels)}
    P = np.zeros((len(t), len(labels)))
    for r, w in zip(runs, weights or [1.0] * len(runs)):
        col = np.full(len(t), idx["none"])
        for s in sorted(r, key=lambda s: s["start_s"]):
            col[(t >= s["start_s"]) & (t < s["end_s"])] = idx[m.get(s["action"], s["action"])]
        P[np.arange(len(t)), col] += w
    return t, labels, P / np.maximum(P.sum(1, keepdims=True), 1e-9)


def viterbi(P: np.ndarray, switch: float, eps: float = 0.02) -> np.ndarray:
    """몫 표 → 장면마다 동작 번호. 동작을 바꿀 때마다 switch 만큼 깎는다 (모든 동작 사이 같게)."""
    L = np.log(P * (1 - eps * P.shape[1]) + eps)
    n, K = L.shape
    D, B = L[0].copy(), np.zeros((n, K), int)
    for i in range(1, n):
        best, arg = D.max(), int(D.argmax())
        stay = D >= best - switch
        B[i] = np.where(stay, np.arange(K), arg)
        D = np.where(stay, D, best - switch) + L[i]
    path = [int(D.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(B[i, path[-1]]))
    return np.array(path[::-1])


def steps_of(t: np.ndarray, labels: List[str], path: np.ndarray, step_s: float = STEP_S) -> List[dict]:
    out: List[dict] = []
    for i, k in enumerate(path):
        a = labels[k]
        if a == "none":
            continue
        if out and out[-1]["action"] == a and abs(out[-1]["end_s"] - t[i]) < step_s / 2:
            out[-1]["end_s"] = round(float(t[i] + step_s), 3)
        else:
            out.append({"start_s": round(float(t[i]), 3), "end_s": round(float(t[i] + step_s), 3), "action": a})
    return out


def fuse(runs: Sequence[List[dict]], t0: float, t1: float, weights: Optional[Sequence[float]] = None, switch: float = 3.0,
         merge: Optional[Dict[str, str]] = None) -> List[dict]:
    t, labels, P = table(runs, t0, t1, weights, merge=merge)
    out = steps_of(t, labels, viterbi(P, switch))
    for s in out:   # 이 단계 동안 벌들이 같은 동작을 낸 몫 (여러 벌이 같게 본 단계가 더 잘 맞는다 — 정답 예시 고르기에 쓸 수 있다)
        sel = (t >= s["start_s"]) & (t < s["end_s"])
        s["agree"] = round(float(P[sel, labels.index(s["action"])].mean()), 3)
    return out
