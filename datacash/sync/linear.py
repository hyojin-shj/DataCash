"""두 시계 사이 직선 사상: phone = a·(watch − w_ref) + b. 짝 지은 사건 시각들로 강건하게 맞춘다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

DRIFT_SD_PPM = 30.0   # 폰·워치(·서버) 수정 발진기 차이의 사전 퍼짐 — 기기마다 ±20ppm 안팎


@dataclass
class ClockMap:
    a: float                  # 기울기 (1 / (1 + 드리프트))
    b: float                  # w_ref 일 때 폰 시계 (ns, float)
    w_ref: int
    n: int                    # 맞추는 데 쓴 짝 수
    resid_ms: float           # 짝들의 잔차 RMS
    method: str

    def __call__(self, watch_ns) -> np.ndarray:
        w = np.asarray(watch_ns, np.int64) - self.w_ref
        return (self.b + self.a * w.astype(np.float64)).astype(np.int64)

    @property
    def drift_ppm(self) -> float:
        return (1.0 / self.a - 1.0) * 1e6

    def to_dict(self) -> dict:
        return {"method": self.method, "n": self.n, "resid_ms": round(self.resid_ms, 3),
                "drift_ppm": round(self.drift_ppm, 2)}


def _slope(x: np.ndarray, y: np.ndarray, drift_sd_ppm: Optional[float]):
    """기울기·절편. drift_sd_ppm: None 이면 자유, 0 이면 드리프트 0(오프셋만), 그 밖엔 드리프트 0 을 중심으로 그만큼 퍼진
    사전값을 둔 능선 회귀 — 짧은 녹화(1~2분)는 수 ms 잔차만으로도 기울기가 ±200ppm 씩 흔들리는데, 실제 폰·워치 수정 발진기
    차이는 수십 ppm 안이다(같은 폰·워치 쌍의 녹화 30편을 34분에 걸쳐 보면 시계 차이에 추세가 거의 없다). 녹화가 길수록 자료가 이긴다."""
    xm, ym = x.mean(), y.mean()
    sxx = float(np.sum((x - xm) ** 2))
    sxy = float(np.sum((x - xm) * (y - ym)))
    if drift_sd_ppm is not None and drift_sd_ppm <= 0:
        a = 1.0
    elif drift_sd_ppm is None:
        a = sxy / sxx
    else:
        r = (y - ym) - sxy / sxx * (x - xm)
        lam = max(float(np.mean(r ** 2)), 1e10) / (drift_sd_ppm * 1e-6) ** 2    # 잔차 바닥 0.1ms
        a = (sxy + lam) / (sxx + lam)
    return a, ym - a * xm


def fit(watch_ns, phone_ns, method: str, iters: int = 3, k: float = 3.0, min_n: int = 2,
        drift_sd_ppm: Optional[float] = None) -> Optional[ClockMap]:
    """반복 재가중 최소제곱 (잔차가 중앙값의 k 배를 넘는 짝은 버린다). 짝이 하나뿐이면 드리프트 0 으로 오프셋만.
    drift_sd_ppm 은 _slope 참고."""
    w = np.asarray(watch_ns, np.int64)
    p = np.asarray(phone_ns, np.int64)
    if len(w) == 0:
        return None
    w_ref = int(w[0])
    x = (w - w_ref).astype(np.float64)
    y = (p - (p[0] - (w[0] - w_ref))).astype(np.float64)  # 큰 수를 피하려 같은 기준으로 옮긴다
    shift = float(p[0] - (w[0] - w_ref))
    keep = np.ones(len(x), bool)
    a, b = 1.0, float(np.median(y - x))
    for _ in range(iters):
        if keep.sum() >= max(min_n, 2) and np.ptp(x[keep]) > 0:
            a, b = _slope(x[keep], y[keep], drift_sd_ppm)
        elif keep.any():
            a, b = 1.0, float(np.median(y[keep] - x[keep]))
        r = y - (a * x + b)
        med = np.median(np.abs(r[keep])) + 1e5   # 0.1ms 바닥
        keep = np.abs(r) <= k * med
    r = (y - (a * x + b))[keep]
    return ClockMap(float(a), float(b + shift), w_ref, int(keep.sum()), float(np.sqrt(np.mean(r ** 2)) / 1e6), method)
