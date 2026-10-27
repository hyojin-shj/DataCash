"""워치 능동 진동 펄스 — 펄스마다 손목 반응(시작 시각·울림 크기·사라지는 속도·에너지)을 잰다.
시작 시각은 모터 지연을 재고 폰·워치 시계를 맞추는 사건으로 쓴다. 쥐는 세기는 gripsense.py (GripSense 방식 창 특징).

입력: 워치 IMU (워치 시계), 진동 펄스 명령 시각(워치 시계).
펄스마다:
  onset    명령 뒤 실제로 떨림이 시작한 시각 (모터 지연을 재는 셈, 동기화의 사건으로도 쓴다)
  amp      울리는 동안(0.05초) 고주파 성분의 RMS
  decay_ms 끝난 뒤 울림 크기가 1/e 로 줄기까지 시간
  energy   반응 전체 에너지
IMU 속도가 모터 주파수(150~200Hz)의 두 배보다 느리면 떨림은 엇갈려(에일리어싱) 보이지만, 크기·사그라드는 정도는 그대로 잴 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

PULSE_S = 0.05


@dataclass
class PulseResponse:
    cmd_ns: int
    onset_ns: Optional[int]
    amp: float
    decay_ms: float
    energy: float
    valid: bool


def _highpass(acc: np.ndarray, rate: float) -> np.ndarray:
    """손목 움직임(수 Hz)을 빼고 진동 성분만. 축마다 거른 뒤 크기."""
    cut = min(30.0, rate * 0.2)
    sos = butter(4, cut, btype="high", fs=rate, output="sos")
    hp = sosfiltfilt(sos, acc, axis=0)
    return np.linalg.norm(hp, axis=1)


def pulse_responses(imu_ns: np.ndarray, acc: np.ndarray, cmd_ns: np.ndarray) -> List[PulseResponse]:
    if imu_ns is None or len(imu_ns) < 32:
        return []
    rate = 1e9 / np.median(np.diff(imu_ns))
    mag = _highpass(acc[:, :3], rate)
    out = []
    for c in np.asarray(cmd_ns, np.int64):
        i0 = int(np.searchsorted(imu_ns, c - 0.05e9))
        i1 = int(np.searchsorted(imu_ns, c))
        i2 = int(np.searchsorted(imu_ns, c + 0.25e9))
        if i2 - i1 < 8 or i1 - i0 < 4:
            out.append(PulseResponse(int(c), None, 0.0, 0.0, 0.0, False))
            continue
        base = mag[i0:i1]
        b_mu, b_sd = float(np.mean(base)), float(np.std(base)) + 1e-6
        seg = mag[i1:i2]
        env = np.convolve(seg, np.ones(3) / 3, mode="same")
        above = np.nonzero(env > b_mu + 6 * b_sd)[0]
        if not len(above):
            out.append(PulseResponse(int(c), None, 0.0, 0.0, 0.0, False))
            continue
        k0 = int(above[0])
        onset = int(imu_ns[i1 + k0])
        n_on = max(2, int(PULSE_S * rate))
        on = seg[k0:k0 + n_on]
        amp = float(np.sqrt(np.mean(on ** 2)))
        tail = env[k0 + n_on:]
        thr = (amp / np.e)
        below = np.nonzero(tail < max(thr, b_mu + 2 * b_sd))[0]
        decay_ms = float((below[0] if len(below) else len(tail)) / rate * 1000)
        energy = float(np.sum(seg[k0:] ** 2) / rate)
        out.append(PulseResponse(int(c), onset, amp, decay_ms, energy, True))
    return out
