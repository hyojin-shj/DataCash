"""④ 음향 사건 찾기 — 음향 비콘(처프)과 워치 진동음(톤 버스트)을 정합 필터로 샘플 단위까지 찾는다.

비콘: 폰 스피커가 낸 18→20kHz 처프를 폰 마이크와 워치 마이크가 같이 듣는다. 폰 마이크에 들린 시각은 '실제로 소리가 난 시각'
(스피커 지연이 빠진다, BeepBeep 의 자기 녹음 원리). 두 마이크 시각의 차이는 시계 차이 + 소리가 날아간 시간(40cm ≈ 1.2ms).
마이크 샘플 속도가 40kHz 보다 낮은 기기(워치 16kHz 등)는 18kHz 를 못 들어 비콘을 못 쓴다 → 진동 펄스·시계로 간다.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt


def chirp(rate: int, f0: float, f1: float, dur: float) -> np.ndarray:
    t = np.arange(int(rate * dur)) / rate
    k = (f1 - f0) / dur
    return (np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t)) * np.hanning(len(t))).astype(np.float32)


def tone(rate: int, f: float, dur: float) -> np.ndarray:
    t = np.arange(int(rate * dur)) / rate
    return (np.sin(2 * np.pi * f * t) * np.hanning(len(t))).astype(np.float32)


def bandpass(x: np.ndarray, rate: int, lo: float, hi: float) -> np.ndarray:
    hi = min(hi, rate / 2 * 0.98)
    if lo >= hi:
        return np.zeros_like(x)
    sos = butter(4, [lo, hi], btype="band", fs=rate, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def find_events(x: np.ndarray, rate: int, template: np.ndarray, expected_s: List[float], window_s: float,
                band: Optional[tuple] = None, min_snr: float = 6.0) -> List[Optional[float]]:
    """기대 시각(오디오 시작 기준 초) 둘레 ±window_s 에서 template 이 시작하는 시각(초, 샘플 사이 보간)을 찾는다.
    못 찾으면(정점이 둘레 중앙값의 min_snr 배 미만) None."""
    if band is not None:
        x = bandpass(x, rate, *band)
    out = []
    tpl = template[::-1].astype(np.float32)
    for e in expected_s:
        a = max(0, int((e - window_s) * rate))
        b = min(len(x), int((e + window_s) * rate) + len(template))
        if b - a <= len(template) * 2:
            out.append(None)
            continue
        c = np.abs(fftconvolve(x[a:b], tpl, mode="valid"))
        i = int(np.argmax(c))
        noise = np.median(c) + 1e-12
        if c[i] < min_snr * noise:
            out.append(None)
            continue
        d = 0.0
        if 0 < i < len(c) - 1:
            y0, y1, y2 = c[i - 1], c[i], c[i + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        out.append((a + i + d) / rate)
    return out


def can_hear(rate: int, f_hi: float) -> bool:
    return rate / 2 > f_hi * 1.02
