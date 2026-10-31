"""접촉음 + 손목 충격 — 내려놓음·부딪힘·문지름처럼 소리가 나는 접촉의 시각을 ms 단위로.

소리: 1~8kHz 대역 에너지가 짧게 치솟는 순간(스펙트럼 변화량). 음향 비콘(18~20kHz)·진동음(170Hz)은 대역 밖이라 섞이지 않는다.
손목: 0.3~30Hz 밖 고주파 충격 중 진동 펄스 시각이 아닌 것.
둘이 40ms 안에 함께 있으면 '확인된 접촉'.
워치 마이크: 손목 IMU(100Hz)는 내려놓는 짧은 충격을 거의 못 담는다 (녹화 14편에서 충격 1개). 대신 손 가까이 있는
  워치 마이크에도 같은 소리가 잡히면 확인된 접촉으로 본다 — 폰 소리 사건의 40% 가 워치 소리와 25ms 안에 맞는다 (무작위 11%).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def audio_onsets(x: np.ndarray, rate: int, start_ns: int, k: float = 8.0, min_gap_s: float = 0.15, ns_at=None) -> List[dict]:
    """ns_at: 오디오 안 위치(초) → 기기 시계 ns (Audio.ns_at). 없으면 첫 샘플 시각 + 공칭 속도."""
    if x is None or len(x) < rate // 10:
        return []
    sos = butter(4, [1000, min(8000, rate / 2 * 0.95)], btype="band", fs=rate, output="sos")
    y = sosfiltfilt(sos, x)
    hop = int(rate * 0.005)
    e = np.sqrt(np.convolve(y ** 2, np.ones(hop) / hop, mode="same"))[::hop]
    floor = np.median(e) + 1e-9
    flux = np.maximum(0, np.diff(e, prepend=e[0]))
    # 바닥 소음의 k 배 이상 커지면서 빠르게 솟는 곳만 (k=8 ≈ 18dB)
    peaks, _ = find_peaks(e, height=k * floor, distance=max(1, int(min_gap_s / 0.005)))
    peaks = [p for p in peaks if flux[max(0, p - 4):p + 1].max() > 0.3 * e[p]]
    at = ns_at or (lambda sec: start_ns + sec * 1e9)
    return [{"t_ns": int(at(p * hop / rate)), "strength": round(float(e[p] / floor), 1)} for p in peaks]


def wrist_impacts(t_ns: np.ndarray, acc: np.ndarray, pulse_ns: Optional[np.ndarray] = None, k: float = 10.0) -> List[dict]:
    if t_ns is None or len(t_ns) < 64:
        return []
    rate = 1e9 / np.median(np.diff(t_ns))
    jerk = np.linalg.norm(np.diff(acc[:, :3], axis=0), axis=1)
    jerk = np.concatenate([[0.0], jerk])
    if pulse_ns is not None and len(pulse_ns):             # 진동 펄스 구간(명령~0.25초)은 뺀다
        for p in pulse_ns:
            a, b = np.searchsorted(t_ns, p), np.searchsorted(t_ns, p + 0.25e9)
            jerk[a:b] = 0.0
    live = jerk[jerk > 0]
    med = np.median(live) + 1e-9
    mad = np.median(np.abs(live - med)) + 1e-9
    thr = med + k * 4 * mad
    peaks, _ = find_peaks(jerk, height=thr, distance=max(1, int(0.15 * rate)))
    return [{"t_ns": int(t_ns[p]), "strength": round(float(jerk[p] / thr), 2)} for p in peaks]


def confirmed_contacts(audio_ev: List[dict], wrist_ev_phone_ns: List[int], tol_ms: float = 40.0) -> List[dict]:
    """소리 사건 중 손목 충격(폰 시계로 옮긴 것)과 tol 안에 겹치는 것."""
    w = np.array(sorted(wrist_ev_phone_ns), np.int64)
    out = []
    for e in audio_ev:
        if len(w):
            j = int(np.argmin(np.abs(w - e["t_ns"])))
            d = abs(int(w[j]) - e["t_ns"]) / 1e6
            if d <= tol_ms:
                out.append({**e, "wrist_ns": int(w[j]), "gap_ms": round(d, 2)})
    return out


def mic_contacts(audio_ev: List[dict], watch_ev_phone_ns: List[int], tol_ms: float = 25.0,
                 max_shift_ms: float = 100.0) -> List[dict]:
    """폰 소리 사건 중 워치 마이크 소리 사건(폰 시계로 옮긴 것)과 겹치는 것.
    두 마이크의 사건 시각에는 녹화마다 거의 일정한 차이(10~25ms — 마이크 처리 지연)가 있어,
    max_shift 안에서 가장 가까운 짝들의 중앙값을 빼고 tol 안이면 맞는 것으로 본다 (짝이 3개 미만이면 빼지 않는다)."""
    w = np.array(sorted(watch_ev_phone_ns), np.int64)
    if not len(w) or not audio_ev:
        return []
    near = [int(w[np.argmin(np.abs(w - e["t_ns"]))]) - e["t_ns"] for e in audio_ev]
    near = [d for d in near if abs(d) <= max_shift_ms * 1e6]
    shift = float(np.median(near)) if len(near) >= 3 else 0.0
    out = []
    for e in audio_ev:
        j = int(np.argmin(np.abs(w - shift - e["t_ns"])))
        d = abs(w[j] - shift - e["t_ns"]) / 1e6
        if d <= tol_ms:
            out.append({**e, "watch_mic_ns": int(w[j]), "gap_ms": round(float(d), 2), "shift_ms": round(shift / 1e6, 1)})
    return out
