"""② 폰↔워치 시각 주고받기. 폰이 보낸 시각·워치가 받은 워치 시각·폰이 답을 받은 시각.
워치 시각은 왕복의 가운데에 찍혔다고 보고, 왕복이 짧은(네트워크가 덜 붐빈) 측정만 쓴다.
한계: 가는 길과 오는 길의 지연이 다르면 그 차이의 절반만큼 치우친다 — 그래서 음향 비콘으로 다듬는다."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .linear import ClockMap, fit


def rtt_map(rows: np.ndarray, keep_quantile: float = 0.3) -> Optional[ClockMap]:
    if rows is None or len(rows) < 2:
        return None
    send, watch, recv = rows[:, 0], rows[:, 1], rows[:, 2]
    rtt = recv - send
    thr = np.quantile(rtt, keep_quantile)
    sel = rtt <= max(thr, np.min(rtt))
    mid = send[sel] + (recv[sel] - send[sel]) // 2
    m = fit(watch[sel], mid, "clock_rtt")
    if m is not None:
        m.rtt_ms_median = float(np.median(rtt) / 1e6)
    return m
