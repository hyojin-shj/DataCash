"""② 폰↔워치 시각 주고받기. 폰이 보낸 시각·워치가 받은 워치 시각·폰이 답을 받은 시각.
워치 시각은 왕복의 가운데에 찍혔다고 보고, 왕복이 짧은(네트워크가 덜 붐빈) 측정만 쓴다.
한계: 가는 길과 오는 길의 지연이 다르면 그 차이의 절반만큼 치우친다 — 그래서 음향 비콘으로 다듬는다."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .linear import DRIFT_SD_PPM, ClockMap, fit


def rtt_map(rows: np.ndarray, keep_quantile: float = 0.3) -> Optional[ClockMap]:
    if rows is None or len(rows) < 2:
        return None
    send, watch, recv = rows[:, 0], rows[:, 1], rows[:, 2]
    rtt = recv - send
    thr = np.quantile(rtt, keep_quantile)
    sel = rtt <= max(thr, np.min(rtt))
    mid = send[sel] + (recv[sel] - send[sel]) // 2
    m = fit(watch[sel], mid, "clock_rtt", drift_sd_ppm=DRIFT_SD_PPM)
    if m is not None:
        m.rtt_ms_median = float(np.median(rtt) / 1e6)
    return m


def server_map(rows: np.ndarray, method: str) -> Optional[ClockMap]:
    """기기↔서버 시각 주고받기 (device_send_ns, server_ns, device_recv_ns) → 기기 시계 → 서버 시계 사상."""
    if rows is None or len(rows) < 2:
        return None
    send, srv, recv = rows[:, 0], rows[:, 1], rows[:, 2]
    rtt = recv - send
    sel = rtt <= max(np.quantile(rtt, 0.3), np.min(rtt))
    mid = send[sel] + (recv[sel] - send[sel]) // 2
    return fit(mid, srv[sel], method, drift_sd_ppm=DRIFT_SD_PPM)        # 입력: 기기 시각 → 출력: 서버 시각


def via_server(phone_rows: np.ndarray, watch_rows: np.ndarray) -> Optional[ClockMap]:
    """폰·워치가 각자 서버와 시각을 주고받은 기록으로 워치 → 폰 사상을 만든다 (둘이 직접 통신하지 않아도 된다).
    워치 → 서버 → 폰. 오차는 두 왕복의 비대칭이 더해진 만큼 (와이파이에서 수 ms) → 음향 비콘·진동 펄스로 다듬는다."""
    pw, pp = server_map(watch_rows, "watch_server"), server_map(phone_rows, "phone_server")
    if pw is None or pp is None:
        return None
    w = np.linspace(watch_rows[0, 0], watch_rows[-1, 2], 50).astype(np.int64)
    s_ns = pw(w)
    # 서버 → 폰: pp 의 역함수
    p_ns = (pp.w_ref + (s_ns.astype(np.float64) - pp.b) / pp.a).astype(np.int64)
    m = fit(w, p_ns, "clock_via_server", drift_sd_ppm=DRIFT_SD_PPM)
    m.resid_ms = float(np.hypot(pw.resid_ms, pp.resid_ms))
    m.rtt_ms_median = float(np.median(np.r_[watch_rows[:, 2] - watch_rows[:, 0], phone_rows[:, 2] - phone_rows[:, 0]]) / 1e6)
    return m
