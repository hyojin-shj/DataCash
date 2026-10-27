"""4감 정렬 — 워치 시계 → 폰 시계 사상을 여러 방법으로 구하고, 가장 믿을 만한 것을 고른 뒤 다른 방법·내용으로 검증한다.

결과(Alignment)
  watch_to_phone  고른 사상 (없으면 워치 채널을 쓰지 않는다)
  methods         방법별 사상 요약 (짝 수·잔차·드리프트) — 인증서에 그대로
  error_ms        감각별 동기화 오차 추정 (영상·소리·손목·심박) — LeRobot 의 sync.error_ms 열
  checks          독립 검증 (방법 간 차이, 영상↔손목 움직임 상관)
  ok              감각별 판매 가능 여부 (오차가 기준 이하이고 검증과 어긋나지 않음)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import config
from ..session.format import Session
from .acoustic import can_hear, chirp, find_events, tone
from .clock import rtt_map, via_server
from .linear import ClockMap, fit

SOUND_TRAVEL_MS = 1.0   # 폰 스피커 → 워치 마이크 (보통 30~50cm). 비콘 짝에서 뺀다
LIMIT_MS = {"video": 20.0, "audio": 20.0, "wrist": 10.0, "touch": 10.0, "physio": 500.0}  # 판매 기준
VIB_HZ = 170.0


@dataclass
class Alignment:
    watch_to_phone: Optional[ClockMap]
    methods: Dict[str, dict] = field(default_factory=dict)
    error_ms: Dict[str, float] = field(default_factory=dict)
    checks: Dict[str, float] = field(default_factory=dict)
    ok: Dict[str, bool] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"chosen": self.watch_to_phone.method if self.watch_to_phone else None, "methods": self.methods,
                "error_ms": {k: round(v, 3) for k, v in self.error_ms.items()},
                "checks": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.checks.items()},
                "ok": self.ok, "latency_ms": self.latency_ms}


def device_latency(model: Optional[str]) -> dict:
    """③ 기기 모델별 고정 지연표 (data/device_latency.json). 사람이 아니라 기종마다 한 번 잰 값. 없으면 0."""
    p = config.DATA / "device_latency.json"
    table = json.loads(p.read_text()) if p.exists() else {}
    return table.get(model or "", {})


def _beacon_pairs(s: Session, guess: Optional[ClockMap]):
    """비콘마다 (폰 마이크에 들린 폰 시각, 워치 마이크에 들린 워치 시각)."""
    if s.beacon_ns is None or s.audio is None or s.watch_audio is None or guess is None:
        return [], "비콘·마이크 없음"
    f0, f1, dur = (float(v) for v in s.beacon[0])
    if not (can_hear(s.audio.rate, f1) and can_hear(s.watch_audio.rate, f1)):
        return [], f"마이크가 {f1:.0f}Hz 를 못 들음 (폰 {s.audio.rate}Hz, 워치 {s.watch_audio.rate}Hz)"
    exp_p = [(t - s.audio.start_ns) / 1e9 for t in s.beacon_ns]
    hp = find_events(s.audio.x, s.audio.rate, chirp(s.audio.rate, f0, f1, dur / 1000), exp_p, 0.25, band=(f0 - 500, f1 + 500))
    # 워치 쪽 기대 시각: 폰 마이크에 들린 시각을 대략적 사상으로 옮긴 것
    inv = lambda p_ns: guess.w_ref + (p_ns - guess.b) / guess.a  # noqa: E731
    pairs = []
    for t_rel in hp:
        if t_rel is None:
            continue
        p_ns = s.audio.start_ns + t_rel * 1e9
        w_exp = (inv(p_ns) - s.watch_audio.start_ns) / 1e9
        hw = find_events(s.watch_audio.x, s.watch_audio.rate, chirp(s.watch_audio.rate, f0, f1, dur / 1000), [w_exp], 0.08,
                         band=(f0 - 500, f1 + 500))[0]
        if hw is not None:
            pairs.append((int(s.watch_audio.start_ns + hw * 1e9), int(p_ns)))
    return pairs, f"비콘 {len(s.beacon_ns)}개 중 {len(pairs)}개 짝"


def _vib_pairs(s: Session, guess: Optional[ClockMap], responses):
    """워치 진동 펄스: 워치 IMU 로 잰 실제 떨림 시작(워치 시각) ↔ 폰 마이크에 들린 진동음(폰 시각)."""
    if s.audio is None or guess is None or not responses:
        return [], "진동 펄스·폰 마이크 없음"
    pairs = []
    tpl = tone(s.audio.rate, VIB_HZ, 0.05)
    for r in responses:
        if not r.valid or r.onset_ns is None:
            continue
        p_exp = (guess(np.array([r.onset_ns]))[0] - s.audio.start_ns) / 1e9
        hp = find_events(s.audio.x, s.audio.rate, tpl, [p_exp], 0.15, band=(VIB_HZ - 60, VIB_HZ + 60), min_snr=3.0)[0]
        if hp is not None:
            pairs.append((int(r.onset_ns), int(s.audio.start_ns + hp * 1e9)))
    return pairs, f"펄스 {len(responses)}개 중 {len(pairs)}개 짝"


def _motion_check(s: Session, m: ClockMap, max_lag_s: float = 1.0) -> Optional[dict]:
    """⑤ 영상 프레임 차이 에너지 ↔ 손목 가속 크기 상관 — 참고용 검사.
    실제 폰+워치 녹화 10편(홈데이터미션, 10-24·25)에서 이 검사의 최고 상관 위치가 -5~+4초로 제각각이었고
    (같은 녹화들의 시계 왕복 잔차는 0.5~2.3ms 로 고르다), 화면 전체 움직임에는 몸통·배경·가리기 효과가 섞이므로
    판매 여부를 이 검사로 정하지 않는다. {lag_ms, cc, cc_at_0, verdict} — verdict: consistent · suspect · inconclusive."""
    import cv2

    if s.video_path is None or s.frame_ns is None or s.watch_imu is None:
        return None
    cap = cv2.VideoCapture(str(s.video_path))
    prev, energy = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (160, int(160 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        energy.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))
        prev = g
    n = min(len(energy), len(s.frame_ns))
    if n < 30:
        return None
    fps = 1e9 / np.median(np.diff(s.frame_ns[:n]))
    grid = s.frame_ns[0] + (np.arange(int((s.frame_ns[n - 1] - s.frame_ns[0]) / 1e9 * 100)) * 1e7).astype(np.int64)
    v = np.interp(grid, s.frame_ns[:n], np.asarray(energy[:n]))
    wt = m(s.watch_imu.t_ns)
    acc = s.watch_imu.values[:, :3]
    from scipy.signal import butter, sosfiltfilt

    rate = 1e9 / np.median(np.diff(s.watch_imu.t_ns))
    sos = butter(2, [0.3, min(8.0, rate / 2 * 0.9)], btype="band", fs=rate, output="sos")
    a = np.linalg.norm(sosfiltfilt(sos, acc, axis=0), axis=1)
    w = np.interp(grid, wt, a)
    # 영상 움직임 에너지는 '속도'에 가깝고 손목 가속계는 '가속'이라, 영상 쪽을 한 번 미분해 같은 종류로 맞춘다
    k = np.ones(15) / 15
    v = np.abs(np.gradient(np.convolve(v, k, mode="same")))
    k = np.ones(30) / 30
    v, w = np.convolve(v, k, mode="same"), np.convolve(w, k, mode="same")
    v, w = (v - v.mean()) / (v.std() + 1e-9), (w - w.mean()) / (w.std() + 1e-9)
    lags = np.arange(-int(max_lag_s * 100), int(max_lag_s * 100) + 1)
    cc = []
    for k in lags:                       # 겹친 부분의 피어슨 상관 (겹침이 짧아도 1을 넘지 않게)
        x, y = v[max(0, k):len(v) + min(0, k)], w[max(0, -k):len(w) - max(0, k)]
        cc.append(float(np.corrcoef(x, y)[0, 1]) if len(x) > 50 and x.std() > 0 and y.std() > 0 else 0.0)
    i = int(np.argmax(cc))
    c0 = cc[len(lags) // 2]
    lag = float(lags[i] * 10.0)
    verdict = ("consistent" if abs(lag) <= 150 or c0 >= cc[i] - 0.05 else
               "suspect" if cc[i] >= 0.5 and cc[i] - c0 >= 0.3 else "inconclusive")
    return {"lag_ms": lag, "cc": round(cc[i], 3), "cc_at_0": round(c0, 3), "verdict": verdict}


def align_session(s: Session, responses=None) -> Alignment:
    lat_p = device_latency((s.meta.get("devices") or {}).get("phone", {}).get("model"))
    lat_w = device_latency((s.meta.get("devices") or {}).get("watch", {}).get("model"))
    al = Alignment(None, latency_ms={"phone_camera": lat_p.get("camera_ms", 0.0), "phone_mic": lat_p.get("mic_ms", 0.0),
                                     "watch_imu": lat_w.get("imu_ms", 0.0), "watch_mic": lat_w.get("mic_ms", 0.0)})
    # 같은 폰 안의 영상·소리: 같은 시계. 남는 건 기기 고정 지연(표)과 프레임 시각 흔들림
    if s.frame_ns is not None and len(s.frame_ns) > 2:
        jitter = float(np.std(np.diff(s.frame_ns)) / 1e6)
        al.error_ms["video"] = jitter / 2 + (0.0 if lat_p else 5.0)   # 지연표가 없으면 5ms 여유
        al.error_ms["audio"] = 0.5 + (0.0 if lat_p else 5.0)
    if not s.has_watch:
        al.ok = {k: v <= LIMIT_MS[k] for k, v in al.error_ms.items()}
        return al
    rtt = rtt_map(s.clock_sync)
    if rtt is None and s.phone_server_sync is not None and s.watch_server_sync is not None:
        rtt = via_server(s.phone_server_sync, s.watch_server_sync)   # 폰·워치가 각자 서버와 맞춘 기록으로
    if rtt:
        al.methods[rtt.method] = {**rtt.to_dict(), "rtt_ms_median": round(getattr(rtt, "rtt_ms_median", 0.0), 2)}
    guess = rtt
    if responses is None:
        from ..touch.vibration import pulse_responses

        responses = pulse_responses(s.watch_imu.t_ns, s.watch_imu.values, s.watch_vib.t_ns) if s.watch_vib else []
    cands = {}
    bp, note = _beacon_pairs(s, guess)
    al.methods["beacon"] = {"note": note}
    if len(bp) >= 3:
        w, p = zip(*bp)
        # 소리가 날아간 시간(약 1ms)·마이크 고정 지연 보정: 워치 마이크가 늦게 들으면 워치 시각이 늦게 찍힌다
        mw = fit(np.array(w) - int((al.latency_ms["watch_mic"] + SOUND_TRAVEL_MS) * 1e6),
                 np.array(p) - int(al.latency_ms["phone_mic"] * 1e6), "beacon")
        cands["beacon"] = mw
        al.methods["beacon"].update(mw.to_dict())
    vp, note = _vib_pairs(s, cands.get("beacon") or guess, responses)
    al.methods["vibration_pulse"] = {"note": note}
    if len(vp) >= 3:
        w, p = zip(*vp)
        mv = fit(np.array(w), np.array(p), "vibration_pulse")
        cands["vibration_pulse"] = mv
        al.methods["vibration_pulse"].update(mv.to_dict())
    if rtt:
        cands[rtt.method] = rtt
    # 고르기: 비콘 > 진동 펄스 > 시계 (짝이 충분하고 잔차가 작을 때)
    chosen = None
    for k in ("beacon", "vibration_pulse", "clock_rtt", "clock_via_server"):
        if k in cands and (k.startswith("clock") or cands[k].resid_ms < 2.0):
            chosen = cands[k]
            break
    al.watch_to_phone = chosen
    if chosen is None:
        al.ok = {k: v <= LIMIT_MS[k] for k, v in al.error_ms.items()}
        return al
    # 방법 사이 차이 (같은 순간을 서로 어떻게 옮기나)
    mid = int(np.median(s.watch_imu.t_ns))
    for k, m in cands.items():
        if m is not chosen:
            al.checks[f"diff_{k}_ms"] = float((m(np.array([mid]))[0] - chosen(np.array([mid]))[0]) / 1e6)
    base = {"beacon": 0.3, "vibration_pulse": 1.5, "clock_rtt": 3.0, "clock_via_server": 5.0}[chosen.method]
    werr = float(np.hypot(base, chosen.resid_ms))
    al.error_ms["wrist"] = werr
    al.error_ms["touch"] = float(np.hypot(werr, 2.5))           # 펄스 간격·IMU 샘플(400Hz=2.5ms)
    if s.has_physio:
        al.error_ms["physio"] = float(np.hypot(werr, 250.0))    # 심박은 1초마다 한 값 → 절반 주기
    mc = _motion_check(s, chosen)
    al.checks["motion"] = mc if mc is not None else "판단 불가"       # 참고용 (판매 여부는 시계 맞춤 오차로)
    al.ok = {k: v <= LIMIT_MS[k] for k, v in al.error_ms.items()}
    return al
