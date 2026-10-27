"""4감 자동 동기화: 사용자 동작 없이 워치 시계를 폰 시계로 1ms 안에, 드리프트까지."""

import numpy as np

from datacash.session import load_session
from datacash.session.synth import watch_to_phone_truth
from datacash.sync import align_session
from datacash.sync.clock import rtt_map
from datacash.sync.linear import fit


def test_linear_fit_rejects_outliers():
    w = np.arange(0, 20, 1, dtype=np.int64) * 1_000_000_000 + 5_000_000_000_000
    p = (1_000_000_000_000 + (w - w[0]) / (1 + 40e-6)).astype(np.int64)
    p[5] += 50_000_000  # 50ms 엇나간 짝 하나
    m = fit(w, p, "t")
    assert m.n == 19 and abs(m.drift_ppm - 40) < 1 and m.resid_ms < 0.01


def test_beacon_beats_clock_and_pulses_agree(synth_session):
    d, truth = synth_session
    s = load_session(d)
    al = align_session(s)
    assert al.watch_to_phone.method == "beacon"
    mid = s.watch_imu.t_ns[len(s.watch_imu.t_ns) // 2]
    err = abs(al.watch_to_phone(np.array([mid]))[0] - watch_to_phone_truth(truth, [mid])[0]) / 1e6
    assert err < 1.0, err                                        # 비콘: 1ms 안
    assert abs(al.watch_to_phone.drift_ppm - 40) < 5             # 드리프트도 찾는다
    assert al.methods["vibration_pulse"]["n"] >= 30              # 촉각 펄스가 동기 신호를 겸한다
    assert abs(al.checks["diff_vibration_pulse_ms"]) < 2.0
    rtt = rtt_map(s.clock_sync)
    rtt_err = abs(rtt(np.array([mid]))[0] - watch_to_phone_truth(truth, [mid])[0]) / 1e6
    assert rtt_err < 5.0                                         # 시계 주고받기만으로도 수 ms
    assert all(al.ok.values())


def test_low_rate_watch_mic_falls_back(tmp_path):
    """워치 마이크가 16kHz 면 18kHz 비콘을 못 듣는다 → 진동 펄스로."""
    from datacash.session.synth import generate

    truth = generate(tmp_path / "s", duration=14.0, watch_audio_rate=16000)
    s = load_session(tmp_path / "s")
    al = align_session(s)
    assert "못 들음" in al.methods["beacon"]["note"]
    assert al.watch_to_phone.method == "vibration_pulse"
    mid = s.watch_imu.t_ns[len(s.watch_imu.t_ns) // 2]
    assert abs(al.watch_to_phone(np.array([mid]))[0] - watch_to_phone_truth(truth, [mid])[0]) / 1e6 < 3.0


def test_clock_via_server_without_direct_link(tmp_path):
    """폰·워치가 직접 통신하지 않고 각자 서버와 시각을 주고받아도 워치 → 폰 사상이 수 ms 안."""
    from datacash.session.synth import generate
    from datacash.sync.clock import via_server

    truth = generate(tmp_path / "s", duration=14.0, via_server=True)
    s = load_session(tmp_path / "s")
    assert s.clock_sync is None and s.phone_server_sync is not None
    m = via_server(s.phone_server_sync, s.watch_server_sync)
    mid = s.watch_imu.t_ns[len(s.watch_imu.t_ns) // 2]
    assert abs(m(np.array([mid]))[0] - watch_to_phone_truth(truth, [mid])[0]) / 1e6 < 6.0
    al = align_session(s)
    assert "clock_via_server" in al.methods and al.watch_to_phone.method == "beacon"   # 비콘이 더 정확하면 비콘
