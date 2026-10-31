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


def test_event_times_survive_real_boottime_magnitude():
    """실제 폰 시계는 10^15ns 쯤 — 찾은 시각이 float32 면 더할 때 268ms 단위로 뭉개진다 (실제 녹화에서 비콘이 다 버려지던 원인)."""
    from datacash.session.format import Audio
    from datacash.sync.acoustic import chirp, find_events

    rate = 48000
    x = np.zeros(rate * 3, np.float32)
    c = chirp(rate, 18000, 20000, 0.05)
    at = 1.2345
    x[int(at * rate):int(at * rate) + len(c)] += c
    t = find_events(x, rate, c, [1.2], 0.3, band=(17500, 20500))[0]
    assert type(t) is float
    a = Audio(x, rate, 2_562_125_975_249_302, "phone")
    assert abs(float(a.ns_at(t)) - (a.start_ns + at * 1e9)) < 0.05e6        # 0.05ms 안


def test_short_noisy_fit_does_not_invent_drift():
    """1분 녹화·2ms 잔차로는 기울기가 수백 ppm 씩 흔들린다 → 드리프트 사전값이 실제 수정 발진기 범위로 붙잡는다."""
    from datacash.sync.linear import DRIFT_SD_PPM

    rng = np.random.default_rng(3)
    w = np.sort(rng.uniform(0, 60e9, 30)).astype(np.int64) + 5_000_000_000_000
    p = w + 1_000_000_000 + rng.normal(0, 2e6, len(w)).astype(np.int64)        # 진짜 드리프트 0
    free = fit(w, p, "t")
    held = fit(w, p, "t", drift_sd_ppm=DRIFT_SD_PPM)
    assert abs(held.drift_ppm) < abs(free.drift_ppm) and abs(held.drift_ppm) < 15
    assert fit(w, p, "t", drift_sd_ppm=0.0).drift_ppm == 0.0


def test_audio_anchors_map_samples_to_clock():
    """녹음 중 기준점이 있으면 마이크 샘플 시계가 기기 시계와 어긋나도(100ppm) 샘플 → 시각이 맞다."""
    from datacash.session.format import Audio

    rate, start = 48000, 9_000_000_000_000
    true_rate = rate * (1 + 100e-6)
    anchors = np.array([[k * rate, start + int(k * rate / true_rate * 1e9)] for k in range(61)], np.int64)
    a = Audio(np.zeros(rate * 60, np.float32), rate, start, "watch", anchors)
    t_true = start + 50 * rate / true_rate * 1e9
    assert abs(float(a.ns_at(50.0)) - t_true) < 1e3                         # 기준점으로: 1µs 안
    assert abs(float(a.sec_at(t_true)) - 50.0) < 1e-6
    legacy = Audio(a.x, rate, start, "watch")
    assert abs(float(legacy.ns_at(50.0)) - t_true) > 4e6                    # 첫 시각만: 5ms 밀림


def test_legacy_audio_without_anchors_fits_offset_only(tmp_path):
    """옛 녹화(소리 시각이 녹음 처음 하나뿐): 소리로 잰 기울기를 믿지 않고 오프셋만 — 그래도 녹화 가운데는 1ms 안."""
    import json

    from datacash.session.synth import generate

    truth = generate(tmp_path / "s", duration=14.0)
    for name in ("audio.json", "watch_audio.json"):
        p = tmp_path / "s" / name
        info = json.loads(p.read_text())
        info.pop("anchors")
        p.write_text(json.dumps(info))
    s = load_session(tmp_path / "s")
    al = align_session(s)
    assert al.watch_to_phone.method == "beacon" and al.watch_to_phone.drift_ppm == 0.0
    mid = s.watch_imu.t_ns[len(s.watch_imu.t_ns) // 2]
    assert abs(al.watch_to_phone(np.array([mid]))[0] - watch_to_phone_truth(truth, [mid])[0]) / 1e6 < 1.0
    assert al.error_ms["wrist"] > 3.0                                       # 기본 오차도 그만큼 크게 적는다
