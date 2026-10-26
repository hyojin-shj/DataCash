"""촉각: 워치 능동 진동으로 쥐는 힘, 영상으로 잡음·놓음, 소리 + 손목으로 내려놓음."""

import numpy as np

from datacache.session import load_session
from datacache.sync import align_session
from datacache.touch.contact import contact_phases, phase_events
from datacache.touch.sounds import audio_onsets, confirmed_contacts, wrist_impacts
from datacache.touch.vibration import calibrate, pulse_responses


def test_grip_force_from_active_vibration(synth_session):
    d, truth = synth_session
    s = load_session(d)
    rs = pulse_responses(s.watch_imu.t_ns, s.watch_imu.values, s.watch_vib.t_ns)
    assert sum(r.valid for r in rs) == len(rs)
    lag = np.median([(r.onset_ns - r.cmd_ns) / 1e6 for r in rs])
    assert abs(lag - truth["motor_latency_s"] * 1000) < 3          # 모터 지연을 잰다
    cal = calibrate(rs, s.meta["calibration"]["presses"])
    assert cal is not None and cal.resid_n < 1.0
    t = [(r.cmd_ns - truth["W0"]) / 1e9 for r in rs]
    held = [cal.force(r) for r, x in zip(rs, t) if 6.3 < x < 9.8]
    free = [cal.force(r) for r, x in zip(rs, t) if x > 10.5]
    assert abs(np.median(held) - truth["grip_force_n"]) < 1.0      # 8N 을 1N 안에
    assert np.median(free) < 0.5


def test_vision_contact_phases(synth_tracks):
    hand, objs, t = synth_tracks
    ev = phase_events(contact_phases(hand, objs, 30.0), t)
    hold = [e for e in ev if e["phase"] == "hold"]
    assert hold and 5.9 < hold[0]["t_s"] < 6.4                      # 6초에 잡음
    after = [e for e in ev if e["t_s"] > hold[0]["t_s"] and e["phase"] != "hold"]
    assert after and 9.9 < after[0]["t_s"] < 10.3                   # 10초에 놓기 시작


def test_putdown_confirmed_by_sound_and_wrist(synth_session):
    d, truth = synth_session
    s = load_session(d)
    al = align_session(s)
    ao = audio_onsets(s.audio.x, s.audio.rate, s.audio.start_ns)
    wi = wrist_impacts(s.watch_imu.t_ns, s.watch_imu.values, s.watch_vib.t_ns)
    wp = list(al.watch_to_phone(np.array([e["t_ns"] for e in wi])))
    c = confirmed_contacts(ao, wp)
    assert len(c) == 1 and abs((c[0]["t_ns"] - truth["P0"]) / 1e9 - truth["putdown_s"]) < 0.02
    assert c[0]["gap_ms"] < 5
