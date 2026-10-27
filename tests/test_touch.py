"""촉각: 워치 능동 진동으로 쥐는 힘, 영상으로 잡음·놓음, 소리 + 손목으로 내려놓음."""

import json

import numpy as np

from datacache.session import load_session
from datacache.sync import align_session
from datacache.touch.contact import contact_phases, phase_events
from datacache.touch.sounds import audio_onsets, confirmed_contacts, wrist_impacts
from datacache.touch.gripsense import GripModel, labels_for, train, window_features
from datacache.touch.vibration import pulse_responses


def test_motor_pulses_measured(synth_session):
    """워치 진동 펄스마다 반응을 찾고 모터 지연을 잰다 (동기화 사건으로도 쓴다)."""
    d, truth = synth_session
    s = load_session(d)
    rs = pulse_responses(s.watch_imu.t_ns, s.watch_imu.values, s.watch_vib.t_ns)
    assert sum(r.valid for r in rs) == len(rs)
    lag = np.median([(r.onset_ns - r.cmd_ns) / 1e6 for r in rs])
    assert abs(lag - truth["motor_latency_s"] * 1000) < 3


def test_gripsense_levels_from_self_rated_calibration(synth_session):
    """GripSense (UIST 2012): 자이로·가속도 고주파 90%값 + 4~12Hz 떨림 분산 → 사람마다 결정 트리.
    저울 없이 '가볍게·평소처럼·꽉·손 놓기'로 학습하고, 다른 시각의 쥐기(8N)와 놓은 손을 가린다."""
    d, truth = synth_session
    s = load_session(d)
    w = window_features(s.watch_imu.t_ns, s.watch_imu.values, s.watch_vib.t_ns)
    W0, dr = truth["W0"], truth["drift"]
    segs = [{"start_ns": int(W0 + a * (1 + dr) * 1e9), "end_ns": int(W0 + b * (1 + dr) * 1e9), "level": lv} for a, b, lv in truth["calib"]]
    y = labels_for(w, segs)
    feats = {lv: w.x[y == lv].mean(0) for lv in range(4)}
    assert feats[0][0] > feats[1][0] > feats[2][0] > feats[3][0]          # 세게 쥘수록 모터 떨림(자이로 고주파)이 줄어든다
    assert feats[3][4] > feats[1][4]                                      # 세게 쥘수록 버티는 떨림(저주파 분산)이 크다
    model = train(w, y)
    assert model is not None and model.accuracy_cv > 0.7
    tau = (w.t_ns - W0) / 1e9
    lv, score = model.predict(w.x)
    held, free = (tau > 6.5) & (tau < 9.5), tau > 16.5
    assert np.bincount(lv[held], minlength=4).argmax() == 2 and 1.5 < score[held].mean() < 2.5   # 8N → 평소처럼
    assert (lv[free] == 0).mean() > 0.8
    again = GripModel.from_dict(json.loads(json.dumps(model.to_dict())))                        # JSON 으로 저장해도 같다
    assert (again.predict(w.x)[0] == lv).all()


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


def test_calibration_session_then_reuse(tmp_path):
    """보정 세션(폰 화면이 안내한 '손 놓기·가볍게·평소처럼·꽉' 구간, 폰 시계) → 사람별 모델 저장 → 보정 구간 없는 다음 녹화에 그 모델."""
    import pyarrow.parquet as pq

    from datacache.jobs import run_session
    from datacache.session.synth import generate

    store = tmp_path / "grip_calibration"
    cal_dir = tmp_path / "s_cal"
    truth = generate(cal_dir, duration=18.0, via_server=True, seed=7)
    m = json.loads((cal_dir / "meta.json").read_text())
    m["mission"] = {"id": "grip_calibration"}
    m["devices"]["phone"]["id"] = "phoneA"
    (cal_dir / "meta.json").write_text(json.dumps(m))
    rep = run_session(cal_dir, tmp_path / "o_cal", use_api=False, calibration_store=store)
    assert rep.get("calibration", {}).get("method") == "gripsense" and rep["calibration"]["accuracy_cv"] > 0.7, rep

    nxt = tmp_path / "s_next"
    generate(nxt, duration=18.0, via_server=True, seed=7, grip_n=truth["level_n"][3])     # 같은 사람, 이번엔 꽉
    m = json.loads((nxt / "meta.json").read_text())
    m["devices"]["phone"]["id"] = "phoneA"
    m.pop("calibration")
    (nxt / "meta.json").write_text(json.dumps(m))
    run_session(nxt, tmp_path / "o_next", use_api=False, calibration_store=store)
    t = pq.read_table(tmp_path / "o_next" / "lerobot" / "data" / "chunk-000" / "file-000.parquet")
    touch = np.array(t.column("observation.touch").to_pylist())
    ts = np.array(t.column("timestamp").to_pylist())
    held = (ts > 6.5) & (ts < 9.5)
    assert np.median(touch[held, 1]) == 3 and touch[held, 3].all()       # 꽉
