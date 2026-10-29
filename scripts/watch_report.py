"""워치 감각 점검 — 모아 둔 녹화(또는 지정한 녹화)마다 API 없이 워치 쪽만 본다.

  python scripts/watch_report.py [녹화 ...]
  동기화      폰↔워치 시계 맞춤 방법·오차, 음향 비콘·진동 펄스 짝 수
  촉각        진동 펄스가 손목 센서에 잡힌 비율·지연, 쥐는 세기 (이 사람·이 워치 보정으로 · 영상 없이 워치만):
              쥔 창 비율, 쥔 동안 하·중·상 비율과 점수 중앙값
  생체        심박 평균·범위
  소리        워치 마이크 길이·크기
결과: data/server/watch_report.json (녹화별) · 화면에는 미션별 요약. 영상과 합친 '닿음 + 쥐는 세기' 는 처리(--process-collected) 뒤에.
"""

import json
import sys
import wave
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacash.jobs import calibration_key  # noqa: E402
from datacash.session import load_session  # noqa: E402
from datacash.sync import align_session  # noqa: E402
from datacash.touch.gripsense import GripModel, window_features  # noqa: E402
from datacash.touch.vibration import pulse_responses  # noqa: E402

ROOT = Path("data/server")
warnings.filterwarnings("ignore")


def one(sid: str) -> dict:
    s = load_session(ROOT / "sessions" / sid)
    r = {"session": sid, "mission": (s.meta.get("mission") or {}).get("id"),
         "seconds": round(float((s.frame_ns[-1] - s.frame_ns[0]) / 1e9), 1) if s.frame_ns is not None else None}
    if s.watch_imu is None or s.watch_vib is None:
        return {**r, "watch": False}
    t, imu, cmd = s.watch_imu.t_ns, s.watch_imu.values, np.asarray(s.watch_vib.t_ns, np.int64)
    pr = pulse_responses(t, imu, cmd)
    al = align_session(s, pr).to_dict()
    m = al["methods"]
    lat = [(p.onset_ns - p.cmd_ns) / 1e6 for p in pr if p.valid]
    r["sync"] = {"chosen": al["chosen"], "error_ms": al["error_ms"], "motion_check": (al.get("checks") or {}).get("motion"),
                 "beacon": m.get("beacon", {}).get("note") or f"{m.get('beacon', {}).get('n')}개 짝",
                 "server_resid_ms": m.get("clock_via_server", {}).get("resid_ms")}
    r["vibration"] = {"pulses": len(cmd), "detected": round(float(np.mean([p.valid for p in pr])), 3) if pr else 0,
                      "latency_ms": round(float(np.median(lat)), 1) if lat else None,
                      "imu_hz": round((len(t) - 1) / ((t[-1] - t[0]) / 1e9))}
    key = calibration_key(s.meta)
    cal = ROOT / "grip_calibration" / f"{key}.json"
    if cal.exists():
        model = GripModel.from_dict(json.loads(cal.read_text()))
        win = window_features(t, imu, cmd)
        ok = win.x[:, 8] > 0.5
        lv, sc = model.predict(win.x[ok], win.t_ns[ok])
        held = lv > 0
        r["grip"] = {"calibration": key, "windows": int(ok.sum()), "held_share": round(float(held.mean()), 3) if len(lv) else 0,
                     "low_mid_high": [round(float(np.mean(lv[held] == k)), 3) for k in (1, 2, 3)] if held.any() else None,
                     "score_median": round(float(np.median(sc[held])), 1) if held.any() else None,
                     "note": "영상 없이 워치만 (안 쥠 판단이 약함 — 처리 뒤 영상 닿음과 합친다)"}
    else:
        r["grip"] = {"calibration": None, "note": "이 폰·워치 보정이 없음 — 쥐는 세기 맞추기를 먼저"}
    if s.watch_hr is not None and len(s.watch_hr.values):
        hr = s.watch_hr.values[:, 0]
        hr = hr[hr > 0]
        r["heart_rate"] = {"samples": int(len(hr)), "mean": round(float(hr.mean()), 1) if len(hr) else None,
                           "range": [float(hr.min()), float(hr.max())] if len(hr) else None}
    wav = ROOT / "sessions" / sid / "watch_audio.wav"
    if wav.exists():
        w = wave.open(str(wav))
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        r["watch_audio"] = {"seconds": round(w.getnframes() / w.getframerate(), 1), "peak": int(np.abs(x).max()) if len(x) else 0}
    return r


def main():
    sids = sys.argv[1:] or sorted(l.name for m in (ROOT / "collected").iterdir() if m.is_dir() for l in m.iterdir())
    out = []
    for sid in sids:
        try:
            out.append(one(sid))
        except Exception as e:  # noqa: BLE001
            out.append({"session": sid, "error": f"{type(e).__name__}: {e}"})
        x = out[-1]
        g, v, sy = x.get("grip") or {}, x.get("vibration") or {}, x.get("sync") or {}
        print(f"{x.get('mission', '?'):15s} {sid} {x.get('seconds')}s · 동기화 {sy.get('chosen')} ±{(sy.get('error_ms') or {}).get('touch')}ms"
              f" · 진동 {v.get('pulses')}번 잡힘 {v.get('detected')} 지연 {v.get('latency_ms')}ms"
              f" · 쥠 {g.get('held_share')} 하중상 {g.get('low_mid_high')} 점수 {g.get('score_median')}"
              f" · 심박 {(x.get('heart_rate') or {}).get('mean')} · {x.get('error', '')}", flush=True)
    (ROOT / "watch_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    by = {}
    for x in out:
        if "error" in x or not x.get("watch", True):
            continue
        b = by.setdefault(x["mission"], {"n": 0, "sec": 0.0, "held": [], "score": [], "hr": [], "det": []})
        b["n"] += 1
        b["sec"] += x.get("seconds") or 0
        b["det"].append((x.get("vibration") or {}).get("detected") or 0)
        if (x.get("grip") or {}).get("held_share") is not None:
            b["held"].append(x["grip"]["held_share"])
        if (x.get("grip") or {}).get("score_median") is not None:
            b["score"].append(x["grip"]["score_median"])
        if (x.get("heart_rate") or {}).get("mean"):
            b["hr"].append(x["heart_rate"]["mean"])
    print("\n미션별")
    for k, b in by.items():
        med = lambda v: round(float(np.median(v)), 2) if v else None  # noqa: E731
        print(f"  {k:15s} 녹화 {b['n']} · {b['sec'] / 60:.1f}분 · 진동 잡힘 {med(b['det'])} · 쥔 비율 {med(b['held'])}"
              f" · 쥔 동안 점수 {med(b['score'])} · 심박 {med(b['hr'])}")


if __name__ == "__main__":
    main()
