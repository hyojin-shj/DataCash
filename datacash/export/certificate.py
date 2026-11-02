"""품질 인증서 — 녹화마다 어떻게 만들었고 얼마나 믿을 만한지 (JSON + 사람이 읽는 마크다운).

출처·동의(이름 없이), 4감 동기화(방법·감각별 오차·검증), 촉각(보정·접촉 사건), 생체 요약, 라벨(단계·검증), 개인정보 처리, 데이터셋.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def _fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def to_markdown(c: dict, lang: str = "ko") -> str:
    if lang != "ko":
        return to_markdown_en(c)
    L = [f"# DataCash 품질 인증서 — {c['session_id']}", "",
         f"- 발급: {c['issued_at']} · 미션: {c.get('mission', {}).get('title', '-')} · 등급: {c['level']}",
         f"- 데이터셋: LeRobot v3.0, 에피소드 {c['dataset']['episodes']}개, {c['dataset']['frames']}장면 ({c['dataset']['fps']}fps)",
         f"- 채널: {', '.join(c['dataset']['features'])}", ""]
    s = c["sync"]
    L += ["## 4감 동기화 (사용자 동작 없이 자동)", "",
          f"고른 방법: **{s.get('chosen') or '없음 (폰 채널만)'}**", "",
          "| 방법 | 짝 수 | 잔차 (ms) | 드리프트 (ppm) | 비고 |", "|---|---|---|---|---|"]
    for k, m in s.get("methods", {}).items():
        L.append(f"| {k} | {m.get('n', '-')} | {_fmt(m.get('resid_ms'), 3)} | {_fmt(m.get('drift_ppm'))} | {m.get('note', '')} |")
    L += ["", "| 감각 | 동기화 오차 추정 (ms) | 판매 기준 통과 |", "|---|---|---|"]
    for k, v in s.get("error_ms", {}).items():
        L.append(f"| {k} | {_fmt(v)} | {'예' if s.get('ok', {}).get(k) else '아니오'} |")
    if s.get("checks"):
        L += ["", "검증: " + ", ".join(f"{k} = {v}" for k, v in s["checks"].items())]
    t = c.get("touch") or {}
    L += ["", "## 촉각 (촉각 센서 없이)", "",
          f"- 쥐는 세기: 워치 능동 진동 + GripSense 방식(UIST 2012) 특징, 펄스 {t.get('pulses', 0)}개 (유효 {t.get('pulses_valid', 0)}), "
          f"사람별 모델 {'있음 — 교차 검증 ' + _fmt((t.get('grip_model') or {}).get('accuracy_cv'), 2) + ', 창 ' + str((t.get('grip_model') or {}).get('n_windows')) + '개' if t.get('grip_model') else '없음 (GripSense 특징만 판매)'}",
          f"- 영상 접촉 사건: {len(t.get('contact_events', []))}개 · 확인된 접촉음(소리 + 손목 충격): {len(t.get('confirmed_sounds', []))}개",
          "- 판매 이름: '접촉 이벤트 + 워치로 잰 쥐는 세기 하·중·상 (0~100 점수)' (촉각 센서 데이터가 아님). 손끝 압력 분포는 제공하지 않는다"]
    h = c.get("human_state")
    if h:
        L += ["", "## 몸 감각 (생체 동의)", "", "- " + ", ".join(f"{k} {v}" for k, v in h.get("physio", {}).items()),
              f"- 본인 응답: {json.dumps(h.get('self_report'), ensure_ascii=False)}", "- 판정·조언 라벨 없음 (원값과 본인 응답만)"]
    st = c.get("steps") or []
    if st:
        L += ["", "## 단계 라벨 (Muse Spark, 감각으로 검증)", "", "| 구간 (초) | 지시문 | 검증 |", "|---|---|---|"]
        for x in st:
            ev = ", ".join(f"{k} {'O' if v else 'X'}" for k, v in (x.get("evidence") or {}).items())
            L.append(f"| {x['start_s']:.1f}~{x['end_s']:.1f} | {x['instruction']} | {x.get('verified', '-')} ({ev}) |")
    mk = c.get("mistakes")
    if mk:
        L += ["", "실수 후보 (Muse Spark): " + "; ".join(f"{m['t_s']:.1f}초 {m['description']}"
                                                     + (" — 참여자 표시와 일치" if m.get("confirmed_by_participant") else "") for m in mk)]
    o = c.get("outcome")
    if o:
        L += ["", "## 성공·실패 (본인 표시)", "", f"- 결과: {o.get('outcome')}, 다시 한 횟수: {o.get('retries')}, "
              f"녹화 중 '실수 표시': {len(o.get('self_marks_s', []))}번 {o.get('self_marks_s', [])}"]
    p = c.get("privacy")
    if p:
        L += ["", "## 개인정보 처리", "", "- 원본 영상·소리는 처리하는 동안만 서버에 있고 끝나면 지운다 (판매용은 얼굴을 바꾸고 가린 영상·개인정보를 끈 소리만)"
              " — 해커톤 시연 서버는 원본 ↔ 치환 비교를 보여 주려고 지우지 않는다",
              "- " + json.dumps(p, ensure_ascii=False)]
    cons = c.get("consent") or {}
    L += ["", "## 출처·동의", "", f"- 영상 동의: {cons.get('video')}, 생체 동의: {cons.get('bio')}, 연령대: {cons.get('age_band', '-')} (이름·얼굴 정보 없음)",
          f"- 기기: {json.dumps(c.get('devices'), ensure_ascii=False)}"]
    return "\n".join(L) + "\n"


def _note_en(note: str) -> str:
    """동기화 방법 비고 ('비콘 16개 중 16개 짝') → 영어."""
    m = re.match(r"(비콘|펄스) (\d+)개 중 (\d+)개 짝", note or "")
    if not m:
        return note or ""
    return f"{m.group(3)} of {m.group(2)} {'beacons' if m.group(1) == '비콘' else 'pulses'} paired"


def _age_en(v) -> str:
    """'19~29세' → '19–29', '30대' → '30s'."""
    v = str(v)
    return re.sub(r"(\d+)대", r"\1s", v).replace("세", "").replace("~", "–")


def to_markdown_en(c: dict) -> str:
    """해외 구매 기업·심사용 영어판 (같은 certificate.json 에서)."""
    yn = lambda v: "yes" if v else "no"
    L = [f"# DataCash quality certificate — {c['session_id']}", "",
         f"- Issued: {c['issued_at']} · Mission: {c.get('mission', {}).get('title', '-')} · Tier: {c['level']}",
         f"- Dataset: LeRobot v3.0, {c['dataset']['episodes']} episode(s), {c['dataset']['frames']} frames ({c['dataset']['fps']} fps)",
         f"- Channels: {', '.join(c['dataset']['features'])}", ""]
    s = c["sync"]
    L += ["## Multi-sense sync (automatic, no user action)", "",
          f"Chosen method: **{s.get('chosen') or 'none (phone channels only)'}**", "",
          "| Method | Pairs | Residual (ms) | Drift (ppm) | Note |", "|---|---|---|---|---|"]
    for k, m in s.get("methods", {}).items():
        L.append(f"| {k} | {m.get('n', '-')} | {_fmt(m.get('resid_ms'), 3)} | {_fmt(m.get('drift_ppm'))} | {_note_en(m.get('note', ''))} |")
    L += ["", "| Sense | Estimated sync error (ms) | Meets sale threshold |", "|---|---|---|"]
    for k, v in s.get("error_ms", {}).items():
        L.append(f"| {k} | {_fmt(v)} | {yn(s.get('ok', {}).get(k))} |")
    if s.get("checks"):
        L += ["", "Cross-checks: " + ", ".join(f"{k} = {v}" for k, v in s["checks"].items())]
    t = c.get("touch") or {}
    g = t.get("grip_model")
    L += ["", "## Touch (without a touch sensor)", "",
          f"- Grip force: active watch vibration + GripSense-style features (UIST 2012), {t.get('pulses', 0)} pulses ({t.get('pulses_valid', 0)} valid), "
          f"per-person model {'yes — cross-validated ' + _fmt(g.get('accuracy_cv'), 2) + ', ' + str(g.get('n_windows')) + ' windows' if g else 'none (GripSense features only)'}",
          f"- Video contact events: {len(t.get('contact_events', []))} · confirmed contact sounds (sound + wrist impact): {len(t.get('confirmed_sounds', []))}",
          "- Sold as 'contact events + watch-measured grip force low · mid · high (0–100 score)' — not tactile-sensor data. No fingertip pressure maps"]
    h = c.get("human_state")
    if h:
        ph = {k: (_age_en(v) if k == "age_band" else v) for k, v in h.get("physio", {}).items()}
        L += ["", "## Body state (biometric consent)", "", "- " + ", ".join(f"{k} {v}" for k, v in ph.items()),
              f"- Self-report: {json.dumps(h.get('self_report'), ensure_ascii=False)}", "- No judgement or advice labels (raw values and self-report only)"]
    st = c.get("steps") or []
    if st:
        L += ["", "## Step labels (Muse Spark, verified by sensors)", "", "| Span (s) | Instruction | Verified |", "|---|---|---|"]
        for x in st:
            ev = ", ".join(f"{k} {'O' if v else 'X'}" for k, v in (x.get("evidence") or {}).items())
            L.append(f"| {x['start_s']:.1f}–{x['end_s']:.1f} | {x['instruction']} | {x.get('verified', '-')} ({ev}) |")
    mk = c.get("mistakes")
    if mk:
        L += ["", "Mistake candidates (Muse Spark): " + "; ".join(f"{m['t_s']:.1f}s {m['description']}"
                                                              + (" — matches participant's mark" if m.get("confirmed_by_participant") else "") for m in mk)]
    o = c.get("outcome")
    if o:
        L += ["", "## Success / failure (self-reported)", "", f"- Outcome: {o.get('outcome')}, retries: {o.get('retries')}, "
              f"'mistake' marks during recording: {len(o.get('self_marks_s', []))} {o.get('self_marks_s', [])}"]
    p = c.get("privacy")
    if p:
        L += ["", "## Privacy processing", "", "- Raw video and audio stay on the server only while processing and are then deleted (what is sold is face-swapped, masked video and privacy-filtered audio)"
              " — the hackathon demo server keeps them to show the raw ↔ swapped comparison",
              "- " + json.dumps(p, ensure_ascii=False)]
    cons = c.get("consent") or {}
    L += ["", "## Source and consent", "", f"- Video consent: {cons.get('video')}, biometric consent: {cons.get('bio')}, age band: {_age_en(cons.get('age_band', '-'))} (no name or face data)",
          f"- Devices: {json.dumps(c.get('devices'), ensure_ascii=False)}"]
    return "\n".join(L) + "\n"


def write(cert: dict, out_dir) -> None:
    out = Path(out_dir)
    cert.setdefault("issued_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (out / "certificate.json").write_text(json.dumps(cert, indent=2, ensure_ascii=False, default=float))
    (out / "certificate.md").write_text(to_markdown(cert))
