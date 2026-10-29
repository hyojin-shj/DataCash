"""품질 인증서 — 녹화마다 어떻게 만들었고 얼마나 믿을 만한지 (JSON + 사람이 읽는 마크다운).

출처·동의(이름 없이), 4감 동기화(방법·감각별 오차·검증), 촉각(보정·접촉 사건), 생체 요약, 라벨(단계·검증), 개인정보 처리, 데이터셋.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def to_markdown(c: dict) -> str:
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


def write(cert: dict, out_dir) -> None:
    out = Path(out_dir)
    cert.setdefault("issued_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (out / "certificate.json").write_text(json.dumps(cert, indent=2, ensure_ascii=False, default=float))
    (out / "certificate.md").write_text(to_markdown(cert))
