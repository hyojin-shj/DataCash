"""모델 진화 — 가중치 학습 없이, 쓸수록 정답 예시가 쌓여 Spark 1.3 라벨이 정확해진다.

  ① 1차 라벨        Muse Spark 1.3 (Meta API, 영상을 직접 본다) — 모든 녹화
  ② 2차 검증        Muse Spark Open 816B (Baseten → Fireworks) 가 같은 동작 구간의 장면 24장을 따로 라벨하고(훑기),
                    모든 단계를 그 구간 안 장면 4장으로 한 번 더 본다(촘촘히) → second_opinion. 세 번 본 것으로 다수결:
                    셋이 같고 센서로 검증 = 정답(gold). 훑기·촘촘히가 같게 Spark 1.3 과 다르면 바로잡고 '어려운 예시'(feedback/,
                    source=review)로 남긴다. 이의 신청은 Spark Open 추론 high 가 다시 본다 (judge.py).
  ③ 진화            다음 녹화부터 Spark 1.3 질문에 같은 미션의 정답 예시(gold_examples)와 지난 바로잡음(correction_hints)을 붙인다.
                    API 모델은 가중치를 바꿀 수 없으니 예시로 배운다.
  ④ 증명            녹화마다 'Spark 1.3 이 Spark Open 과 같은 동작으로 본 비율'을 기록 (agreement) → 운영자 화면의 선.
                    일치율이 오르면 바로잡을 단계가 줄어 2차 검증 비용도 준다.
기록: data/evolve/crosscheck/<녹화>.json (2차 검증), feedback/ (바로잡음), registry.json (agreement·데이터셋)
Glimmer(오픈 웨이트 30B) 학습 데이터 만들기(build_dataset)·평가는 남겨 둔다 — Fireworks 업로드가 풀리면 scripts/fireworks_train.py.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import config
from .understand.spark import PRIMITIVES

ROOT = config.DATA / "evolve"
READY_ACCURACY = 0.8      # 평가용 단계 정확도가 이보다 높아야 Spark 대신 쓴다
READY_EXAMPLES = 200      # 그리고 학습한 단계가 이만큼은 있어야
EVAL_EVERY = 4            # 녹화 4개 중 1개는 평가용 (녹화 단위로 나눠야 점수가 부풀지 않는다)
SIDE = 448                # 학습·평가 장면 크기 (긴 쪽)
WINDOW_S = 0.5            # ④ 새 녹화를 이만큼씩 잘라 묻는다 (사람 정답 단계는 평균 1.25초 — 1초 창은 거칠다)
STAY = 1 - WINDOW_S / 1.25   # 창 하나 뒤에도 같은 동작일 확률 (순서 디코딩) — 평균 단계 길이에서
ASK_PREV = False          # 질문에 앞 단계 이름을 넣나 — 넣으면 실제로 쓸 때 자기 답을 넣어야 해서 틀림이 이어진다 (10-31 v81: 사람 정답을
                          # 넣으면 61.5%, 자기 답이면 53.3%). 대신 순서는 순서 디코딩(decode)으로 쓴다
AUGMENT = 6               # 학습용 단계 하나에서 장면 3장을 뽑는 구간을 조금씩 바꿔 이만큼 만든다 (데이터가 적을 때 늘리기)
HARD_WEIGHT = 3           # 이의 신청에서 바로잡은 단계(모델이 틀리기 쉬운 '어려운 문제')는 이만큼 더 넣는다
FRAMES = 8                # 단계마다 보여 줄 장면 수 (10-31: 3 → 8 — Spark 처럼 흐름을 보게. 첫 장은 단계 바로 앞)
PRE_S = 0.3               # 첫 장을 단계 시작보다 이만큼 앞에서 — 집기(물체가 면 위에 있다가 손으로) ↔ 놓기(반대)를 가리려면 직전 모습이 필요
HARD_ACTIONS = {"grasp", "lift", "transport", "place", "release"}   # 서로 가장 많이 헷갈리는 동작 — 학습에 두 배로

QUESTION = """Frames in time order from one step of a robot-learning video (the first frame is just before the step), mission: "{mission}".
Gray areas are privacy masks; ignore them. Which manipulation primitive is the hand doing in this step?
Answer JSON {{"action": one of: {prims}}}."""

DEFINITIONS = """Definitions: reach = hand moves toward the object, not touching yet; grasp = fingers close on the object;
lift = raise the held object; pull = draw it toward you or out of something (e.g. a tissue out of its box); push = move it away;
twist = rotate it (e.g. a lid); press = push a button or surface down; squeeze; tear; wipe; insert = fit it into place;
transport = carry the held object through the air; place = set the held object down on a surface; handover = pass it to the
other hand or a person; release = fingers open and let go; open / close = a door, drawer, lid or container;
other = none of these."""
STYLES = {"basic": "", "defs": DEFINITIONS, "hint": "", "hint_defs": DEFINITIONS}
STYLE = "hint"              # 학습·평가·쓰기에 같은 질문 — 미션 동작 순서 힌트 (Spark 에게 주는 것과 같은 정보, 10-29 평가에서 가장 좋음)
HINT = "Expected primitive order in this mission (a hint, not a rule; repetitions are normal): {order}."


def mission_primitives(mission_id: str) -> List[str]:
    """미션의 예상 동작 순서 (missions.json) — Spark 에게도 주는 힌트. 모르면 빈 목록."""
    try:
        ms = json.loads((Path(__file__).with_name("missions.json")).read_text())["missions"]
        return next((m.get("primitives") or [] for m in ms if m["id"] == mission_id), [])
    except Exception:
        return []

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["action"],
          "properties": {"action": {"type": "string", "enum": PRIMITIVES}}}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------- 이의 신청에서 바로잡은 라벨 ----------

def add_correction(sid: str, fixes: List[dict], mission: str = "") -> None:
    """이의 신청에서 바로잡은 단계를 남긴다 — 다음 학습의 정답, Spark 에게 보여 줄 예시."""
    d = ROOT / "feedback"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sid}.json"
    old = json.loads(p.read_text())["fixes"] if p.exists() else []
    p.write_text(json.dumps({"sid": sid, "mission": mission, "fixes": old + fixes, "at": _now()}, indent=1, ensure_ascii=False))


def corrections(sid: str) -> List[dict]:
    p = ROOT / "feedback" / f"{sid}.json"
    return json.loads(p.read_text())["fixes"] if p.exists() else []


def gold_examples(mission_id: str, limit: int = 3) -> str:
    """Spark 1.3 프롬프트에 붙일 같은 미션의 정답 예시 — 두 모델이 같게 보고 센서로 검증된 단계 순서 (최근 녹화부터)."""
    d = ROOT / "crosscheck"
    if not mission_id or not d.exists():
        return ""
    lines = []
    for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        r = json.loads(p.read_text())
        if r.get("mission_id") != mission_id:
            continue
        seq = [x for x in r["rows"] if x["agree"] and x["verified"] and x.get("confirmed", True)
               and x["spark13"] != "other"]           # '기타'는 EPIC 정답과 0~3% 만 맞는다 (docs/label_eval.md 정답 예시 기준)
        if len(seq) < 2:
            continue
        lines.append("- " + " -> ".join(f'{x["spark13"]} {x["end_s"] - x["start_s"]:.1f}s' for x in seq[:14]))
        if len(lines) >= limit:
            break
    return ("Verified step sequences from past recordings of this same mission (action and duration; "
            "use them as a guide to granularity, not as the answer):\n" + "\n".join(lines)) if lines else ""


def correction_hints(limit: int = 6) -> str:
    """Spark 프롬프트에 붙일 '지난 바로잡음' (가중치를 바꿀 수 없는 API 모델은 예시로 배운다). 없으면 빈 문자열.
    2차 검증이 바로잡은 것(source=review)은 뺀다 — EPIC 정답으로 재 보니 바로잡은 뒤에도 28% 만 맞아, 예시로 가르치면 틀린 걸 가르친다.
    라벨 자체는 그대로 바로잡고(21% → 28%) feedback/ 에도 남긴다. 붙이는 것은 이의 신청에서 인정된 것만."""
    d = ROOT / "feedback"
    if not d.exists():
        return ""
    lines = []
    for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        r = json.loads(p.read_text())
        for f in r["fixes"]:
            if f.get("source") == "review":
                continue
            lines.append(f'- in a "{r.get("mission") or "manipulation"}" video, a step labeled "{f["was"]}" was actually "{f["action"]}"'
                         + (f' ({f["note"]})' if f.get("note") else ""))
    return ("Past corrections (confirmed on appeal or by a larger reviewer model; avoid repeating them):\n"
            + "\n".join(lines[:limit])) if lines else ""


# ---------- ② 2차 검증 (Spark Open) ----------

def compare(steps: List[dict], so_steps: List[dict], tol: float) -> List[dict]:
    """Spark 1.3 단계마다: 그 구간 ± tol(Spark Open 장면 간격) 안에 Spark Open 도 같은 동작을 봤는가.
    (Spark Open 은 장면 16장만 봐 경계가 0.5~1초 거칠다 — 가장 많이 겹치는 단계 하나만 보면 같은 동작도 어긋난다.)"""
    def overlap(a, b):
        return max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))

    rows = []
    for s in steps:
        win = {"start_s": s["start_s"] - tol, "end_s": s["end_s"] + tol}
        near = [n for n in so_steps if overlap(win, n) > 0]
        best = max(near, key=lambda n: overlap(s, n), default=None)
        same = any(n["action"] == s["action"] for n in near)
        rows.append({"start_s": s["start_s"], "end_s": s["end_s"], "spark13": s["action"],
                     "spark_open": s["action"] if same else (best["action"] if best else None), "agree": same,
                     "verified": bool(s.get("verified"))})
    return rows


def summarize(name: str, mission: str, rows: List[dict], so_steps: List[dict], mission_id: str = "") -> dict:
    return {"recording": name, "mission": mission, "mission_id": mission_id, "steps": len(rows), "agree": sum(x["agree"] for x in rows),
            "gold": sum(bool(x["agree"] and x["verified"] and x.get("confirmed", True)) for x in rows), "rows": rows,
            "spark_open_steps": so_steps}


def second_opinion(sid: str, video, t0: float, mission: str, mission_id: str, steps: List[dict], log=print) -> Optional[dict]:
    """Spark Open 이 같은 구간을 따로 라벨 → 단계마다 같은 동작인지.
    다르게 본 단계는 Spark Open 이 그 구간 안 장면 4장을 촘촘히 다시 보고(adjudicate) 정한다 — 16장 훑기는 짧은 단계(쥐기·들기)를
    놓치기 쉬워서, 훑은 답으로 바로 바꾸지 않는다. 판단이 Spark 1.3 과 같으면 유지(정답 후보), 다르면 그 동작으로 바로잡는다
    (steps 를 그 자리에서 고친다: action · labeled_by · relabeled_from · spark_open). 결과 crosscheck/<녹화>.json, 바로잡음 feedback/.
    Spark Open 을 부를 수 없으면 None (Spark 1.3 라벨 그대로)."""
    from .understand import spark_open

    if not steps or not spark_open.available():
        return None
    order = mission_primitives(mission_id)
    so = spark_open.label_frames(video, t0, None, mission, primitives=order)
    fr = so.get("frames") or []
    tol = (fr[-1] - fr[0]) / max(1, len(fr) - 1) if len(fr) > 1 else 0.7
    rows = compare(steps, so["steps"], tol)
    first = sum(r["agree"] for r in rows)
    # 모든 단계를 그 구간 안 장면 4장으로 촘촘히 한 번 더 (크레딧이 넉넉해 정확도를 산다) → 세 번 본 것으로 다수결
    verdicts = spark_open.adjudicate(video, [{"start_s": st["start_s"], "end_s": st["end_s"],
                                              "options": [r["spark13"]] + ([r["spark_open"]] if r["spark_open"] else [])}
                                             for st, r in zip(steps, rows)], mission, order)
    disputes = [k for k, r in enumerate(rows) if not r["agree"] and r["spark_open"]]
    fixes = []
    for k, (r, st, v) in enumerate(zip(rows, steps, verdicts)):
        r["verdict"] = v
        if r["agree"]:                        # 두 모델이 같게 봄: 촘촘히 본 것도 같으면 세 번 확인된 정답, 다르면 정답에서 뺀다
            r["confirmed"] = v is None or v == r["spark13"]
        elif v == r["spark13"]:               # 다르게 봤는데 촘촘히 보니 Spark 1.3 이 맞음 → 유지 (2:1)
            r["agree"], r["upheld"], r["confirmed"] = True, True, True
        elif v and (v == r["spark_open"] or not r["spark_open"]):   # 훑기와 촘촘히가 같음 → 바로잡음 (2:1)
            fixes.append({"start_s": st["start_s"], "end_s": st["end_s"], "was": st["action"], "action": v,
                          "note": f"2nd-pass review (scan {r['spark_open']}, close look {v})", "source": "review"})
            st.update(action=v, labeled_by=so.get("model"), relabeled_from=r["spark13"])
            r["confirmed"] = False
        else:                                 # 셋이 다 다름 → 그대로 두되 정답에서 뺀다
            r["confirmed"] = False
    for st, r in zip(steps, rows):
        st["spark_open"] = {"action": r.get("verdict") or r["spark_open"], "agree": bool(r["agree"] and r.get("confirmed", True)),
                            "model": so.get("model")}
    rep = {**summarize(sid, mission, rows, so["steps"], mission_id), "first_pass_agree": first}
    (ROOT / "crosscheck").mkdir(parents=True, exist_ok=True)
    (ROOT / "crosscheck" / f"{sid}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    if fixes:
        add_correction(sid, fixes, mission)
    record_agreement(sid, mission_id, rep["steps"], first, so.get("model", ""))
    gold = sum(bool(r["agree"] and r["verified"] and r.get("confirmed", True)) for r in rows)
    log(f"[{sid}] 2차 검증 {so.get('model')}: 처음 같은 동작 {first}/{rep['steps']} · 다르게 본 {len(disputes)}단계 중 "
        f"유지 {sum(1 for k in disputes if rows[k].get('upheld'))} · 바로잡음 {len(fixes)} · 세 번 확인된 정답 {gold}")
    return {"model": so.get("model"), "steps": rep["steps"], "agree": first, "disputed": len(disputes),
            "upheld": sum(1 for k in disputes if rows[k].get("upheld")), "relabeled": len(fixes), "gold": gold}


def finish_gold(sid: str, steps: List[dict]) -> Optional[int]:
    """센서 검증(verify)은 2차 검증 뒤 처리에서 붙는다 → 끝난 단계의 verified 를 crosscheck 에 옮기고 정답 수를 다시 센다.
    (이걸 안 하면 verified 가 모두 False 로 남아 정답 0, 다음 녹화에 붙일 정답 예시도 없다.)"""
    p = ROOT / "crosscheck" / f"{sid}.json"
    if not p.exists():
        return None
    rep = json.loads(p.read_text())
    if len(rep["rows"]) == len(steps):        # 경계 다듬기(refine.pick)는 시각만 옮기고 단계 수·순서는 그대로 → 순서로 맞춘다
        for r, s in zip(rep["rows"], sorted(steps, key=lambda s: s["start_s"])):
            r["verified"] = bool(s.get("verified"))
    else:
        ok = {round(s["start_s"], 2) for s in steps if s.get("verified")}
        for r in rep["rows"]:
            r["verified"] = round(r["start_s"], 2) in ok
    rep["gold"] = sum(bool(r["agree"] and r["verified"] and r.get("confirmed", True)) for r in rep["rows"])
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    return rep["gold"]


def record_agreement(sid: str, mission_id: str, steps: int, agree: int, model: str) -> None:
    """④ 녹화마다 Spark 1.3 ↔ Spark Open 일치 — 붙인 정답 예시 수와 함께 (예시가 늘수록 오르는지 본다)."""
    reg = _registry()
    hints = sum(1 for p in (ROOT / "crosscheck").glob("*.json")) if (ROOT / "crosscheck").exists() else 0
    reg["agreement"] = [a for a in reg.get("agreement", []) if a["sid"] != sid] + [
        {"sid": sid, "mission": mission_id, "steps": steps, "agree": agree, "rate": round(agree / steps, 3) if steps else None,
         "examples_before": max(0, hints - 1), "reviewer": model, "at": _now()}]
    _save(reg)


# ---------- ① 데이터 쌓기 ----------

def _frames(video, t0: float, t1: float, n: int = FRAMES) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    out = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + (t1 - t0) * (k + 0.5) / n) * 1000)
        ok, f = cap.read()
        if ok:
            s = SIDE / max(f.shape[:2])
            out.append(cv2.resize(f, None, fx=s, fy=s) if s < 1 else f)
    cap.release()
    return out


def labeled_steps(out_dir) -> Optional[dict]:
    """처리 결과 폴더 → {'id', 'mission', 'video', 'steps': [{start_s, end_s, action, source}]} (정답이 될 단계만)."""
    out = Path(out_dir)
    cert, video = out / "certificate.json", out / "masked_swap.mp4"
    if not cert.exists() or not video.exists():
        return None
    c = json.loads(cert.read_text())
    m = c.get("mission")
    mission = (m.get("title_en") or m.get("title") or m.get("id")) if isinstance(m, dict) else (m or "")
    mid = m.get("id") if isinstance(m, dict) else ""
    human = ROOT / "human" / f"{out.name}.json"
    if human.exists():                           # 운영자가 /label 에서 단 사람 정답이 있으면 그것만 (Spark 라벨은 사람 정답과 40~50% 만 맞아 그대로 배우면 Spark 수준에 머문다)
        h = json.loads(human.read_text())
        if h.get("excluded"):
            return None
        steps = [{"start_s": s["start_s"], "end_s": s["end_s"], "action": s["action"], "source": "human"}
                 for s in h["steps"] if s["action"] in PRIMITIVES and s["end_s"] - s["start_s"] >= 0.2]
        return {"id": out.name, "mission": mission, "mission_id": mid, "video": video, "steps": steps} if steps else None
    steps =[{"start_s": s["start_s"], "end_s": s["end_s"], "action": s["action"], "source": "spark+sensors"}
             for s in c.get("steps") or [] if s.get("verified") and s.get("action") in PRIMITIVES]
    cross = ROOT / "crosscheck" / f"{out.name}.json"
    if cross.exists():                           # 교차 검증(scripts/crosscheck.py): 더 큰 Spark Open 도 같은 동작이라고 본 단계만
        ok = {(round(r["start_s"], 3), r["spark13"]) for r in json.loads(cross.read_text())["rows"] if r["agree"] and r.get("confirmed", True)}
        steps = [{**s, "source": "spark+sensors+spark_open"} for s in steps if (round(s["start_s"], 3), s["action"]) in ok]
    for f in corrections(out.name):              # 이의 신청에서 바로잡은 단계가 우선 (더 높은 모델로 다시 본 정답)
        steps = [s for s in steps if not (s["start_s"] < f["end_s"] and f["start_s"] < s["end_s"])]
        if f.get("action") in PRIMITIVES:
            steps.append({"start_s": f["start_s"], "end_s": f["end_s"], "action": f["action"], "source": f.get("source", "appeal")})
    steps = sorted((s for s in steps if s["end_s"] - s["start_s"] >= 0.2), key=lambda s: s["start_s"])
    return {"id": out.name, "mission": mission, "mission_id": mid, "video": video, "steps": steps} if steps else None


def build_dataset(out_dirs, log=print) -> Optional[dict]:
    """처리된 녹화들 → data/evolve/sft/{train,eval}.jsonl + images/. 녹화가 같으면 다시 만들지 않는다."""
    recs = [r for r in (labeled_steps(d) for d in sorted(out_dirs)) if r]
    if not recs:
        return None
    reg = _registry()
    key = sorted(f"{r['id']}:{len(r['steps'])}:{len(corrections(r['id']))}" for r in recs)
    key.append(f"frames{FRAMES}-pre{PRE_S}-{'prev' if ASK_PREV else 'noprev'}-hard2")      # 질문·장면 방식이 바뀌면 다시 만든다
    if reg["datasets"] and reg["datasets"][-1]["key"] == key:
        return reg["datasets"][-1]
    sft = ROOT / "sft"
    img = sft / "images"
    img.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "eval": []}
    for i, r in enumerate(recs):
        split = "eval" if (i % EVAL_EVERY == EVAL_EVERY - 1) else "train"
        for k, s in enumerate(r["steps"]):
            d = s["end_s"] - s["start_s"]
            # 평가용은 한 번만, 학습용은 구간을 조금씩 옮겨 여러 벌 (같은 동작의 다른 순간) · 어려운 문제는 더
            n = 1 if split == "eval" else AUGMENT * (HARD_WEIGHT if s["source"] in ("appeal", "review") else 1) \
                * (2 if s["action"] in HARD_ACTIONS else 1)
            prev = (r["steps"][k - 1]["action"] if k else None) if ASK_PREV else ""
            for v in range(n):
                j0 = 0.0 if v == 0 else ((v * 37) % 11 - 5) / 50 * d          # -10% ~ +10% 이동
                j1 = 0.0 if v == 0 else ((v * 53) % 11 - 5) / 50 * d
                paths = []
                for j, f in enumerate(_frames(r["video"], max(0.0, s["start_s"] - PRE_S + j0), s["end_s"] + j1)):
                    p = img / f"{r['id']}_{k}_{v}_{j}.jpg"
                    cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    paths.append(f"images/{p.name}")
                if len(paths) < FRAMES:
                    continue
                rows[split].append({"messages": [
                    {"role": "user", "content": [{"type": "image", "path": p} for p in paths]
                     + [{"type": "text", "text": question(r["mission"], order=mission_primitives(r.get("mission_id", "")), prev=prev)}]},
                    {"role": "assistant", "content": json.dumps({"action": s["action"]})}],
                    "meta": {"recording": r["id"], "mission_id": r.get("mission_id", ""), "step": k, "variant": v, "source": s["source"],
                             "prev": prev}})
    for split, rs in rows.items():
        (sft / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rs))
    # 학습 전 모델은 어떤 녹화도 보지 않았으니 정답 단계 전체(늘린 벌 빼고 한 벌씩)로도 잰다 — 숫자가 덜 흔들린다
    every = [x for x in rows["train"] + rows["eval"] if x["meta"].get("variant", 0) == 0]
    (sft / "all.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in every))
    counts = {}
    for x in rows["train"]:
        a = json.loads(x["messages"][1]["content"])["action"]
        counts[a] = counts.get(a, 0) + 1
    entry = {"version": len(reg["datasets"]) + 1, "built_at": _now(), "key": key, "recordings": len(recs),
             "train_steps": len(rows["train"]), "eval_steps": len(rows["eval"]),
             "from_appeals": sum(x["meta"]["source"] == "appeal" for x in rows["train"] + rows["eval"]),
             "train_actions": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}
    reg["datasets"].append(entry)
    _save(reg)
    log(f"Glimmer 학습 데이터 v{entry['version']}: 녹화 {len(recs)}개 · 학습 단계 {entry['train_steps']} · 평가 단계 {entry['eval_steps']}")
    return entry


# ---------- ② 실력 재기 ----------

def question(mission: str, style: Optional[str] = None, order: Optional[List[str]] = None, prev: Optional[str] = "") -> str:
    """prev: 바로 앞 단계 동작 (None 이면 첫 단계, "" 이면 말하지 않음 — 예전 데이터와 같은 질문)."""
    style = style or STYLE
    extra = STYLES.get(style, "")
    q = QUESTION.format(mission=mission, prims=", ".join(PRIMITIVES))
    if style.startswith("hint") and order:
        q += "\n" + HINT.format(order=" -> ".join(order))
    if prev != "":
        q += "\n" + (f"Previous step: {prev}." if prev else "This is the first step.")
    return q + ("\n" + extra if extra else "")


def _dist(r) -> dict:
    """답(JSON {"action": ...})의 동작 이름이 시작하는 토큰 자리 top_logprobs → {동작: 확률}. 첫 토큰이 여러 동작에 맞으면 나눈다."""
    toks = (r.choices[0].logprobs.content or []) if getattr(r.choices[0], "logprobs", None) else []
    txt, k = "", None
    for i, t in enumerate(toks):
        if txt.replace(" ", "").endswith('"action":"'):
            k = i
            break
        txt += t.token
    out = {}
    for c in (toks[k].top_logprobs or []) if k is not None else []:
        s = c.token.strip().strip('"')
        match = [a for a in PRIMITIVES if s and a.startswith(s)]
        for a in match:
            out[a] = out.get(a, 0.0) + float(np.exp(c.logprob)) / len(match)
    if not out:
        a = json.loads(r.choices[0].message.content or "{}").get("action")
        out = {a: 1.0} if a in PRIMITIVES else {}
    z = sum(out.values()) or 1.0
    return {a: v / z for a, v in out.items()}


def _ask_dist(client, images: List[np.ndarray], mission: str, order: Optional[List[str]] = None,
              model: str = "muse-glimmer") -> dict:
    """한 구간의 동작 확률 (순서 디코딩용). 실패하면 빈 dict (그 창은 순서 확률만으로)."""
    parts = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
             base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]).decode()}} for f in images]
    parts.append({"type": "text", "text": question(mission, None, order, "")})
    try:
        r = client.chat.completions.create(model=model, max_tokens=300, temperature=0, logprobs=True, top_logprobs=5,
                                           extra_body={"reasoning_effort": "low"},
                                           response_format={"type": "json_schema", "json_schema": {"name": "step", "schema": SCHEMA}},
                                           messages=[{"role": "user", "content": parts}])
        return _dist(r)
    except Exception:
        return {}


def _ask(client, images: List[np.ndarray], mission: str, style: Optional[str] = None, reasoning: str = "low",
         order: Optional[List[str]] = None, text: Optional[str] = None, prev: Optional[str] = "") -> Optional[str]:
    """text: 질문을 그대로 (학습 데이터에 적힌 것) — 없으면 mission·order·prev 로 만든다."""
    parts = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
             base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]).decode()}} for f in images]
    parts.append({"type": "text", "text": text or question(mission, style, order, prev)})
    try:
        r = client.chat.completions.create(model="muse-glimmer", max_tokens=300 if reasoning == "low" else 1500, temperature=0,
                                           extra_body={"reasoning_effort": reasoning},
                                           response_format={"type": "json_schema", "json_schema": {"name": "step", "schema": SCHEMA}},
                                           messages=[{"role": "user", "content": parts}])
        a = json.loads(r.choices[0].message.content or "{}").get("action")
        return a if a in PRIMITIVES else None
    except Exception:
        return None


def evaluate(adapter: Optional[str] = None, split: str = "eval", client=None, log=print, style: Optional[str] = None,
             reasoning: str = "low", keep_running: bool = False) -> Optional[dict]:
    """Glimmer(어댑터를 붙이면 그 버전)가 평가용 단계를 맞히는 정도. 결과를 기록에 남긴다."""
    from openai import OpenAI

    from .understand import glimmer

    path = ROOT / "sft" / f"{split}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    if not rows:
        return None
    started = False
    if client is None:
        if not glimmer.ensure(log=log, lora=adapter):
            return None
        started = True
        client = OpenAI(base_url=glimmer.URL, api_key="local", timeout=300)
    try:
        y, p = [], []
        for x in rows:
            content = x["messages"][0]["content"]
            imgs = [cv2.imread(str(ROOT / "sft" / c["path"])) for c in content if c["type"] == "image"]
            mission = content[-1]["text"].split('mission: "', 1)[-1].split('"', 1)[0]
            y.append(json.loads(x["messages"][1]["content"])["action"])
            order = mission_primitives((x.get("meta") or {}).get("mission_id", ""))
            p.append(_ask(client, imgs, mission, style, reasoning, order, text=None if style else content[-1]["text"]) or "none")
    finally:
        if started and not keep_running:
            glimmer.stop()
    yt, yp = np.array(y), np.array(p)
    f1s = []
    for a in sorted(set(y)):
        tp, fp, fn = ((yp == a) & (yt == a)).sum(), ((yp == a) & (yt != a)).sum(), ((yp != a) & (yt == a)).sum()
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    reg = _registry()
    train_steps = reg["datasets"][-1]["train_steps"] if reg["datasets"] else 0
    acc = float((yt == yp).mean())
    entry = {"version": len(reg["evals"]) + 1, "at": _now(),
             "model": "muse-glimmer-30b-q4" + (f" + {Path(adapter).name}" if adapter else " (학습 전)"),
             "question": style or STYLE, "reasoning": reasoning,
             "adapter": str(adapter) if adapter else None, "dataset": reg["datasets"][-1]["version"] if reg["datasets"] else None,
             "split": split, "steps": len(y), "accuracy": round(acc, 3), "macro_f1": round(float(np.mean(f1s)), 3),
             "baseline_accuracy": round(float(max(np.mean(yt == a) for a in set(y))), 3),
             "confusions": [{"truth": a, "said": b} for a, b in zip(y, p) if a != b][:20],
             "ready": bool(adapter) and acc >= READY_ACCURACY and train_steps >= READY_EXAMPLES}
    reg["evals"].append(entry)
    _save(reg)
    log(f"Glimmer 평가 v{entry['version']} ({entry['model']}): 정확도 {entry['accuracy']} (기준선 {entry['baseline_accuracy']}) · 단계 {len(y)}")
    return entry


# ---------- 순서 디코딩: 단계마다 Glimmer 확률 × 미션별 '다음 동작' 확률 → 녹화 전체를 한 번에 (Viterbi) ----------
# Glimmer 가 틀리는 것은 대부분 바로 옆 단계와의 혼동이다 (다가가기 → 들기, 쥐기 → 들기, 당기기 → 쥐기 …, 10-31 v81 평가).
# 미션은 순서가 거의 정해져 있어서 (reach → grasp → lift → transport → place → release 되풀이) 사람 정답에서 배운 순서로 바로잡는다.

def transitions(exclude=(), alpha: float = 0.5, beta: float = 4.0) -> dict:
    """사람 정답(human/)에서 미션별 동작 잇기 확률. exclude: 빼 둘 녹화 (평가용). 미션 것이 적으면 전체 것으로 메운다.
    {'start': {mission: {a: p}}, 'next': {mission: {a: {b: p}}}, 'g_start', 'g_next'} — 미션 '' 는 전체."""
    seqs = []
    for p in sorted((ROOT / "human").glob("*.json")) if (ROOT / "human").exists() else []:
        h = json.loads(p.read_text())
        if h.get("excluded") or h["sid"] in exclude or not h["steps"]:
            continue
        cert = config.DATA / "server" / "out" / h["sid"] / "certificate.json"
        m = (json.loads(cert.read_text()).get("mission") or {}) if cert.exists() else {}
        seqs.append((m.get("id", "") if isinstance(m, dict) else "", [s["action"] for s in h["steps"]]))
    K = len(PRIMITIVES)

    def count(ss):
        st, nx = {}, {}
        for seq in ss:
            st[seq[0]] = st.get(seq[0], 0) + 1
            for a, b in zip(seq, seq[1:]):
                nx.setdefault(a, {})[b] = nx.get(a, {}).get(b, 0) + 1
        return st, nx

    gs, gn = count([q for _, q in seqs])
    g_start = {a: (gs.get(a, 0) + alpha) / (sum(gs.values()) + alpha * K) for a in PRIMITIVES}
    g_next = {a: {b: (gn.get(a, {}).get(b, 0) + alpha) / (sum(gn.get(a, {}).values()) + alpha * K) for b in PRIMITIVES} for a in PRIMITIVES}
    out = {"start": {}, "next": {}, "g_start": g_start, "g_next": g_next}
    for mid in {m for m, _ in seqs}:
        ms, mn = count([q for m, q in seqs if m == mid])
        out["start"][mid] = {a: (ms.get(a, 0) + beta * g_start[a]) / (sum(ms.values()) + beta) for a in PRIMITIVES}
        out["next"][mid] = {a: {b: (mn.get(a, {}).get(b, 0) + beta * g_next[a][b]) / (sum(mn.get(a, {}).values()) + beta)
                                for b in PRIMITIVES} for a in PRIMITIVES}
    return out


def decode(dists: List[dict], mission_id: str, trans: dict, weight: float = 1.0, floor: float = 1e-3,
           stay: Optional[float] = None) -> List[str]:
    """dists: 단계(또는 창)마다 {동작: 확률} (Glimmer). weight: 순서 확률을 얼마나 믿을지 (0 이면 단계마다 가장 높은 것).
    stay: 단계가 아니라 고른 창마다일 때, 다음 창도 같은 동작일 확률."""
    if not dists:
        return []
    st = trans["start"].get(mission_id, trans["g_start"])
    nx = trans["next"].get(mission_id, trans["g_next"])
    A = PRIMITIVES
    em = [np.log(np.array([d.get(a, 0.0) for a in A]) + floor) for d in dists]
    D = em[0] + weight * np.log([st[a] for a in A])
    T = np.array([[nx[a][b] for b in A] for a in A])
    if stay is not None:
        T = stay * np.eye(len(A)) + (1 - stay) * T
    T = weight * np.log(T)
    back = []
    for e in em[1:]:
        cand = D[:, None] + T
        back.append(cand.argmax(0))
        D = cand.max(0) + e
    path = [int(D.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return [A[k] for k in reversed(path)]


# ---------- ④ 쓰기 ----------

def ready_adapter() -> Optional[str]:
    """Spark 대신 써도 되는 어댑터 (평가에서 준비됨이 나온 가장 최근 것). 없으면 None."""
    good = [e for e in _registry()["evals"] if e.get("ready") and e.get("adapter") and Path(e["adapter"]).exists()]
    return good[-1]["adapter"] if good else None


def label_video(video, mission: str, t0: float = 0.0, client=None, order: Optional[List[str]] = None, mission_id: str = "",
                model: str = "muse-glimmer", trans: Optional[dict] = None) -> List[dict]:
    """학습한 Glimmer 로 새 녹화에 단계 라벨: WINDOW_S 마다 장면을 물어 동작 확률을 받고, 미션 순서와 함께 녹화 전체를
    한 번에 정한다 (decode). 같은 동작이 이어진 창은 한 단계로."""
    cap = cv2.VideoCapture(str(video))
    dur = cap.get(7) / (cap.get(5) or 30)
    cap.release()
    ts = list(np.arange(t0, dur - 0.2, WINDOW_S))
    dists = [_ask_dist(client, _frames(video, max(0.0, t - PRE_S), min(dur, t + WINDOW_S)), mission, order, model) for t in ts]
    path = decode(dists, mission_id, trans or transitions(), stay=STAY)
    steps: List[dict] = []
    for t, a in zip(ts, path):
        if steps and steps[-1]["action"] == a:
            steps[-1]["end_s"] = float(min(dur, t + WINDOW_S))
        else:
            steps.append({"action": a, "start_s": float(t), "end_s": float(min(dur, t + WINDOW_S))})
    return [s for s in steps if s["action"] != "other"]


def export_fireworks(split: str = "train") -> Path:
    """Axolotl 형식(이미지 경로) → Fireworks 형식(이미지를 base64 data URI 로 안에 넣기). 파일 경로를 돌려준다."""
    src, dst = ROOT / "sft" / f"{split}.jsonl", ROOT / "sft" / f"fireworks_{split}.jsonl"
    with open(dst, "w") as out:
        for line in src.read_text().splitlines():
            x = json.loads(line)
            parts = []
            for c in x["messages"][0]["content"]:
                if c["type"] == "image":
                    b = base64.b64encode((ROOT / "sft" / c["path"]).read_bytes()).decode()
                    parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}})
                else:
                    parts.append(c)
            out.write(json.dumps({"messages": [{"role": "user", "content": parts}, x["messages"][1]]}) + "\n")
    return dst


def record_training(entry: dict) -> None:
    """클라우드 학습(③) 기록 — 운영자 화면에 보인다."""
    reg = _registry()
    reg["trainings"] = [t for t in reg.get("trainings", []) if t.get("job") != entry.get("job")] + [entry]
    _save(reg)


def _registry() -> dict:
    p = ROOT / "registry.json"
    r = json.loads(p.read_text()) if p.exists() else {}
    return {"datasets": r.get("datasets", []), "evals": r.get("evals", []), "trainings": r.get("trainings", []),
            "agreement": r.get("agreement", [])}


def _save(reg: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "registry.json").write_text(json.dumps(reg, indent=1, ensure_ascii=False))
