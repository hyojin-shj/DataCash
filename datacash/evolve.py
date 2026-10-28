"""모델 진화 — Meta 오픈 웨이트 Muse Glimmer 를 우리 라벨로 키운다.

  ① 데이터 쌓기 (이 맥)   검증된 Spark 단계 + 이의 신청에서 Spark 1.3 최대 추론이 바로잡은 단계 →
                         단계마다 장면 3장(처음·가운데·끝) + 질문 → 정답 동작. Glimmer 공식 학습 형식(Axolotl messages JSONL).
                         녹화가 처리될 때마다 서버가 다시 만든다. 학습용·평가용은 녹화 단위로 나눈다.
  ② 실력 재기 (이 맥)     llama.cpp 로 띄운 Glimmer(학습 전, 또는 어댑터를 붙인 것)가 평가용 단계를 얼마나 맞히나.
  ③ 학습 (클라우드 GPU)   QLoRA — 공식 문서 기준 H100 1장 (train/glimmer_qlora.yaml, scripts/train_glimmer.sh).
                         맥에서는 학습하지 않는다 (공식 학습 도구가 CUDA 전용).
  ④ 쓰기 (이 맥)          학습한 어댑터(GGUF)를 llama.cpp 에 붙여 띄우고 ②로 다시 잰다.
                         평가 정확도가 READY_ACCURACY 이상이고 학습 단계가 READY_EXAMPLES 개 이상이면
                         새 녹화의 단계 라벨을 Spark 대신 Glimmer 가 붙인다 (API 비용 0).
기록: data/evolve/registry.json (데이터셋·평가 버전), data/evolve/sft/ (학습 데이터), data/evolve/adapters/ (어댑터)
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
WINDOW_S = 1.0            # ④ 새 녹화를 이만큼씩 잘라 묻는다
AUGMENT = 6               # 학습용 단계 하나에서 장면 3장을 뽑는 구간을 조금씩 바꿔 이만큼 만든다 (데이터가 적을 때 늘리기)
HARD_WEIGHT = 3           # 이의 신청에서 바로잡은 단계(모델이 틀리기 쉬운 '어려운 문제')는 이만큼 더 넣는다

QUESTION = """Frames from one step of a robot-learning video (start, middle, end), mission: "{mission}".
Gray areas are privacy masks; ignore them. Which manipulation primitive is the hand doing in this step?
Answer JSON {{"action": one of: {prims}}}."""

DEFINITIONS = """Definitions: reach = hand moves toward the object, not touching yet; grasp = fingers close on the object;
lift = raise the held object; pull = draw it toward you or out of something (e.g. a tissue out of its box); push = move it away;
twist = rotate it (e.g. a lid); press = push a button or surface down; squeeze; tear; wipe; insert = fit it into place;
transport = carry the held object through the air; place = set the held object down on a surface; handover = pass it to the
other hand or a person; release = fingers open and let go; other = none of these."""
STYLES = {"basic": "", "defs": DEFINITIONS}
STYLE = "basic"             # 학습·평가·쓰기에 같은 질문 (scripts/evolve.py styles 로 고른다)

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


def correction_hints(limit: int = 6) -> str:
    """Spark 프롬프트에 붙일 '지난 바로잡음' (가중치를 바꿀 수 없는 API 모델은 예시로 배운다). 없으면 빈 문자열."""
    d = ROOT / "feedback"
    if not d.exists():
        return ""
    lines = []
    for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        r = json.loads(p.read_text())
        for f in r["fixes"]:
            lines.append(f'- in a "{r.get("mission") or "manipulation"}" video, a step labeled "{f["was"]}" was actually "{f["action"]}"'
                         + (f' ({f["note"]})' if f.get("note") else ""))
    return ("Past corrections confirmed on appeal (avoid repeating them):\n" + "\n".join(lines[:limit])) if lines else ""


# ---------- ① 데이터 쌓기 ----------

def _frames(video, t0: float, t1: float, n: int = 3) -> List[np.ndarray]:
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
    steps = [{"start_s": s["start_s"], "end_s": s["end_s"], "action": s["action"], "source": "spark+sensors"}
             for s in c.get("steps") or [] if s.get("verified") and s.get("action") in PRIMITIVES]
    cross = ROOT / "crosscheck" / f"{out.name}.json"
    if cross.exists():                           # 교차 검증(scripts/crosscheck.py): 더 큰 Spark Open 도 같은 동작이라고 본 단계만
        ok = {(round(r["start_s"], 3), r["spark13"]) for r in json.loads(cross.read_text())["rows"] if r["agree"]}
        steps = [{**s, "source": "spark+sensors+spark_open"} for s in steps if (round(s["start_s"], 3), s["action"]) in ok]
    for f in corrections(out.name):              # 이의 신청에서 바로잡은 단계가 우선 (더 높은 모델로 다시 본 정답)
        steps = [s for s in steps if not (s["start_s"] < f["end_s"] and f["start_s"] < s["end_s"])]
        if f.get("action") in PRIMITIVES:
            steps.append({"start_s": f["start_s"], "end_s": f["end_s"], "action": f["action"], "source": "appeal"})
    steps = sorted((s for s in steps if s["end_s"] - s["start_s"] >= 0.2), key=lambda s: s["start_s"])
    return {"id": out.name, "mission": mission, "video": video, "steps": steps} if steps else None


def build_dataset(out_dirs, log=print) -> Optional[dict]:
    """처리된 녹화들 → data/evolve/sft/{train,eval}.jsonl + images/. 녹화가 같으면 다시 만들지 않는다."""
    recs = [r for r in (labeled_steps(d) for d in sorted(out_dirs)) if r]
    if not recs:
        return None
    reg = _registry()
    key = sorted(f"{r['id']}:{len(r['steps'])}:{len(corrections(r['id']))}" for r in recs)
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
            n = 1 if split == "eval" else AUGMENT * (HARD_WEIGHT if s["source"] == "appeal" else 1)
            for v in range(n):
                j0 = 0.0 if v == 0 else ((v * 37) % 11 - 5) / 50 * d          # -10% ~ +10% 이동
                j1 = 0.0 if v == 0 else ((v * 53) % 11 - 5) / 50 * d
                paths = []
                for j, f in enumerate(_frames(r["video"], max(0.0, s["start_s"] + j0), s["end_s"] + j1)):
                    p = img / f"{r['id']}_{k}_{v}_{j}.jpg"
                    cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    paths.append(f"images/{p.name}")
                if len(paths) < 3:
                    continue
                rows[split].append({"messages": [
                    {"role": "user", "content": [{"type": "image", "path": p} for p in paths]
                     + [{"type": "text", "text": question(r["mission"])}]},
                    {"role": "assistant", "content": json.dumps({"action": s["action"]})}],
                    "meta": {"recording": r["id"], "step": k, "variant": v, "source": s["source"]}})
    for split, rs in rows.items():
        (sft / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rs))
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

def question(mission: str, style: Optional[str] = None) -> str:
    extra = STYLES.get(style or STYLE, "")
    return QUESTION.format(mission=mission, prims=", ".join(PRIMITIVES)) + ("\n" + extra if extra else "")


def _ask(client, images: List[np.ndarray], mission: str, style: Optional[str] = None, reasoning: str = "low") -> Optional[str]:
    parts = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
             base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]).decode()}} for f in images]
    parts.append({"type": "text", "text": question(mission, style)})
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
            p.append(_ask(client, imgs, mission, style, reasoning) or "none")
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


# ---------- ④ 쓰기 ----------

def ready_adapter() -> Optional[str]:
    """Spark 대신 써도 되는 어댑터 (평가에서 준비됨이 나온 가장 최근 것). 없으면 None."""
    good = [e for e in _registry()["evals"] if e.get("ready") and e.get("adapter") and Path(e["adapter"]).exists()]
    return good[-1]["adapter"] if good else None


def label_video(video, mission: str, t0: float = 0.0, client=None) -> List[dict]:
    """학습한 Glimmer 로 새 녹화에 단계 라벨: WINDOW_S 마다 장면 3장을 묻고 같은 동작이 이어지면 한 단계로."""
    cap = cv2.VideoCapture(str(video))
    dur = cap.get(7) / (cap.get(5) or 30)
    cap.release()
    steps: List[dict] = []
    t = t0
    while t < dur - 0.2:
        a = _ask(client, _frames(video, t, min(dur, t + WINDOW_S)), mission) or "other"
        if steps and steps[-1]["action"] == a:
            steps[-1]["end_s"] = min(dur, t + WINDOW_S)
        else:
            steps.append({"action": a, "start_s": t, "end_s": min(dur, t + WINDOW_S)})
        t += WINDOW_S
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
    return {"datasets": r.get("datasets", []), "evals": r.get("evals", []), "trainings": r.get("trainings", [])}


def _save(reg: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "registry.json").write_text(json.dumps(reg, indent=1, ensure_ascii=False))
