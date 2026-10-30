"""공개 정답으로 단계 라벨 채점 — EPIC-KITCHENS-100 (사람이 붙인 동작 구간) 을 우리 라벨 흐름에 넣는다.

  python scripts/eval_epic.py pick  [--n 8]        # 동작이 촘촘한 60초 구간 고르기 (참여자마다 하나) → data/eval/epic/windows.json
  python scripts/eval_epic.py fetch                # 고른 구간만 부분 읽기로 받아 640px 로 줄인다 (원본 영상 통째로 받지 않는다)
  python scripts/eval_epic.py run                  # Spark 1.3 → Spark Open 2차 검증 (처리와 같은 흐름) → 결과 저장 (비용 있음)
  python scripts/eval_epic.py score                # 저장된 결과로 채점만 (비용 없음) → data/eval/epic/report.json

영상: 공식 서버(data.bris.ac.uk)는 연결당 초당 25KB 라 1분 구간에 1시간 → Hugging Face 미러(awsaf49/epic_kitchens_100, 확장 영상
P01_101 등만 있음)에서 받는다. 정답은 공식 GitHub 의 EPIC_100_train.csv · EPIC_100_validation.csv (학습에 쓰지 않으니 train 정답도 채점에만).
라이선스: EPIC-KITCHENS 는 CC BY-NC 4.0 — 내부 확인용으로만 쓴다. 영상·결과는 data/ 밖으로 내보내지 않고, 학습·판매·정답 예시에 쓰지 않는다
(2차 검증 기록은 data/eval/epic/evolve 에 따로 — data/evolve 의 정답 예시·바로잡음에 섞이지 않게).
EPIC 동사는 우리 기본 동작보다 거칠다 ('take' = 뻗기+쥐기+들기) → 양쪽을 같은 거친 묶음으로 바꿔 채점한다 (MERGE_*).
센서가 없으니 감각 검증(verified)은 못 잰다 — 대신 'Spark Open 도 같게 봤다'가 맞는 비율을 verified_precision 자리에 잰다.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacash import evolve  # noqa: E402
from datacash.media import run as ffmpeg_run  # noqa: E402
from datacash.understand.score import pooled, score  # noqa: E402

D = Path("data/eval/epic")
URL = "https://huggingface.co/datasets/awsaf49/epic_kitchens_100/resolve/main/{p}/videos/{v}.MP4"
WIN_S = 60.0
MISSION = "Everyday kitchen manipulation (first-person video)"

# EPIC 동사 묶음(verb_class key) → 거친 묶음
MERGE_EPIC = {"take": "take", "remove": "take", "put": "put", "insert": "put", "throw": "put",
              "open": "articulate", "close": "articulate", "turn-on": "articulate", "turn-off": "articulate",
              "press": "articulate", "turn": "articulate", "adjust": "articulate",
              "wash": "wash", "dry": "wash", "pour": "pour", "fill": "pour", "empty": "pour", "move": "move"}
# 우리 기본 동작 → 거친 묶음
MERGE_OURS = {"reach": "take", "grasp": "take", "lift": "take", "place": "put", "release": "put", "insert": "put",
              "pull": "articulate", "push": "articulate", "twist": "articulate", "press": "articulate",
              "wipe": "wash", "pour": "pour", "transport": "move", "handover": "move"}
COARSE = set(MERGE_EPIC.values())
# 너그러운 대응: 주걱으로 미는 것('push')을 EPIC 은 'move', 꺼내는 것('pull')을 'take' 로 부르는 일이 많다 — 대응표 탓 오답이 얼마인지 보려고
LENIENT = {**MERGE_OURS, "push": "move", "pull": "take"}


def _ts(x: str) -> float:
    h, m, s = x.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _gold_table() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(D / f"EPIC_100_{x}.csv") for x in ("train", "validation")])
    keys = pd.read_csv(D / "EPIC_100_verb_classes.csv").set_index("id")["key"]
    d["a"], d["b"] = d.start_timestamp.map(_ts), d.stop_timestamp.map(_ts)
    d["coarse"] = d.verb_class.map(keys).map(lambda k: MERGE_EPIC.get(k, "other"))
    have = {Path(f).stem for f in json.loads((D / "hf_videos.json").read_text())}   # 미러에 있는 영상만
    return d[d.video_id.isin(have)]


def pick(n: int) -> None:
    """참여자마다 '거친 묶음에 드는 동작이 가장 촘촘하고 다른 동작이 적은' 60초 구간 하나."""
    d = _gold_table()
    best = []
    for vid, g in d.groupby("video_id"):
        for t0 in range(0, int(g.b.max() - WIN_S), 10):
            w = g[(g.a >= t0) & (g.b <= t0 + WIN_S)]
            good = int((w.coarse != "other").sum())
            best.append({"video_id": vid, "participant": vid.split("_")[0], "t0": float(t0),
                         "score": good - 2 * int((w.coarse == "other").sum()), "segments": int(len(w)), "mapped": good})
    best.sort(key=lambda x: -x["score"])
    seen, out = set(), []
    if SET != "dev":                         # 확인용은 방법을 고를 때 쓴 참여자(부엌)를 뺀다
        seen = {w["participant"] for w in json.loads((D / "windows.json").read_text())}
    for b in best:
        if b["participant"] not in seen:
            seen.add(b["participant"])
            out.append(b)
        if len(out) >= n:
            break
    D.mkdir(parents=True, exist_ok=True)
    _windows_path().write_text(json.dumps(out, indent=1))
    for w in out:
        print(f"{w['video_id']} {w['t0']:.0f}~{w['t0'] + WIN_S:.0f}초  동작 {w['segments']} (묶음에 드는 {w['mapped']})")


def _clip(w: dict) -> Path:
    return D / "clips" / f"{w['video_id']}_{int(w['t0'])}.mp4"


def fetch() -> None:
    for w in json.loads(_windows_path().read_text()):
        dst = _clip(w)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = URL.format(p=w["participant"], v=w["video_id"])
        ffmpeg_run(["-ss", w["t0"], "-t", WIN_S, "-i", url, "-vf", "scale=640:-2", "-r", 30, "-an",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", 23, dst])
        print(f"{dst} {dst.stat().st_size / 1e6:.1f}MB")


def gold_steps(w: dict) -> list:
    d = _gold_table()
    g = d[(d.video_id == w["video_id"]) & (d.a >= w["t0"]) & (d.b <= w["t0"] + WIN_S)]
    return [{"start_s": float(r.a - w["t0"]), "end_s": float(r.b - w["t0"]), "action": r.coarse, "narration": r.narration}
            for r in g.itertuples()]


def run(redo: bool) -> None:
    from datacash.understand.spark import understand_clip

    evolve.ROOT = D / "evolve"                 # 정답 예시·바로잡음·일치율 기록이 실제 서비스 기록에 섞이지 않게
    for w in json.loads((D / "windows.json").read_text()):
        clip, dst = _clip(w), D / "runs" / f"{_clip(w).stem}.json"
        if dst.exists() and not redo:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        u = understand_clip(clip, MISSION, WIN_S)
        first = copy.deepcopy(u["steps"])
        so = evolve.second_opinion(clip.stem, clip, 0.0, MISSION, "", u["steps"], log=print)
        dst.write_text(json.dumps({"window": w, "spark13": first, "final": u["steps"], "second_opinion": so}, indent=1, ensure_ascii=False))
        print(f"{clip.stem}: Spark 1.3 {len(first)}단계 · 2차 검증 {so}")


def _ours(steps: list, agreed_as_verified: bool = False, merge: dict = MERGE_OURS) -> list:
    out = []
    for s in steps:
        x = {"start_s": s["start_s"], "end_s": s["end_s"], "action": merge.get(s["action"], "other")}
        if agreed_as_verified:
            x["verified"] = bool((s.get("spark_open") or {}).get("agree"))
        out.append(x)
    return out


def confusion(gold: list, pred: list) -> dict:
    """0.1초 장면마다 (정답 묶음, 모델 묶음) 개수 — 어디서 틀리나."""
    import numpy as np

    from datacash.understand.score import STEP_S, _at, _norm

    g, p = _norm(gold, None), _norm(pred, None)
    t = np.arange(g[0]["start_s"], g[-1]["end_s"], STEP_S)
    c = {}
    for x, y in zip(_at(g, t), _at(p, t)):
        if x != "none":
            c[f"{x}->{y}"] = c.get(f"{x}->{y}", 0) + 1
    return c


def do_score() -> None:
    rep = {"spark13": [], "final": [], "lenient": []}
    rows, conf = [], {}
    for p in sorted((D / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        g = gold_steps(r["window"])
        a = score(g, _ours(r["spark13"]), gold_only=True)
        b = score(g, _ours(r["final"], agreed_as_verified=True), gold_only=True)
        c = score(g, _ours(r["final"], merge=LENIENT), gold_only=True)
        rep["spark13"].append(a)
        rep["final"].append(b)
        rep["lenient"].append(c)
        for k, v in confusion(g, _ours(r["final"])).items():
            conf[k] = conf.get(k, 0) + v
        rows.append({"clip": p.stem, "spark13": a, "final": b})
        print(f"{p.stem}: 정답 {a['gold_steps']} · 장면 정확도 {a['frame_acc']:.2f} → {b['frame_acc']:.2f} · F1@50 {a['f1@50']:.2f} → {b['f1@50']:.2f}"
              f" · 경계 {a['boundary_ms_median']}ms · 두 모델 같게 본 단계가 맞는 비율 {b['verified_precision']}")
    out = {"spark13": pooled(rep["spark13"]), "final": pooled(rep["final"]), "final_lenient": pooled(rep["lenient"]),
           "confusion": dict(sorted(conf.items(), key=lambda kv: -kv[1])), "clips": rows,
           "note": "EPIC-KITCHENS-100 (CC BY-NC 4.0, internal check only); coarse classes " + ", ".join(sorted(COARSE))}
    (D / "report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("Spark 1.3 혼자:", out["spark13"])
    print("2차 검증 뒤:  ", out["final"])
    print("너그러운 대응:", out["final_lenient"])
    tot = sum(conf.values())
    print("자주 틀린 곳 (장면 비율):", ", ".join(f"{k} {v / tot:.0%}" for k, v in list(out["confusion"].items())[:12]))


# ---------- 실험: Spark 1.3 질문·자르기를 바꿔 같은 구간으로 채점 (2차 검증은 효과가 작아 뺀다) ----------

TIMING = ("Timing matters as much as the action name. Put each boundary at the real moment the motion changes "
          "(first touch, object leaves the surface, fingers open). Do not split the clip into even intervals. "
          "Short steps (often under 1 second) are normal; when the person quickly repeats take and put, label each one separately.")
EXPERIMENTS = {"base": {"chunk": None, "extra": ""},              # 처음과 같은 질문 — 다시 돌려 우연한 흔들림을 잰다
               "timing": {"chunk": None, "extra": TIMING},        # 경계·길이 지시
               "chunk20": {"chunk": 20.0, "extra": ""},           # 20초씩 잘라 보낸다
               "chunk20_timing": {"chunk": 20.0, "extra": TIMING},
               "chunk10": {"chunk": 10.0, "extra": "", "pad": 1.0},       # 10초씩, 앞뒤 1초 겹쳐 보내고 가운데 10초 것만 쓴다
               "chunk20_slow": {"chunk": 20.0, "extra": "", "slow": 2.0},  # 20초를 0.5배속(40초)으로 늘려 보내고 시각을 2로 나눈다
               "chunk20_r2": {"chunk": 20.0, "extra": ""},        # chunk20 을 두 번 더 → 세 번 다수결 (vote3)
               "chunk20_r3": {"chunk": 20.0, "extra": ""}}
# 동작 이름 실험 (10-30): 지금 처리(chunk20_slow) 위에 — 어휘 넓히기 · 물체 상태로 집기/놓기 가르기
STATE = ("Name each step by how the object's state changes: object goes from a surface (or container) into the hand = "
         "grasp/lift; object goes from the hand onto a surface (or into a container) = place/release; object stays in the hand "
         "and only moves = transport; hand works a door, drawer, lid, tap or knob = the articulation primitive. "
         "Look at the first and last moments of each step before naming it.")
VOCAB = {"open": "open a door, drawer, lid, bag or container", "close": "close a door, drawer, lid or container",
         "wash": "wash, rinse or scrub under water", "stir": "stir or mix with a tool", "cut": "cut, slice, chop or peel"}
EXPERIMENTS.update({"slow_vocab": {"chunk": 20.0, "extra": "", "slow": 2.0, "vocab": True},
                    "slow_state": {"chunk": 20.0, "extra": STATE, "slow": 2.0},
                    "slow_vocab_state": {"chunk": 20.0, "extra": STATE, "slow": 2.0, "vocab": True}})
MERGE_OURS.update({"open": "articulate", "close": "articulate", "wash": "wash"})      # stir · cut 은 'other'
LENIENT.update({"open": "articulate", "close": "articulate", "wash": "wash"})


def _use_vocab() -> None:
    """Spark 1.3 의 동작 목록(스키마·질문)에 VOCAB 을 더한다 — 프로세스 전체에 걸리므로 vocab 변형은 따로 돌린다."""
    from datacash.understand import spark

    new = {k: v for k, v in VOCAB.items() if k not in spark.PRIMITIVES}   # open·close 는 10-30 부터 처리에 들어 있다
    for k in new:
        spark.PRIMITIVES.insert(spark.PRIMITIVES.index("other"), k)
    spark.SCHEMA["properties"]["steps"]["items"]["properties"]["action"]["enum"] = spark.PRIMITIVES
    add = ", ".join(f"{k} ({v})" for k, v in new.items())
    if new and add not in spark.PROMPT:
        spark.PROMPT = spark.PROMPT.replace("container), other.", f"container), {add}, other.")


VOTES = {"vote3": ["chunk20", "chunk20_r2", "chunk20_r3"]}          # 0.1초마다 다수결 (같은 표면 첫 번째 것)
SET = "dev"                                                         # dev: 방법 고르는 8구간 · test: 고른 방법을 확인하는 새 8구간


def _windows_path() -> Path:
    return D / ("windows.json" if SET == "dev" else f"windows_{SET}.json")


def _exp_dir() -> Path:
    return D / ("exp" if SET == "dev" else f"exp_{SET}")


def _sub(clip: Path, t0: float, dur: float, slow: float) -> Path:
    dst = D / "tmp" / f"{clip.stem}_{t0:.0f}_{slow:g}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = "scale=640:-2" + (f",setpts={slow}*PTS" if slow != 1 else "")
    ffmpeg_run(["-ss", t0, "-t", dur, "-i", clip, "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", 23,
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst])
    return dst


def _label(clip: Path, chunk, extra: str, pad: float = 0.0, slow: float = 1.0) -> list:
    from datacash.understand.spark import understand_clip

    if not chunk:
        return understand_clip(clip, MISSION, WIN_S, extra=extra)["steps"]
    steps, t0 = [], 0.0
    while t0 < WIN_S - 1:
        a, b = max(0.0, t0 - pad), min(WIN_S, t0 + chunk + pad)       # 보내는 구간 (겹침 포함)
        sub = _sub(clip, a, b - a, slow)
        for s in understand_clip(sub, MISSION, (b - a) * slow, extra=extra)["steps"]:
            x = {**s, "start_s": a + s["start_s"] / slow, "end_s": a + s["end_s"] / slow}
            mid = (x["start_s"] + x["end_s"]) / 2
            if t0 <= mid < t0 + chunk:                                 # 가운데가 이 조각에 든 단계만 (겹친 곳 두 번 세지 않게)
                steps.append(x)
        sub.unlink(missing_ok=True)
        t0 += chunk
    return sorted(steps, key=lambda s: s["start_s"])


def vote(runs: list, step_s: float = 0.1) -> list:
    """같은 구간을 여러 번 라벨한 것 → 0.1초마다 가장 많이 나온 동작 (같은 표면 첫 번째 것) → 이어진 단계로."""
    import numpy as np

    from datacash.understand.score import _at

    t = np.arange(0.0, WIN_S, step_s)
    cols = [_at(sorted(r, key=lambda s: s["start_s"]), t) for r in runs]
    lab = []
    for k in range(len(t)):
        vs = [c[k] for c in cols]
        lab.append(max(dict.fromkeys(vs), key=vs.count))
    out = []
    for k, a in enumerate(lab):
        if a == "none":
            continue
        if out and out[-1]["action"] == a and abs(out[-1]["end_s"] - t[k]) < step_s / 2:
            out[-1]["end_s"] = float(t[k] + step_s)
        else:
            out.append({"start_s": float(t[k]), "end_s": float(t[k] + step_s), "action": a})
    return out


def exp(names: list) -> None:
    from concurrent.futures import ThreadPoolExecutor

    evolve.ROOT = D / "evolve_empty"            # 지난 바로잡음 힌트 없이 — 변형끼리 같은 조건
    ws = json.loads(_windows_path().read_text())
    todo = [n for n in names if n in EXPERIMENTS and not all((_exp_dir() / n / f"{_clip(w).stem}.json").exists() for w in ws)]
    vocab = {bool(EXPERIMENTS[n].get("vocab")) for n in todo}
    if len(vocab) > 1:
        raise SystemExit("어휘를 넓힌 변형(vocab)과 아닌 변형은 따로 돌린다 (동작 목록이 프로세스 전체에 걸린다)")
    if vocab == {True}:
        _use_vocab()
    runs = [n for n in names if n in EXPERIMENTS] + [r for n in names for r in VOTES.get(n, [])]
    jobs = [(n, w) for n in dict.fromkeys(runs) for w in ws if not (_exp_dir() / n / f"{_clip(w).stem}.json").exists()]

    def one(job):
        n, w = job
        dst = _exp_dir() / n / f"{_clip(w).stem}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        e = EXPERIMENTS[n]
        try:
            steps = _label(_clip(w), e["chunk"], e["extra"], e.get("pad", 0.0), e.get("slow", 1.0))
        except Exception as ex:
            print(f"{n} {dst.stem} 실패: {type(ex).__name__} {ex}")
            return
        dst.write_text(json.dumps({"window": w, "steps": steps}, ensure_ascii=False))
        print(f"{n} {dst.stem}: {len(steps)}단계")

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, jobs))
    exp_score(names)


def _exp_steps(n: str, stem: str):
    if n in VOTES:
        rs = [_exp_dir() / r / f"{stem}.json" for r in VOTES[n]]
        if not all(p.exists() for p in rs):
            return None
        return vote([json.loads(p.read_text())["steps"] for p in rs])
    p = _exp_dir() / n / f"{stem}.json"
    return json.loads(p.read_text())["steps"] if p.exists() else None


def exp_score(names: list) -> None:
    ws = json.loads(_windows_path().read_text())
    res, per = {}, {}
    if SET == "dev":
        res["first_run"] = pooled([score(gold_steps(json.loads(p.read_text())["window"]), _ours(json.loads(p.read_text())["spark13"]),
                                         gold_only=True) for p in sorted((D / "runs").glob("*.json"))])
    for n in names:
        rs, lens = [], []
        for w in ws:
            st = _exp_steps(n, _clip(w).stem)
            if st is None:
                continue
            r = score(gold_steps(w), _ours(st), gold_only=True)
            rs.append(r)
            per.setdefault(_clip(w).stem, {})[n] = r["frame_acc"]
            lens += [s["end_s"] - s["start_s"] for s in st]
        res[n] = {**pooled(rs), "mean_step_s": round(sum(lens) / len(lens), 2) if lens else None}
    (D / f"exp_report{'' if SET == 'dev' else '_' + SET}.json").write_text(json.dumps({"pooled": res, "per_clip": per}, indent=1, ensure_ascii=False))
    print(f"{'변형':16} 구간 장면정확도 F1@10 F1@50 순서 경계ms 단계길이")
    for n, r in res.items():
        print(f"{n:16} {r.get('recordings')}  {r.get('frame_acc')}  {r.get('f1@10')}  {r.get('f1@50')}  {r.get('edit')}  "
              f"{r.get('boundary_ms_median')}  {r.get('mean_step_s', '-')}")
    print("구간별 장면 정확도:")
    for k, v in per.items():
        print(f"  {k:14} " + " ".join(f"{n}={a:.2f}" for n, a in v.items()))


# ---------- 경계 다듬기 (datacash/understand/refine.py): 손 멈춤으로 맞추기(snap) · 경계 장면 고르기(pick) ----------

def hands(_=None) -> None:
    """구간마다 손 관절 (MediaPipe, 이 맥) → data/eval/epic/hands/<구간>.npy  (장면, 손 2, 21, 3) — 없는 손은 nan."""
    import numpy as np

    from datacash.labels.pose import extract_labels

    for w in json.loads(_windows_path().read_text()):
        dst = D / "hands" / f"{_clip(w).stem}.npy"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        fr = extract_labels(_clip(w))
        a = np.full((len(fr), 2, 21, 3), np.nan, np.float32)
        for i, f in enumerate(fr):
            for k in (0, 1):
                if f.hands[k] is not None:
                    a[i, k] = f.hands[k]
        np.save(dst, a)
        print(f"{dst.stem}: 장면 {len(fr)} · 손 보인 장면 {float(np.mean(np.isfinite(a[:, :, 0, 0]).any(axis=1))):.2f}")


def _events(stem: str) -> list:
    import cv2
    import numpy as np

    from datacash.understand.refine import hand_events

    a = np.load(D / "hands" / f"{stem}.npy")
    cap = cv2.VideoCapture(str(D / "clips" / f"{stem}.mp4"))
    fps, width = cap.get(5) or 30.0, cap.get(3) or 640
    cap.release()
    hs = [[None if not np.isfinite(a[i, k, 0, 0]) else a[i, k] for k in (0, 1)] for i in range(len(a))]
    return hand_events(hs, fps, width)


def refine(base: str, tol: float) -> None:
    """base 변형의 단계에 snap · pick · pick 뒤 snap 을 붙여 새 변형으로 저장하고 채점한다."""
    from concurrent.futures import ThreadPoolExecutor

    from datacash.understand.refine import pick, snap

    ws = json.loads(_windows_path().read_text())

    def one(w):
        stem = _clip(w).stem
        st = _exp_steps(base, stem)
        if st is None:
            return
        ev = _events(stem)
        for name, fn in ((f"{base}+snap", lambda: snap(st, ev, tol)),
                         (f"{base}+pick", lambda: pick(_clip(w), st, MISSION)),
                         (f"{base}+pick+snap", None)):
            dst = _exp_dir() / name / f"{stem}.json"
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if fn is None:
                picked = json.loads((_exp_dir() / f"{base}+pick" / f"{stem}.json").read_text())["steps"]
                out = snap(picked, ev, tol)
            else:
                out = fn()
            dst.write_text(json.dumps({"window": w, "steps": out}, ensure_ascii=False))
        print(f"{stem}: 손 멈춤 사건 {len(ev)}개")

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, ws))
    exp_score([base, f"{base}+snap", f"{base}+pick", f"{base}+pick+snap"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "fetch", "run", "score", "exp", "exp_score", "hands", "refine"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--only", default=",".join(EXPERIMENTS), help="exp: 돌릴 변형 (쉼표, vote3 도 된다)")
    ap.add_argument("--base", default="chunk20", help="refine: 다듬을 변형")
    ap.add_argument("--tol", type=float, default=0.4, help="refine: 손 멈춤으로 옮길 수 있는 거리 (초)")
    ap.add_argument("--set", default="dev", help="dev (방법 고르기) · test (새 구간으로 확인 — pick 은 dev 구간의 참여자를 뺀다)")
    a = ap.parse_args()
    global SET
    SET = a.set
    {"pick": lambda: pick(a.n), "fetch": fetch, "run": lambda: run(a.redo), "score": do_score,
     "exp": lambda: exp(a.only.split(",")), "exp_score": lambda: exp_score(a.only.split(",")),
     "hands": hands, "refine": lambda: refine(a.base, a.tol)}[a.cmd]()


if __name__ == "__main__":
    main()
