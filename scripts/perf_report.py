"""성능 그래프 — 저장된 결과로만 다시 채점해 그린다 (API 호출 없음, 재현: python scripts/perf_report.py).

① 라벨 정확도 (EPIC-KITCHENS 새 8구간, 사람 정답): 방법을 하나씩 더할 때 장면 정확도·경계 오차
② 혼동 행렬: 처음(60초 한 번) ↔ 지금(다섯 벌 합치기) — 어디서 틀리나
③ Glimmer 진화: 같은 평가 122단계(사람 정답)에서 학습 전 → v79 → v80
④ 서비스 모니터링: 녹화마다 Spark 1.3 ↔ Spark Open 일치율, 처리 시간, 얼굴 '같은 사람' 장면 비율 (datacash/perf.py 와 같은 계산)
결과: docs/perf/*.{en,ko}.png, docs/perf/benchmarks.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_epic as E  # noqa: E402
from datacash import perf  # noqa: E402
from datacash.understand.fuse import fuse  # noqa: E402
from datacash.understand.score import pooled, score  # noqa: E402

OUT = ROOT / "docs" / "perf"
INK, MUTED, LINE, ACC, WARN, OK = "#1C1C1E", "#8E8E93", "#E5E5EA", "#FF5F5F", "#FFAF5F", "#2E9BF0"
plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "Helvetica Neue", "DejaVu Sans"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 150})

# 지금 처리와 같은 다섯 벌 (기본 질문 0·7·13초 + 상태 질문 0·10초, switch 1) — docs/label_eval.md 실험 6
ENSEMBLE = ["chunk20_slow", "slow_o7", "slow_o13", "slow_state", "state_o10"]
STAGES = [("base", "60 s in one call", "60초 한 번에"),
          ("chunk20_slow", "+ 20 s chunks at 0.5× speed", "+ 20초 조각 · 0.5배속"),
          ("chunk20_slow+pick", "+ boundary frame pick (Spark Open)", "+ 경계 장면 고르기 (Spark Open)"),
          ("ensemble", "+ 5-run fusion (Viterbi)", "+ 다섯 벌 합치기 (Viterbi)")]
T = {"en": {"acc": "Frame accuracy (%)", "bnd": "Boundary error, median (ms)",
            "t1": "Step labels vs human ground truth — EPIC-KITCHENS held-out 8 clips",
            "conf": "Where labels go wrong (0.1 s frames, row = human truth)", "first": "First: 60 s in one call",
            "now": "Now: 5-run fusion", "pred": "Muse Spark label", "truth": "Human truth",
            "g": "Muse Glimmer 30B fine-tuning — same 122 human-labelled steps",
            "gacc": "Step accuracy (%)", "gl": ["Majority\nclass", "Glimmer\nbefore", "v79\n(Spark labels)", "v80\n(human labels)", "v82 + order\ndecoding"],
            "agree": "Spark 1.3 ↔ Spark Open agreement per clip", "clip": "Clip (processing order)", "last5": "last-5 mean",
            "time": "Processing time per clip (s)", "reid": "Frames still 'same person' after face swap (%)",
            "mon": "Live service monitoring (operator page shows the same)", "warn": "alert"},
     "ko": {"acc": "장면 정확도 (%)", "bnd": "경계 오차 중앙값 (ms)",
            "t1": "단계 라벨 vs 사람 정답 — EPIC-KITCHENS 새 8구간",
            "conf": "어디서 틀리나 (0.1초 장면, 행 = 사람 정답)", "first": "처음: 60초 한 번에",
            "now": "지금: 다섯 벌 합치기", "pred": "Muse Spark 라벨", "truth": "사람 정답",
            "g": "Muse Glimmer 30B 학습 — 같은 평가 122단계 (사람 정답)",
            "gacc": "단계 정확도 (%)", "gl": ["가장 많은\n동작만", "학습 전", "v79\n(Spark 라벨)", "v80\n(사람 정답)", "v82 + 순서\n디코딩"],
            "agree": "녹화마다 Spark 1.3 ↔ Spark Open 일치율", "clip": "녹화 (처리 순서)", "last5": "최근 5편 평균",
            "time": "녹화 하나 처리 시간 (초)", "reid": "치환 뒤에도 '같은 사람'인 얼굴 장면 (%)",
            "mon": "서비스 모니터링 (운영자 화면과 같은 값)", "warn": "경고"}}
CLASSES = ["take", "put", "articulate", "move", "wash", "pour", "other"]


def epic_test() -> dict:
    """새 8구간을 다시 채점: 단계별 장면 정확도·경계, 처음·지금 혼동 행렬."""
    E.SET = "test"
    ws = json.loads(E._windows_path().read_text())
    res, conf = {}, {"base": {}, "ensemble": {}}
    for key, *_ in STAGES:
        rs = []
        for w in ws:
            stem = E._clip(w).stem
            if key == "ensemble":
                runs = [E._exp_steps(n, stem) for n in ENSEMBLE]
                if any(r is None for r in runs):
                    continue
                st = fuse(runs, 0.0, E.WIN_S, switch=1.0)
            else:
                st = E._exp_steps(key, stem)
                if st is None:
                    continue
            g = E.gold_steps(w)
            rs.append(score(g, E._ours(st), gold_only=True))
            if key in conf:
                for k, v in E.confusion(g, E._ours(st)).items():
                    conf[key][k] = conf[key].get(k, 0) + v
        res[key] = pooled(rs)
    return {"stages": res, "confusion": conf}


# 다른 감각·개인정보는 문서에 남긴 측정값을 그대로 (출처 함께) — 운영자 화면 '성능' 카드
CITED = [
    {"key": "sync", "en": "Watch ↔ video sync error", "ko": "워치 ↔ 영상 시간 오차", "before": 24.7, "after": 3.7, "unit": "ms",
     "better": "lower", "limit": 10, "source": "docs/sync_eval.md"},
    {"key": "faceswap", "en": "Face swap time per clip", "ko": "녹화 하나 얼굴 치환 시간", "before": 636, "after": 84, "unit": "s",
     "better": "lower", "source": "docs/privacy_eval.md"},
    {"key": "sam_cost", "en": "SAM 3.1 cost, 4 clips", "ko": "SAM 3.1 비용 (녹화 4편)", "before": 2.92, "after": 0.27, "unit": "$",
     "better": "lower", "source": "README (Privacy)"},
    {"key": "mask_decode", "en": "SAM mask decode per clip", "ko": "SAM 마스크 해독 (녹화 하나)", "before": 593, "after": 8, "unit": "s",
     "better": "lower", "source": "README (Numba)"},
    {"key": "grip3", "en": "Grip level low·mid·high", "ko": "쥐는 세기 하·중·상", "before": 33, "after": 83, "unit": "%",
     "better": "higher", "note": "chance 33%, 3 test recordings", "source": "README (Touch)"},
]


def glimmer() -> list:
    """같은 122단계 평가 (docs/label_eval.md '사람 정답으로 다시 학습'). 학습 전은 registry 에서, v79·v80 은 H200 채점 기록."""
    reg = json.loads((ROOT / "data/evolve/registry.json").read_text())
    pre = [e for e in reg["evals"] if e.get("dataset") == 80 and e.get("steps") == 122]
    base, before = (pre[-1]["baseline_accuracy"], pre[-1]["accuracy"]) if pre else (0.197, 0.418)
    # v82e3 + 미션별 순서 디코딩: 순서 무게 다섯 개 평균 64.1% (범위 60.7–67.2%, docs/label_eval.md 'v81–v83')
    return [base, before, 0.328, 0.541, 0.641]


def plot_all(b: dict, live: dict, lang: str) -> None:
    t = T[lang]
    OUT.mkdir(parents=True, exist_ok=True)
    # ① 정확도·경계
    st = b["epic"]["stages"]
    names = [s[1] if lang == "en" else s[2] for s in STAGES]
    acc = [100 * st[k]["frame_acc"] for k, *_ in STAGES]
    bnd = [st[k]["boundary_ms_median"] for k, *_ in STAGES]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    y = np.arange(len(names))[::-1]
    ax[0].barh(y, acc, color=[MUTED, OK, OK, ACC], height=.6)
    for yi, a in zip(y, acc):
        ax[0].text(a + .8, yi, f"{a:.1f}%", va="center", fontweight="bold")
    ax[0].set_yticks(y, names); ax[0].set_xlim(0, 70); ax[0].set_xlabel(t["acc"])
    ax[1].barh(y, bnd, color=[MUTED, OK, ACC, OK], height=.6)
    for yi, v in zip(y, bnd):
        ax[1].text(v + 6, yi, f"{v:.0f} ms", va="center", fontweight="bold")
    ax[1].set_yticks(y, [""] * len(y)); ax[1].set_xlim(0, max(bnd) * 1.25); ax[1].set_xlabel(t["bnd"])
    fig.suptitle(t["t1"], fontweight="bold", x=.02, ha="left")
    fig.tight_layout(); fig.savefig(OUT / f"label_accuracy.{lang}.png"); plt.close(fig)

    # ② 혼동 행렬 두 개
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for a, key, title in [(ax[0], "base", t["first"]), (ax[1], "ensemble", t["now"])]:
        c = b["epic"]["confusion"][key]
        rows = [x for x in CLASSES if any(k.startswith(x + "->") for k in c)]
        cols = rows + [x for x in ["none"] if any(k.endswith("->none") for k in c)]
        M = np.array([[c.get(f"{r}->{q}", 0) for q in cols] for r in rows], float)
        R = M / np.maximum(M.sum(1, keepdims=True), 1)
        a.imshow(R, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(rows)):
            for j in range(len(cols)):
                if M[i, j]:
                    a.text(j, i, f"{100 * R[i, j]:.0f}", ha="center", va="center", fontsize=9,
                           color="white" if R[i, j] > .55 else INK)
        diag = b["epic"]["stages"][key]["frame_acc"]
        a.set_xticks(range(len(cols)), cols, rotation=35, ha="right"); a.set_yticks(range(len(rows)), rows)
        a.set_xlabel(t["pred"]); a.set_ylabel(t["truth"]); a.set_title(f"{title} — {100 * diag:.0f}%")
        a.spines[:].set_visible(False)
    fig.suptitle(t["conf"], fontweight="bold", x=.02, ha="left")
    fig.tight_layout(); fig.savefig(OUT / f"confusion.{lang}.png"); plt.close(fig)

    # ③ Glimmer
    g = b["glimmer"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(range(len(g)), [100 * x for x in g], color=[LINE, MUTED, WARN, OK, ACC][:len(g)], width=.6)
    for i, x in enumerate(g):
        ax.text(i, 100 * x + 1.5, f"{100 * x:.1f}%", ha="center", fontweight="bold")
    ax.set_xticks(range(len(g)), t["gl"][:len(g)]); ax.set_ylim(0, 75); ax.set_ylabel(t["gacc"]); ax.set_title(t["g"], loc="left")
    fig.tight_layout(); fig.savefig(OUT / f"glimmer.{lang}.png"); plt.close(fig)

    # ④ 서비스 모니터링 세 줄
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ag = live["agreement"]
    x = np.arange(1, len(ag) + 1)
    ax[0].plot(x, [100 * a["rate"] for a in ag], "o", color=OK, ms=4, alpha=.7)
    ax[0].plot(x, [100 * a["last5"] for a in ag], color=INK, lw=2, label=t["last5"])
    ax[0].axhline(100 * perf.LIMITS["agreement_last5"], color=ACC, ls="--", lw=1, label=t["warn"])
    ax[0].set_ylim(0, 105); ax[0].set_title(t["agree"], loc="left", fontsize=11); ax[0].set_xlabel(t["clip"]); ax[0].legend(frameon=False, fontsize=9)
    ss = live["sessions"]
    xs = np.arange(1, len(ss) + 1)
    ax[1].bar(xs, [s["total_s"] for s in ss], color=[ACC if "slow" in s["alert"] else MUTED for s in ss])
    ax[1].axhline(live["summary"]["time_median_s"] * perf.LIMITS["time_x_median"], color=ACC, ls="--", lw=1)
    ax[1].set_title(t["time"], loc="left", fontsize=11); ax[1].set_xlabel(t["clip"])
    ax[2].bar(xs, [100 * (s["reid"] or 0) for s in ss], color=[ACC if (s["reid"] or 0) > perf.LIMITS["reid"] else OK for s in ss])
    ax[2].axhline(100 * perf.LIMITS["reid"], color=ACC, ls="--", lw=1)
    ax[2].set_title(t["reid"], loc="left", fontsize=11); ax[2].set_xlabel(t["clip"])
    fig.suptitle(t["mon"], fontweight="bold", x=.02, ha="left")
    fig.tight_layout(); fig.savefig(OUT / f"monitoring.{lang}.png"); plt.close(fig)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "server"
    b = {"epic": epic_test(), "glimmer": glimmer(), "cited": CITED,
         "note": "EPIC-KITCHENS-100 held-out 8 clips (CC BY-NC 4.0, internal check only); Glimmer: 122 human-labelled steps from 8 held-out recordings"}
    live = perf.collect(root)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmarks.json").write_text(json.dumps({**b, "live_summary": live["summary"]}, indent=1, ensure_ascii=False))
    for lang in ("en", "ko"):
        plot_all(b, live, lang)
    for k, *_ in STAGES:
        s = b["epic"]["stages"][k]
        print(f"{k:20} 장면 {s['frame_acc']:.3f}  경계 {s['boundary_ms_median']}ms  F1@25 {s['f1@25']}")
    print("Glimmer", b["glimmer"], "· 모니터링", live["summary"])


if __name__ == "__main__":
    main()
