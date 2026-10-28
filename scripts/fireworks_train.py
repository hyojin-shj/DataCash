"""③ Glimmer 학습을 Fireworks 에서 — 데이터 올리기 → LoRA 지도 학습(SFT) 시작 → 끝날 때까지 지켜보기 → 기록.

  python scripts/fireworks_train.py [--version v2] [--epochs 3] [--max-cost 5]
Fireworks 키는 .env 의 FIREWORKS_API_KEY. 학습만 한다 (Muse Glimmer 30B SFT 는 11/30 까지 무료).
학습한 모델을 띄워 평가하는 것(전용 GPU, 유료)은 따로 한다.
안전장치: Fireworks 가 알려 주는 예상 비용이 --max-cost 달러를 넘으면 바로 취소한다.
기록: data/evolve/registry.json 의 trainings (운영자 화면 '모델 진화'), 진행 로그는 data/evolve/fireworks.log
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacash import config, evolve  # noqa: E402

API = "https://api.fireworks.ai/v1"
BASE = "accounts/fireworks/models/muse-glimmer-30b"
DONE = {"JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "JOB_STATE_EARLY_STOPPED",
        "JOB_STATE_DELETED"}
LOG = evolve.ROOT / "fireworks.log"


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def client() -> httpx.Client:
    config.load_env()
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        sys.exit("FIREWORKS_API_KEY 가 .env 에 없음")
    return httpx.Client(base_url=API, headers={"Authorization": f"Bearer {key}"}, timeout=120)


def account(c: httpx.Client) -> str:
    return c.get("/accounts").raise_for_status().json()["accounts"][0]["name"]      # accounts/<id>


def upload(c: httpx.Client, acc: str, path: Path, did: str) -> str:
    n = sum(1 for _ in open(path))
    r = c.post(f"/{acc}/datasets", json={"datasetId": did, "dataset": {"displayName": did, "exampleCount": str(n), "format": "CHAT"}})
    if r.status_code not in (200, 409):
        r.raise_for_status()
    with open(path, "rb") as f:
        c.post(f"/{acc}/datasets/{did}:upload", files={"file": (path.name, f, "application/jsonl")}).raise_for_status()
    for _ in range(60):
        st = c.get(f"/{acc}/datasets/{did}").json().get("state")
        if st == "READY":
            break
        time.sleep(5)
    log(f"데이터 올림 {did}: {n}개 ({path.stat().st_size / 1e6:.1f}MB) 상태 {st}")
    return f"{acc}/datasets/{did}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="출력 모델 이름 끝 (기본: 학습 데이터 버전)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--max-cost", type=float, default=5.0)
    a = ap.parse_args()

    reg = evolve._registry()
    if not reg["datasets"]:
        sys.exit("학습 데이터가 없음 — python scripts/evolve.py build")
    ds = reg["datasets"][-1]
    ver = a.version or f"v{ds['version']}-{datetime.now().strftime('%m%d%H%M')}"
    train, ev = evolve.export_fireworks("train"), evolve.export_fireworks("eval")
    c = client()
    acc = account(c)
    stamp = datetime.now().strftime("%m%d%H%M")
    train_ds = upload(c, acc, train, f"datacash-steps-train-{stamp}")
    eval_ds = upload(c, acc, ev, f"datacash-steps-eval-{stamp}")
    out_model = f"{acc}/models/datacash-glimmer-{ver}".replace(".", "-").lower()
    body = {"baseModel": BASE, "dataset": train_ds, "evaluationDataset": eval_ds, "outputModel": out_model,
            "epochs": a.epochs, "loraRank": a.lora_rank, "learningRate": 2e-4, "maxContextLength": 8192,
            "displayName": f"DataCash Glimmer steps {ver}"}
    job = c.post(f"/{acc}/supervisedFineTuningJobs", json=body).raise_for_status().json()
    jid = job["name"]
    log(f"학습 시작 {jid} → {out_model} (학습 {ds['train_steps']}개 · 평가 {ds['eval_steps']}개 · epochs {a.epochs} · LoRA r{a.lora_rank})")
    entry = {"job": jid, "output_model": out_model, "dataset": ds["version"], "base": BASE, "epochs": a.epochs,
             "lora_rank": a.lora_rank, "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "state": job.get("state")}
    evolve.record_training(entry)

    cost_checked = False
    while True:
        time.sleep(60)
        j = c.get(f"/{jid}").json()
        st, est = j.get("state"), j.get("estimatedCost")
        prog = (j.get("jobProgress") or {})
        if not cost_checked and est:
            usd = float(est.get("units", 0) or 0) + float(est.get("nanos", 0) or 0) / 1e9 if isinstance(est, dict) else float(est)
            log(f"예상 비용 ${usd:.2f}")
            cost_checked = True
            if usd > a.max_cost:
                c.post(f"/{jid}:cancel")
                log(f"예상 비용이 ${a.max_cost} 를 넘어 취소함")
                entry.update(state="CANCELLED_COST", estimated_usd=usd)
                evolve.record_training(entry)
                return
            entry["estimated_usd"] = usd
        entry.update(state=st, progress=prog)
        evolve.record_training(entry)
        log(f"상태 {st} {json.dumps(prog, ensure_ascii=False)[:160]}")
        if st in DONE:
            entry["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if st != "JOB_STATE_COMPLETED":
                entry["error"] = json.dumps(j.get("status") or {}, ensure_ascii=False)[:300]
            evolve.record_training(entry)
            log(f"끝 {st} — 모델 {out_model}")
            return


if __name__ == "__main__":
    main()
