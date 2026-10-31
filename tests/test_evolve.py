"""모델 진화(Glimmer): 검증·바로잡은 라벨 → 학습 데이터(Axolotl 형식) → 평가 기록 → 준비된 어댑터만 Spark 대신."""

import json
from types import SimpleNamespace

import cv2
import numpy as np

from datacash import evolve


def _recording(out, n_steps=3, actions=("reach", "grasp", "release")):
    out.mkdir(parents=True)
    vw = cv2.VideoWriter(str(out / "masked_swap.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for i in range(30 * (n_steps + 1)):        # 끝 단계 뒤로 1초 — 장면을 단계 끝까지(늘린 벌은 조금 더) 뽑는다
        vw.write(np.full((120, 160, 3), (i * 3) % 255, np.uint8))
    vw.release()
    steps = [{"action": a, "start_s": k, "end_s": k + 1, "verified": True} for k, a in enumerate(actions)]
    steps.append({"action": "lift", "start_s": 2.5, "end_s": 2.9, "verified": False})       # 검증 안 된 단계는 빠진다
    (out / "certificate.json").write_text(json.dumps({"steps": steps, "mission": {"title_en": "pick cup"}}))


class FakeGlimmer:
    """질문을 받으면 정해 둔 답을 차례로 낸다."""
    def __init__(self, answers):
        self.answers = list(answers)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        a = self.answers.pop(0) if self.answers else "other"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"action": a})))])


def test_dataset_eval_and_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve, "ROOT", tmp_path / "evolve")
    dirs = [tmp_path / f"r{k}" for k in range(4)]
    for d in dirs:
        _recording(d)
    evolve.add_correction("r0", [{"start_s": 1, "end_s": 2, "was": "grasp", "action": "lift", "note": ""}], "pick cup")
    ds = evolve.build_dataset(dirs, log=lambda *a: None)
    # 학습용은 단계마다 AUGMENT 벌, 이의 신청에서 바로잡은 단계는 HARD_WEIGHT 배 더, 헷갈리는 동작(HARD_ACTIONS)은 두 배 · 평가용은 한 벌
    a = evolve.AUGMENT
    hard = a * evolve.HARD_WEIGHT * 2                       # r0 의 lift (바로잡음 · 헷갈리는 동작)
    assert ds["recordings"] == 4 and ds["eval_steps"] == 3
    assert ds["train_steps"] == (a + hard + 2 * a) + 2 * (a + 2 * a + 2 * a) and ds["from_appeals"] == hard
    rows = [json.loads(l) for l in (tmp_path / "evolve" / "sft" / "train.jsonl").read_text().splitlines()]
    assert [c["type"] for c in rows[0]["messages"][0]["content"]] == ["image"] * evolve.FRAMES + ["text"]
    assert "This is the first step." in rows[0]["messages"][0]["content"][-1]["text"]
    assert sum(json.loads(r["messages"][1]["content"]) == {"action": "lift"} for r in rows) == hard   # 바로잡은 정답
    fw = [json.loads(l) for l in evolve.export_fireworks("train").read_text().splitlines()]
    assert fw[0]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert evolve.build_dataset(dirs, log=lambda *a: None)["version"] == 1           # 같으면 새로 안 만든다

    base = evolve.evaluate(client=FakeGlimmer(["reach", "other", "other"]), log=lambda *a: None)
    assert base["accuracy"] == round(1 / 3, 3) and not base["ready"] and base["model"].endswith("(학습 전)")
    assert evolve.ready_adapter() is None

    adapter = tmp_path / "v1.gguf"
    adapter.write_bytes(b"x")
    monkeypatch.setattr(evolve, "READY_EXAMPLES", 5)
    tuned = evolve.evaluate(adapter=str(adapter), client=FakeGlimmer(["reach", "grasp", "release"]), log=lambda *a: None)
    assert tuned["accuracy"] == 1.0 and tuned["ready"] and evolve.ready_adapter() == str(adapter)


def test_label_video_merges_windows(tmp_path):
    _recording(tmp_path / "r")
    steps = evolve.label_video(tmp_path / "r" / "masked_swap.mp4", "pick cup",
                               client=FakeGlimmer(["reach", "grasp", "grasp"]))
    assert [s["action"] for s in steps] == ["reach", "grasp"] and steps[1]["end_s"] >= 2.9
