"""이의 신청 — 자동 확인 → 검토자 확인 → 결과.

신청 사유
  reward      보상이 적어요       자동: 받아들인 분량·등급·보너스로 보상을 다시 계산해 덜 줬으면 바로 차액 인정
  failed      처리가 실패했어요    자동: 처리를 다시 대기열에 넣는다 (다시 처리한 결과로 끝)
  privacy     가려야 할 게 보여요   자동: 그 녹화를 바로 판매에서 뺀다(withheld) → 검토자가 확인
  label       라벨·단계가 틀려요   Muse Spark Open(Fireworks, 없으면 Spark 1.3 최대 추론)으로 다시 라벨 → 달라진 단계가 있으면 인정·바로잡음 (datacash/judge.py), 아니면 검토자
  failed(반려)                    같은 모델로 다시 보고 미션 동작이 나오면 검수를 건너뛰고 다시 처리
  other       기타                검토자 확인
저장: data/server/appeals/<id>.json — 기기 id, 녹화 id, 사유, 단계 기록, 상태, 인정된 추가 보상.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from . import pricing
from .judge import needs_review_model, review_model_name

DAILY_LIMIT = 5
REASONS = {"reward": "보상이 적어요", "failed": "처리가 실패했어요", "privacy": "가려야 할 게 보여요",
           "label": "라벨·단계가 틀려요", "other": "기타"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Appeals:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, aid: str) -> dict:
        p = self.root / f"{aid}.json"
        if not aid.isalnum() or not p.exists():
            raise KeyError(aid)
        return json.loads(p.read_text())

    def save(self, a: dict) -> dict:
        (self.root / f"{a['id']}.json").write_text(json.dumps(a, indent=2, ensure_ascii=False))
        return a

    def all(self) -> List[dict]:
        return sorted((json.loads(p.read_text()) for p in self.root.glob("*.json")), key=lambda a: a["created"], reverse=True)

    def for_device(self, device: str) -> List[dict]:
        return [a for a in self.all() if a["device"] == device]

    def file(self, device: str, sid: str, reason: str, detail: str, status: dict,
             requeue: Callable[[str], None], withhold: Callable[[str], None], review_model: bool = False) -> dict:
        """신청을 받고 바로 자동 확인까지 한다."""
        if reason not in REASONS:
            raise ValueError("사유")
        mine = self.for_device(device)
        if any(a["sid"] == sid and a["state"] in ("auto", "judging", "review", "reprocessing") for a in mine):
            raise ValueError("이미 진행 중인 이의 신청이 있음")
        day = datetime.now(timezone.utc).timestamp() - 86400
        if sum(datetime.fromisoformat(a["created"]).timestamp() > day for a in mine) >= DAILY_LIMIT:
            raise ValueError(f"이의 신청은 하루 {DAILY_LIMIT}번까지예요")
        a = {"id": secrets.token_hex(5), "device": device, "sid": sid, "reason": reason, "reason_ko": REASONS[reason],
             "detail": detail[:500], "created": _now(), "state": "auto", "extra_reward": 0,
             "steps": [{"stage": "filed", "at": _now(), "note": "신청 받음"}]}
        st = status.get("state")
        if reason == "reward" and st == "done":
            want = expected_reward(status)
            got = int(status.get("reward") or 0)
            if want > got:
                return self._finish(a, "accepted", f"자동 확인: 다시 계산하니 {want:,}원 — 차액 {want - got:,}원 인정", want - got)
            self._step(a, "auto", f"자동 확인: 다시 계산해도 {got:,}원 (분량 {status.get('minutes')}분 · {status.get('level')}) → 검토자 확인")
        elif review_model and needs_review_model(reason, status):
            self._step(a, "auto", f"더 높은 모델({review_model_name()})로 영상을 다시 보고 있어요")
            a["state"] = "judging"
            return self.save(a)
        elif reason == "failed" and st in ("failed", "rejected"):
            requeue(sid)
            self._step(a, "auto", "자동 확인: 처리를 다시 시작했어요 — 끝나면 결과가 바뀌어요")
            a["state"] = "reprocessing"
            return self.save(a)
        elif reason == "privacy":
            withhold(sid)
            self._step(a, "auto", "자동 확인: 이 녹화를 판매에서 바로 뺐어요 → 검토자가 가리기를 확인해요")
        else:
            self._step(a, "auto", "자동 확인으로 판단할 수 없어 검토자에게 넘겼어요")
        a["state"] = "review"
        return self.save(a)

    def judged(self, aid: str, res: dict) -> dict:
        """다시 보기(judge.judge) 결과를 반영한다."""
        a = self.get(aid)
        if res["decision"] == "accept":
            a["corrections"] = res.get("corrections", [])
            return self._finish(a, "accepted", res["note"], int(res.get("extra") or 0))
        self._step(a, "review_model", res["note"])
        a["state"] = "reprocessing" if res["decision"] == "reprocess" else "review"
        return self.save(a)

    def decide(self, aid: str, accept: bool, extra_reward: int, note: str) -> dict:
        a = self.get(aid)
        if a["state"] not in ("review", "reprocessing"):
            raise ValueError("이미 끝난 신청")
        return self._finish(a, "accepted" if accept else "rejected", "검토자: " + (note or ("인정" if accept else "인정하지 않음")),
                            max(0, int(extra_reward)) if accept else 0)

    def reprocessed(self, sid: str, status: dict) -> None:
        """다시 처리가 끝나면 그 녹화의 'failed' 신청을 닫는다."""
        for a in self.all():
            if a["sid"] == sid and a["state"] == "reprocessing":
                ok = status.get("state") == "done"
                self._finish(a, "accepted" if ok else "review",
                             "다시 처리 성공 · 보상 %s원" % format(int(status.get("reward") or 0), ",") if ok
                             else "다시 처리도 실패 → 검토자 확인", 0)

    def _step(self, a: dict, stage: str, note: str) -> None:
        a["steps"].append({"stage": stage, "at": _now(), "note": note})

    def _finish(self, a: dict, state: str, note: str, extra: int) -> dict:
        self._step(a, "result" if state != "review" else "auto", note)
        a["state"], a["extra_reward"] = state, extra
        return self.save(a)


def expected_reward(status: dict) -> int:
    """받아들인 분량·등급·오늘 보너스로 다시 계산한 보상."""
    try:
        return pricing.reward(float(status.get("minutes") or 0), status.get("level") or "basic",
                              today=bool(status.get("today_bonus")))
    except (KeyError, ValueError):
        return 0
