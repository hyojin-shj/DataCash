"""보상 입금 (시험) — 폰마다 받을 계좌 표시(가린 번호)만 서버에 두고, 승인된 녹화의 보상을 보낸 것으로 기록한다.

계좌번호 전체는 폰에만 저장되고 서버에는 'KB국민 ****1234' 같은 표시만 온다. 시험 버전이라 실제 송금은 하지 않는다.
녹화 상태(status)의 payout: {status: sent | waiting_account, amount, label, at}
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

LABEL = re.compile(r"^[^\d]{1,12} \*{4}\d{4}$")     # 'KB국민 ****1234' — 번호 전체가 오면 받지 않는다
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Payouts:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, device: str) -> Path:
        return self.root / f"{device}.json"

    def account(self, device: str) -> Optional[dict]:
        p = self._p(device)
        return json.loads(p.read_text()) if p.exists() else None

    def set_account(self, device: str, label: str) -> dict:
        if not LABEL.match(label or ""):
            raise ValueError("계좌 표시는 '은행 ****1234' 꼴로만 받아요 (번호 전체는 보내지 않아요)")
        a = {"method": "bank", "label": label, "at": _now()}
        with _lock:
            self._p(device).write_text(json.dumps(a, ensure_ascii=False))
        return a

    def settle(self, device: Optional[str], status: dict) -> Optional[dict]:
        """승인된 녹화의 보상(+ 이의 신청으로 인정된 추가 보상) → 입금 기록. 이미 보냈으면 그대로."""
        amount = int(status.get("reward") or 0) + int(status.get("appeal_extra") or 0)
        if status.get("state") != "done" or amount <= 0 or not device:
            return status.get("payout")
        old = status.get("payout") or {}
        if old.get("status") == "sent" and old.get("amount") == amount:
            return old
        acc = self.account(device)
        if acc:
            return {"status": "sent", "amount": amount, "label": acc["label"], "at": _now()}
        return {"status": "waiting_account", "amount": amount}

    def summary(self, device: str, statuses: Iterable[dict]) -> dict:
        total = waiting = 0
        for s in statuses:
            p = s.get("payout") or {}
            if p.get("status") == "sent":
                total += int(p.get("amount") or 0)
            elif p.get("status") == "waiting_account":
                waiting += int(p.get("amount") or 0)
        acc = self.account(device)
        return {"label": acc["label"] if acc else "", "total": total, "waiting": waiting}
