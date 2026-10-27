"""Meta Model API 공통 클라이언트 (OpenAI SDK 호환, https://api.meta.ai/v1).

비용 (dev.meta.ai/docs/pricing-rate-limits, 2026-10-27 확인, Standard mode — 보낸 내용은 Meta 학습에 쓰이지 않는다):
  muse-spark-1.3 / 1.2   입력 $1.25 · 출력 $4.25 / 100만 토큰 (캐시 입력 $0.15)
  muse-image-1.0         $0.01 / 장
  muse-voice-transcribe  $0.18 / 시간 (초 단위 내림)
  sam-3.1                이미지 $2.50 / 1,000장, 영상 $0.20 / 1,000프레임 (요청당 최소 50프레임)
Muse Wonder 는 한국에서 쓸 수 없다 (모델 문서).

호출 전에 예상 비용으로 한도를 확인하고(넘으면 BudgetExceeded), 호출 뒤 실제 사용량으로 장부에 적는다.
한도에 걸려도 파이프라인은 API 없는 단계로 계속 가도록, 부르는 쪽이 BudgetExceeded 를 잡는다.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .. import config

PRICE = {
    "muse-spark-1.3": (1.25e-6, 4.25e-6), "muse-spark-1.2": (1.25e-6, 4.25e-6),
    "muse-image-1.0": 0.01, "muse-voice-transcribe-1.0": 0.18 / 3600,
    "sam-3.1-image": 2.5e-3, "sam-3.1-frame": 2.0e-4,
}


class BudgetExceeded(RuntimeError):
    pass


def _ledger_rows():
    if not config.LEDGER.exists():
        return []
    return [json.loads(x) for x in config.LEDGER.read_text().splitlines() if x.strip()]


def spent_usd() -> float:
    return round(sum(r.get("usd", 0.0) for r in _ledger_rows()), 6)


def record(model: str, usd: float, what: str, **extra) -> None:
    config.LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "usd": round(usd, 6), "what": what, **extra}
    with open(config.LEDGER, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_budget(estimate_usd: float) -> None:
    if spent_usd() + estimate_usd > config.API_BUDGET_USD:
        raise BudgetExceeded(f"API 한도 ${config.API_BUDGET_USD} (사용 ${spent_usd():.4f} + 예상 ${estimate_usd:.4f})")


def data_url(path, mime: Optional[str] = None) -> str:
    p = Path(path)
    mime = mime or {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
                    ".mp4": "video/mp4", ".mov": "video/quicktime", ".wav": "audio/wav"}[p.suffix.lower()]
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


class MetaClient:
    """OpenAI SDK 클라이언트를 감싼다. 키는 .env 의 MODEL_API_KEY."""

    def __init__(self, timeout: float = 600):
        from openai import OpenAI

        self.client = OpenAI(base_url=config.API_BASE, api_key=config.api_key(), timeout=timeout)

    # ---- 공통 ----
    def models(self) -> list:
        return [m.id for m in self.client.models.list().data]

    def upload(self, path) -> str:
        """Files API (purpose=user_data). 같은 영상을 여러 번 분할할 때 다시 올리지 않게 id 를 돌려준다."""
        with open(path, "rb") as f:
            return self.client.files.create(file=f, purpose="user_data").id

    def responses(self, model: str, content: list, estimate_usd: float, what: str,
                  metadata: Optional[Dict[str, str]] = None, **kw) -> Any:
        check_budget(estimate_usd)
        r = self.client.responses.create(model=model, input=[{"type": "message", "role": "user", "content": content}],
                                         **({"metadata": metadata} if metadata else {}), **kw)
        usd = estimate_usd
        if model in PRICE and isinstance(PRICE[model], tuple) and getattr(r, "usage", None):
            pi, po = PRICE[model]
            usd = (r.usage.input_tokens or 0) * pi + (r.usage.output_tokens or 0) * po
        record(model, usd, what, usage=getattr(getattr(r, "usage", None), "model_dump", lambda: None)())
        return r

    def chat_json(self, model: str, messages: list, schema: dict, name: str, what: str,
                  estimate_usd: float = 0.02, **kw) -> dict:
        """구조화 출력 (Chat Completions response_format=json_schema). JSON 을 dict 로."""
        check_budget(estimate_usd)
        r = self.client.chat.completions.create(
            model=model, messages=messages,
            response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}, **kw)
        pi, po = PRICE.get(model, (0, 0))
        u = r.usage
        record(model, (u.prompt_tokens or 0) * pi + (u.completion_tokens or 0) * po if u else estimate_usd, what,
               usage=u.model_dump() if u else None)
        return json.loads(r.choices[0].message.content)
