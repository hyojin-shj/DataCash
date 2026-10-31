"""Muse Image (POST /v1/images/generations) — 실존하지 않는 가상 인물 사진 생성. 장당 $0.01."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import List

from .client import PRICE, MetaClient, check_budget, record

MODEL = "muse-image-1.0"


def generate(prompt: str, out_dir, n: int = 1, size: str = "1024x1024", client: MetaClient = None,
             what: str = "image") -> List[Path]:
    check_budget(n * PRICE[MODEL])
    c = (client or MetaClient()).client
    t0 = time.time()
    r = c.images.generate(model=MODEL, prompt=prompt, n=n, size=size, response_format="b64_json",
                          extra_body={"output_format": "png"})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, d in enumerate(r.data):
        p = out_dir / f"{what}_{i}.png"
        p.write_bytes(base64.b64decode(d.b64_json))
        paths.append(p)
    record(MODEL, len(paths) * PRICE[MODEL], f"{what} x{len(paths)}", secs=time.time() - t0)
    return paths
