"""Baseten 모델: datacash.privacy.swap_job.run_local 을 GPU 에서. 영상은 S3 임시 주소로 받고 결과도 임시 주소에 올린다 (원본을 남기지 않는다)."""

import json
import os
import tempfile
import traceback
from pathlib import Path

import httpx


def _boxes(d, nested=False):
    if not d:
        return None
    if nested:                                           # 장면 → [(추적 id, 상자)]
        return {int(k): [(v[0], tuple(v[1])) for v in vs] for k, vs in d.items()}
    return {int(k): tuple(v) for k, v in d.items()}


class Model:
    def __init__(self, **kwargs):
        pass

    def load(self):
        os.environ.setdefault("LP_ROOT", "/app/liveportrait")
        from datacash.privacy.faceswap import LivePortraitSwap

        LivePortraitSwap()                               # 가중치를 미리 GPU 에 (첫 요청이 덜 기다리게)

    def predict(self, req: dict) -> dict:
        from datacash.privacy import swap_job

        donors = Path(swap_job.__file__).parents[1] / "donors"
        try:
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                for k, name in (("video_url", "video.mp4"), ("masked_url", "masked.mp4")):
                    with httpx.stream("GET", req[k], timeout=300) as r:
                        r.raise_for_status()
                        with open(d / name, "wb") as f:
                            for b in r.iter_bytes():
                                f.write(b)
                out = swap_job.run_local(d / "video.mp4", d / "masked.mp4", d / "out.mp4",
                                         gender=req.get("gender"), age_band=req.get("age_band"),
                                         mine=donors / f"{req['mine']}.png" if req.get("mine") else None,
                                         family={k: donors / f"{v}.png" for k, v in (req.get("family") or {}).items()},
                                         participant_boxes=_boxes(req.get("participant_boxes")),
                                         consented_boxes=_boxes(req.get("consented_boxes"), nested=True))
                with open(d / "out.mp4", "rb") as f:
                    httpx.put(req["out_put_url"], content=f.read(), headers={"Content-Type": "video/mp4"}, timeout=300).raise_for_status()
                return json.loads(json.dumps(out, default=str))
        except Exception as e:
            traceback.print_exc()
            return {"error": f"{type(e).__name__}: {str(e)[:300]}"}
