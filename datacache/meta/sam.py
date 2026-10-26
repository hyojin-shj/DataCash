"""SAM 3.1 분할 (이미지·영상). 결과를 화면 크기 마스크(bool) + 상자 + 신뢰도로 돌려준다.

요청: Responses API, 사용자 메시지 하나에 input_text(명사구 하나) + input_image / input_video.
응답: 한 줄에 한 프레임 `<Nf>` + 물체마다 box·mask 토큰 → 공식 파서 meta_sam_parser 로 읽는다.
마스크 래스터는 상자 범위(bounds) 안의 작은 그림이라, 상자 크기로 늘려 화면 크기 마스크에 붙인다.
영상은 한 요청에 최대 15,000프레임, 프레임당 물체 16개까지 추적. 영상 비용은 프레임 수로(최소 50).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .client import PRICE, MetaClient, data_url

MODEL = "sam-3.1"


@dataclass
class SamObject:
    object_id: str
    box: tuple                      # (x1, y1, x2, y2) 원본 픽셀
    confidence: Optional[float]
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # (H, W) bool, 화면 크기

    @property
    def area(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else 0


def _raster(record, frame_w: int, frame_h: int) -> np.ndarray:
    from meta_sam_parser import decode_mask_to_raster

    m = record.mask
    r = np.frombuffer(decode_mask_to_raster(m), dtype=np.uint8).reshape(m.height, m.width)
    b = record.bounds
    x0, y0 = max(0, int(round(b.left))), max(0, int(round(b.top)))
    x1, y1 = min(frame_w, int(round(b.right))), min(frame_h, int(round(b.bottom)))
    full = np.zeros((frame_h, frame_w), bool)
    if x1 > x0 and y1 > y0:
        full[y0:y1, x0:x1] = cv2.resize(r, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return full


def _parse(text: str, media: str, frame_w: int, frame_h: int) -> Dict[int, List[SamObject]]:
    from meta_sam_parser import CompletedOutcome, image_segmentation_format, video_segmentation_format

    fmt = image_segmentation_format() if media == "image" else video_segmentation_format()
    p = fmt.create_parser()
    p.push(text if text.endswith("\n") else text + "\n", emit=False)
    res = p.finish(CompletedOutcome()).result
    frames: Dict[int, Dict[str, SamObject]] = {}
    for rec in res.records:
        fi = rec.frame.frame_index if getattr(rec, "frame", None) is not None else 0
        objs = frames.setdefault(fi, {})
        o = objs.setdefault(rec.object_id, SamObject(rec.object_id, (0, 0, 0, 0), None))
        if rec.kind == "box":
            o.box = (rec.left, rec.top, rec.right, rec.bottom)
            o.confidence = rec.confidence if rec.confidence is not None else o.confidence
        elif rec.kind == "mask":
            o.mask = _raster(rec, frame_w, frame_h)
            o.confidence = rec.confidence if rec.confidence is not None else o.confidence
    return {k: list(v.values()) for k, v in sorted(frames.items())}


def _text(r) -> str:
    t = getattr(r, "output_text", None)
    if t:
        return t
    parts = []
    for item in getattr(r, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") == "output_text":
                parts.append(c.text)
    return "".join(parts)


class Sam:
    def __init__(self, client: Optional[MetaClient] = None):
        self.c = client or MetaClient()

    def image(self, image_path, concept: str) -> List[SamObject]:
        """이미지 한 장에서 명사구 하나에 맞는 모든 물체."""
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        r = self.c.responses(MODEL, [{"type": "input_text", "text": concept},
                                     {"type": "input_image", "image_url": data_url(image_path)}],
                             estimate_usd=PRICE["sam-3.1-image"], what=f"sam image '{concept}' {Path(image_path).name}",
                             metadata={"include_confidence": "true", "mask_encoding": "one_bit"})
        return _parse(_text(r), "image", w, h).get(0, [])

    def video(self, video_path, concept: str, file_id: Optional[str] = None) -> Dict[int, List[SamObject]]:
        """영상 전체에서 명사구 하나를 추적. {프레임 번호: [물체]}. 물체 id 는 영상 안에서 유지된다."""
        cap = cv2.VideoCapture(str(video_path))
        w, h, n = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
        cap.release()
        part = {"type": "input_video", "file_id": file_id} if file_id else {"type": "input_video", "video_url": data_url(video_path)}
        r = self.c.responses(MODEL, [{"type": "input_text", "text": concept}, part],
                             estimate_usd=max(50, n) * PRICE["sam-3.1-frame"],
                             what=f"sam video '{concept}' {Path(video_path).name} {n}f",
                             metadata={"include_confidence": "true", "mask_encoding": "one_bit"})
        return _parse(_text(r), "video", w, h)
