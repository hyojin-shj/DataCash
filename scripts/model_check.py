"""열린 모델 확인 + 모델마다 첫 호출 (공개 영상만 보낸다). 결과는 화면에, 비용은 data/api_ledger.jsonl 에.

  python scripts/model_check.py
"""

import json
import time

from datacache import config
from datacache.media import clip_mp4, wav
from datacache.meta import MetaClient, spent_usd
from datacache.meta.client import data_url
from datacache.meta.image import generate
from datacache.meta.sam import Sam
from datacache.meta.voice import transcribe

CHECK = config.DATA / "check"


def main():
    c = MetaClient()
    print("열린 모델:", c.models())
    wash = config.OPENDATA / "dishwashing" / "Washing_Dishes.webmhd.webm"  # CC BY-SA 4.0, Leet289
    clip = clip_mp4(wash, CHECK / "wash_8s.mp4", start_s=10, dur_s=8, max_side=640)

    import cv2
    cap = cv2.VideoCapture(str(wash))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ok, f = cap.read()
    cv2.imwrite(str(CHECK / "wash.jpg"), f)
    s = Sam(c)
    for concept in ("person", "hand", "plate"):
        objs = s.image(CHECK / "wash.jpg", concept)
        print("sam-3.1", concept, [(o.object_id, [int(v) for v in o.box], o.confidence) for o in objs])

    schema = {"type": "object", "additionalProperties": False, "required": ["task", "steps"], "properties": {
        "task": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                             "required": ["start_s", "end_s", "instruction"],
                                             "properties": {"start_s": {"type": "number"}, "end_s": {"type": "number"},
                                                            "instruction": {"type": "string"}}}}}}
    t = time.time()
    out = c.chat_json("muse-spark-1.3", [{"role": "user", "content": [
        {"type": "text", "text": "This clip shows a household task. Describe the task and list the steps with start/end seconds."},
        {"type": "video_url", "video_url": {"url": data_url(clip)}}]}], schema, "steps", "model_check spark")
    print("muse-spark-1.3", round(time.time() - t, 1), "s", json.dumps(out, ensure_ascii=False))

    latin = config.OPENDATA / "cooking" / "Michāēl_coquus_est_-_Mike_is_a_cook_-_60_Latin_mini-stories_-1.webm"
    tr = transcribe(wav(latin, CHECK / "latin.wav", start_s=5, dur_s=40))
    print("muse-voice-transcribe-1.0", tr["turns"][:3])

    print("muse-image-1.0", generate("Frontal portrait photo of a fictional person, neutral expression, plain background",
                                     CHECK, what="model_check"))
    print("누적 비용 $", spent_usd())


if __name__ == "__main__":
    main()
