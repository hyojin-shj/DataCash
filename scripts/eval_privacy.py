"""SAM 3.1 가리기 평가 — 라이선스 확인된 Wikimedia Commons 영상만 (data/opendata, 출처는 manifest.json).

지표
  others_redetected   가린 영상의 표본 장면에서 SAM 이 참여자 아닌 사람을 다시 찾은 장면 비율 (독립 재검사)
  hands_kept_pct      원본에서 MediaPipe 가 찾은 손이 가린 영상에서도 찾아지는 비율 (참여자 손이 남는가)
  masked_area_pct     가린 넓이 평균
  비교: '얼굴만 블러'는 다른 사람의 몸 전체가 남는다 → 다른 사람 영역 중 얼굴 상자 밖 넓이 비율(노출)로 함께 적는다

  python scripts/eval_privacy.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from datacache import config
from datacache.media import clip_mp4
from datacache.meta import spent_usd
from datacache.meta.client import BudgetExceeded
from datacache.meta.sam import Sam
from datacache.privacy import mask_video

CLIPS = [
    ("home_kitchen_wide", "dishwashing/_Yum-Cast_Our_Daily_Life_Let_s_clean_and_do_the_dishes_.webm", 0.15, ["plate", "chair"]),
    ("gloves_dishwashing", "dishwashing/_Yum-Cast_Our_Daily_Life_Let_s_clean_and_do_the_dishes_.webm", 0.62, ["plate", "rubber glove"]),
    ("food_shop", "dishwashing/Dishes_you_can_t_miss_in_Xi_an.webm", 0.50, ["bowl", "plate"]),
]
OUT = config.DATA / "eval" / "privacy"


def hands_count(frames):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.MODELS / "hand_landmarker.task")), num_hands=4,
        min_hand_detection_confidence=0.3))
    n = [len(lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))).hand_landmarks)
         for f in frames]
    lm.close()
    return n


def sample(video, k=6):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(7))
    out = []
    for i in np.linspace(0, n - 1, k).astype(int):
        cap.set(1, int(i))
        ok, f = cap.read()
        out.append((int(i), f))
    return out


def main(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    sam = Sam()
    results = {}
    for name, rel, at, manip in CLIPS:
        if only and name not in only:
            continue
        src = config.OPENDATA / rel
        n = int(cv2.VideoCapture(str(src)).get(7))
        fps = cv2.VideoCapture(str(src)).get(5) or 25
        clip = clip_mp4(src, OUT / f"{name}.mp4", start_s=at * n / fps, dur_s=8, max_side=640, fps=15, audio=False)
        t = time.time()
        try:
            r = mask_video(clip, OUT / f"{name}_masked.mp4", manipulated=manip, sam=sam)
        except BudgetExceeded as e:
            results[name] = {"error": str(e)}
            continue
        rep = dict(r.report, seconds=round(time.time() - t, 1), manipulated=r.object_names)
        so, sm = sample(clip), sample(r.masked_video)
        ho, hm = hands_count([f for _, f in so]), hands_count([f for _, f in sm])
        rep["hands_original"], rep["hands_masked"] = ho, hm
        rep["hands_kept_pct"] = round(100 * min(sum(hm), sum(ho)) / max(1, sum(ho)), 1)
        redet, imgs = 0, []
        for (i, fo), (_, fm) in zip(so, sm):
            p = OUT / f"{name}_m{i}.jpg"
            cv2.imwrite(str(p), fm)
            try:
                found = [o for o in sam.image(p, "person") if (o.confidence or 0) >= 0.5]
            except BudgetExceeded:
                found = []
            others = max(0, len(found) - (1 if r.participant is not None else 0))
            redet += others > 0
            imgs.append(np.hstack([cv2.resize(fo, (320, 240)), cv2.resize(fm, (320, 240))]))
        rep["others_redetected_frames"] = f"{redet}/{len(so)}"
        cv2.imwrite(str(OUT / f"{name}_sheet.jpg"), np.vstack(imgs[::2]))
        results[name] = rep
        print(name, json.dumps(rep, ensure_ascii=False), flush=True)
    results["_spent_usd"] = spent_usd()
    (OUT / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("spent", spent_usd())


if __name__ == "__main__":
    main(sys.argv[1:] or None)
