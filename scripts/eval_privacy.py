"""SAM 3.1 가리기 평가 — 라이선스 확인된 Wikimedia Commons 영상만 (data/opendata, 출처는 manifest.json).

지표
  others_redetected   가린 영상의 표본 장면에서 SAM 이 참여자 아닌 사람을 다시 찾은 장면 비율 (독립 재검사)
                      참여자 상자와 겹치거나(IoU≥0.3) 대부분 그 안에 든(≥60%) 검출은 참여자 몸, 절반 이상 회색인 검출은 가린 실루엣으로 본다
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
    ("game_lounge_screens", "extra/2023-03-31_Retro_Games_Lounge.webm", 0.30, ["game controller"]),
    ("mirror_bathroom", "extra/Man_in_hotel_bathroom_with_mirrors_all_around_turning_and_making_video_thereof.webm", 0.30, ["towel"]),
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


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def contained(a, b):
    """a 상자 넓이 중 b 안에 든 비율 (참여자 팔·손처럼 참여자 상자 안의 조각은 참여자)."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return max(0, x2 - x1) * max(0, y2 - y1) / area if area > 0 else 0.0


def visible_frac(img, box):
    """검출 상자 안에서 회색으로 가려지지 않은 픽셀 비율 — 가린 실루엣을 다시 '사람'으로 잡은 건 노출이 아니다."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    roi = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if roi.size == 0:
        return 0.0
    gray = np.all(np.abs(roi.astype(int) - 128) <= 6, axis=2)
    return float(1 - gray.mean())


MIN_PERSON_PX = 40   # 이보다 작은 '사람'은 얼굴이 몇 픽셀 → 알아볼 수 없음. 따로 센다 (small_redetected)


def is_other(o, pb, img):
    if pb is not None and (iou(o.box, pb) >= 0.3 or contained(o.box, pb) >= 0.6 or contained(pb, o.box) >= 0.5):
        return False                       # 참여자: 같은 상자, 참여자 몸의 일부, 또는 참여자(손만 잡힌)를 품은 몸 전체
    return visible_frac(img, o.box) >= 0.5   # 대부분 가려진 실루엣이면 노출 아님


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
    rp = OUT / "report.json"
    results = json.loads(rp.read_text()) if rp.exists() else {}
    for name, rel, at, manip in CLIPS:
        if only and name not in only:
            continue
        src = config.OPENDATA / rel
        n = int(cv2.VideoCapture(str(src)).get(7))
        fps = cv2.VideoCapture(str(src)).get(5) or 25
        clip = clip_mp4(src, OUT / f"{name}.mp4", start_s=at * n / fps, dur_s=8, max_side=640, fps=15, audio=False)
        t = time.time()
        from datacache.labels import extract_labels
        from datacache.pipeline import FINGERTIPS

        pts = [[h[FINGERTIPS, :2] for h in f.hands if h is not None] for f in extract_labels(clip)]   # 앱 처리와 같게 손끝을 넘긴다
        try:
            r = mask_video(clip, OUT / f"{name}_masked.mp4", manipulated=manip, sam=sam, hand_pts=pts)
        except BudgetExceeded as e:
            results[name] = {"error": str(e)}
            continue
        rep = dict(r.report, seconds=round(time.time() - t, 1), manipulated=r.object_names)
        so, sm = sample(clip), sample(r.masked_video)
        ho, hm = hands_count([f for _, f in so]), hands_count([f for _, f in sm])
        rep["hands_original"], rep["hands_masked"] = ho, hm
        rep["hands_kept_pct"] = round(100 * min(sum(hm), sum(ho)) / max(1, sum(ho)), 1)
        redet, small, imgs = 0, 0, []
        for (i, fo), (_, fm) in zip(so, sm):
            p = OUT / f"{name}_m{i}.jpg"
            cv2.imwrite(str(p), fm)
            try:
                found = [o for o in sam.image(p, "person") if (o.confidence or 0) >= 0.5]
            except BudgetExceeded:
                found = []
            pb = r.participant_boxes.get(i)
            oth = [o for o in found if is_other(o, pb, fm)]
            others = sum(1 for o in oth if o.box[3] - o.box[1] >= MIN_PERSON_PX)
            small += len(oth) > 0 and others == 0
            rep.setdefault("redetect_detail", []).append({"frame": i, "participant_box": [round(v) for v in pb] if pb else None,
                                                          "found": [{"box": [round(v) for v in o.box], "conf": round(o.confidence or 0, 2),
                                                                     "height_px": round(o.box[3] - o.box[1]),
                                                                     "other": is_other(o, pb, fm)} for o in found]})
            redet += others > 0
            imgs.append(np.hstack([cv2.resize(fo, (320, 240)), cv2.resize(fm, (320, 240))]))
        rep["others_redetected_frames"] = f"{redet}/{len(so)}"
        rep["small_redetected_frames"] = f"{small}/{len(so)}"
        cv2.imwrite(str(OUT / f"{name}_sheet.jpg"), np.vstack(imgs[::2]))
        results[name] = rep
        print(name, json.dumps(rep, ensure_ascii=False), flush=True)
    results["_spent_usd"] = spent_usd()
    (OUT / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("spent", spent_usd())


if __name__ == "__main__":
    main(sys.argv[1:] or None)
