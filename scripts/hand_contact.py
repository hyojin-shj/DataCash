"""손-물체 접촉 신호 — 100DOH 손·물체 검출기(Shan et al., CVPR 2020)로 장면마다 손이 무엇에 닿아 있는지.

  train/handobj/venv/bin/python scripts/hand_contact.py <영상.mp4> [<영상.mp4> ...] [--fps 5] [--out 폴더]
  → <폴더>/<영상 이름>.json  {"fps", "detector", "frames": [{"t", "hands": [{"side", "score", "state", "probs", "box", "obj_box"}]}]}

state: N 아무것도 안 닿음 · S 자기 몸 · O 다른 사람 · P 들고 다닐 수 있는 물체 · F 고정된 물체(가구 등).
집기 ↔ 놓기 경계를 가리는 데 쓴다: N → P 로 바뀌는 순간은 집기, P → N 은 놓기.

검출기: github.com/ddshan/hand_object_detector (MIT) 를 train/handobj/hand_object_detector 에, C 확장 대신 torchvision 연산으로 바꿔 (맥 CPU·MPS).
가중치: faster_rcnn_1_8_132028.pth (handobj_100K+ego). 원저자 Google Drive 가 지워져 GitHub 미러에서 받았고,
두 미러의 해시(SHA-256 3444e3b9…, MD5 f0b494c2…)가 같은 것을 확인했다. 불러올 때 weights_only (파일 안 코드 실행 막음).
ego 학습 데이터는 EPIC-KITCHENS-55(2018) 장면 — 우리 채점 구간(EPIC-100 확장 영상, 2020)과 겹치지 않는다.
이 맥 venv(train/handobj/venv)에서 돈다 — 프로젝트 venv 와 따로.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "train" / "handobj" / "hand_object_detector"
sys.path.insert(0, str(ROOT / "lib"))
from model.faster_rcnn.resnet import resnet  # noqa: E402
from model.roi_layers import nms  # noqa: E402
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes  # noqa: E402
from model.utils.blob import im_list_to_blob  # noqa: E402
from model.utils.config import cfg, cfg_from_file, cfg_from_list  # noqa: E402

WEIGHTS = ROOT / "models" / "res101_handobj_100K" / "pascal_voc" / "faster_rcnn_1_8_132028.pth"
CLASSES = np.asarray(["__background__", "targetobject", "hand"])
STATES = "NSOPF"
HAND_T, OBJ_T = 0.5, 0.5


def load(device: str):
    cfg_from_file(str(ROOT / "cfgs" / "res101.yml"))
    cfg_from_list(["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"])
    cfg.USE_GPU_NMS = False
    cfg.TEST.SCALES = (450,)            # 640px 영상 — 키우지 않는다
    cfg.TEST.RPN_POST_NMS_TOP_N = 60    # 제안 상자 300 → 60: 맥 CPU 장면당 4.6 → 1.1초, 손·접촉 결과 같음 (손은 눈에 띄는 물체라)
    net = resnet(CLASSES, 101, pretrained=False, class_agnostic=False)
    net.create_architecture()
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    net.load_state_dict(ck["model"])
    if "pooling_mode" in ck:
        cfg.POOLING_MODE = ck["pooling_mode"]
    net.to(device)
    net.eval()          # 이 resnet 의 train() 은 self 를 돌려주지 않는다
    return net


def _blob(im):
    x = im.astype(np.float32) - cfg.PIXEL_MEANS
    s = float(cfg.TEST.SCALES[0]) / min(x.shape[:2])
    if round(s * max(x.shape[:2])) > cfg.TEST.MAX_SIZE:
        s = float(cfg.TEST.MAX_SIZE) / max(x.shape[:2])
    return im_list_to_blob([cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)]), s


@torch.no_grad()
def detect(net, im, device: str) -> list:
    blob, s = _blob(im)
    data = torch.from_numpy(blob).permute(0, 3, 1, 2).to(device)
    info = torch.tensor([[blob.shape[1], blob.shape[2], s]], dtype=torch.float32, device=device)
    z = torch.zeros(1, 1, 5, device=device)
    rois, cls_prob, bbox_pred, _, _, _, _, _, extra = net(data, info, z, torch.zeros(1, dtype=torch.long, device=device), z)
    scores, boxes = cls_prob[0].cpu(), rois[0, :, 1:5].cpu()
    contact = torch.softmax(extra[0][0][0].cpu(), -1)               # (상자, 5) N S O P F
    side = (torch.sigmoid(extra[2][0][0].cpu()) > 0.5).float().view(-1)
    deltas = bbox_pred[0].cpu().view(-1, 4) * torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) + torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    pred = clip_boxes(bbox_transform_inv(boxes[None], deltas.view(1, -1, 4 * len(CLASSES)), 1), info.cpu(), 1)[0] / s
    out = {}
    for j, name in ((1, "obj"), (2, "hand")):
        keep_i = torch.nonzero(scores[:, j] > (HAND_T if name == "hand" else OBJ_T)).view(-1)
        if not keep_i.numel():
            out[name] = []
            continue
        bx, sc = pred[keep_i, j * 4:(j + 1) * 4], scores[keep_i, j]
        k = nms(bx, sc, cfg.TEST.NMS)
        out[name] = [(keep_i[i].item(), bx[i].tolist(), float(sc[i])) for i in k.tolist()]
    hands = []
    for i, b, sc in out["hand"]:
        p = contact[i].tolist()
        st = STATES[int(np.argmax(p))]
        ob = None
        if st in "PF" and out["obj"]:          # 손 가운데에서 가장 가까운 물체 상자
            c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            ob = min(out["obj"], key=lambda o: np.linalg.norm(c - [(o[1][0] + o[1][2]) / 2, (o[1][1] + o[1][3]) / 2]))[1]
        hands.append({"side": "R" if side[i] > 0 else "L", "score": round(sc, 3), "state": st,
                      "probs": {k: round(v, 3) for k, v in zip(STATES, p)}, "box": [round(v, 1) for v in b],
                      "obj_box": [round(v, 1) for v in ob] if ob else None})
    return hands


def run(net, video: Path, fps: float, device: str) -> dict:
    cap = cv2.VideoCapture(str(video))
    vfps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, k, step = [], 0, vfps / fps
    nxt = 0.0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k >= nxt - 1e-6:
            frames.append({"t": round(k / vfps, 3), "hands": detect(net, im, device)})
            nxt += step
        k += 1
    cap.release()
    return {"fps": fps, "detector": "100doh_handobj_100K+ego", "video_frames": n, "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="기본: 영상 옆 contact/ 폴더")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    net = load(a.device)
    for v in map(Path, a.videos):
        dst = Path(a.out or v.parent / "contact") / f"{v.stem}.json"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        r = run(net, v, a.fps, a.device)
        dst.write_text(json.dumps(r))
        fr = r["frames"]
        hand = np.mean([bool(f["hands"]) for f in fr]) if fr else 0
        port = np.mean([any(h["state"] == "P" for h in f["hands"]) for f in fr]) if fr else 0
        print(f"{v.stem}: 장면 {len(fr)} · {time.time() - t:.0f}초 · 손 {hand:.2f} · 물체 쥠(P) {port:.2f}", flush=True)


if __name__ == "__main__":
    main()
