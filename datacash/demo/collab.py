"""활용 예시 영상 — '사람을 이해하며 같이 나르는 휴머노이드' (시연 마지막 20초).

세 칸
  왼쪽   데이터셋의 가린 장면 (사람이 바구니·컵을 든다)
  가운데 MuJoCo: 사람 손(살색)과 휴머노이드 손(회색)이 함께 바구니를 든다. 바구니 높이·위치는 데이터의 물체 영역(SAM)을,
         휴머노이드 팔 굵기·색은 로봇이 맡은 비율을 따른다
  오른쪽 같은 시간축 그래프: 실제 심박 · 로봇이 예상한 심박 여유 상승 · 로봇 분담
로봇은 사람에게 조언하지 않는다 — 예상이 높으면 자기가 더 든다 (effort.robot_share).
지금은 합성 녹화로 만든 원리 시연이다. 워치로 찍은 실제 녹화가 쌓이면 같은 코드로 다시 만든다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from ..media import run
from .effort import FEATURES, WINDOW_S, robot_share

XML = """
<mujoco>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1" diffuse=".9 .9 .9"/>
    <geom type="plane" size="3 3 .1" rgba=".96 .95 .94 1"/>
    <body name="table" pos="0 0 .4"><geom type="box" size=".7 .4 .02" rgba=".85 .78 .7 1"/></body>
    <body name="human_torso" pos="-.9 0 .9"><geom type="capsule" fromto="0 0 0 0 0 .55" size=".16" rgba="1 .8 .7 1"/>
      <geom type="sphere" pos="0 0 .75" size=".13" rgba="1 .8 .7 1"/></body>
    <body name="robot_torso" pos=".9 0 .9"><geom type="capsule" fromto="0 0 0 0 0 .55" size=".16" rgba=".55 .58 .62 1"/>
      <geom type="box" pos="0 0 .76" size=".12 .1 .1" rgba=".55 .58 .62 1"/>
      <geom type="box" pos="-.12 0 .78" size=".01 .06 .02" rgba="1 .37 .37 1"/></body>
    <body name="basket" mocap="true" pos="0 0 .6"><geom type="box" size=".3 .22 .14" rgba="1 .69 .37 1"/></body>
    <body name="hand_h" mocap="true" pos="-.34 0 .7"><geom type="sphere" size=".06" rgba="1 .8 .7 1"/></body>
    <body name="hand_r" mocap="true" pos=".34 0 .7"><geom type="sphere" size=".06" rgba=".45 .48 .52 1"/></body>
    <body name="arm_h" mocap="true" pos="-.6 0 1.1"><geom type="capsule" fromto="0 0 0 .3 0 -.4" size=".05" rgba="1 .8 .7 1"/></body>
    <body name="arm_r" mocap="true" pos=".6 0 1.1"><geom name="arm_r_g" type="capsule" fromto="0 0 0 -.3 0 -.4" size=".05" rgba=".55 .58 .62 1"/></body>
  </worldbody>
</mujoco>
"""


def _chart(t, hr, pred, share, cur, size=(420, 480)):
    W, H = size
    img = np.full((H, W, 3), 255, np.uint8)
    lanes = [(40, 170, "heart rate (bpm)", (247, 92, 142)), (200, 320, "predicted HRR in 3s (%)", (240, 155, 46)),
             (350, 460, "robot share (%)", (95, 95, 255))]
    x = lambda i: int(30 + (W - 45) * i / max(1, len(t) - 1))  # noqa: E731
    for (y0, y1, label, color), arr, lo, hi in zip(lanes, (hr, pred, np.array(share) * 100), (55, -5, 45), (130, 40, 95)):
        cv2.putText(img, label, (30, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.rectangle(img, (30, y0), (W - 15, y1), (235, 235, 238), 1)
        pts = [(x(i), int(y1 - (min(hi, max(lo, v)) - lo) / (hi - lo) * (y1 - y0))) for i, v in enumerate(arr) if np.isfinite(v)]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
    cv2.line(img, (x(cur), 30), (x(cur), H - 10), (60, 60, 60), 1)
    cv2.putText(img, f"{t[cur]:.1f}s  share {share[cur] * 100:.0f}%", (34, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def render(dataset, model_json, out_mp4, episode: int = 0):
    import mujoco

    ds = Path(dataset)
    info = json.loads((ds / "meta" / "info.json").read_text())
    fps = info["fps"]
    t = pq.read_table(ds / "data" / "chunk-000" / "file-000.parquet")
    ep = np.array(t.column("episode_index").to_pylist())
    sel = np.nonzero(ep == episode)[0]
    touch = np.array(t.column("observation.touch").to_pylist())[sel]
    phys = np.array(t.column("observation.physio").to_pylist())[sel]
    objs = np.array(t.column("observation.objects").to_pylist())[sel]
    wrist = np.array(t.column("observation.imu.wrist").to_pylist())[sel]
    ts = np.array(t.column("timestamp").to_pylist())[sel]
    coef = np.array(json.loads(Path(model_json).read_text())["coef"])
    motion = np.abs(np.linalg.norm(wrist[:, :3], axis=1) - np.median(np.linalg.norm(wrist[:, :3], axis=1)))
    w = int(WINDOW_S * fps)
    pred, share = np.full(len(sel), np.nan), np.full(len(sel), 0.5)
    for i in range(w, len(sel)):
        g = touch[i - w:i, 2] * touch[i - w:i, 3]     # 쥐는 세기 점수(0~3) × 있음
        x = [g.mean(), float(np.mean(touch[i - w:i, 4] == 3)), motion[i - w:i].mean(), phys[i, 2], 1.0]
        pred[i] = phys[i, 2] + float(np.dot(coef, x))          # 몇 초 뒤 예상 HRR (지금 + 예상 상승)
        carrying = touch[i, 4] in (3, 4) or (touch[i, 3] > 0 and touch[i, 1] >= 2)
        share[i] = robot_share(pred[i]) if carrying else 0.5
    share = np.convolve(share, np.ones(9) / 9, mode="same")

    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    rnd = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, .8]
    cam.distance, cam.azimuth, cam.elevation = 3.0, 90, -15
    ids = {n: m.body(n).mocapid[0] for n in ("basket", "hand_h", "hand_r", "arm_h", "arm_r")}
    vid = cv2.VideoCapture(str(ds / "videos" / "observation.images.ego" / "chunk-000" / "file-000.mp4"))
    vid.set(cv2.CAP_PROP_POS_FRAMES, int(sel[0]))
    tmp = Path(out_mp4).with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (480 + 640 + 420, 480))
    base_y = None
    for k in range(len(sel)):
        ok, frame = vid.read()
        if not ok:
            break
        o = objs[k]
        cy = (o[1] + o[3]) / 2 if o[4] > 0 else None
        cx = (o[0] + o[2]) / 2 if o[4] > 0 else 0.5
        if cy is not None and base_y is None:
            base_y = cy
        lift = 0.0 if cy is None or base_y is None else float(np.clip((base_y - cy) * 3.0, 0, 0.6))
        z = 0.6 + lift
        xb = float(np.clip((cx - 0.5) * 0.6, -0.3, 0.3))
        d.mocap_pos[ids["basket"]] = [xb, 0, z]
        d.mocap_pos[ids["hand_h"]] = [xb - 0.34, 0, z + 0.12]
        d.mocap_pos[ids["hand_r"]] = [xb + 0.34, 0, z + 0.12]
        d.mocap_pos[ids["arm_h"]] = [xb - 0.64, 0, z + 0.52]
        d.mocap_pos[ids["arm_r"]] = [xb + 0.64, 0, z + 0.52]
        m.geom("arm_r_g").size[0] = 0.04 + 0.06 * (share[k] - 0.5) / 0.4      # 더 맡을수록 팔이 굵어진다
        m.geom("arm_r_g").rgba[:] = [0.55 + 0.45 * (share[k] - 0.5) / 0.4, 0.58, 0.62, 1]
        mujoco.mj_forward(m, d)
        rnd.update_scene(d, cam)
        sim = cv2.cvtColor(rnd.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(sim, f"robot carries {share[k] * 100:.0f}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 230), 2, cv2.LINE_AA)
        left = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
        pad = np.full((480, 480, 3), 30, np.uint8)
        y0 = (480 - left.shape[0]) // 2
        pad[max(0, y0):max(0, y0) + min(480, left.shape[0])] = left[:480]
        vw.write(np.hstack([pad, sim, _chart(ts, phys[:, 0], pred, share, k)]))
    vw.release()
    run(["-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4])
    tmp.unlink()
    return {"frames": len(sel), "max_share": float(np.nanmax(share)), "features": FEATURES}
