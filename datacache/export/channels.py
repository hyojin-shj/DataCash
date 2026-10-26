"""장면 격자(폰 시계, 일정한 fps) 위에 4감 채널을 모은다 — LeRobot 내보내기의 재료.

채널 (이름 = LeRobot feature 키)
  observation.hands            126  왼손·오른손 × 21점 × (x/너비, y/높이, z)
  observation.hands_present      2  1 검출, 0.5 짧은 끊김 보간, 0 없음
  observation.body              99  33점 × (x/너비, y/높이, 보임)
  observation.body_world        99  33점 × (x, y, z) 미터, 골반 가운데 원점
  observation.imu.phone          6  폰 가속(m/s²)·자이로(rad/s)
  observation.imu.wrist          6  손목 가속·자이로 (워치 → 폰 시계로 옮김)
  observation.audio              3  소리 세기(dBFS) · 소리 사건 시작 · 확인된 접촉음(손목 충격과 겹침)
  observation.touch              5  접촉(0~1) · 쥐는 힘(N) · 쥐는 힘 있음 · 접촉 단계(0~5) · 물체 칸
  observation.touch.vibration    4  능동 진동 반응: 울림 크기 · 사그라드는 시간(ms) · 에너지 · 있음
  observation.physio             6  심박(bpm) · 있음 · HRR(%) · 있음 · 피부온도 변화(°C) · 있음
  observation.objects           20  물체 4칸 × (x1, y1, x2, y2 비율, 있음)  — SAM 3.1
  annotation.events              4  '실수했어요' · 확인된 접촉 · 단계 경계 · 라벨 검증됨
  annotation.step                1  단계 번호 (에피소드 정보의 steps 문장 목록, -1 없음)
  annotation.primitive           1  조작 기본 동작 번호 (PRIMITIVES: reach·grasp·lift·pull·…·release, -1 없음)
  sync.error_ms                  4  감각별 동기화 오차 (영상 · 소리 · 손목·촉각 · 생체)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..touch.contact import PHASES
from ..understand.spark import PRIMITIVES

POSE_JOINTS = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
               "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
               "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
               "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
               "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
HAND_JOINTS = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip", "index_tip",
               "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
               "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
N_OBJ = 4

NAMES: Dict[str, List[str]] = {
    "observation.hands": [f"{s}_{j}_{c}" for s in ("left", "right") for j in HAND_JOINTS for c in ("x", "y", "z")],
    "observation.hands_present": ["left", "right"],
    "observation.body": [f"{j}_{c}" for j in POSE_JOINTS for c in ("x", "y", "vis")],
    "observation.body_world": [f"{j}_{c}" for j in POSE_JOINTS for c in ("x", "y", "z")],
    "observation.imu.phone": ["ax", "ay", "az", "gx", "gy", "gz"],
    "observation.imu.wrist": ["ax", "ay", "az", "gx", "gy", "gz"],
    "observation.audio": ["level_dbfs", "onset", "contact_sound"],
    "observation.touch": ["contact", "grip_force_n", "grip_valid", "phase", "object_slot"],
    "observation.touch.vibration": ["amp", "decay_ms", "energy", "valid"],
    "observation.physio": ["hr_bpm", "hr_valid", "hrr_pct", "hrr_valid", "skin_delta_c", "skin_valid"],
    "observation.objects": [f"obj{k}_{c}" for k in range(N_OBJ) for c in ("x1", "y1", "x2", "y2", "present")],
    "annotation.events": ["self_mark", "contact_confirmed", "step_boundary", "label_verified"],
    "annotation.step": ["step_index"],
    "annotation.primitive": ["primitive_index"],
    "sync.error_ms": ["video", "audio", "wrist_touch", "physio"],
}

# 등급별로 빼는 채널 (기본: 폰만으로 가능한 것, 고품질: + 손목, 생체: 전부)
LEVEL_DROP = {
    "basic": {"observation.imu.wrist", "observation.touch.vibration", "observation.physio"},
    "premium": {"observation.physio"},
    "bio": set(),
}


@dataclass
class EpisodeChannels:
    task: str
    grid_ns: np.ndarray                     # (n,) 폰 시계
    src_frame: np.ndarray                   # (n,) 원본 영상 프레임 번호
    data: Dict[str, np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)   # 에피소드 정보 (steps, outcome, human_state, sync, touch 보정 ...)


def make_grid(frame_ns: np.ndarray, start_ns: int, end_ns: int, fps: float):
    n = int(np.floor((end_ns - start_ns) / 1e9 * fps)) + 1
    grid = (start_ns + np.arange(n) * (1e9 / fps)).astype(np.int64)
    idx = np.clip(np.searchsorted(frame_ns, grid), 1, len(frame_ns) - 1)
    left = frame_ns[idx - 1]
    src = np.where(np.abs(grid - left) <= np.abs(frame_ns[idx] - grid), idx - 1, idx)
    return grid, src


def interp(grid: np.ndarray, t_ns: np.ndarray, v: np.ndarray, max_gap_s: float = 0.25) -> tuple:
    """선형 보간 + 가장 가까운 샘플이 max_gap 보다 멀면 없음."""
    if t_ns is None or len(t_ns) == 0:
        return np.zeros((len(grid), v.shape[1] if v is not None and v.ndim > 1 else 1), np.float32), np.zeros(len(grid), bool)
    x = (t_ns - grid[0]).astype(np.float64)
    g = (grid - grid[0]).astype(np.float64)
    out = np.column_stack([np.interp(g, x, v[:, k]) for k in range(v.shape[1])]).astype(np.float32)
    j = np.clip(np.searchsorted(t_ns, grid), 1, len(t_ns) - 1)
    gap = np.minimum(np.abs(grid - t_ns[j - 1]), np.abs(t_ns[j] - grid)) / 1e9
    ok = (gap <= max_gap_s) & (grid >= t_ns[0] - max_gap_s * 1e9) & (grid <= t_ns[-1] + max_gap_s * 1e9)
    return out * ok[:, None], ok


def hand_rows(labels, src, size) -> tuple:
    w, h = size
    H = np.zeros((len(src), 126), np.float32)
    P = np.zeros((len(src), 2), np.float32)
    for r, i in enumerate(src):
        if labels is None or i >= len(labels):
            continue
        f = labels[i]
        for k in (0, 1):
            if f.hands[k] is not None:
                a = np.asarray(f.hands[k], np.float32).copy()
                a[:, 0] /= w
                a[:, 1] /= h
                H[r, k * 63:(k + 1) * 63] = a.ravel()
                P[r, k] = f.hands_state[k]
    return H, P


def body_rows(labels, src, size) -> tuple:
    w, h = size
    B = np.zeros((len(src), 99), np.float32)
    BW = np.zeros((len(src), 99), np.float32)
    for r, i in enumerate(src):
        if labels is None or i >= len(labels) or labels[i].body is None:
            continue
        a = np.asarray(labels[i].body, np.float32).copy()
        a[:, 0] /= w
        a[:, 1] /= h
        B[r] = a.ravel()
        if labels[i].body_world is not None:
            BW[r] = np.asarray(labels[i].body_world, np.float32).ravel()
    return B, BW


def object_rows(objects, src, size) -> np.ndarray:
    w, h = size
    O = np.zeros((len(src), N_OBJ * 5), np.float32)
    for r, i in enumerate(src):
        if objects is None or i >= len(objects):
            continue
        for k, b in list(objects[i].items())[:N_OBJ]:
            if 0 <= k < N_OBJ:
                O[r, k * 5:(k + 1) * 5] = [b[0] / w, b[1] / h, b[2] / w, b[3] / h, 1.0]
    return O


def touch_rows(contact_frames, src, grid, grip_events, vib_events) -> tuple:
    """grip_events: [(폰 ns, 힘 N)] 펄스마다. 0.6초 안에 펄스가 있었을 때만 쥐는 힘 있음."""
    T = np.zeros((len(src), 5), np.float32)
    V = np.zeros((len(src), 4), np.float32)
    T[:, 4] = -1
    for r, i in enumerate(src):
        if contact_frames is not None and i < len(contact_frames):
            cf = contact_frames[i]
            T[r, 0], T[r, 3], T[r, 4] = cf.contact, cf.phase, cf.object
    if grip_events:
        gt = np.array([g[0] for g in grip_events], np.int64)
        gv = np.array([g[1] for g in grip_events], np.float32)
        j = np.searchsorted(gt, grid, side="right") - 1
        ok = (j >= 0) & ((grid - gt[np.clip(j, 0, None)]) <= 0.6e9)
        T[ok, 1] = gv[j[ok]]
        T[ok, 2] = 1.0
    if vib_events:
        vt = np.array([v[0] for v in vib_events], np.int64)
        vv = np.array([v[1:] for v in vib_events], np.float32)
        j = np.searchsorted(vt, grid, side="right") - 1
        ok = (j >= 0) & ((grid - vt[np.clip(j, 0, None)]) <= 0.6e9)
        V[ok, :3] = vv[j[ok]]
        V[ok, 3] = 1.0
    return T, V


def audio_rows(audio, grid, onsets, confirmed, fps) -> np.ndarray:
    A = np.zeros((len(grid), 3), np.float32)
    A[:, 0] = -120.0
    if audio is not None:
        from scipy.signal import butter, sosfiltfilt

        # 들리는 소리 세기만 (12kHz 아래) — 사람 귀에 안 들리는 음향 비콘(18~20kHz)은 빼고 잰다
        x = sosfiltfilt(butter(4, min(12000, audio.rate / 2 * 0.9), btype="low", fs=audio.rate, output="sos"), audio.x)
        half = int(audio.rate / fps / 2)
        pos = ((grid - audio.start_ns) / 1e9 * audio.rate).astype(np.int64)
        for r, p in enumerate(pos):
            a, b = max(0, p - half), min(len(x), p + half)
            if b > a:
                A[r, 0] = 20 * np.log10(np.sqrt(np.mean(x[a:b] ** 2)) + 1e-6)
    step = 1e9 / fps
    for e in onsets or []:
        r = int(np.searchsorted(grid, e["t_ns"] - step / 2))
        if 0 <= r < len(grid):
            A[r, 1] = 1.0
    for e in confirmed or []:
        r = int(np.searchsorted(grid, e["t_ns"] - step / 2))
        if 0 <= r < len(grid):
            A[r, 2] = 1.0
    return A


def event_rows(grid, fps, marks_ns, confirmed, steps, verified_steps) -> tuple:
    E = np.zeros((len(grid), 4), np.float32)
    S = np.full((len(grid), 1), -1, np.float32)
    P = np.full((len(grid), 1), -1, np.float32)
    step = 1e9 / fps
    for t in marks_ns or []:
        r = int(np.searchsorted(grid, t - step / 2))
        if 0 <= r < len(grid):
            E[r, 0] = 1.0
    for e in confirmed or []:
        r = int(np.searchsorted(grid, e["t_ns"] - step / 2))
        if 0 <= r < len(grid):
            E[r, 1] = 1.0
    t_s = (grid - grid[0]) / 1e9
    for k, st in enumerate(steps or []):
        sel = (t_s >= st["start_s"]) & (t_s < st["end_s"])
        S[sel, 0] = k
        if st.get("action") in PRIMITIVES:
            P[sel, 0] = PRIMITIVES.index(st["action"])
        r = int(np.searchsorted(t_s, st["start_s"]))
        if 0 <= r < len(grid):
            E[r, 2] = 1.0
        if verified_steps and k < len(verified_steps) and verified_steps[k]:
            E[sel, 3] = 1.0
    return E, S, P


def phase_name(p: float) -> str:
    return PHASES[int(p)] if 0 <= int(p) < len(PHASES) else "none"
