"""SAM 비용 줄이기: 초당 10장면만 보내고 사이 장면은 가까운 장면의 마스크로, 같은 영상·명사구는 저장된 응답으로."""

import cv2
import numpy as np

from datacache.meta.sam import Sam, SamObject
from datacache.privacy.mask import GRAY, mask_video


def _video(path, n=30, fps=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(n):
        f = np.full((120, 160, 3), 200, np.uint8)
        cv2.rectangle(f, (10 + i, 40), (40 + i, 80), (0, 0, 255), -1)
        vw.write(f)
    vw.release()


class FakeSam:
    def __init__(self):
        self.calls, self.frames = [], []
        self.c = self

    def upload(self, p):
        return "fid"

    def video(self, path, concept, file_id=None):
        n = int(cv2.VideoCapture(str(path)).get(7))
        self.calls.append(concept)
        self.frames.append(n)
        if concept != "computer monitor":
            return {}
        out = {}
        for k in range(n):
            m = np.zeros((120, 160), bool)
            m[40:80, 10 + 3 * k:40 + 3 * k] = True
            out[k] = [SamObject("1", (10 + 3 * k, 40, 40 + 3 * k, 80), 0.9, m)]
        return out


def test_thin_and_spread(tmp_path):
    v = tmp_path / "v.mp4"
    _video(v)
    sam = FakeSam()
    r = mask_video(v, tmp_path / "m.mp4", sam=sam)
    assert r.report["sam_every"] == 3
    assert max(sam.frames) == 10                       # 30장면 중 10장면만 SAM 에
    cap = cv2.VideoCapture(str(tmp_path / "m.mp4"))
    for i in range(30):
        f = cap.read()[1]
        if i in (1, 2, 16, 29):                         # SAM 이 안 본 장면도 빨간 상자가 가려진다
            red = (f[:, :, 2] > 150) & (f[:, :, 1] < 100)
            assert red.sum() < 20, i


def test_cache(tmp_path, monkeypatch):
    v = tmp_path / "v.mp4"
    _video(v, n=5)
    sent = []

    class C:
        def upload(self, p):
            sent.append("up")
            return "fid"

        def responses(self, *a, **k):
            sent.append("api")
            return type("R", (), {"output_text": ""})()

    s = Sam(client=C(), cache_dir=tmp_path / "cache")
    s.video(v, "person", file_id=lambda: s.c.upload(v))
    s.video(v, "person", file_id=lambda: s.c.upload(v))
    assert sent == ["up", "api"]
