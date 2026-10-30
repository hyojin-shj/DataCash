"""SAM 비용 줄이기: 초당 10장면만 보내고 사이 장면은 가까운 장면의 마스크로, 같은 영상·명사구는 저장된 응답으로."""

import cv2
import numpy as np

from datacash.meta.sam import Sam, SamObject
from datacash.privacy.mask import GRAY, mask_video


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
    assert r.report["sam_plan"]["computer monitor"] == 10
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


def test_plan_skips_and_slows(tmp_path):
    """혼자 찍은 영상: 사람은 SAM 에 묻지 않고, 화면은 초당 1장면."""
    from datacash.privacy.scan import sam_frames, sam_plan

    v = tmp_path / "v.mp4"
    _video(v, n=60)
    plan = sam_plan({"frames": 4, "max_people": 1, "seen": {"tv": 2}}, sensitive=("computer monitor", "mirror"),
                    manipulated=("cup",))
    assert plan["person"] == 0 and plan["child"] == 0 and plan["computer monitor"] == 1.0 and plan["cup"] == 2.0
    sam = FakeSam()
    r = mask_video(v, tmp_path / "m.mp4", sam=sam, plan=plan, concepts_sensitive=("computer monitor", "mirror"),
                   manipulated=("cup",))
    assert "person" not in sam.calls and r.report["sam_plan"]["person"] == 0
    assert max(sam.frames) <= 4                        # 60장면(2초) 중 초당 1~2장면만
    assert sam_frames(plan, 30.0, 3600) == 120 + 120 + 240  # 2분: 화면·거울 초당 1장면 + 컵 초당 2장면
    assert sam_plan(None)["person"] == 10.0          # 훑지 못하면 예전처럼 다 묻는다


def test_fast_mask_decode_same_as_official():
    """Numba 해독기 = 공식 meta_sam_parser 해독기 (비트 단위)."""
    from types import SimpleNamespace

    from meta_sam_parser._mask_codec import _encode_raster, decode_mask_to_raster

    from datacash.meta.mask_fast import decode

    rng = np.random.default_rng(0)
    for w, h in ((37, 23), (120, 80)):
        img = np.zeros((h, w), np.uint8)
        cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 4), 20, 0, 360, 1, -1)
        img[rng.random((h, w)) < 0.02] ^= 1
        m = SimpleNamespace(encoding="one_bit", width=w, height=h, payload=_encode_raster(img.ravel().tolist(), w, h))
        official = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(h, w)
        assert np.array_equal(decode(m), official) and np.array_equal(official, img)


def test_object_name_fallback(tmp_path):
    """미션의 물체 이름(cup)으로 0장면이면 다른 이름들을 차례로 — 찾은 이름의 상자가 물체 채널에 들어간다."""
    v = tmp_path / "v.mp4"
    _video(v, n=30)

    class Pitcher(FakeSam):
        def video(self, path, concept, file_id=None):
            out = FakeSam.video(self, path, "computer monitor" if concept == "pitcher" else "-", file_id)
            self.calls[-1] = concept
            return out

    sam = Pitcher()
    r = mask_video(v, tmp_path / "m.mp4", sam=sam, manipulated=("cup",), concepts_people=(), concepts_child=(),
                   concepts_sensitive=(), name_objects=lambda: ["metal jug", "pitcher", "package"])
    assert sam.calls == ["cup", "metal jug", "pitcher"]            # 찾으면 멈춘다
    assert r.report["object_fallback"]["found"] == "pitcher"
    assert sum(1 for o in r.objects if o) == 30
