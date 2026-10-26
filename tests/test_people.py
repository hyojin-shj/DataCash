"""본인 얼굴 확인 · 동의 인원 · 이의 신청 (서버, Meta API 없이). 얼굴은 저장소의 가상 인물 사진으로."""

import base64
import json

import cv2
import numpy as np
import pytest

from datacache import config

DONORS = config.ROOT / "datacache" / "donors"
pytestmark = pytest.mark.skipif(not (config.MODELS / "yunet.onnx").exists(), reason="얼굴 모델이 없음 (data/models)")
DEV = {"X-DC-Token": "t", "X-DC-Device": "phoneA1"}


def jpg(name: str) -> bytes:
    return cv2.imencode(".jpg", cv2.imread(str(DONORS / name)))[1].tobytes()


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from datacache.server.app import create_app

    return TestClient(create_app(tmp_path, "t", use_api=False)), tmp_path


def test_owner_face_and_people(client):
    c, root = client
    assert c.post("/api/face/check", content=jpg("donor0.png"), headers=DEV).json()["registered"] is False
    assert c.post("/api/face/owner", content=jpg("donor0.png"), headers=DEV).json()["registered"]
    same = c.post("/api/face/check", content=jpg("donor0.png"), headers=DEV).json()
    other = c.post("/api/face/check", content=jpg("donor5.png"), headers=DEV).json()
    assert same["match"] and not other["match"]
    blank = cv2.imencode(".jpg", np.full((300, 300, 3), 200, np.uint8))[1].tobytes()
    assert c.post("/api/face/owner", content=blank, headers=DEV).status_code == 422

    body = {"name": "엄마", "relation": "가족", "consent_video": True, "consent_sale": True,
            "photo_b64": base64.b64encode(jpg("donor2.png")).decode()}
    assert c.post("/api/people", json={**body, "consent_sale": False}, headers=DEV).status_code == 400
    p = c.post("/api/people", json=body, headers=DEV).json()
    lst = c.get("/api/people", headers=DEV).json()
    assert lst["owner_registered"] and [x["name"] for x in lst["people"]] == ["엄마"]
    assert "embedding" not in json.dumps(lst)                      # 특징값은 밖으로 안 나간다
    assert not list(root.rglob("*.jpg")) and not list(root.rglob("*.png"))   # 사진은 저장하지 않는다
    assert c.get("/api/people", headers={**DEV, "X-DC-Device": "phoneB2"}).json()["people"] == []   # 다른 폰은 못 본다
    assert c.delete(f"/api/people/{p['id']}", headers=DEV).status_code == 200
    assert c.get("/api/people", headers=DEV).json()["people"] == []


def test_tracks_identified_owner_family_stranger(tmp_path):
    """한 영상의 세 사람 추적: 본인 → 참여자, 동의한 가족 → 가리지 않음, 모르는 사람 → 알아보지 못함(가림)."""
    from datacache.meta.sam import SamObject
    from datacache.people import People, embed_image
    from datacache.privacy.mask import identify_tracks

    ppl = People(tmp_path)
    face = {n: cv2.resize(cv2.imread(str(DONORS / f"donor{n}.png")), (240, 240)) for n in (0, 2, 5)}
    ppl.set_owner("phoneA1", embed_image(face[0])[0])
    ppl.add("phoneA1", "엄마", "가족", True, True, embed_image(face[2])[0])
    video = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 540))
    tracks = {}
    for fi in range(12):
        f = np.full((540, 960, 3), 230, np.uint8)
        objs = []
        for k, (n, x) in enumerate(((0, 40), (2, 360), (5, 680))):
            y = 60 + fi * 3
            f[y:y + 240, x:x + 240] = face[n]
            m = np.zeros((540, 960), bool)
            m[y:y + 480, x:x + 240] = True
            objs.append(SamObject(str(k), (x, y, x + 240, y + 480), 0.9, m))
        tracks[fi] = objs
        vw.write(f)
    vw.release()
    got = identify_tracks(video, tracks, ppl.matcher("phoneA1"))
    assert got["0"][0] == "owner" and got["1"][0] == "consented" and "2" not in got


def _session(root, sid, device, status):
    d = root / "sessions" / sid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"devices": {"phone": {"id": device}}, "mission": {"id": "pull_tissue"}}))
    (root / "status").mkdir(exist_ok=True)
    (root / "status" / f"{sid}.json").write_text(json.dumps(status))


def test_appeals_auto_then_review(client):
    c, root = client
    _session(root, "s_reward", "phoneA1", {"state": "done", "minutes": 1.0, "level": "bio", "reward": 300, "today_bonus": True})
    a = c.post("/api/sessions/s_reward/appeal", json={"reason": "reward", "detail": ""}, headers=DEV).json()
    assert a["state"] == "accepted" and a["extra_reward"] == 1250 - 300          # 50,000/60 × 1.5 = 1,250원
    assert c.post("/api/sessions/s_reward/appeal", json={"reason": "reward"},
                  headers={**DEV, "X-DC-Device": "phoneB2"}).status_code == 404    # 남의 녹화엔 못 한다

    _session(root, "s_priv", "phoneA1", {"state": "done", "minutes": 1.0, "level": "basic", "reward": 330})
    p = c.post("/api/sessions/s_priv/appeal", json={"reason": "privacy", "detail": "거울에 동생이 보여요"}, headers=DEV).json()
    assert p["state"] == "review" and json.loads((root / "status" / "s_priv.json").read_text())["withheld"]
    assert c.post("/api/sessions/s_priv/appeal", json={"reason": "privacy"}, headers=DEV).status_code == 400   # 진행 중
    d = c.post(f"/api/admin/appeals/{p['id']}?token=t", json={"accept": True, "extra_reward": 0, "note": "다시 가림"}).json()
    assert d["state"] == "accepted" and [s["stage"] for s in d["steps"]] == ["filed", "auto", "result"]
    assert json.loads((root / "status" / "s_priv.json").read_text())["withheld"]   # 인정 → 판매에서 계속 빠짐
    mine = c.get("/api/appeals", headers=DEV).json()["appeals"]
    assert {x["sid"] for x in mine} == {"s_reward", "s_priv"}
