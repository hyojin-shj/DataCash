"""본인 얼굴 확인 · 동의 인원 · 이의 신청 (서버, Meta API 없이). 얼굴은 저장소의 가상 인물 사진으로."""

import base64
import json

import cv2
import numpy as np
import pytest

from datacash import config
from datacash import people as people_mod

DONORS = config.ROOT / "datacash" / "donors"
pytestmark = pytest.mark.skipif(not (config.MODELS / "yunet.onnx").exists(), reason="얼굴 모델이 없음 (data/models)")
DEV = {"X-DC-Token": "t", "X-DC-Device": "phoneA1"}


def jpg(name: str) -> bytes:
    return cv2.imencode(".jpg", cv2.imread(str(DONORS / name)))[1].tobytes()


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from datacash.server.app import create_app

    return TestClient(create_app(tmp_path, "t", use_api=False)), tmp_path


def b64(name: str) -> str:
    return base64.b64encode(jpg(name)).decode()


def test_people_consent_face_donor(client):
    c, root = client
    assert c.post("/api/face/check", content=jpg("donor0.png"), headers=DEV).json()["registered"] is False
    me = {"name": "나", "relation": "본인", "adult": True, "age_band": "30대", "agreed": True, "bio": True}
    assert c.post("/api/people", json={**me, "adult": False}, headers=DEV).status_code == 400        # 아이·미성년 불가
    assert c.post("/api/people", json={**me, "agreed": False}, headers=DEV).status_code == 400
    p_me = c.post("/api/people", json=me, headers=DEV).json()
    assert p_me["face"] == "none" and p_me["bio_consent"]["agreed"]
    assert c.post("/api/people", json=me, headers=DEV).status_code == 400                          # 본인은 하나
    mom = {"name": "엄마", "relation": "가족", "adult": True, "age_band": "50대", "agreed": True, "bio": True}
    assert c.post("/api/people", json=mom, headers=DEV).status_code == 400                         # 생체는 본인만
    p_mom = c.post("/api/people", json={**mom, "bio": False}, headers=DEV).json()

    blank = base64.b64encode(cv2.imencode(".jpg", np.full((300, 300, 3), 200, np.uint8))[1].tobytes()).decode()
    bad = c.post(f"/api/people/{p_me['id']}/face", json={"photos_b64": [blank, blank, b64("donor0.png")]}, headers=DEV).json()
    assert bad["face"] == "failed" and bad["face_error"]
    ok = c.post(f"/api/people/{p_me['id']}/face", json={"photos_b64": [b64("donor0.png")] * 3}, headers=DEV).json()
    assert ok["face"] == "enrolled"
    same = c.post("/api/face/check", content=jpg("donor0.png"), headers=DEV).json()
    other = c.post("/api/face/check", content=jpg("donor5.png"), headers=DEV).json()
    assert same["match"] and not other["match"]

    assert c.post(f"/api/people/{p_mom['id']}/donor", json={"donor": "nope"}, headers=DEV).status_code == 400
    assert c.post(f"/api/people/{p_mom['id']}/donor", json={"donor": "donor3"}, headers=DEV).json()["donor"] == "donor3"
    assert c.post(f"/api/people/{p_me['id']}/consent", json={"bio": False}, headers=DEV).json()["bio_consent"]["agreed"] is False
    lst = c.get("/api/people", headers=DEV).json()
    assert lst["owner_registered"] and [x["name"] for x in lst["people"]] == ["나", "엄마"] and len(lst["donors"]) == len(list(people_mod.DONORS_DIR.glob("donor*.png"))) >= 30
    assert "embedding" not in json.dumps(lst)                      # 특징값은 밖으로 안 나간다
    assert not list(root.rglob("*.jpg")) and not list(root.rglob("*.png"))   # 사진은 저장하지 않는다
    assert c.get("/api/people", headers={**DEV, "X-DC-Device": "phoneB2"}).json()["people"] == []
    assert c.get("/api/donors/donor3.png").status_code == 200 and c.get("/api/donors/..%2Fx.png").status_code == 404
    assert c.delete(f"/api/people/{p_mom['id']}", headers=DEV).status_code == 200
    assert [x["name"] for x in c.get("/api/people", headers=DEV).json()["people"]] == ["나"]


def test_tracks_identified_owner_family_stranger(tmp_path):
    """한 영상의 세 사람 추적: 본인 → 참여자, 동의한 가족 → 가리지 않음, 모르는 사람 → 알아보지 못함(가림)."""
    from datacash.meta.sam import SamObject
    from datacash.people import People, embed_image
    from datacash.privacy.mask import identify_tracks

    ppl = People(tmp_path)
    face = {n: cv2.resize(cv2.imread(str(DONORS / f"donor{n}.png")), (240, 240)) for n in (0, 2, 5)}
    jp = {n: cv2.imencode(".jpg", face[n])[1].tobytes() for n in face}
    me = ppl.add("phoneA1", "나", "본인", True, "30대", True)
    mom = ppl.add("phoneA1", "엄마", "가족", True, "50대", True)
    ppl.enroll("phoneA1", me["id"], [jp[0]] * 3)
    ppl.enroll("phoneA1", mom["id"], [jp[2]] * 3)
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
    assert got["0"][0] == "owner" and got["1"][0] == "consented" and got["1"][2] == mom["id"] and "2" not in got


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


def test_payout_waits_for_account_then_sends(client):
    c, root = client
    st = c.app.state.store
    _session(root, "s_pay", "phoneA1", {"state": "done", "minutes": 1.0, "level": "bio", "reward": 830})
    assert st.settle("s_pay")["payout"] == {"status": "waiting_account", "amount": 830}
    assert c.get("/api/payout/account", headers=DEV).json() == {"label": "", "total": 0, "waiting": 830}
    assert c.post("/api/payout/account", json={"label": "KB국민 1234567890"}, headers=DEV).status_code == 400   # 번호 전체는 안 받음
    r = c.post("/api/payout/account", json={"label": "KB국민 ****7890"}, headers=DEV).json()
    assert r["sent_now"] == 1 and r["total"] == 830 and r["waiting"] == 0
    p = c.get("/api/sessions/s_pay/status", headers=DEV).json()["payout"]
    assert p["status"] == "sent" and p["label"] == "KB국민 ****7890"
    _session(root, "s_low", "phoneA1", {"state": "done", "minutes": 1.0, "level": "bio", "reward": 300, "today_bonus": True})
    st.settle("s_low")
    c.post("/api/sessions/s_low/appeal", json={"reason": "reward"}, headers=DEV)            # 자동 인정 +950원
    assert c.get("/api/sessions/s_low/status", headers=DEV).json()["payout"]["amount"] == 1250
