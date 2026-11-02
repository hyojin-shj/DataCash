"""모으기만 모드: 올라온 녹화를 처리하지 않고 미션별로 보관 → 나중에 한꺼번에 처리 줄에 넣는다."""

import json


def _upload(c, sid, mission):
    h = {"X-DC-Token": "t"}
    c.put(f"/api/sessions/{sid}/files/meta.json", content=json.dumps({"session_id": sid, "mission": {"id": mission}}), headers=h)
    c.put(f"/api/sessions/{sid}/files/video.mp4", content=b"\x00" * 16, headers=h)
    return c.post(f"/api/sessions/{sid}/process", headers=h).json()


def test_collect_keeps_by_mission_then_processes(tmp_path):
    from fastapi.testclient import TestClient

    from datacash.server.app import create_app

    c = TestClient(create_app(tmp_path, "t", use_api=False, collect=True))
    assert _upload(c, "20261029_200000", "pour_water")["state"] == "collected"
    assert _upload(c, "20261029_200100", "pack_box")["state"] == "collected"
    link = tmp_path / "collected" / "pour_water" / "20261029_200000"
    assert link.is_symlink() and (link / "meta.json").exists()          # 원본은 sessions/ 그대로, 미션별 목록은 링크
    st = c.app.state.store
    assert st.collected() == [("pack_box", "20261029_200100"), ("pour_water", "20261029_200000")]

    app2 = create_app(tmp_path, "t", use_api=False, process_collected=True)
    assert app2.state.store.status("20261029_200000")["state"] in ("queued", "processing", "failed", "rejected", "done")
    assert app2.state.store.collected() == [] or app2.state.store.status("20261029_200000")["state"] != "collected"


def test_late_watch_upload_keeps_collected(tmp_path):
    """폰이 보관됨이 된 뒤 워치 파일이 늦게 올라와도 보관됨 그대로 (한꺼번에 처리에서 빠지지 않게)."""
    from fastapi.testclient import TestClient

    from datacash.server.app import create_app

    c = TestClient(create_app(tmp_path, "t", use_api=False, collect=True))
    assert _upload(c, "20261029_210000", "twist_cap")["state"] == "collected"
    c.put("/api/sessions/20261029_210000/files/watch_imu.csv", content=b"t_ns\n", headers={"X-DC-Token": "t"})
    assert c.app.state.store.status("20261029_210000")["state"] == "collected"


def test_app_token_cannot_open_admin(tmp_path, monkeypatch):
    """나눠 주는 앱의 참여자 토큰으로는 미션은 받지만 운영자 화면 API 는 못 연다."""
    from fastapi.testclient import TestClient

    from datacash.server.app import create_app

    monkeypatch.setenv("DATACASH_APP_TOKEN", "app")
    c = TestClient(create_app(tmp_path, "admin", use_api=False))
    assert c.get("/api/missions", headers={"X-DC-Token": "app"}).status_code == 200
    assert c.get("/api/admin/sessions?token=app").status_code == 401
    assert c.get("/api/admin/sessions?token=admin").status_code == 200


def test_missions_localized_by_lang(tmp_path):
    """앱은 X-DC-Lang, 웹 카탈로그는 ?lang= 으로 미션 제목·안내를 그 언어로. id 는 그대로, 모르는 언어는 한국어 원문."""
    from fastapi.testclient import TestClient

    from datacash.server.app import create_app, localize_mission

    m = {"id": "x", "title": "티슈 뽑기", "title_en": "Pull a tissue", "guide": "가이드", "purpose": "쓰임",
         "i18n": {"ja": {"title": "ティッシュ", "guide": "ガイド", "purpose": "用途"}}}
    assert localize_mission(m, "ja-JP")["title"] == "ティッシュ"
    assert localize_mission(m, "en")["title"] == "Pull a tissue" and localize_mission(m, "en")["guide"] == "가이드"
    assert localize_mission(m, "ko") is m and localize_mission(m, "xx")["title"] == "티슈 뽑기"
    assert m["title"] == "티슈 뽑기"

    c = TestClient(create_app(tmp_path, "t", use_api=False))
    ko = c.get("/api/missions", headers={"X-DC-Token": "t"}).json()["missions"]
    ja = c.get("/api/missions", headers={"X-DC-Token": "t", "X-DC-Lang": "ja"}).json()["missions"]
    assert [x["id"] for x in ko] == [x["id"] for x in ja]
    assert ko[0]["title"] == "티슈 뽑기" and ja[0]["title"] == "ティッシュを引き出す"
    es = {x["id"]: x for x in c.get("/api/market/catalog?lang=es").json()["missions"]}
    assert es["pull_tissue"]["title"] == "Sacar un pañuelo" and es["pull_tissue"]["title_en"] == "Pull a tissue"
    zh = {x["id"]: x for x in c.get("/api/market/catalog?lang=zh-CN").json()["missions"]}
    assert zh["lift_box"]["guide"].startswith("抬起箱子")


def test_rejection_reason_in_app_language():
    from datacash.server.app import localize_error

    e = "동작 부분이 너무 짧아요 (1.2초 · 3초 이상 찍어 주세요) · 너무 어두워요 · 화면이 거의 까매요 (렌즈를 가리지 않았는지 봐 주세요)"
    assert localize_error(e, "en") == "The action part is too short (1.2 s · record at least 3 s) · Too dark · the screen is almost black (check the lens isn't covered)"
    assert localize_error(e, "ko") == e and localize_error(e, None) == e
    assert localize_error("RuntimeError: x", "ja") == "RuntimeError: x"
