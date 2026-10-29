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
