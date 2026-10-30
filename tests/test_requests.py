"""자연어 의뢰 → 미션 (Spark 대답은 흉내): 예산에 맞춘 견적, 참여 조건에 맞는 사람에게만 미션이 열린다."""

import urllib.parse

import pytest

PLAN = {"ready": True, "title": "택배 상하차", "domain": "물류", "robot_use": "상하차",
        "sensors": ["video", "pose", "wrist", "touch"], "eligibility": {"age_bands": ["19~29세", "30대", "40대", "50대"], "gender": "any"},
        "budget_krw": 5_000_000, "hours": 0, "questions": [],
        "missions": [{"title": "상자 들어 선반에 올리기", "title_en": "Lift a box onto a shelf", "guide": "30초~1분 · 3번",
                      "purpose": "상자 들기", "objects": ["cardboard box", "shelf"],
                      "primitives": ["reach", "grasp", "lift", "transport", "place", "release"],
                      "situations": ["가벼운 상자", "무거운 상자", "허리 높이", "어깨 높이", "한 손", "두 손"]}]}


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from datacash.server.app import create_app

    return TestClient(create_app(tmp_path, "t", use_api=False)), tmp_path


def test_budget_fit_quote():
    from datacash.requests import fit_quote

    q = fit_quote(PLAN)
    assert q["within_budget"] and q["total"] <= 5_000_000 and q["hours"] >= 10 and q["custom"]
    assert q["sensors"] == ["video", "pose", "wrist", "touch"] and q["level"] == "premium"
    more = fit_quote({**PLAN, "budget_krw": 10_000_000})
    assert more["hours"] > q["hours"]


def test_request_opens_missions_for_eligible_people(client):
    c, root = client
    assert c.post("/api/market/request", json={"plan": PLAN, "org": "", "email": "x"}).status_code == 400
    r = c.post("/api/market/request", json={"plan": PLAN, "org": "한빛제조", "email": "buyer@hanbit.co.kr",
                                             "messages": [{"role": "user", "content": "택배 상하차 데이터"}]}).json()
    assert r["status"] == "collecting" and r["missions"][0]["target_hours"] > 0
    mid = r["missions"][0]["id"]

    def visible(age, gender=None):
        h = {"X-DC-Token": "t", "X-DC-Age": urllib.parse.quote(age)}          # 머리글은 ASCII 라 연령대는 % 인코딩
        if gender:
            h["X-DC-Gender"] = gender
        return [m["id"] for m in c.get("/api/missions", headers=h).json()["missions"]]

    assert mid in visible("30대") and mid in visible("50대", "female")
    assert mid not in visible("60대 이상")                                   # 50대 이하만
    m = next(x for x in c.get("/api/missions", headers={"X-DC-Token": "t"}).json()["missions"] if x["id"] == "pull_tissue")
    assert m["primitives"]                                                   # 기본 미션은 누구에게나
    cat = {x["id"]: x for x in c.get("/api/market/catalog").json()["missions"]}
    assert cat[mid]["title"] == "상자 들어 선반에 올리기"
    assert c.get(f"/api/market/request/{r['id']}").json()["missions"][0]["collected_hours"] == 0
    g = {**PLAN, "eligibility": {"age_bands": ["19~29세", "30대", "40대", "50대", "60대 이상"], "gender": "female"}}
    r2 = c.post("/api/market/request", json={"plan": g, "org": "B", "email": "b@b.co"}).json()
    assert r2["missions"][0]["id"] in visible("30대", "female") and r2["missions"][0]["id"] not in visible("30대", "male")


def test_session_uses_request_mission(tmp_path):
    from datacash.jobs import _mission

    m = _mission({"mission": {"id": "req_x_0"}}, extra=[{"id": "req_x_0", "title": "상자", "title_en": "box", "objects": ["box"],
                                                         "primitives": ["reach", "release"]}])
    assert m["objects"] == ["box"] and m["primitives"] == ["reach", "release"]


def _cat():
    return {"missions": [{"id": "lift_box", "title": "상자 들어 올려 옮기기", "hours": {"basic": 30.0, "premium": 30.0, "bio": 0.0}},
                         {"id": "fold_cloth", "title": "수건 개기", "hours": {"basic": 0.0, "premium": 0.0, "bio": 0.0}}]}


def test_instant_reads_request_without_model():
    from datacash.requests import instant

    p = instant([{"role": "user", "content": "물류 회사예요. 택배 상자 상하차, 50대 이하 여성, 예산 300만 원, 쥐는 세기 꼭"}], _cat())
    assert p["missions"][0]["id"] == "lift_box" and p["budget_krw"] == 3_000_000
    assert p["eligibility"] == {"age_bands": ["19~29세", "30대", "40대", "50대"], "gender": "female"}
    assert "touch" in p["sensors"] and p["source"] == "instant"


def test_options_existing_new_mix_and_rarity():
    from datacash.requests import instant, options

    o = options(instant([{"role": "user", "content": "택배 상자 옮기기 데이터 20시간"}], _cat()), _cat())
    assert o["recommended"] == "existing" and o["options"]["existing"]["setup"] == 0       # 보관량으로 충분 → 기존, 착수비 없음
    o = options(instant([{"role": "user", "content": "택배 상자 옮기기 데이터 50시간"}], _cat()), _cat())
    assert o["recommended"] == "mix" and o["options"]["mix"]["existing_hours"] == 30.0
    o = options(instant([{"role": "user", "content": "90대 어르신이 상자 옮기는 데이터 20시간"}], _cat()), _cat())
    assert o["recommended"] == "new" and "existing" not in o["options"] and o["rarity"] == "very_rare"
    normal = options(instant([{"role": "user", "content": "수건 개기 20시간"}], _cat()), _cat())["options"]["new"]["total"]
    assert o["options"]["new"]["total"] > normal * 2                                        # 특이 조건은 비싸게


def test_tiny_stock_is_not_offered():
    """보관 데이터가 1시간도 안 되면 기존·섞기를 내놓지 않는다."""
    from datacash.requests import instant, options

    cat = {"missions": [{"id": "lift_box", "title": "상자", "hours": {"basic": 0.03, "premium": 0.03, "bio": 0.0}}]}
    o = options(instant([{"role": "user", "content": "택배 상자 옮기기 데이터, 예산 500만원"}], cat), cat)
    assert set(o["options"]) == {"new"} and o["why"] == "stock_small" and o["stock_hours"] == 0.03
