"""DataCache 서버 — 모바일 앱(일반 사용자)의 업로드·처리·보상과, 데스크톱 웹(데이터 구매 기업)의 카탈로그·견적·주문·내려받기.

  앱 (머리글 X-DC-Token)
    GET  /api/missions                       미션 목록과 보상 (받아들인 분당 등급별)
    PUT  /api/sessions/{sid}/files/{name}    녹화 파일 하나 (정해진 이름만)
    POST /api/sessions/{sid}/process         처리 시작 (가리기 → 관절 → 이해 → 4감 내보내기)
    GET  /api/sessions/{sid}/status          상태·보상·인증서 요약
    (아래는 머리글 X-DC-Device: 폰 기기 id 도 함께)
    POST /api/face/check                     녹화 1단계 얼굴 확인 (사진 JPEG 본문 → 등록한 본인인지)
    GET  /api/people                         동의 인원 목록 (+ 고를 수 있는 가상 인물)
    POST /api/people                         동의 인원 추가 {name, relation, adult, age_band, agreed, bio}
    POST /api/people/{pid}/face              얼굴 등록 {photos_b64: [정면, 왼쪽, 오른쪽]} → 특징값만 저장, 사진은 버림
    POST /api/people/{pid}/donor             가상 인물 고르기 {donor: "donor3" | "auto"}
    POST /api/people/{pid}/consent           생체 신호 동의·철회 {bio} (본인만)
    DELETE /api/people/{pid}                 동의 철회 (특징값 삭제)
    GET  /api/donors/{id}.png                가상 인물 사진 (실존하지 않는 사람)
    POST /api/sessions/{sid}/appeal          이의 신청 {reason, detail} → 바로 자동 확인
    GET  /api/appeals                        내 이의 신청 (단계 기록)
    GET  /api/sessions/{sid}/preview/{k}     가상 얼굴 미리보기: 판매용(가리고 바꾼) 영상의 k번째(0~5) 고른 장면 JPEG (내 녹화만)
    GET  /api/payout/account                 받은 보상 합계 · 계좌 연결 기다리는 보상 · 계좌 표시
    POST /api/payout/account                 받을 계좌 표시 {label: "KB국민 ****1234"} → 기다리던 보상을 보냄 (시험 입금)
  구매 기업 (공개 읽기)
    GET  /                                   구매 기업용 웹
    GET  /api/market/catalog                 미션별 시간·등급·채널 (처리된 녹화의 인증서에서 계산, 지어낸 숫자 없음)
    GET  /api/market/sample/{sid}/timeline   예시 에피소드의 4감 타임라인 (촉각·심박·소리·단계)
    GET  /api/market/sample/{sid}/frame/{i}  예시 장면 (가린 영상에서만)
    GET  /api/market/certificate/{sid}       품질 인증서 (마크다운)
    POST /api/market/quote                   견적 {missions, level, hours, custom, refine}
    POST /api/market/orders                  주문 {org, email, missions, level, hours}
    GET  /api/market/orders/{oid}            주문 상태 (승인되면 일회용 내려받기 주소)
    GET  /api/market/download/{key}          묶음 zip (7일, 5번)
  운영자 (?token=)
    GET  /admin                              주문 관리 화면
    GET  /api/admin/orders                   주문 목록
    POST /api/admin/orders/{oid}/approve     승인 → 묶음 만들기
    GET  /api/admin/appeals                  이의 신청 목록
    POST /api/admin/appeals/{aid}            검토자 결정 {accept, extra_reward, note}
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Queue
from typing import List, Optional

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from .. import config, pricing
from ..appeals import Appeals
from ..payout import Payouts
from ..people import People, embed_jpeg

FILES = {"meta.json", "video.mp4", "video_frames.csv", "audio.wav", "audio.json", "phone_imu.csv", "beacon.csv",
         "watch_imu.csv", "watch_audio.wav", "watch_audio.json", "watch_vib.csv", "watch_hr.csv", "watch_skin.csv",
         "clock_sync.csv", "phone_server_sync.csv", "watch_server_sync.csv", "watch_meta.json"}
WATCH_WAIT_S = 240   # 폰 업로드 뒤 워치 업로드를 기다리는 최대 시간
SID = re.compile(r"^[A-Za-z0-9_\-]{3,64}$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
WEB = config.ROOT / "web"


class Store:
    def __init__(self, root: Path, token: str, use_api: bool = True):
        self.root, self.token, self.use_api = root, token, use_api
        for d in ("sessions", "out", "status", "orders", "bundles"):
            (root / d).mkdir(parents=True, exist_ok=True)
        self.q: Queue = Queue()
        self.lock = threading.Lock()
        self.live: dict = {}          # 지금 녹화 중인 세션 {sid, started, watch_joined}
        self.people = People(root / "people")
        self.appeals = Appeals(root / "appeals")
        self.payouts = Payouts(root / "payout")
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- 상태 ----
    def status(self, sid: str) -> dict:
        p = self.root / "status" / f"{sid}.json"
        return json.loads(p.read_text()) if p.exists() else {"state": "none"}

    def set_status(self, sid: str, **kw) -> dict:
        with self.lock:
            st = self.status(sid)
            st.update(kw, updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            (self.root / "status" / f"{sid}.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
            return st

    def _worker(self):
        from ..jobs import run_session

        while True:
            sid = self.q.get()
            st0 = self.status(sid)
            if st0.get("watch_joined") and not st0.get("watch_done"):   # 워치가 같이 찍었으면 워치 업로드를 기다린다
                t0 = time.time()
                while time.time() - t0 < WATCH_WAIT_S and not self.status(sid).get("watch_done"):
                    time.sleep(2)
            self.set_status(sid, state="processing")
            try:
                rep = run_session(self.root / "sessions" / sid, self.root / "out" / sid, use_api=self.use_api,
                                  calibration_store=self.root / "grip_calibration", people=self.people)
                fin = self.set_status(sid, state="done" if "error" not in rep else "rejected", **rep)
            except Exception as e:
                fin = self.set_status(sid, state="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
            self.appeals.reprocessed(sid, fin)
            self.settle(sid)

    def settle(self, sid: str) -> dict:
        """승인된 녹화의 보상을 입금 기록으로 (계좌가 없으면 기다림)."""
        s = self.status(sid)
        p = self.payouts.settle(self.device_of(sid), s)
        return self.set_status(sid, payout=p) if p != s.get("payout") else s

    def device_sessions(self, device: str) -> List[str]:
        return [d.name for d in (self.root / "sessions").iterdir() if d.is_dir() and self.device_of(d.name) == device]

    def device_of(self, sid: str) -> Optional[str]:
        p = self.root / "sessions" / sid / "meta.json"
        return ((json.loads(p.read_text()).get("devices") or {}).get("phone") or {}).get("id") if p.exists() else None

    # ---- 카탈로그 (처리된 녹화의 인증서에서) ----
    def accepted(self) -> List[dict]:
        rows = []
        for p in sorted((self.root / "out").glob("*/certificate.json")):
            sid = p.parent.name
            if self.status(sid).get("withheld"):      # 개인정보 이의 신청으로 판매에서 뺀 녹화
                continue
            c = json.loads(p.read_text())
            st = self.status(sid)
            if st.get("withdrawn"):
                continue
            rows.append({"sid": sid, "mission": (c.get("mission") or {}).get("id"), "level": c["level"],
                         "minutes": c["dataset"]["frames"] / c["dataset"]["fps"] / 60, "episodes": c["dataset"]["episodes"],
                         "features": c["dataset"]["features"], "sync": c["sync"].get("chosen"),
                         "sync_error_ms": c["sync"].get("error_ms", {}),
                         "verified_steps": sum(1 for s in (c.get("steps") or []) if s.get("verified") is True),
                         "steps": len(c.get("steps") or []), "grip_calibrated": bool((c.get("touch") or {}).get("grip_model")),
                         "contacts": len((c.get("touch") or {}).get("contact_events", [])),
                         "synthetic": "synthetic" in json.dumps(c.get("devices"))})
        return rows

    def catalog(self) -> dict:
        ms = json.loads((Path(__file__).parents[1] / "missions.json").read_text())["missions"]
        rows = self.accepted()
        out = []
        for m in ms:
            mine = [r for r in rows if r["mission"] == m["id"]]
            hours = {lv: round(sum(r["minutes"] for r in mine if pricing.LEVELS.index(r["level"]) >= pricing.LEVELS.index(lv)) / 60, 3)
                     for lv in pricing.LEVELS}
            out.append({**{k: m[k] for k in ("id", "title", "title_en", "guide", "purpose", "objects", "primitives")},
                        "recordings": len(mine), "hours": hours, "episodes": sum(r["episodes"] for r in mine),
                        "verified_steps": sum(r["verified_steps"] for r in mine), "steps": sum(r["steps"] for r in mine),
                        "samples": [r["sid"] for r in sorted(mine, key=lambda r: (-pricing.LEVELS.index(r["level"]), -r["steps"], r["synthetic"]))][:4],
                        "synthetic_only": bool(mine) and all(r["synthetic"] for r in mine)})
        extra = sorted({r["mission"] for r in rows} - {m["id"] for m in ms})
        for mid in extra:
            mine = [r for r in rows if r["mission"] == mid]
            out.append({"id": mid, "title": mid, "title_en": mid, "guide": "", "purpose": "", "objects": [], "primitives": [],
                        "recordings": len(mine), "hours": {lv: round(sum(r["minutes"] for r in mine) / 60, 3) for lv in pricing.LEVELS},
                        "episodes": sum(r["episodes"] for r in mine), "verified_steps": sum(r["verified_steps"] for r in mine),
                        "steps": sum(r["steps"] for r in mine), "samples": [r["sid"] for r in mine][:4],
                        "synthetic_only": all(r["synthetic"] for r in mine)})
        return {"brand": config.BRAND, "levels": [{"id": lv, "name": pricing.LEVEL_KO[lv], "desc": pricing.LEVEL_DESC[lv],
                                                   "per_hour": pricing.PER_HOUR[lv], "reward_per_hour": pricing.REWARD_PER_HOUR[lv]}
                                                  for lv in pricing.LEVELS],
                "missions": out, "totals": {"recordings": len(rows), "hours": round(sum(r["minutes"] for r in rows) / 60, 3)},
                "pricing": {"custom_mult": pricing.CUSTOM_MULT, "custom_min_hours": pricing.CUSTOM_MIN_HOURS,
                            "custom_setup": pricing.CUSTOM_SETUP, "discounts": pricing.DISCOUNTS}}

    # ---- 주문 ----
    def _order_path(self, oid: str) -> Path:
        if not re.match(r"^[A-Za-z0-9_\-]{8,40}$", oid):
            raise HTTPException(404, "없는 주문")
        return self.root / "orders" / f"{oid}.json"

    def order(self, oid: str) -> dict:
        p = self._order_path(oid)
        if not p.exists():
            raise HTTPException(404, "없는 주문")
        return json.loads(p.read_text())

    def save_order(self, o: dict) -> None:
        self._order_path(o["id"]).write_text(json.dumps(o, indent=2, ensure_ascii=False))

    def sources(self, missions: List[str], level: str) -> List[Path]:
        lv = pricing.LEVELS.index(level)
        return [self.root / "out" / r["sid"] for r in self.accepted()
                if (not missions or r["mission"] in missions) and pricing.LEVELS.index(r["level"]) >= lv]

    def build_bundle(self, oid: str) -> None:
        from ..export.bundle import merge

        o = self.order(oid)
        try:
            srcs = self.sources(o["missions"], o["level"])
            if not srcs:
                raise RuntimeError("조건에 맞는 녹화가 없음")
            work = self.root / "bundles" / oid
            ds = merge(srcs, work / "dataset", o["level"])
            zp = shutil.make_archive(str(work / f"datacache_{oid}"), "zip", ds)
            shutil.rmtree(ds)
            o.update(status="ready", file=zp, size_mb=round(os.path.getsize(zp) / 1e6, 2), key=secrets.token_urlsafe(24),
                     expires=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat(timespec="seconds"), downloads=0,
                     recordings=len(srcs))
        except Exception as e:
            o.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
        self.save_order(o)


class QuoteIn(BaseModel):
    missions: List[str] = Field(default_factory=list)
    level: str = "basic"
    hours: float = 1.0
    custom: bool = False
    refine: bool = False


class LiveIn(BaseModel):
    sid: str


class PersonIn(BaseModel):
    name: str = Field(min_length=1, max_length=20)
    relation: str = Field(max_length=10)
    adult: bool
    age_band: str = Field(max_length=10)
    agreed: bool
    bio: bool = False


class FaceIn(BaseModel):
    photos_b64: List[str] = Field(min_length=1, max_length=3)


class DonorIn(BaseModel):
    donor: str = Field(max_length=20)


class BioIn(BaseModel):
    bio: bool


class AccountIn(BaseModel):
    label: str = Field(max_length=40)


class AppealIn(BaseModel):
    reason: str
    detail: str = Field("", max_length=500)


class DecisionIn(BaseModel):
    accept: bool
    extra_reward: int = Field(0, ge=0, le=100_000)
    note: str = Field("", max_length=300)


class OrderIn(QuoteIn):
    org: str
    email: str
    note: str = ""


def create_app(root: Optional[Path] = None, token: Optional[str] = None, use_api: bool = True) -> FastAPI:
    root = Path(root or config.DATA / "server")
    token = token or os.environ.get("DATACACHE_TOKEN") or secrets.token_urlsafe(12)
    st = Store(root, token, use_api)
    app = FastAPI(title="DataCache")
    app.state.store = st

    def auth(t: Optional[str]):
        if t != st.token:
            raise HTTPException(401, "토큰이 맞지 않음")

    # ---- 앱 ----
    @app.get("/api/missions")
    def missions(x_dc_token: Optional[str] = Header(None)):
        auth(x_dc_token)
        ms = json.loads((Path(__file__).parents[1] / "missions.json").read_text())["missions"]
        cat = {m["id"]: m for m in st.catalog()["missions"]}
        today = pricing.today_mission([m["id"] for m in ms])
        for m in ms:
            m["today"] = m["id"] == today
            have = cat.get(m["id"], {}).get("recordings", 0)
            m["reward_per_minute"] = {lv: round(pricing.REWARD_PER_HOUR[lv] / 60) for lv in pricing.LEVELS}
            m["scarce"] = have < 3          # 아직 적게 모인 미션 → 보너스
            m["bonus_pct"] = int(pricing.BONUS_SCARCE * 100) if m["scarce"] else 0
        return {"missions": ms}

    @app.put("/api/sessions/{sid}/files/{name}")
    async def upload(sid: str, name: str, request: Request, x_dc_token: Optional[str] = Header(None)):
        auth(x_dc_token)
        if not SID.match(sid) or name not in FILES:
            raise HTTPException(400, "올릴 수 없는 파일")
        d = root / "sessions" / sid
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (name + ".part")
        n = 0
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                n += len(chunk)
                if n > 4 << 30:
                    raise HTTPException(413, "파일이 너무 큼")
                f.write(chunk)
        tmp.replace(d / name)
        st.set_status(sid, state="uploading")
        return {"ok": True, "bytes": n}

    @app.post("/api/sessions/{sid}/process")
    def start(sid: str, x_dc_token: Optional[str] = Header(None)):
        auth(x_dc_token)
        d = root / "sessions" / sid
        if not SID.match(sid) or not (d / "meta.json").exists():
            raise HTTPException(400, "meta.json 이 필요함")
        calib = (json.loads((d / "meta.json").read_text()).get("mission") or {}).get("id") == "grip_calibration"
        if not calib and not (d / "video.mp4").exists():
            raise HTTPException(400, "video.mp4 가 필요함")
        st.q.put(sid)
        return st.set_status(sid, state="queued")

    # ---- 폰·워치 연결 (서로 직접 통신하지 않는다: 둘 다 서버와 시각을 주고받고, 서버가 녹화 시작·끝을 알려 준다) ----
    @app.get("/api/time")
    def server_time(x_dc_token: Optional[str] = Header(None)):
        auth(x_dc_token)
        return {"ns": time.monotonic_ns()}

    @app.post("/api/live/start")
    def live_start(b: LiveIn, x_dc_token: Optional[str] = Header(None)):
        auth(x_dc_token)
        if not SID.match(b.sid):
            raise HTTPException(400, "세션 id")
        st.live = {"sid": b.sid, "state": "recording", "started": time.time(), "watch_joined": False}
        return st.live

    @app.post("/api/live/stop")
    def live_stop(b: LiveIn, x_dc_token: Optional[str] = Header(None)):
        auth(x_dc_token)
        if st.live.get("sid") == b.sid:
            st.live["state"] = "stopped"
            st.set_status(b.sid, watch_joined=bool(st.live.get("watch_joined")))
        return st.live

    @app.get("/api/live")
    def live(role: str = "phone", x_dc_token: Optional[str] = Header(None)):
        auth(x_dc_token)
        if role == "watch" and st.live.get("state") == "recording":
            st.live["watch_joined"] = True
        return st.live or {"state": "idle"}

    @app.post("/api/sessions/{sid}/watch_done")
    def watch_done(sid: str, x_dc_token: Optional[str] = Header(None)):
        auth(x_dc_token)
        return st.set_status(sid, watch_done=True)

    @app.get("/api/sessions/{sid}/status")
    def status(sid: str, x_dc_token: Optional[str] = Header(None)):
        auth(x_dc_token)
        return st.status(sid)

    # ---- 얼굴 확인 · 동의 인원 · 이의 신청 (기기 id 단위) ----
    def device(d: Optional[str]) -> str:
        if not d or not re.match(r"^[A-Za-z0-9_\-]{6,64}$", d):
            raise HTTPException(400, "기기 id 가 필요함 (X-DC-Device)")
        return d

    FACE_ERR = {"no_face": "얼굴을 찾지 못했어요", "too_small": "얼굴이 너무 작아요 — 조금 더 가까이", "bad_image": "사진을 읽지 못했어요"}

    async def face_of(request: Request):
        data = await request.body()
        if not data or len(data) > 8 << 20:
            raise HTTPException(400, "사진이 필요함")
        emb, why = embed_jpeg(data)
        if emb is None:
            raise HTTPException(422, FACE_ERR.get(why, why))
        return emb

    @app.post("/api/face/check")
    async def face_check(request: Request, x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        return st.people.check_owner(device(x_dc_device), await face_of(request))

    @app.get("/api/people")
    def people_list(x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        return st.people.listing(device(x_dc_device))

    def person_call(fn):
        try:
            return fn()
        except KeyError:
            raise HTTPException(404, "없는 사람")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/people")
    def people_add(b: PersonIn, x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        dev = device(x_dc_device)
        return person_call(lambda: st.people.add(dev, b.name, b.relation, b.adult, b.age_band, b.agreed, b.bio))

    @app.post("/api/people/{pid}/face")
    def people_face(pid: str, b: FaceIn, x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        dev = device(x_dc_device)
        import base64

        try:
            photos = [base64.b64decode(x, validate=True) for x in b.photos_b64]
        except ValueError:
            raise HTTPException(400, "사진")
        if any(len(x) > 8 << 20 for x in photos):
            raise HTTPException(413, "사진이 너무 큼")
        return person_call(lambda: st.people.enroll(dev, pid, photos))

    @app.post("/api/people/{pid}/donor")
    def people_donor(pid: str, b: DonorIn, x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        dev = device(x_dc_device)
        return person_call(lambda: st.people.set_donor(dev, pid, b.donor))

    @app.post("/api/people/{pid}/consent")
    def people_bio(pid: str, b: BioIn, x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        dev = device(x_dc_device)
        return person_call(lambda: st.people.set_bio(dev, pid, b.bio))

    @app.get("/api/donors/{name}")
    def donor_image(name: str):
        f = Path(__file__).parents[1] / "donors" / name
        if not re.match(r"^donor\d+\.png$", name) or not f.exists():
            raise HTTPException(404)
        return FileResponse(f, media_type="image/png")

    @app.delete("/api/people/{pid}")
    def people_remove(pid: str, x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        if not st.people.remove(device(x_dc_device), pid):
            raise HTTPException(404)
        return {"removed": pid}

    @app.post("/api/sessions/{sid}/appeal")
    def appeal(sid: str, b: AppealIn, x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        dev = device(x_dc_device)
        if not SID.match(sid) or st.device_of(sid) != dev:
            raise HTTPException(404, "내 녹화가 아님")

        def requeue(s):
            st.set_status(s, state="queued")
            st.q.put(s)

        try:
            a = st.appeals.file(dev, sid, b.reason, b.detail, st.status(sid), requeue, lambda s: st.set_status(s, withheld=True))
        except ValueError as e:
            raise HTTPException(400, str(e))
        credit_appeal(a)
        return a

    def credit_appeal(a: dict) -> None:
        """인정된 추가 보상 → 그 녹화의 입금에 더한다."""
        if a.get("state") == "accepted" and a.get("extra_reward"):
            s = st.status(a["sid"])
            st.set_status(a["sid"], appeal_extra=int(s.get("appeal_extra") or 0) + int(a["extra_reward"]))
            st.settle(a["sid"])

    @app.get("/api/sessions/{sid}/preview/{k}")
    def preview(sid: str, k: int, x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        if not SID.match(sid) or st.device_of(sid) != device(x_dc_device) or not 0 <= k < 6:
            raise HTTPException(404)
        out = root / "out" / sid
        v = next((p for p in (out / "masked_swap.mp4", out / "masked.mp4") if p.exists()), None)
        if v is None:
            raise HTTPException(404, "아직 처리 중이에요")
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.linspace(n * 0.1, n * 0.9, 6)[k]))
        ok, f = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(404)
        f = cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1])))
        return Response(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), media_type="image/jpeg")

    @app.get("/api/payout/account")
    def payout_get(x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        dev = device(x_dc_device)
        return st.payouts.summary(dev, (st.status(s) for s in st.device_sessions(dev)))

    @app.post("/api/payout/account")
    def payout_set(b: AccountIn, x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        dev = device(x_dc_device)
        try:
            st.payouts.set_account(dev, b.label)
        except ValueError as e:
            raise HTTPException(400, str(e))
        sent = 0
        for sid in st.device_sessions(dev):           # 계좌를 기다리던 보상을 지금 보낸다
            before = (st.status(sid).get("payout") or {}).get("status")
            after = (st.settle(sid).get("payout") or {}).get("status")
            sent += before == "waiting_account" and after == "sent"
        return {**st.payouts.summary(dev, (st.status(s) for s in st.device_sessions(dev))), "sent_now": sent}

    @app.get("/api/appeals")
    def appeals_mine(x_dc_token: Optional[str] = Header(None), x_dc_device: Optional[str] = Header(None)):
        auth(x_dc_token)
        return {"appeals": st.appeals.for_device(device(x_dc_device))}

    # ---- 구매 기업 ----
    @app.get("/", response_class=HTMLResponse)
    def home():
        return (WEB / "index.html").read_text()

    @app.get("/admin", response_class=HTMLResponse)
    def admin_page():
        return (WEB / "admin.html").read_text()

    @app.get("/static/{name}")
    def static(name: str):
        p = WEB / "static" / name
        if not re.match(r"^[A-Za-z0-9_.\-]+$", name) or not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/market/catalog")
    def catalog():
        return st.catalog()

    def _ds(sid: str) -> Path:
        p = root / "out" / sid / "lerobot"
        if not SID.match(sid) or not p.exists():
            raise HTTPException(404, "없는 녹화")
        return p

    @app.get("/api/market/sample/{sid}/timeline")
    def timeline(sid: str, episode: int = 0):
        import pyarrow.parquet as pq

        from ..export.lerobot import read_episode_info

        ds = _ds(sid)
        t = pq.read_table(ds / "data" / "chunk-000" / "file-000.parquet")
        ep = np.array(t.column("episode_index").to_pylist())
        sel = ep == episode
        if not sel.any():
            raise HTTPException(404, "없는 에피소드")

        def col(name, k=None):
            if name not in t.column_names:
                return None
            a = np.array(t.column(name).to_pylist(), np.float64)[sel]
            return (a if k is None else a[:, k]).round(3).tolist()

        info = read_episode_info(ds)[episode]
        return {"t": col("timestamp"), "grip_score": col("observation.touch", 2), "grip_level": col("observation.touch", 1),
                "contact": col("observation.touch", 0), "phase": col("observation.touch", 4), "hr": col("observation.physio", 0), "hrr": col("observation.physio", 2),
                "audio_db": col("observation.audio", 0), "contact_sound": col("observation.audio", 2),
                "self_mark": col("annotation.events", 0), "step": col("annotation.step"),
                "steps": info.get("steps"), "sync": info.get("sync"), "outcome": info.get("outcome"),
                "touch": info.get("touch"), "scene": info.get("scene"), "frames": int(sel.sum())}

    @app.get("/api/market/sample/{sid}/frame/{i}")
    def frame(sid: str, i: int):
        ds = _ds(sid)
        cap = cv2.VideoCapture(str(ds / "videos" / "observation.images.ego" / "chunk-000" / "file-000.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i))
        ok, f = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(404)
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return Response(buf.tobytes(), media_type="image/jpeg")

    @app.get("/api/market/certificate/{sid}", response_class=PlainTextResponse)
    def cert(sid: str):
        p = root / "out" / sid / "certificate.md"
        if not SID.match(sid) or not p.exists():
            raise HTTPException(404)
        return p.read_text()

    @app.post("/api/market/quote")
    def quote(q: QuoteIn):
        try:
            r = pricing.quote(q.hours, q.level, q.custom, q.refine)
        except ValueError as e:
            raise HTTPException(400, str(e))
        avail = round(sum(r_["minutes"] for r_ in st.accepted()
                          if (not q.missions or r_["mission"] in q.missions)
                          and pricing.LEVELS.index(r_["level"]) >= pricing.LEVELS.index(q.level)) / 60, 3)
        return {**r, "available_hours": avail, "missions": q.missions}

    @app.post("/api/market/orders")
    def order(o: OrderIn):
        if not o.org.strip() or not EMAIL.match(o.email):
            raise HTTPException(400, "기관 이름과 이메일을 확인해 주세요")
        q = pricing.quote(o.hours, o.level, o.custom, o.refine)
        rec = {"id": secrets.token_urlsafe(12), "org": o.org.strip()[:120], "email": o.email.strip()[:200],
               "note": o.note[:1000], "missions": o.missions, "level": o.level, "quote": q, "status": "pending",
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        st.save_order(rec)
        return {k: rec[k] for k in ("id", "status", "quote")}

    @app.get("/api/market/orders/{oid}")
    def order_status(oid: str):
        o = st.order(oid)
        out = {k: o.get(k) for k in ("id", "status", "quote", "size_mb", "expires", "downloads", "error", "recordings")}
        if o.get("status") == "ready":
            out["download"] = f"/api/market/download/{o['key']}"
        return out

    @app.get("/api/market/download/{key}")
    def download(key: str):
        for p in (root / "orders").glob("*.json"):
            o = json.loads(p.read_text())
            if o.get("key") == key and o.get("status") == "ready":
                if datetime.fromisoformat(o["expires"]) < datetime.now(timezone.utc) or o.get("downloads", 0) >= 5:
                    raise HTTPException(410, "내려받기 기간·횟수가 끝났습니다")
                o["downloads"] = o.get("downloads", 0) + 1
                st.save_order(o)
                return FileResponse(o["file"], filename=Path(o["file"]).name, media_type="application/zip")
        raise HTTPException(404)

    # ---- 운영자 ----
    @app.get("/api/admin/orders")
    def admin_orders(token: str = Query("")):
        auth(token)
        return {"orders": sorted((json.loads(p.read_text()) for p in (root / "orders").glob("*.json")),
                                 key=lambda o: o["created"], reverse=True)}

    @app.post("/api/admin/orders/{oid}/approve")
    def approve(oid: str, bg: BackgroundTasks, token: str = Query("")):
        auth(token)
        o = st.order(oid)
        o["status"] = "building"
        st.save_order(o)
        bg.add_task(st.build_bundle, oid)
        return {"id": oid, "status": "building"}

    @app.get("/api/admin/appeals")
    def admin_appeals(token: str = Query("")):
        auth(token)
        return {"appeals": st.appeals.all()}

    @app.post("/api/admin/appeals/{aid}")
    def admin_decide(aid: str, b: DecisionIn, token: str = Query("")):
        auth(token)
        try:
            a = st.appeals.decide(aid, b.accept, b.extra_reward, b.note)
        except KeyError:
            raise HTTPException(404)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if a["reason"] == "privacy" and not b.accept:      # 가리기에 문제 없다고 판단 → 다시 판매
            st.set_status(a["sid"], withheld=False)
        credit_appeal(a)
        return a

    return app


def main():
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser("datacache-serve")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--root", default=None)
    ap.add_argument("--no-api", action="store_true", help="Meta API 를 부르지 않는다 (시험용)")
    a = ap.parse_args()
    config.load_env()
    token = os.environ.get("DATACACHE_TOKEN") or secrets.token_urlsafe(12)
    print(f"DataCache 서버 토큰: {token}  (앱 X-DC-Token, 운영자 /admin?token=)")
    uvicorn.run(create_app(a.root, token, not a.no_api), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
