"""DataCache 서버 — 모바일 앱(일반 사용자)의 업로드·처리·보상과, 데스크톱 웹(데이터 구매 기업)의 카탈로그·견적·주문·내려받기.

  앱 (머리글 X-DC-Token)
    GET  /api/missions                       미션 목록과 보상 (받아들인 분당 등급별)
    PUT  /api/sessions/{sid}/files/{name}    녹화 파일 하나 (정해진 이름만)
    POST /api/sessions/{sid}/process         처리 시작 (가리기 → 관절 → 이해 → 4감 내보내기)
    GET  /api/sessions/{sid}/status          상태·보상·인증서 요약
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
                                  calibration_store=self.root / "grip_calibration")
                self.set_status(sid, state="done" if "error" not in rep else "rejected", **rep)
            except Exception as e:
                self.set_status(sid, state="failed", error=f"{type(e).__name__}: {str(e)[:300]}")

    # ---- 카탈로그 (처리된 녹화의 인증서에서) ----
    def accepted(self) -> List[dict]:
        rows = []
        for p in sorted((self.root / "out").glob("*/certificate.json")):
            c = json.loads(p.read_text())
            sid = p.parent.name
            st = self.status(sid)
            if st.get("withdrawn"):
                continue
            rows.append({"sid": sid, "mission": (c.get("mission") or {}).get("id"), "level": c["level"],
                         "minutes": c["dataset"]["frames"] / c["dataset"]["fps"] / 60, "episodes": c["dataset"]["episodes"],
                         "features": c["dataset"]["features"], "sync": c["sync"].get("chosen"),
                         "sync_error_ms": c["sync"].get("error_ms", {}),
                         "verified_steps": sum(1 for s in (c.get("steps") or []) if s.get("verified") is True),
                         "steps": len(c.get("steps") or []), "grip_calibrated": bool((c.get("touch") or {}).get("calibration")),
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
        for m in ms:
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
        return {"t": col("timestamp"), "grip_n": col("observation.touch", 1), "contact": col("observation.touch", 0),
                "phase": col("observation.touch", 3), "hr": col("observation.physio", 0), "hrr": col("observation.physio", 2),
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
