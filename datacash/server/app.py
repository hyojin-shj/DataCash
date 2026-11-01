"""DataCash 서버 — 모바일 앱(일반 사용자)의 업로드·처리·보상과, 데스크톱 웹(데이터 구매 기업)의 카탈로그·견적·주문·내려받기.

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
    GET  /api/missions 머리글 X-DC-Age·X-DC-Gender — 의뢰 미션은 참여 조건에 맞는 사람에게만
  구매 기업 의뢰 (자연어 → 미션)
    POST /api/market/request/chat            채팅 {messages} → Muse Spark 가 의뢰를 미션·참여 조건·센서·견적으로
    POST /api/market/request                 의뢰 확정 {plan, org, email, messages} → 미션을 앱에 연다
    GET  /api/market/request/{rid}           의뢰 진행 (미션별 모은 시간 / 목표)
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
from ..requests import Requests, eligible
from ..people import People, embed_jpeg
from ..understand.spark import PRIMITIVES

LANGS = ("en", "ko", "ja", "zh-CN", "es")


def norm_lang(lang: Optional[str]) -> str:
    """'ko-KR'·'zh'·'zh-Hans'·'ja_JP' → 지원 언어 코드. 모르면 ''(한국어 원문 그대로)."""
    s = (lang or "").strip().replace("_", "-").lower()
    if not s:
        return ""
    if s.startswith("zh"):
        return "zh-CN"
    head = s.split("-")[0]
    return head if head in LANGS else ""


def localize_mission(m: dict, lang: Optional[str]) -> dict:
    """미션의 title·guide·purpose 를 lang 으로 바꾼 사본. 번역이 없으면 en → title_en, 그것도 없으면 한국어 원문."""
    lang = norm_lang(lang)
    if not lang or lang == "ko":
        return m
    tr = (m.get("i18n") or {}).get(lang) or {}
    out = dict(m)
    for k in ("title", "guide", "purpose"):
        if tr.get(k):
            out[k] = tr[k]
        elif lang == "en" and k == "title" and m.get("title_en"):
            out[k] = m["title_en"]
    sl = tr.get("situations") or (m.get("i18n") or {}).get("en", {}).get("situations")
    if sl and len(sl) == len(m.get("situations") or []):   # 보이는 이름만 (상황 자체는 한국어 그대로 — 앱의 완료 표시·녹화 메타의 열쇠)
        out["situation_labels"] = sl
    return out


FILES = {"meta.json", "video.mp4", "video_frames.csv", "audio.wav", "audio.json", "phone_imu.csv", "beacon.csv",
         "watch_imu.csv", "watch_audio.wav", "watch_audio.json", "watch_vib.csv", "watch_hr.csv", "watch_ibi.csv", "watch_skin.csv",
         "clock_sync.csv", "phone_server_sync.csv", "watch_server_sync.csv", "watch_meta.json"}
WATCH_WAIT_S = 240   # 폰 업로드 뒤 워치 업로드를 기다리는 최대 시간
SID = re.compile(r"^[A-Za-z0-9_\-]{3,64}$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
WEB = config.ROOT / "web"


class Store:
    def __init__(self, root: Path, token: str, use_api: bool = True, collect: bool = False):
        self.root, self.token, self.use_api = root, token, use_api
        self.collect = collect        # 모으기만: 올라온 녹화를 처리하지 않고 미션별로 보관 (나중에 --process-collected 로 한꺼번에)
        for d in ("sessions", "out", "status", "orders", "bundles"):
            (root / d).mkdir(parents=True, exist_ok=True)
        self.q: Queue = Queue()
        self.lock = threading.Lock()
        self.train_lock = threading.Lock()
        self.judge_lock = threading.Lock()
        self.live: dict = {}          # 지금 녹화 중인 세션 {sid, started, watch_joined}
        self.people = People(root / "people")
        self.appeals = Appeals(root / "appeals")
        self.payouts = Payouts(root / "payout")
        self.requests = Requests(root / "requests")
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- 상태 ----
    def status(self, sid: str) -> dict:
        p = self.root / "status" / f"{sid}.json"
        return json.loads(p.read_text()) if p.exists() else {"state": "none"}

    def set_status(self, sid: str, **kw) -> dict:
        with self.lock:
            st = self.status(sid)
            st.update(kw, updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            # 다른 스레드가 읽는 중에 빈 파일을 보지 않게 임시 파일에 쓰고 바꿔 끼운다 (반쯤 쓴 파일을 읽어 작업 스레드가 죽었었다)
            p = self.root / "status" / f"{sid}.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False))
            os.replace(tmp, p)
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
                                  calibration_store=self.root / "grip_calibration", people=self.people,
                                  extra_missions=self.requests.live_missions(),
                                  skip_precheck=bool(self.status(sid).get("skip_precheck")))
                fin = self.set_status(sid, state="done" if "error" not in rep else "rejected", skip_precheck=False, **rep)
                # 실제 서비스(DATACASH_PURGE=1, 다른 사람이 쓰는 내부 시험부터): 승인되면 원본은 바로 지운다.
                # 반려는 이의 신청으로 다시 처리할 수 있게 원본을 남기고, 서버 오류(failed)도 다시 처리하도록 둔다.
                # (운영자가 모아 찍은 시연용 녹화 collected/ 는 원본 ↔ 치환 비교를 위해 남긴다)
                mine = any((self.root / "collected").glob(f"*/{sid}")) if (self.root / "collected").exists() else False
                if os.environ.get("DATACASH_PURGE") == "1" and fin.get("state") == "done" and not mine:
                    from ..jobs import purge_originals

                    fin = self.set_status(sid, originals_deleted=purge_originals(self.root / "sessions" / sid, self.root / "out" / sid),
                                          originals_deleted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            except Exception as e:
                fin = self.set_status(sid, state="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
            self.appeals.reprocessed(sid, fin)
            self.settle(sid)
            if fin.get("state") == "done":           # 모델 진화: 검증된 라벨이 늘었으니 (Fireworks 가 풀리면 쓸) 오픈 웨이트 학습 데이터를 뒤에서 다시 만든다
                threading.Thread(target=self.retrain, daemon=True).start()

    def keep(self, sid: str) -> dict:
        """모으기만 모드: 처리(가리기·라벨·학습) 없이 원본 그대로 두고 collected/<미션>/<녹화> 로 미션별 목록을 만든다."""
        meta = json.loads((self.root / "sessions" / sid / "meta.json").read_text())
        mid = (meta.get("mission") or {}).get("id") or "unknown"
        d = self.root / "collected" / mid
        d.mkdir(parents=True, exist_ok=True)
        link = d / sid
        if not link.exists():
            link.symlink_to(Path("../../sessions") / sid)
        return self.set_status(sid, state="collected", mission=mid, collected_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def collected(self) -> list:
        """모아 둔 녹화 [(미션, 녹화)] — 아직 처리하지 않은 것만."""
        base = self.root / "collected"
        return [(m.name, l.name) for m in sorted(base.iterdir()) if m.is_dir() for l in sorted(m.iterdir())
                if self.status(l.name).get("state") in ("collected", "uploading", "queued", "processing")] if base.exists() else []
        # uploading: 예전 서버가 늦은 워치 파일로 덮어쓴 것 · queued/processing: 처리 중에 서버를 다시 켰을 때

    def credit(self, a: dict) -> None:
        """인정된 추가 보상 → 그 녹화의 입금에 더한다."""
        if a.get("state") == "accepted" and a.get("extra_reward"):
            s = self.status(a["sid"])
            self.set_status(a["sid"], appeal_extra=int(s.get("appeal_extra") or 0) + int(a["extra_reward"]))
            self.settle(a["sid"])

    def judge_appeal(self, aid: str) -> None:
        """이의 신청을 더 높은 모델(Spark Open, 없으면 Spark 1.3 최대 추론)로 다시 본다 (datacash/judge.py). 한 번에 하나씩."""
        from ..judge import judge

        with self.judge_lock:
            a = self.appeals.get(aid)
            sid = a["sid"]
            try:
                res = judge(a, self.status(sid), self.root / "sessions" / sid, self.root / "out" / sid)
            except Exception as e:
                res = {"decision": "review", "note": f"다시 보기 실패 ({type(e).__name__}) → 검토자 확인", "extra": 0, "corrections": []}
            a = self.appeals.judged(aid, res)
            if res["decision"] == "reprocess":
                self.set_status(sid, state="queued", skip_precheck=True)
                self.q.put(sid)
            elif res["decision"] == "accept":
                self.credit(a)
                if res.get("corrections"):              # 바로잡은 라벨 → 모델 진화: 정답 · 다음 Spark 1.3 질문의 예시
                    from .. import evolve

                    evolve.add_correction(sid, res["corrections"], self.status(sid).get("mission", ""))
                    threading.Thread(target=self.retrain, daemon=True).start()

    def retrain(self) -> None:
        from .. import evolve

        with self.train_lock:
            try:
                evolve.build_dataset(sorted(p for p in (self.root / "out").iterdir() if p.is_dir()), log=print)
            except Exception as e:
                print(f"오픈 웨이트 학습 데이터 만들기 실패: {type(e).__name__}: {e}")

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

    def base_missions(self) -> List[dict]:
        return json.loads((Path(__file__).parents[1] / "missions.json").read_text())["missions"]

    def all_missions(self) -> List[dict]:
        """기본 미션 + 구매 기업 의뢰로 열린 미션."""
        return self.base_missions() + self.requests.live_missions()

    def catalog(self, lang: Optional[str] = None) -> dict:
        ms = [localize_mission(m, lang) for m in self.all_missions()]
        rows = self.accepted()
        out = []
        for m in ms:
            mine = [r for r in rows if r["mission"] == m["id"]]
            hours = {lv: round(sum(r["minutes"] for r in mine if pricing.LEVELS.index(r["level"]) >= pricing.LEVELS.index(lv)) / 60, 3)
                     for lv in pricing.LEVELS}
            out.append({**{k: m[k] for k in ("id", "title", "title_en", "guide", "purpose", "objects", "primitives")},
                        "sponsor": m.get("sponsor", ""),
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
                                                   "per_hour": pricing.PER_HOUR[lv], "reward_per_hour": pricing.REWARD_PER_HOUR[lv],
                                                   "sensors": pricing.PRESETS[lv]}
                                                  for lv in pricing.LEVELS],
                "sensors": [{"id": k, "name": v[0], "desc": v[1], "per_hour": v[2], "channels": v[3]} for k, v in pricing.SENSORS.items()],
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

    def sources(self, missions: List[str], level: str, sensors: Optional[list] = None) -> List[Path]:
        """미션 · 고른 센서 묶음의 채널이 다 들어 있는 녹화."""
        need = pricing.required_channels(pricing.sensors_for(level if level in pricing.LEVELS else None, sensors))
        return [self.root / "out" / r["sid"] for r in self.accepted()
                if (not missions or r["mission"] in missions) and need <= set(r["features"])]

    def available_hours(self, missions: List[str], level: str, sensors: Optional[list] = None) -> float:
        need = pricing.required_channels(pricing.sensors_for(level if level in pricing.LEVELS else None, sensors))
        return round(sum(r["minutes"] for r in self.accepted()
                         if (not missions or r["mission"] in missions) and need <= set(r["features"])) / 60, 3)

    def build_bundle(self, oid: str) -> None:
        from ..export.bundle import merge

        o = self.order(oid)
        try:
            srcs = self.sources(o["missions"], o["level"], o.get("sensors"))
            if not srcs:
                raise RuntimeError("조건에 맞는 녹화가 없음")
            work = self.root / "bundles" / oid
            ds = merge(srcs, work / "dataset", o["level"], o.get("sensors"))
            zp = shutil.make_archive(str(work / f"datacash_{oid}"), "zip", ds)
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
    sensors: Optional[List[str]] = None       # 센서 묶음을 골라 사기 (없으면 등급의 조합)
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


class HumanStep(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    action: str
    objects: List[str] = Field(default_factory=list, max_length=5)


class HumanIn(BaseModel):
    steps: List[HumanStep] = Field(max_length=300)
    note: str = Field("", max_length=500)
    seconds: float = Field(0, ge=0)  # 이 녹화를 보며 고친 시간 (화면이 계산) — 너무 짧으면 Spark 라벨을 그대로 저장한 것일 수 있다
    excluded: bool = False          # 정답으로 쓸 수 없는 녹화 (예: 동작이 거의 안 보임) — 채점·학습에서 뺀다


class DonorIn(BaseModel):
    donor: str = Field(max_length=20)


class BioIn(BaseModel):
    bio: bool


class ChatMsg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    messages: List[ChatMsg] = Field(min_length=1, max_length=24)
    lang: str = Field("en", pattern="^(en|ko|ja|zh-CN|es)$")


class RequestIn(BaseModel):
    plan: dict
    org: str = Field(max_length=120)
    email: str = Field(max_length=200)
    messages: List[ChatMsg] = Field(default_factory=list, max_length=24)


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


CHAT_PER_HOUR, CHAT_PER_DAY = 20, 300


def create_app(root: Optional[Path] = None, token: Optional[str] = None, use_api: bool = True, collect: bool = False,
               process_collected: bool = False) -> FastAPI:
    root = Path(root or config.DATA / "server")
    token = token or os.environ.get("DATACASH_TOKEN") or secrets.token_urlsafe(12)
    st = Store(root, token, use_api, collect)
    if process_collected:                     # 모아 둔 녹화를 한꺼번에 처리 (미션 순서대로)
        for _, sid in st.collected():
            st.set_status(sid, state="queued")
            st.q.put(sid)
    app = FastAPI(title="DataCash")
    app.state.store = st

    app_token = os.environ.get("DATACASH_APP_TOKEN") or ""   # 나눠 주는 앱에 넣는 참여자 토큰 (운영자 화면은 못 연다)

    def auth(t: Optional[str]):
        if t != st.token and not (app_token and t == app_token):
            raise HTTPException(401, "토큰이 맞지 않음")

    chat_calls: list = []

    def admin_auth(t: Optional[str]):
        if t != st.token:
            raise HTTPException(401, "토큰이 맞지 않음")

    # ---- 앱 ----
    @app.get("/api/missions")
    def missions(x_dc_token: Optional[str] = Header(None), x_dc_age: Optional[str] = Header(None),
                 x_dc_gender: Optional[str] = Header(None), x_dc_lang: Optional[str] = Header(None)):
        auth(x_dc_token)
        import urllib.parse

        age = urllib.parse.unquote(x_dc_age or "")
        base = st.base_missions()
        ms = base + [m for m in st.requests.live_missions() if eligible(m, age, x_dc_gender)]
        cat = {m["id"]: m for m in st.catalog()["missions"]}
        today = pricing.today_mission([m["id"] for m in base])
        for m in ms:
            m["today"] = m["id"] == today
            have = cat.get(m["id"], {}).get("recordings", 0)
            m["reward_per_minute"] = {lv: round(pricing.REWARD_PER_HOUR[lv] / 60) for lv in pricing.LEVELS}
            m["scarce"] = have < 3          # 아직 적게 모인 미션 → 보너스
            m["bonus_pct"] = int(pricing.BONUS_SCARCE * 100) if m["scarce"] else 0
            if m.get("request"):           # 의뢰 미션: 목표 시간 대비 모은 시간, 보너스 대신 의뢰 보상
                m["collected_hours"] = round(cat.get(m["id"], {}).get("hours", {}).get("basic", 0), 3)
                m["scarce"], m["bonus_pct"] = False, 0
        prims = json.loads((Path(__file__).parents[1] / "missions.json").read_text()).get("primitives_ko", {})
        return {"missions": [localize_mission(m, x_dc_lang) for m in ms], "primitives_ko": prims}

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
        if st.status(sid).get("state") in ("none", "uploading", "upload_failed", None):   # 폰이 올리는 중일 때만 — 뒤늦은 워치 파일이 보관됨·처리 중을 덮지 않게
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
            if st.status(sid).get("originals_deleted"):
                raise HTTPException(410, "원본은 처리가 끝나 지웠어요 — 폰에서 다시 올려야 다시 처리할 수 있어요")
            raise HTTPException(400, "video.mp4 가 필요함")
        if st.collect and not calib:          # 모으기만: 처리하지 않고 미션별로 보관 (보정은 바로 처리 — API 를 안 쓴다)
            return st.keep(sid)
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
            a = st.appeals.file(dev, sid, b.reason, b.detail, st.status(sid), requeue, lambda s: st.set_status(s, withheld=True),
                                review_model=True)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if a["state"] == "judging":                     # 다시 보기는 뒤에서 (1~2분) — 앱은 이의 신청 목록에서 결과를 본다
            threading.Thread(target=st.judge_appeal, args=(a["id"],), daemon=True).start()
        st.credit(a)
        return a

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

    # ---- 워치 시험 (워치3 타이젠 브라우저로 GripSense 가 되는지) — 토큰 없이 같은 와이파이에서만, 시험 데이터만 받는다 ----
    @app.get("/w", response_class=HTMLResponse)
    def watchtest_page():
        return (WEB / "watchtest.html").read_text()

    @app.post("/api/watchtest/{kind}")
    async def watchtest_post(kind: str, request: Request):
        if kind not in ("check", "segment"):
            raise HTTPException(404)
        body = await request.body()
        if len(body) > 20_000_000:
            raise HTTPException(413)
        d = json.loads(body)
        run = str(d.get("run", ""))
        if not re.match(r"^w\d{6,20}$", run):
            raise HTTPException(400)
        out = root / "watchtest" / run
        out.mkdir(parents=True, exist_ok=True)
        name = "check.json" if kind == "check" else f"seg_r{int(d['round'])}_l{int(d['level'])}.json"
        (out / name).write_text(json.dumps(d))
        return JSONResponse({"ok": True}, headers={"Access-Control-Allow-Origin": "*"})

    @app.get("/api/watchtest/result")
    def watchtest_result(run: str = Query("")):
        from ..touch.watchtest import evaluate_run

        if not re.match(r"^w\d{6,20}$", run):
            raise HTTPException(400)
        return JSONResponse(evaluate_run(root / "watchtest" / run), headers={"Access-Control-Allow-Origin": "*"})

    @app.get("/app.apk")
    def apk():                               # 내부 시험: 참여자 앱 내려받기 (data/server/dist/datacash.apk)
        p = root / "dist" / "datacash.apk"
        if not p.exists():
            raise HTTPException(404, "앱 파일이 아직 없음")
        return FileResponse(p, media_type="application/vnd.android.package-archive", filename="DataCash.apk")

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
    def catalog(lang: Optional[str] = None):
        return st.catalog(lang)

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

    @app.get("/api/market/sample/{sid}/lerobot")
    def lerobot_view(sid: str):
        """LeRobot v3.0 묶음 안을 들여다보기: 파일 목록 · 특징(이름·모양) · 가운데 장면 한 줄의 값 · 작업 이름."""
        import pyarrow.parquet as pq

        ds = _ds(sid)
        info = json.loads((ds / "meta" / "info.json").read_text())
        files = [{"path": str(p.relative_to(ds)), "bytes": p.stat().st_size} for p in sorted(ds.rglob("*")) if p.is_file()]
        t = pq.read_table(ds / "data" / "chunk-000" / "file-000.parquet")
        # 보여 줄 장면: 값이 가장 많이 찬 줄 (손이 안 보이거나 안 쥔 장면은 0 투성이라 설명이 안 된다)
        filled = sum((np.array(t.column(k).to_pylist(), np.float64) != 0).sum(1)
                     for k in ("observation.hands", "observation.touch", "observation.physio", "observation.objects") if k in t.column_names)
        i = int(np.argmax(filled)) if np.ndim(filled) else t.num_rows // 2
        row = {k: v[0] for k, v in t.slice(i, 1).to_pydict().items()}
        feats = []
        for k, f in info["features"].items():
            names = f.get("names") or []
            v = row.get(k)
            if isinstance(v, list):
                v = [round(float(x), 3) for x in v]
            elif isinstance(v, float):
                v = round(v, 3)
            feats.append({"key": k, "dtype": f["dtype"], "shape": f["shape"],
                          "names": names if f["dtype"] != "video" else [], "value": v})
        tasks = pq.read_table(ds / "meta" / "tasks.parquet").column("task").to_pylist()
        return {"codebase_version": info.get("codebase_version"), "robot_type": info.get("robot_type"), "fps": info.get("fps"),
                "total_episodes": info.get("total_episodes"), "total_frames": info.get("total_frames"), "tasks": tasks,
                "frame": i, "files": files, "features": feats}

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
            r = pricing.quote(q.hours, q.level, q.custom, q.refine, q.sensors)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {**r, "available_hours": st.available_hours(q.missions, q.level, q.sensors), "missions": q.missions}

    @app.post("/api/market/orders")
    def order(o: OrderIn):
        if not o.org.strip() or not EMAIL.match(o.email):
            raise HTTPException(400, "기관 이름과 이메일을 확인해 주세요")
        try:
            q = pricing.quote(o.hours, o.level, o.custom, o.refine, o.sensors)
        except ValueError as e:
            raise HTTPException(400, str(e))
        rec = {"id": secrets.token_urlsafe(12), "org": o.org.strip()[:120], "email": o.email.strip()[:200],
               "note": o.note[:1000], "missions": o.missions, "level": q["level"], "sensors": q["sensors"], "quote": q, "status": "pending",
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

    @app.post("/api/market/request/instant")
    def request_instant(b: ChatIn):
        """모델 없이 바로 (0.1초): 문장에서 예산·연령·센서를 읽고 보관 미션과 짝지어 기존·신규·섞기를 보여 준다."""
        from ..requests import instant, options

        cat = st.catalog()
        plan = instant([m.model_dump() for m in b.messages], cat)
        return {"plan": plan, "choice": options(plan, cat)}

    @app.post("/api/market/request/chat")
    def request_chat(b: ChatIn, request: Request):
        if not st.use_api:
            raise HTTPException(503, "Meta API 가 꺼진 서버예요")
        # 공개 주소(터널)로 열려 있어 아무나 부를 수 있다 → 사람(IP)마다 시간당 CHAT_PER_HOUR 번, 하루 전체 CHAT_PER_DAY 번까지
        ip = request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")
        now = time.time()
        calls = [t for t in chat_calls if now - t[1] < 86400]
        chat_calls[:] = calls
        if len(calls) >= CHAT_PER_DAY or sum(1 for i, t in calls if i == ip and now - t < 3600) >= CHAT_PER_HOUR:
            raise HTTPException(429, "잠시 뒤에 다시 해 주세요 (의뢰 대화 횟수 제한)")
        chat_calls.append((ip, now))
        if len(b.messages) > 30 or sum(len(m.content) for m in b.messages) > 8000:
            raise HTTPException(413, "대화가 너무 길어요")
        from ..requests import chat

        try:
            return chat([m.model_dump() for m in b.messages], catalog=st.catalog(), lang=b.lang)
        except Exception as e:
            raise HTTPException(502, f"의뢰를 이해하지 못했어요: {type(e).__name__}")

    @app.post("/api/market/request")
    def request_submit(b: RequestIn):
        if not b.org.strip() or not EMAIL.match(b.email):
            raise HTTPException(400, "기관 이름과 이메일을 확인해 주세요")
        try:
            r = st.requests.submit(b.plan, b.org, b.email, [m.model_dump() for m in b.messages])
        except ValueError as e:
            raise HTTPException(400, str(e))
        return request_view(r)

    def request_view(r: dict) -> dict:
        cat = {m["id"]: m for m in st.catalog()["missions"]}
        return {"id": r["id"], "title": r["title"], "status": r["status"], "quote": r["quote"], "eligibility": r["plan"]["eligibility"],
                "missions": [{"id": m["id"], "title": m["title"], "target_hours": m["target_hours"],
                              "collected_hours": cat.get(m["id"], {}).get("hours", {}).get("basic", 0),
                              "recordings": cat.get(m["id"], {}).get("recordings", 0)} for m in r["missions"]]}

    @app.get("/api/market/request/{rid}")
    def request_status(rid: str):
        try:
            return request_view(st.requests.get(rid))
        except KeyError:
            raise HTTPException(404)

    # ---- 운영자 ----
    @app.get("/api/admin/requests")
    def admin_requests(token: str = Query("")):
        admin_auth(token)
        return {"requests": [request_view(r) | {"org": r["org"], "email": r["email"]} for r in st.requests.all()]}

    @app.post("/api/admin/requests/{rid}/close")
    def admin_request_close(rid: str, token: str = Query("")):
        admin_auth(token)
        try:
            return request_view(st.requests.close(rid))
        except KeyError:
            raise HTTPException(404)

    @app.get("/api/admin/orders")
    def admin_orders(token: str = Query("")):
        admin_auth(token)
        return {"orders": sorted((json.loads(p.read_text()) for p in (root / "orders").glob("*.json")),
                                 key=lambda o: o["created"], reverse=True)}

    @app.post("/api/admin/orders/{oid}/approve")
    def approve(oid: str, bg: BackgroundTasks, token: str = Query("")):
        admin_auth(token)
        o = st.order(oid)
        o["status"] = "building"
        st.save_order(o)
        bg.add_task(st.build_bundle, oid)
        return {"id": oid, "status": "building"}

    @app.get("/api/admin/sessions")
    def admin_sessions(token: str = Query("")):
        """녹화 목록: 상태 · 미션 · 보상 · 단계별 처리 시간 · 얼굴 치환(가상 인물·기준 넘은 장면) · 라벨을 붙인 모델."""
        admin_auth(token)
        rows = []
        for p in sorted((root / "status").glob("*.json"), reverse=True):
            s = json.loads(p.read_text())
            sid = p.stem
            cert = root / "out" / sid / "certificate.json"
            fs = {}
            if cert.exists():
                fs = ((json.loads(cert.read_text()).get("privacy") or {}).get("face_swap") or {})
            rows.append({"sid": sid, "state": s.get("state"), "mission": s.get("mission"), "level": s.get("level"),
                         "minutes": s.get("minutes"), "reward": s.get("reward"), "appeal_extra": s.get("appeal_extra", 0),
                         "error": s.get("error"), "updated": s.get("updated"), "seconds": s.get("seconds") or {},
                         "labels_by": s.get("labels_by") or ("muse-spark-1.3" if s.get("state") == "done" else None),
                         "donor": fs.get("donor"), "frames": fs.get("frames"), "swapped": fs.get("swapped"),
                         "over_reid": fs.get("over_reid"), "donor_trial": s.get("donor_trial"),
                         "original": (root / "sessions" / sid / "video.mp4").exists(),
                         "swap": (root / "out" / sid / "masked_swap.mp4").exists()})
        return {"sessions": rows}

    @app.get("/api/admin/sessions/{sid}/video")
    def admin_video(sid: str, kind: str = Query("swap"), token: str = Query("")):
        """운영자 비교용 영상: swap(판매용, 얼굴 치환·가림) · original(원본 — 시연 서버에만 남아 있다)."""
        admin_auth(token)
        if not SID.match(sid) or kind not in ("swap", "original"):
            raise HTTPException(404)
        p = root / "out" / sid / "masked_swap.mp4" if kind == "swap" else root / "sessions" / sid / "video.mp4"
        if not p.exists():
            raise HTTPException(404, "영상이 없음 (원본은 처리 후 지워졌을 수 있음)")
        return FileResponse(p, media_type="video/mp4")

    # ---- 사람 정답: 운영자가 녹화를 보며 단계를 직접 단다 (/label). 우리 녹화의 진짜 정확도 재기 · Glimmer 학습 정답 ----
    HUMAN = config.DATA / "evolve" / "human"
    MTITLE = {m["id"]: m["title"] for m in json.loads((config.ROOT / "datacash" / "missions.json").read_text())["missions"]}

    @app.get("/label", response_class=HTMLResponse)
    def label_page():
        return (WEB / "label.html").read_text()

    @app.get("/api/admin/label")
    def label_list(token: str = Query("")):
        admin_auth(token)
        rows = []
        for p in sorted((root / "status").glob("*.json"), reverse=True):
            s, sid = json.loads(p.read_text()), p.stem
            cert = root / "out" / sid / "certificate.json"
            if s.get("state") != "done" or not cert.exists() or not (root / "out" / sid / "masked_swap.mp4").exists():
                continue
            h = HUMAN / f"{sid}.json"
            hj = json.loads(h.read_text()) if h.exists() else None
            rows.append({"sid": sid, "mission": MTITLE.get(s.get("mission"), s.get("mission")), "minutes": s.get("minutes"),
                         "spark_steps": len(json.loads(cert.read_text()).get("steps") or []),
                         "human_steps": len(hj["steps"]) if hj else None, "labeled_at": hj and hj.get("labeled_at"),
                         "excluded": bool(hj and hj.get("excluded")),
                         "seconds": hj and hj.get("seconds")})
        return {"sessions": rows}

    @app.get("/api/admin/label/{sid}")
    def label_get(sid: str, token: str = Query("")):
        admin_auth(token)
        cert = root / "out" / sid / "certificate.json"
        if not SID.match(sid) or not cert.exists():
            raise HTTPException(404)
        c = json.loads(cert.read_text())
        h = HUMAN / f"{sid}.json"
        names = json.loads((config.ROOT / "datacash" / "missions.json").read_text())["primitives_ko"]
        m = c.get("mission") or {}
        m = m if isinstance(m, dict) else {"id": m}
        return {"sid": sid, "mission": " · ".join(x for x in (MTITLE.get(m.get("id"), m.get("title") or m.get("id")), m.get("situation")) if x), "actions": [[a, names.get(a, a)] for a in PRIMITIVES],
                "spark": [{k: s.get(k) for k in ("start_s", "end_s", "action", "objects")} for s in c.get("steps") or []],
                "human": json.loads(h.read_text()) if h.exists() else None}

    @app.put("/api/admin/label/{sid}")
    def label_put(sid: str, b: HumanIn, token: str = Query("")):
        admin_auth(token)
        if not SID.match(sid) or not (root / "out" / sid / "certificate.json").exists():
            raise HTTPException(404)
        steps = sorted((s.model_dump() for s in b.steps), key=lambda s: s["start_s"])
        for s in steps:
            if s["action"] not in PRIMITIVES or s["end_s"] <= s["start_s"]:
                raise HTTPException(400, f"단계가 맞지 않음: {s}")
            s["start_s"], s["end_s"] = round(s["start_s"], 2), round(s["end_s"], 2)
            s["objects"] = [o.strip()[:40] for o in s["objects"] if o.strip()]
        HUMAN.mkdir(parents=True, exist_ok=True)
        out = {"sid": sid, "steps": [] if b.excluded else steps, "note": b.note, "excluded": b.excluded, "seconds": round(b.seconds),
               "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (HUMAN / f"{sid}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return out

    @app.get("/api/admin/costs")
    def admin_costs(token: str = Query("")):
        """Meta API 사용액: 합계·한도·모델별·날짜별·최근 호출."""
        admin_auth(token)
        from ..meta.client import _ledger_rows

        rows = _ledger_rows()
        by_model, by_day = {}, {}
        for r in rows:
            by_model[r["model"]] = round(by_model.get(r["model"], 0) + r.get("usd", 0), 4)
            d = str(r.get("t", ""))[:10]
            by_day[d] = round(by_day.get(d, 0) + r.get("usd", 0), 4)
        from ..understand import spark_open

        fw = [json.loads(l) for l in spark_open.LEDGER.read_text().splitlines() if l.strip()] if spark_open.LEDGER.exists() else []
        def credit(name):          # Spark Open 을 부른 곳마다 (예전 기록은 provider 가 없어 Fireworks)
            rs = [r for r in fw if r.get("provider", "fireworks") == name]
            return {"total": round(sum(r.get("usd", 0) for r in rs), 4), "calls": len(rs), "credit": 100.0, "estimated_price": True,
                    "recent": [{k: r.get(k) for k in ("t", "model", "usd", "what")} for r in rs[-6:]][::-1]}

        return {"fireworks": credit("fireworks"), "baseten": credit("baseten"), "total": round(sum(r.get("usd", 0) for r in rows), 4), "budget": config.API_BUDGET_USD, "calls": len(rows),
                "by_model": dict(sorted(by_model.items(), key=lambda kv: -kv[1])), "by_day": by_day,
                "recent": [{k: r.get(k) for k in ("t", "model", "usd", "what")} for r in rows[-12:]][::-1]}

    @app.get("/api/admin/evolve")
    def admin_evolve(token: str = Query("")):
        """모델 진화: 녹화마다 Spark 1.3 ↔ Spark Open 일치(agreement) · 2차 검증 표 · 바로잡음 · 정답 예시 수."""
        admin_auth(token)
        from .. import evolve

        cross = [json.loads(p.read_text()) for p in sorted((evolve.ROOT / "crosscheck").glob("*.json"))] \
            if (evolve.ROOT / "crosscheck").exists() else []
        fb = [json.loads(p.read_text()) for p in sorted((evolve.ROOT / "feedback").glob("*.json"))] \
            if (evolve.ROOT / "feedback").exists() else []
        fixes = [f for r in fb for f in r["fixes"]]
        from ..understand import spark_open

        return {**evolve._registry(), "reviewer": spark_open.provider_name() or None,
                "fixes": {"review": sum(f.get("source") == "review" for f in fixes), "appeal": sum(f.get("source", "appeal") == "appeal" for f in fixes)},
                "crosscheck": [{k: c[k] for k in ("recording", "mission", "steps", "agree", "gold")}
                               | {"diffs": [{"t": round(r["start_s"], 1), "spark13": r["spark13"], "spark_open": r["spark_open"]}
                                            for r in c["rows"] if not r["agree"]]} for c in cross]}

    @app.get("/api/admin/perf")
    def admin_perf(token: str = Query("")):
        """성능 모니터링: 사람 정답 기준 벤치마크 + 녹화마다 쌓이는 일치율·처리 시간·재식별 · Glimmer 버전 · 경고 (datacash/perf.py)."""
        admin_auth(token)
        from .. import perf

        return perf.collect(root)

    @app.get("/api/admin/appeals")
    def admin_appeals(token: str = Query("")):
        admin_auth(token)
        return {"appeals": st.appeals.all()}

    @app.post("/api/admin/appeals/{aid}")
    def admin_decide(aid: str, b: DecisionIn, token: str = Query("")):
        admin_auth(token)
        try:
            a = st.appeals.decide(aid, b.accept, b.extra_reward, b.note)
        except KeyError:
            raise HTTPException(404)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if a["reason"] == "privacy" and not b.accept:      # 가리기에 문제 없다고 판단 → 다시 판매
            st.set_status(a["sid"], withheld=False)
        st.credit(a)
        return a

    return app


def main():
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser("datacash-serve")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--root", default=None)
    ap.add_argument("--no-api", action="store_true", help="Meta API 를 부르지 않는다 (시험용)")
    ap.add_argument("--collect", action="store_true", help="모으기만: 올라온 녹화를 처리하지 않고 미션별로 보관")
    ap.add_argument("--process-collected", action="store_true", help="모아 둔 녹화를 한꺼번에 처리")
    a = ap.parse_args()
    config.load_env()
    from ..meta.mask_fast import warm

    warm()                               # SAM 마스크 해독기를 미리 컴파일 (처리 중 첫 컴파일이 멈추지 않게)
    token = os.environ.get("DATACASH_TOKEN")
    if token:                            # 정해 준 토큰은 로그에 남기지 않는다
        print("DataCash 서버 토큰: DATACASH_TOKEN 에서 읽음")
    else:
        token = secrets.token_urlsafe(12)
        print(f"DataCash 서버 토큰(임시): {token}  (앱 X-DC-Token, 운영자 /admin?token=)")
    beacon = advertise(a.port)
    try:
        uvicorn.run(create_app(a.root, token, not a.no_api, a.collect, a.process_collected), host=a.host, port=a.port)
    finally:
        if beacon:
            beacon.terminate()


SERVICE = "_datacash._tcp"


def advertise(port: int):
    """같은 와이파이에 'DataCash 서버가 여기 있다'를 알린다 (mDNS) — 맥 IP 가 바뀌어도 앱·워치가 이 이름으로 찾는다.
    맥에 있는 dns-sd 를 쓴다 (설치할 것 없음). 다른 운영체제면 건너뛴다 (앱 설정에서 주소를 직접 넣는다)."""
    import shutil
    import subprocess

    if not shutil.which("dns-sd"):
        return None
    return subprocess.Popen(["dns-sd", "-R", "DataCash", SERVICE, "local", str(port), "app=datacash"],   # 빈 TXT 면 안드로이드가 오류를 남긴다
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
