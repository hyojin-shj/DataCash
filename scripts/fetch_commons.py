"""Wikimedia Commons 에서 라이선스(CC BY·CC BY-SA·CC0·퍼블릭 도메인)가 확인된 영상만 찾아 받는다 — 가리기 평가용.
출처·라이선스는 data/opendata/manifest_extra.json 에 남긴다 (README 출처 표에 옮긴다).

  python scripts/fetch_commons.py "living room television" "mirror room" --max 3
"""

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

from datacash import config

API = "https://commons.wikimedia.org/w/api.php"
OK = re.compile(r"^(CC BY(-SA)? \d|CC0|Public domain)", re.I)
UA = {"User-Agent": "DataCash/0.1 (hackathon research; https://github.com/hyojin-shj/DataCash)"}


def search(q: str, limit: int = 30):
    r = httpx.get(API, params={"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
                               "gsrsearch": f"{q} filetype:video", "gsrlimit": limit, "prop": "imageinfo",
                               "iiprop": "url|size|mime|extmetadata"}, headers=UA, timeout=60)
    pages = (r.json().get("query") or {}).get("pages", {})
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        md = ii.get("extmetadata", {})
        lic = (md.get("LicenseShortName") or {}).get("value", "")
        yield {"title": p["title"], "url": ii.get("url"), "page": ii.get("descriptionurl"), "bytes": ii.get("size", 0),
               "mime": ii.get("mime"), "license": lic, "artist": re.sub("<[^>]+>", "", (md.get("Artist") or {}).get("value", ""))[:200],
               "query": q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--max", type=int, default=2)
    ap.add_argument("--max-mb", type=float, default=60)
    a = ap.parse_args()
    out = config.OPENDATA / "extra"
    out.mkdir(parents=True, exist_ok=True)
    man_p = config.OPENDATA / "manifest_extra.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else []
    have = {m["title"] for m in man}
    for q in a.queries:
        got = 0
        for it in search(q):
            if got >= a.max or it["title"] in have or not OK.match(it["license"]) or it["bytes"] > a.max_mb * 1e6:
                continue
            name = re.sub(r"[^\w.\-]+", "_", it["title"].split(":", 1)[1])
            dst = out / name
            with httpx.stream("GET", it["url"], headers=UA, timeout=300, follow_redirects=True) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for ch in r.iter_bytes():
                        f.write(ch)
            it["file"] = str(dst.relative_to(config.OPENDATA))
            man.append(it)
            have.add(it["title"])
            got += 1
            print(q, "→", it["title"], it["license"], round(it["bytes"] / 1e6, 1), "MB", flush=True)
    man_p.write_text(json.dumps(man, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
