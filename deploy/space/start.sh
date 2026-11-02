#!/bin/bash
# 시작할 때 시연 데이터(seed)를 쓸 수 있는 곳에 깐다. 이미 있으면 그대로 (재시작해도 올라온 녹화가 남게 — 무료 Space 는 디스크가 초기화될 수 있다).
D=${DATACASH_DATA:-/home/user/data}
if [ ! -d "$D/server/out" ]; then
  mkdir -p "$D" && cp -R seed/. "$D/"
fi
mkdir -p "$D/server/dist" && cp -R seed/server/dist/. "$D/server/dist/" 2>/dev/null   # 앱 파일은 늘 새 것으로
exec python -m datacash.server.app --port 7860
