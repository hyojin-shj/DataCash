#!/bin/bash
# 심사·시연용: 서버 + 외부 주소(cloudflared 빠른 터널)를 한 번에 연다.
#   scripts/serve_public.sh          서버가 안 떠 있으면 띄우고, 새 터널을 열어 server.txt 에 주소를 쓴다
# 터널 주소는 열 때마다 바뀐다. 이미 깔린 앱은 저장소의 server.txt(공개 저장소의 raw 주소)에서 새 주소를 읽으니,
# 주소가 바뀌면 server.txt 를 커밋·푸시하면 된다 (앱을 다시 빌드할 필요 없음). 웹 링크도 server.txt 의 주소.
set -e
cd "$(dirname "$0")/.."
D=data/server

if ! curl -s -o /dev/null -m 3 localhost:8800/api/time; then
  DATACASH_TOKEN=$(cat $D/.admin_token) DATACASH_APP_TOKEN=$(cat $D/.app_token) \
    nohup .venv/bin/python -m datacash.server.app --port 8800 > $D/server.log 2>&1 &
  for i in $(seq 1 60); do curl -s -o /dev/null -m 2 localhost:8800/api/time && break; sleep 1; done
fi

pkill -f "cloudflared tunnel --no-autoupdate --url http://localhost:8800" || true
nohup cloudflared tunnel --no-autoupdate --url http://localhost:8800 > $D/tunnel.log 2>&1 &
for i in $(seq 1 30); do
  U=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" $D/tunnel.log | head -1)
  [ -n "$U" ] && break; sleep 1
done
[ -z "$U" ] && { echo "터널 주소를 못 받음 — $D/tunnel.log 확인"; exit 1; }
echo "$U" > server.txt

pgrep -x caffeinate > /dev/null || nohup caffeinate -ims > /dev/null 2>&1 &   # 심사 중 맥이 잠들지 않게 (전원 연결해 둘 것)

echo "웹: $U"
echo "앱: $U/app.apk"
echo "주소가 바뀌었으면: git add server.txt && git commit -m '서버 주소' && git push"
