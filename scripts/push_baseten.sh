#!/bin/bash
# 얼굴 치환 GPU 모델을 Baseten 에 올린다 (deploy/baseten_faceswap). 키는 .env 의 baseten_deploy (Personal 키 — 해커톤 키 baseten_api 는 Spark Open 호출만 된다).
# 끝나면 나오는 predict 주소를 클라우드 서버 비밀값 DATACASH_BASETEN_URL 에 넣는다.
set -e
cd "$(dirname "$0")/.."
T=deploy/baseten_faceswap
P=$T/packages
rm -rf $P && mkdir -p $P/datacash/privacy $P/data/models
cp datacash/__init__.py datacash/config.py datacash/media.py $P/datacash/
cp datacash/privacy/__init__.py datacash/privacy/faceswap.py datacash/privacy/mps_speed.py datacash/privacy/swap_job.py $P/datacash/privacy/
cp -R datacash/donors $P/datacash/donors
cp data/models/face_landmarker.task data/models/yunet.onnx data/models/sface.onnx $P/data/models/

TRUSS=data/truss_venv/bin/truss
[ -x $TRUSS ] || { python3 -m venv data/truss_venv && data/truss_venv/bin/pip install -q truss; }
KEY=$(grep -E '^baseten_deploy=' .env | cut -d= -f2-)
$TRUSS login --api-key "$KEY" > /dev/null
$TRUSS push $T --publish --wait
