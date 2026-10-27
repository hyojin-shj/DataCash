#!/usr/bin/env bash
# ③ Glimmer QLoRA 학습 — 클라우드 NVIDIA GPU(H100 1장)에서 돌린다. 맥에서는 돌리지 않는다 (공식 학습 도구가 CUDA 전용).
#
#   이 맥:     python scripts/evolve.py build                 # 학습 데이터 → data/evolve/sft/
#              (data/evolve/sft 와 train/ 을 GPU 서버로 올린다)
#   GPU 서버:  bash scripts/train_glimmer.sh v1               # 아래
#   이 맥:     (adapters/glimmer-lora-v1.gguf 를 data/evolve/adapters/ 로 받아 온다)
#              python scripts/evolve.py eval --adapter data/evolve/adapters/glimmer-lora-v1.gguf
#
# 공식 문서 기준: Axolotl 0.19.0, flash-attn 2.8.3, CUDA 13 + PyTorch cu130, deepspeed 추가 기능은 설치하지 않는다.
set -euo pipefail
VER="${1:-v1}"

uv venv --python 3.12 .venv-train && source .venv-train/bin/activate
uv pip install "axolotl==0.19.0" "flash-attn==2.8.3" --torch-backend=cu130

axolotl train train/glimmer_qlora.yaml

# llama.cpp 에 붙일 수 있게 LoRA 를 GGUF 로 (맥에서 어댑터를 붙여 띄운다: llama-server --lora …)
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
uv pip install -r /tmp/llama.cpp/requirements/requirements-convert_lora_to_gguf.txt
mkdir -p adapters
python /tmp/llama.cpp/convert_lora_to_gguf.py out/glimmer-lora --outfile "adapters/glimmer-lora-${VER}.gguf" --outtype q8_0
echo "어댑터: adapters/glimmer-lora-${VER}.gguf → 맥의 data/evolve/adapters/ 로 받아 오세요"
