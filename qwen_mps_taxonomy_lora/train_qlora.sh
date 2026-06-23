#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

MODEL_DIR=./models/qwen3-8b-4bit
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}

if [ ! -d "$MODEL_DIR" ]; then
  echo "Converting $BASE_MODEL to 4-bit MLX at $MODEL_DIR"
  mlx_lm.convert --model "$BASE_MODEL" -q --mlx-path "$MODEL_DIR"
else
  echo "Using existing MLX model: $MODEL_DIR"
fi

mlx_lm.lora --config ./configs/qwen3_8b_qlora_32gb.yaml
