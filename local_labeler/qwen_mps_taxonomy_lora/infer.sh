#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

INPUT=${1:-./data_mlx/test_source.csv}
OUTPUT=${2:-./predictions/test_predictions.csv}

python scripts/infer_mlx.py   --input "$INPUT"   --output "$OUTPUT"   --model ./models/qwen3-8b-4bit   --adapter-path ./adapters/taxonomy_qwen3_8b   --prompt-style compact   --max-tokens 900
