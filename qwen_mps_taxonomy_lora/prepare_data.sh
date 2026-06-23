#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

# Replace the path below with your 10k teacher-labeled file.
# It must contain metadata + stage1_reasoning_text + the taxonomy columns from making(2).ipynb.
INPUT=${1:-openai_part_1_annotations/predictions_part_1_final.csv}

python scripts/prepare_mlx_data.py   --input "$INPUT"   --out-dir ./data_mlx   --n 10000   --train-frac 0.90   --valid-frac 0.05   --prompt-style compact   --seed 42
