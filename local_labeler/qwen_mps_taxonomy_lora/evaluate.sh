#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

PRED=${1:-./predictions/test_predictions.csv}
python scripts/evaluate_predictions.py --predictions "$PRED" --out-json ./predictions/metrics.json
