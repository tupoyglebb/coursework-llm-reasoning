#!/usr/bin/env python3
"""CLI-обёртка: считает метрики предсказаний и пишет metrics.json.

Вся логика — в metrics.compute_metrics (общий код с валидацией внутри обучения).
  python scripts/evaluate_predictions.py --predictions ./predictions/test_pred_E0.csv --out-json ./predictions/metrics_E0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted taxonomy columns against teacher labels.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=Path("metrics.json"))
    args = parser.parse_args()

    metrics = compute_metrics(pd.read_csv(args.predictions))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
