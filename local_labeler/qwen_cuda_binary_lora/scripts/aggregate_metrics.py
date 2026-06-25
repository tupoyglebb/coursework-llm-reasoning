#!/usr/bin/env python3
"""CLI-обёртка: сводит metrics_E*.json к метрикам успеха плана и дозаписывает строку в results.csv.

Метрики (см. TIER0_PLAN.md §2):
  - mean_f1_all      : средний macro-F1 по всем 37 полям (PRIMARY). База prod = 0.667
  - mean_f1_collapsed: средний macro-F1 по 14 «схлопнувшимся» полям (SECONDARY). База prod ≈ 0.480
  - mean_f1_good     : средний macro-F1 по 10 «хорошим» полям (GUARDRAIL). База prod ≈ 0.943

Вся логика — в metrics.summarize/append_results_csv (общий код с валидацией внутри обучения).
  python scripts/aggregate_metrics.py --metrics ./predictions/metrics_E0.json --label E0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import append_results_csv, format_summary, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True, help="metrics_E*.json от evaluate_predictions.py")
    ap.add_argument("--label", type=str, required=True, help="метка эксперимента, напр. E0")
    ap.add_argument("--results-csv", type=Path, default=Path("experiments/results.csv"))
    args = ap.parse_args()

    summary = summarize(json.loads(args.metrics.read_text(encoding="utf-8")))
    print(format_summary(args.label, summary))

    append_results_csv(args.results_csv, args.label, summary)
    print(f"\nДописано в {args.results_csv}")


if __name__ == "__main__":
    main()
