#!/usr/bin/env python3
"""Сводит metrics_E*.json к метрикам успеха из плана экспериментов и пишет строку в results.csv.

Метрики (см. TIER0_PLAN.md §2):
  - mean_f1_all      : средний macro-F1 по всем 37 полям (PRIMARY). База prod = 0.667
  - mean_f1_collapsed: средний macro-F1 по 14 «схлопнувшимся» полям (SECONDARY). База prod ≈ 0.480
  - mean_f1_good     : средний macro-F1 по 10 «хорошим» полям (GUARDRAIL). База prod ≈ 0.943

Использование:
  python scripts/aggregate_metrics.py --metrics ./predictions/metrics_E1.json --label E1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Подмножества полей зафиксированы по prod-baseline (metrics.json), чтобы сравнения между
# экспериментами были на одном и том же наборе. НЕ пересчитывать по каждому прогону.
COLLAPSED_FIELDS = [
    "common_interests_prioritization_direction",
    "majority_reliance_causality",
    "moral_style_reasoning_present",
    "source_reliance_direction",
    "incentive_effect_direction",
    "possibility_of_different_opinions",
    "ratio_of_goals_and_means_causality",
    "fear_negative_evaluation_causality",
    "self_deception_causality",
    "common_interests_prioritization_causality",
    "majority_reliance_direction",
    "emotional_reasoning_causality",
    "private_interests_prioritization_direction",
    "overcorrection_concern_causality",
]
GOOD_FIELDS = [
    "treatment_awareness_causality",
    "cruelty_causality",
    "rational_style_reasoning_present",
    "emergency_urgency_causality",
    "social_reward_causality",
    "public_recognition_causality",
    "legality_assessment_causality",
    "privacy_factor_causality",
    "treatment_effect_direction",
    "personal_cost_burden_causality",
]

# Точка отсчёта (prod-модель) для дельт.
BASELINE = {"all": 0.667, "collapsed": 0.480, "good": 0.943}


def mean_f1(per_column: dict, fields: list[str]) -> float:
    vals = [per_column[f]["macro_f1"] for f in fields if f in per_column]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True, help="metrics_E*.json от evaluate_predictions.py")
    ap.add_argument("--label", type=str, required=True, help="метка эксперимента, напр. E1")
    ap.add_argument("--results-csv", type=Path, default=Path("experiments/results.csv"))
    args = ap.parse_args()

    m = json.loads(args.metrics.read_text(encoding="utf-8"))
    pc = m["per_column"]

    row = {
        "label": args.label,
        "mean_f1_all": round(mean_f1(pc, list(pc.keys())), 4),
        "mean_f1_collapsed": round(mean_f1(pc, COLLAPSED_FIELDS), 4),
        "mean_f1_good": round(mean_f1(pc, GOOD_FIELDS), 4),
        "flat_macro_f1": round(m.get("flat_macro_f1", float("nan")), 4),
        "row_exact_match": round(m.get("row_exact_match", float("nan")), 4),
        "prediction_ok_rate": round(m.get("prediction_ok_rate", float("nan")), 4),
        "n_rows": m.get("n_rows"),
    }

    # Печать в консоль.
    print(f"\n=== {args.label} ===")
    print(f"  mean_f1_all       {row['mean_f1_all']:.4f}   (Δ vs prod {row['mean_f1_all']-BASELINE['all']:+.4f})  <- PRIMARY")
    print(f"  mean_f1_collapsed {row['mean_f1_collapsed']:.4f}   (Δ vs prod {row['mean_f1_collapsed']-BASELINE['collapsed']:+.4f})  <- 14 проблемных полей")
    print(f"  mean_f1_good      {row['mean_f1_good']:.4f}   (Δ vs prod {row['mean_f1_good']-BASELINE['good']:+.4f})  <- GUARDRAIL (не должен падать)")
    print(f"  flat_macro_f1     {row['flat_macro_f1']:.4f}   (справочно, для отбора НЕ используем)")
    print(f"  row_exact_match   {row['row_exact_match']:.4f}   ok_rate {row['prediction_ok_rate']:.4f}")
    print(f"\nСтрока для таблицы результатов (markdown):")
    print(f"| {args.label} | {row['mean_f1_all']:.3f} | {row['mean_f1_collapsed']:.3f} | {row['mean_f1_good']:.3f} | {row['row_exact_match']:.3f} | {row['prediction_ok_rate']:.1f} |")

    # Дозапись в CSV-лог.
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.results_csv.exists()
    with args.results_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"\nДописано в {args.results_csv}")


if __name__ == "__main__":
    main()
