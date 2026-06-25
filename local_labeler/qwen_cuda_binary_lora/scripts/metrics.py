#!/usr/bin/env python3
"""Единый источник правды для метрик разметчика.

Содержит логику, которую раньше дублировали evaluate_predictions.py и aggregate_metrics.py:
  - compute_metrics(df)  — per-field macro-F1 + flat-метрики (бывшее тело evaluate_predictions).
  - summarize(metrics)   — mean_f1_all / mean_f1_collapsed / mean_f1_good + дельты к prod (бывший aggregate).
  - append_results_csv() — дозапись строки в experiments/results.csv (та же схема).

И CLI-обёртки (evaluate_predictions.py, aggregate_metrics.py), и валидация внутри обучения
(train_qlora_cuda.py) зовут эти функции — поэтому train-time и test-time считаются одинаково.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from taxonomy_config import OUTPUT_FIELDS

# Подмножества полей зафиксированы по prod-baseline, чтобы сравнения между экспериментами были на
# одном и том же наборе. НЕ пересчитывать по каждому прогону.
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

# Колонки строки results.csv (порядок фиксирован — схема не меняется).
RESULTS_COLUMNS = [
    "label",
    "mean_f1_all",
    "mean_f1_collapsed",
    "mean_f1_good",
    "flat_macro_f1",
    "row_exact_match",
    "prediction_ok_rate",
    "n_rows",
]


def compute_metrics(df: pd.DataFrame) -> dict:
    """Сравнивает pred_<field> с gold-колонками <field> и возвращает per-field + flat метрики.

    Ожидает в df колонки OUTPUT_FIELDS (gold) и pred_<field> (предсказания), опционально prediction_ok.
    """
    missing_true = [c for c in OUTPUT_FIELDS if c not in df.columns]
    missing_pred = [f"pred_{c}" for c in OUTPUT_FIELDS if f"pred_{c}" not in df.columns]
    if missing_true or missing_pred:
        raise ValueError(f"Missing true columns: {missing_true}\nMissing pred columns: {missing_pred}")

    pred_cols = [f"pred_{c}" for c in OUTPUT_FIELDS]
    rows_exact = (df[OUTPUT_FIELDS].astype(int).values == df[pred_cols].astype(int).values).all(axis=1)

    per_col = {}
    for c in OUTPUT_FIELDS:
        y_true = df[c].astype(int)
        y_pred = df[f"pred_{c}"].astype(int)
        labels = sorted(set(y_true.unique().tolist()) | set(y_pred.unique().tolist()))
        per_col[c] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        }

    # Все решения, сплющенные по полям — справочная sanity-метрика, не замена per-field.
    y_true_flat = df[OUTPUT_FIELDS].astype(int).values.reshape(-1)
    y_pred_flat = df[pred_cols].astype(int).values.reshape(-1)
    p, r, f1, _ = precision_recall_fscore_support(y_true_flat, y_pred_flat, average="macro", zero_division=0)

    return {
        "n_rows": int(len(df)),
        "prediction_ok_rate": float(df.get("prediction_ok", pd.Series([True] * len(df))).astype(bool).mean()),
        "row_exact_match": float(rows_exact.mean()),
        "flat_macro_precision": float(p),
        "flat_macro_recall": float(r),
        "flat_macro_f1": float(f1),
        "per_column": per_col,
    }


def mean_f1(per_column: dict, fields: list[str]) -> float:
    vals = [per_column[f]["macro_f1"] for f in fields if f in per_column]
    return sum(vals) / len(vals) if vals else float("nan")


def summarize(metrics: dict) -> dict:
    """Сводит вывод compute_metrics к метрикам успеха из плана (полная точность, без округления)."""
    pc = metrics["per_column"]
    return {
        "mean_f1_all": mean_f1(pc, list(pc.keys())),
        "mean_f1_collapsed": mean_f1(pc, COLLAPSED_FIELDS),
        "mean_f1_good": mean_f1(pc, GOOD_FIELDS),
        "flat_macro_f1": float(metrics.get("flat_macro_f1", float("nan"))),
        "row_exact_match": float(metrics.get("row_exact_match", float("nan"))),
        "prediction_ok_rate": float(metrics.get("prediction_ok_rate", float("nan"))),
        "n_rows": metrics.get("n_rows"),
    }


def format_summary(label: str, summary: dict) -> str:
    """Человекочитаемый блок для консоли (тот же формат, что был в aggregate_metrics)."""
    s = summary
    lines = [
        f"\n=== {label} ===",
        f"  mean_f1_all       {s['mean_f1_all']:.4f}   (Δ vs prod {s['mean_f1_all']-BASELINE['all']:+.4f})  <- PRIMARY",
        f"  mean_f1_collapsed {s['mean_f1_collapsed']:.4f}   (Δ vs prod {s['mean_f1_collapsed']-BASELINE['collapsed']:+.4f})  <- 14 проблемных полей",
        f"  mean_f1_good      {s['mean_f1_good']:.4f}   (Δ vs prod {s['mean_f1_good']-BASELINE['good']:+.4f})  <- GUARDRAIL (не должен падать)",
        f"  flat_macro_f1     {s['flat_macro_f1']:.4f}   (справочно, для отбора НЕ используем)",
        f"  row_exact_match   {s['row_exact_match']:.4f}   ok_rate {s['prediction_ok_rate']:.4f}",
        "\nСтрока для таблицы результатов (markdown):",
        f"| {label} | {s['mean_f1_all']:.3f} | {s['mean_f1_collapsed']:.3f} | {s['mean_f1_good']:.3f} | {s['row_exact_match']:.3f} | {s['prediction_ok_rate']:.1f} |",
    ]
    return "\n".join(lines)


def append_results_csv(path: Path, label: str, summary: dict) -> None:
    """Дозаписывает одну строку в results.csv (создаёт заголовок при первом вызове)."""
    row = {"label": label, "n_rows": summary.get("n_rows")}
    for key in ("mean_f1_all", "mean_f1_collapsed", "mean_f1_good", "flat_macro_f1", "row_exact_match", "prediction_ok_rate"):
        val = summary.get(key, float("nan"))
        row[key] = round(val, 4) if val is not None else None

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)
