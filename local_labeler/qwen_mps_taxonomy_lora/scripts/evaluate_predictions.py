#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from taxonomy_config import OUTPUT_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted taxonomy columns against teacher labels.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=Path("metrics.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    missing_true = [c for c in OUTPUT_FIELDS if c not in df.columns]
    missing_pred = [f"pred_{c}" for c in OUTPUT_FIELDS if f"pred_{c}" not in df.columns]
    if missing_true or missing_pred:
        raise ValueError(f"Missing true columns: {missing_true}\nMissing pred columns: {missing_pred}")

    rows_exact = (df[OUTPUT_FIELDS].astype(int).values == df[[f"pred_{c}" for c in OUTPUT_FIELDS]].astype(int).values).all(axis=1)
    per_col = {}
    for c in OUTPUT_FIELDS:
        y_true = df[c].astype(int)
        y_pred = df[f"pred_{c}"].astype(int)
        labels = sorted(set(y_true.unique().tolist()) | set(y_pred.unique().tolist()))
        per_col[c] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        }

    # Flatten all decisions across fields. Useful as a sanity metric, not a replacement for per-field metrics.
    y_true_flat = df[OUTPUT_FIELDS].astype(int).values.reshape(-1)
    y_pred_flat = df[[f"pred_{c}" for c in OUTPUT_FIELDS]].astype(int).values.reshape(-1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average="macro", zero_division=0
    )

    metrics = {
        "n_rows": int(len(df)),
        "prediction_ok_rate": float(df.get("prediction_ok", pd.Series([True] * len(df))).astype(bool).mean()),
        "row_exact_match": float(rows_exact.mean()),
        "flat_macro_precision": float(p),
        "flat_macro_recall": float(r),
        "flat_macro_f1": float(f1),
        "per_column": per_col,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
