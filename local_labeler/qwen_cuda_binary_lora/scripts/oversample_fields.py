#!/usr/bin/env python3
"""Field-aware oversampling: дублирует строки, в которых редкий класс присутствует.

В отличие от oversample_train.py (балансировка по условиям эксперимента),
этот скрипт работает на уровне отдельных полей разметки. Для каждого
causality-поля находит строки, где значение = 1, и дублирует их до тех пор,
пока соотношение majority/minority не станет <= max_ratio.

Обрабатывает train.jsonl и train_source.csv (val/test не трогаются).

Использование:
  python scripts/oversample_fields.py \
      --in-dir ./data_binary --out-dir ./data_field_balanced \
      --max-ratio 10.0 --seed 42

Параметры:
  --max-ratio   Максимально допустимое соотношение majority/minority после
                оверсамплинга. По умолчанию 10.0.
                Для поля 372/4 при max-ratio=10 minority увеличится с 4 до ~37.
  --seed        Для воспроизводимости перемешивания дублей.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_config import CAUSALITY_COLS


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(records: list[dict], path: Path) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")


def get_causality_values(rec: dict) -> dict[str, int]:
    """Извлекает causality-значения из assistant-сообщения записи."""
    for msg in rec.get("messages", []):
        if msg.get("role") == "assistant":
            try:
                ann = json.loads(msg["content"])
                return {col: int(ann.get(col, 0)) for col in CAUSALITY_COLS}
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return {col: 0 for col in CAUSALITY_COLS}


def oversample_jsonl(records: list[dict], max_ratio: float, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)

    # Для каждого поля собираем индексы строк с minority-классом (1 при дисбалансе 0>>1 или 0 при дисбалансе 1>>0)
    field_values = [get_causality_values(r) for r in records]

    extra: list[dict] = []
    stats: dict[str, dict] = {}

    for col in CAUSALITY_COLS:
        vals = [fv[col] for fv in field_values]
        n0 = vals.count(0)
        n1 = vals.count(1)

        if n0 == 0 or n1 == 0:
            stats[col] = {"n0": n0, "n1": n1, "added": 0, "skipped": "one class absent"}
            continue

        majority_val = 0 if n0 >= n1 else 1
        minority_val = 1 - majority_val
        n_majority = max(n0, n1)
        n_minority = min(n0, n1)

        current_ratio = n_majority / n_minority
        if current_ratio <= max_ratio:
            stats[col] = {"n0": n0, "n1": n1, "added": 0, "ratio": round(current_ratio, 1)}
            continue

        # Сколько копий нужно добавить
        target_minority = int(n_majority / max_ratio)
        to_add = target_minority - n_minority

        minority_indices = [i for i, v in enumerate(vals) if v == minority_val]
        copies = (minority_indices * ((to_add // len(minority_indices)) + 1))[:to_add]
        rng.shuffle(copies)
        extra.extend(records[i] for i in copies)

        stats[col] = {
            "n0": n0, "n1": n1,
            "minority": minority_val, "added": to_add,
            "ratio_before": round(current_ratio, 1),
            "ratio_after": round(n_majority / (n_minority + to_add), 1),
        }

    result = records + extra
    rng.shuffle(result)
    return result, stats


def oversample_csv(df: pd.DataFrame, records_before: int, records_after: int,
                   max_ratio: float, seed: int) -> pd.DataFrame:
    """Та же логика для source-CSV: дублируем строки с редкими значениями."""
    rng = random.Random(seed + 1)
    extra_rows = []

    for col in CAUSALITY_COLS:
        if col not in df.columns:
            continue
        n0 = (df[col] == 0).sum()
        n1 = (df[col] == 1).sum()
        if n0 == 0 or n1 == 0:
            continue
        n_majority = max(n0, n1)
        n_minority = min(n0, n1)
        if n_majority / n_minority <= max_ratio:
            continue
        minority_val = 0 if n0 < n1 else 1
        target_minority = int(n_majority / max_ratio)
        to_add = target_minority - n_minority
        minority_rows = df[df[col] == minority_val]
        copies = pd.concat([minority_rows] * ((to_add // len(minority_rows)) + 1), ignore_index=True).head(to_add)
        extra_rows.append(copies)

    if extra_rows:
        df = pd.concat([df] + extra_rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-ratio", type=float, default=10.0,
                    help="Макс. соотношение majority/minority после оверсамплинга (default: 10)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- train.jsonl ---
    src_jsonl = args.in_dir / "train.jsonl"
    if src_jsonl.exists():
        records = load_jsonl(src_jsonl)
        result, stats = oversample_jsonl(records, args.max_ratio, args.seed)
        save_jsonl(result, args.out_dir / "train.jsonl")
        print(f"train.jsonl: {len(records)} → {len(result)} строк (+{len(result)-len(records)})")
        print("\nСтатистика по полям (только изменившиеся):")
        for col, s in stats.items():
            if s.get("added", 0) > 0:
                print(f"  {col}: {s['n0']}×0 / {s['n1']}×1  "
                      f"ratio {s['ratio_before']} → {s['ratio_after']}  +{s['added']} строк")
    else:
        print(f"Пропуск train.jsonl (не найден)")

    # --- train_source.csv ---
    src_csv = args.in_dir / "train_source.csv"
    if src_csv.exists():
        df = pd.read_csv(src_csv)
        df_out = oversample_csv(df, len(df), 0, args.max_ratio, args.seed)
        df_out.to_csv(args.out_dir / "train_source.csv", index=False)
        print(f"\ntrain_source.csv: {len(df)} → {len(df_out)} строк")

    # val и test копируем без изменений
    for fname in ["valid.jsonl", "test.jsonl", "valid_source.csv", "test_source.csv"]:
        src = args.in_dir / fname
        if src.exists():
            dst = args.out_dir / fname
            dst.write_bytes(src.read_bytes())
            print(f"{fname}: скопирован без изменений")


if __name__ == "__main__":
    main()
