#!/usr/bin/env python3
"""Перебалансировка train.jsonl для E3 (oversampling редких классов) — без правки основного пайплайна.

Читает <in_dir>/train.jsonl, мягко дублирует строки пропорционально редкости их классов по
целевым (схлопнувшимся) полям, и пишет балансированный <out_dir>/train.jsonl.
valid.jsonl и test.jsonl КОПИРУЮТСЯ без изменений (метрики должны остаться сравнимыми).

Стратегия (прозрачная, см. TIER0_PLAN.md E3):
  - для каждого целевого поля считаем частоту классов в train;
  - вес строки = max по целевым полям от (1 / частота её класса)  -> редкие классы тяжелее;
  - досэмплируем (inflate-1)*N дополнительных строк ∝ весу (с возвращением);
  - кап на раздувание задаётся --inflate (по умолчанию 1.5).
Печатает долю миноритарного класса до/после по 14 полям, чтобы эффект был виден (никаких «тихих» правок).

Использование:
  python scripts/oversample_train.py --in-dir ./data_mlx --out-dir ./data_mlx_balanced --inflate 1.5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

# Те же 14 «схлопнувшихся» полей, что в aggregate_metrics.py.
TARGET_FIELDS = [
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


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def labels_of(rec: dict) -> dict:
    """Достаёт JSON ассистента (последнее сообщение) как словарь меток."""
    return json.loads(rec["messages"][-1]["content"])


def minority_fraction(rows_labels: list[dict], field: str) -> float:
    c = Counter(r.get(field, 0) for r in rows_labels)
    if not c:
        return float("nan")
    total = sum(c.values())
    return min(c.values()) / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("data_mlx"))
    ap.add_argument("--out-dir", type=Path, default=Path("data_mlx_balanced"))
    ap.add_argument("--inflate", type=float, default=1.5, help="во сколько раз раздуть train (кап)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    train = load_jsonl(args.in_dir / "train.jsonl")
    labels = [labels_of(r) for r in train]
    n = len(train)

    # Частоты классов по целевым полям.
    freqs: dict[str, dict] = {}
    for f in TARGET_FIELDS:
        c = Counter(lab.get(f, 0) for lab in labels)
        tot = sum(c.values())
        freqs[f] = {v: cnt / tot for v, cnt in c.items()}

    # Вес строки = max инверсной частоты по целевым полям.
    weights = np.array([
        max(1.0 / freqs[f][lab.get(f, 0)] for f in TARGET_FIELDS)
        for lab in labels
    ])
    probs = weights / weights.sum()

    n_extra = int(round((args.inflate - 1.0) * n))
    extra_idx = rng.choice(n, size=n_extra, replace=True, p=probs).tolist() if n_extra > 0 else []

    balanced = train + [train[i] for i in extra_idx]
    rng.shuffle(balanced)
    balanced_labels = labels + [labels[i] for i in extra_idx]

    # Запись.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for rec in balanced:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for name in ("valid.jsonl", "test.jsonl"):
        src = args.in_dir / name
        if src.exists():
            shutil.copy(src, args.out_dir / name)

    # Отчёт до/после.
    print(f"train: {n} -> {len(balanced)} строк (inflate={len(balanced)/n:.2f}x, extra={n_extra})")
    print(f"\nДоля миноритарного класса по целевым полям (было -> стало):")
    report = {}
    for f in TARGET_FIELDS:
        before = minority_fraction(labels, f)
        after = minority_fraction(balanced_labels, f)
        report[f] = {"before": round(before, 4), "after": round(after, 4)}
        print(f"  {f:46s} {before:.3f} -> {after:.3f}")
    (args.out_dir / "balance_report.json").write_text(
        json.dumps({"n_before": n, "n_after": len(balanced), "inflate": args.inflate, "fields": report},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nГотово: {args.out_dir}/train.jsonl  (+ valid/test скопированы без изменений)")
    print("Для E3 используйте data: ./data_mlx_balanced (уже прописано в configs/tier0_E3.yaml).")


if __name__ == "__main__":
    main()
