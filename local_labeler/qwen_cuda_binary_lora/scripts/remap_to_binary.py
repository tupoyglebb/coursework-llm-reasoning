#!/usr/bin/env python3
"""Перемаппирует causality-метки 2 → 1 в чат-jsonl и source-csv файлах.

Зачем: модель обучалась различать 0/1/2, но тест не содержит класса 2,
поэтому оценка де-факто бинарная. Явный переход устраняет несоответствие
между обучением и инференсом.

Использование:
  python scripts/remap_to_binary.py --in-dir ./data --out-dir ./data_binary
  python scripts/remap_to_binary.py --in-dir ./data_balanced --out-dir ./data_balanced_binary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_config import CAUSALITY_COLS


def remap_jsonl(src: Path, dst: Path) -> int:
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    out = []
    for rec in records:
        messages = rec.get("messages", [])
        new_messages = []
        rec_changed = False
        for msg in messages:
            if msg.get("role") == "assistant":
                try:
                    ann = json.loads(msg["content"])
                    for col in CAUSALITY_COLS:
                        if col in ann and ann[col] == 2:
                            ann[col] = 1
                            rec_changed = True
                    msg = {**msg, "content": json.dumps(ann, ensure_ascii=False, separators=(",", ":"))}
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
            new_messages.append(msg)
        out.append({**rec, "messages": new_messages})
        if rec_changed:
            changed += 1
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8")
    return changed


def remap_csv(src: Path, dst: Path) -> int:
    df = pd.read_csv(src)
    changed = 0
    for col in CAUSALITY_COLS:
        if col in df.columns:
            mask = df[col] == 2
            df.loc[mask, col] = 1
            changed += int(mask.sum())
    df.to_csv(dst, index=False)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_changed = 0
    for fname in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
        src = args.in_dir / fname
        if not src.exists():
            continue
        n = remap_jsonl(src, args.out_dir / fname)
        total_changed += n
        print(f"{fname}: {n} записей с изменёнными метками → {args.out_dir / fname}")

    for fname in ["train_source.csv", "valid_source.csv", "test_source.csv"]:
        src = args.in_dir / fname
        if not src.exists():
            continue
        n = remap_csv(src, args.out_dir / fname)
        total_changed += n
        print(f"{fname}: {n} значений 2→1 → {args.out_dir / fname}")

    print(f"\nИтого изменений: {total_changed}")


if __name__ == "__main__":
    main()
