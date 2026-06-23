#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from taxonomy_config import (
    OUTPUT_FIELDS,
    REQUIRED_INPUT_COLS,
    STRATIFY_COLS,
    annotation_from_row,
    assistant_json_from_row,
    build_user_prompt,
    get_system_prompt,
)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported input format: {path}")


def load_inputs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [read_table(p) for p in paths]
    if not frames:
        raise ValueError("No input files provided")
    return pd.concat(frames, ignore_index=True)


def check_required_columns(df: pd.DataFrame) -> None:
    missing_inputs = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    missing_outputs = [c for c in OUTPUT_FIELDS if c not in df.columns]
    if missing_inputs or missing_outputs:
        msg = []
        if missing_inputs:
            msg.append(f"Missing input columns: {missing_inputs}")
        if missing_outputs:
            msg.append(f"Missing output/label columns: {missing_outputs}")
        raise ValueError("\n".join(msg))


def stratified_sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Proportional stratified sample over experiment/group/incentive, with fallback.
    cols = [c for c in STRATIFY_COLS if c in df.columns]
    if not cols:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    sampled_indices: list[int] = []
    grouped = df.groupby(cols, dropna=False)
    total = len(df)

    allocations = []
    for _, g in grouped:
        exact = len(g) / total * n
        base = int(np.floor(exact))
        allocations.append((g.index.to_numpy(), base, exact - base))

    remaining = n - sum(a[1] for a in allocations)
    order = np.argsort([-a[2] for a in allocations])
    for pos in order[:remaining]:
        idxs, base, frac = allocations[pos]
        allocations[pos] = (idxs, base + 1, frac)

    for idxs, k, _ in allocations:
        if k <= 0:
            continue
        k = min(k, len(idxs))
        sampled_indices.extend(rng.choice(idxs, size=k, replace=False).tolist())

    if len(sampled_indices) < n:
        rest = np.array([i for i in df.index if i not in set(sampled_indices)])
        if len(rest):
            sampled_indices.extend(rng.choice(rest, size=min(n - len(sampled_indices), len(rest)), replace=False).tolist())

    rng.shuffle(sampled_indices)
    return df.loc[sampled_indices[:n]].reset_index(drop=True)


def split_df(df: pd.DataFrame, train_frac: float, valid_frac: float, seed: int):
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 <= valid_frac < 1:
        raise ValueError("valid_frac must be between 0 and 1")
    if train_frac + valid_frac >= 1:
        raise ValueError("train_frac + valid_frac must be < 1")

    rng = np.random.default_rng(seed)
    train_idx, valid_idx, test_idx = [], [], []
    cols = [c for c in STRATIFY_COLS if c in df.columns]

    groups = df.groupby(cols, dropna=False) if cols else [(None, df)]
    for _, g in groups:
        idx = g.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_frac))
        n_valid = int(round(n * valid_frac))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_valid = min(max(n_valid, 1), n - n_train - 1)
        train_idx.extend(idx[:n_train])
        valid_idx.extend(idx[n_train : n_train + n_valid])
        test_idx.extend(idx[n_train + n_valid :])

    return (
        df.loc[train_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True),
        df.loc[valid_idx].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True),
        df.loc[test_idx].sample(frac=1.0, random_state=seed + 2).reset_index(drop=True),
    )


def write_jsonl(df: pd.DataFrame, path: Path, prompt_style: str) -> None:
    system_prompt = get_system_prompt(prompt_style)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_prompt(row)},
                    {"role": "assistant", "content": assistant_json_from_row(row)},
                ]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MLX-LM chat JSONL files for taxonomy distillation.")
    parser.add_argument("--input", nargs="+", required=True, type=Path, help="Teacher-labeled CSV/parquet/jsonl files.")
    parser.add_argument("--out-dir", type=Path, default=Path("data_mlx"))
    parser.add_argument("--n", type=int, default=10000, help="Number of examples to use. Set <=0 to use all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.90)
    parser.add_argument("--valid-frac", type=float, default=0.05)
    parser.add_argument("--prompt-style", choices=["compact", "full"], default="compact")
    parser.add_argument("--drop-failed", action="store_true", default=True, help="Drop rows where ok is present and not true.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_inputs(args.input)

    if args.drop_failed and "ok" in df.columns:
        df = df[df["ok"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()

    check_required_columns(df)
    df = df.dropna(subset=["stage1_reasoning_text"]).copy()
    if "row_num" in df.columns:
        df = df.drop_duplicates(subset=["row_num"], keep="first")

    n = None if args.n <= 0 else args.n
    df = stratified_sample(df, n=n, seed=args.seed)

    # Normalize labels before writing, so the exact same hard rules are learned.
    for field in OUTPUT_FIELDS:
        df[field] = [annotation_from_row(row)[field] for _, row in df.iterrows()]

    train_df, valid_df, test_df = split_df(df, args.train_frac, args.valid_frac, args.seed)

    write_jsonl(train_df, args.out_dir / "train.jsonl", args.prompt_style)
    write_jsonl(valid_df, args.out_dir / "valid.jsonl", args.prompt_style)
    write_jsonl(test_df, args.out_dir / "test.jsonl", args.prompt_style)

    train_df.to_csv(args.out_dir / "train_source.csv", index=False)
    valid_df.to_csv(args.out_dir / "valid_source.csv", index=False)
    test_df.to_csv(args.out_dir / "test_source.csv", index=False)

    manifest = {
        "prompt_style": args.prompt_style,
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        "output_fields_count": len(OUTPUT_FIELDS),
        "output_fields": OUTPUT_FIELDS,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
