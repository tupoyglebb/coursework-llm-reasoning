#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from taxonomy_config import (
    OUTPUT_FIELDS,
    build_user_prompt,
    extract_json_object,
    get_system_prompt,
    normalize_annotation_dict,
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


def apply_chat_template(tokenizer, messages):
    # Qwen3 supports enable_thinking=False in HF chat template; older templates may not.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained MLX LoRA annotator and write predictions to CSV.")
    parser.add_argument("--input", type=Path, required=True, help="CSV/parquet/jsonl with metadata + stage1_reasoning_text.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="models/qwen3-8b-4bit")
    parser.add_argument("--adapter-path", type=str, default="adapters/taxonomy_qwen3_8b")
    parser.add_argument("--prompt-style", choices=["compact", "full"], default="compact")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--limit", type=int, default=0, help="For smoke tests. 0 = all rows.")
    args = parser.parse_args()

    from mlx_lm import generate, load

    df = read_table(args.input)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    system_prompt = get_system_prompt(args.prompt_style)
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(row)},
        ]
        prompt = apply_chat_template(tokenizer, messages)
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        try:
            parsed = extract_json_object(raw)
            ann = normalize_annotation_dict(parsed, row=row)
            ok = True
            err = None
        except Exception as e:
            ann = {field: 0 for field in OUTPUT_FIELDS}
            ok = False
            err = str(e)

        rec = row.to_dict()
        rec["prediction_ok"] = ok
        rec["prediction_error"] = err
        rec["raw_prediction"] = raw
        for field in OUTPUT_FIELDS:
            rec[f"pred_{field}"] = ann[field]
        records.append(rec)

    out = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
