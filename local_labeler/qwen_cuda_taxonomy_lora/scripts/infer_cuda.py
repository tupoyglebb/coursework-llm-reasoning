#!/usr/bin/env python3
"""Инференс обученного QLoRA-адаптера на NVIDIA GPU и запись предсказаний в CSV.

CUDA-аналог infer_mlx.py. Формат выходного CSV идентичен (pred_<field>, prediction_ok,
raw_prediction, ...), поэтому scripts/evaluate_predictions.py и aggregate_metrics.py
работают без изменений.

Запуск:
  python scripts/infer_cuda.py --input ./data/test_source.csv --output ./predictions/test_pred_E1.csv \
      --adapter-path ./adapters/tier0_E1 --prompt-style compact --max-tokens 900
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


def build_prompt_ids(tokenizer, messages, device):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
        ).to(device)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter-path", type=str, required=True)
    ap.add_argument("--prompt-style", choices=["compact", "full"], default="compact")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0, help="0 = все строки; >0 — для дымовых тестов/ускорения valid")
    args = ap.parse_args()

    df = read_table(args.input)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    system_prompt = get_system_prompt(args.prompt_style)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(row)},
        ]
        input_ids = build_prompt_ids(tokenizer, messages, model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][input_ids.shape[1]:]
        raw = tokenizer.decode(gen, skip_special_tokens=True)
        try:
            parsed = extract_json_object(raw)
            ann = normalize_annotation_dict(parsed, row=row)
            ok, err = True, None
        except Exception as e:  # noqa: BLE001
            ann = {field: 0 for field in OUTPUT_FIELDS}
            ok, err = False, str(e)

        rec = row.to_dict()
        rec["prediction_ok"] = ok
        rec["prediction_error"] = err
        rec["raw_prediction"] = raw
        for field in OUTPUT_FIELDS:
            rec[f"pred_{field}"] = ann[field]
        records.append(rec)

    out_df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
