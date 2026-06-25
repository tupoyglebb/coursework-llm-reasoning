#!/usr/bin/env python3
"""Инференс обученного LoRA-адаптера на NVIDIA GPU и запись предсказаний в CSV.

Формат выходного CSV идентичен MLX-стеку (pred_<field>, prediction_ok, raw_prediction, ...),
поэтому evaluate_predictions.py / aggregate_metrics.py работают без изменений.

Генерация и парсинг — общий код generation.predict_dataframe (тот же, что в валидации обучения).

  python scripts/infer_cuda.py --input ./data/test_source.csv --output ./predictions/test_pred_E0.csv \
      --adapter-path ./adapters/tier0_E0 --prompt-style compact --max-tokens 900
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from peft import PeftModel

from generation import load_base_model, predict_dataframe
from taxonomy_config import get_system_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter-path", type=str, required=True)
    ap.add_argument("--prompt-style", choices=["compact", "full"], default="compact")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--quantization", choices=["none", "nf4", "int8"], default="nf4",
                    help="ДОЛЖНА совпадать с квантизацией при обучении адаптера")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--limit", type=int, default=0, help="0 = все строки; >0 — для дымовых тестов/ускорения valid")
    args = ap.parse_args()

    df = read_table(args.input)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    logger.info("Загрузка модели %s (quant=%s, attn=%s) + адаптер %s", args.model_id, args.quantization, args.attn, args.adapter_path)
    base, tokenizer = load_base_model(args.model_id, quantization=args.quantization, attn=args.attn)
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    system_prompt = get_system_prompt(args.prompt_style)
    out_df = predict_dataframe(
        model, tokenizer, df, system_prompt,
        max_new_tokens=args.max_tokens, batch_size=args.batch_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    logger.info("Saved: %s (%d rows)", args.output, len(out_df))


if __name__ == "__main__":
    main()
