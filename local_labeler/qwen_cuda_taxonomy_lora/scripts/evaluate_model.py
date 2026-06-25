#!/usr/bin/env python3
"""Самостоятельная генеративная оценка обученного адаптера на тесте (с gold-колонками).

Генерит предсказания моделью, парсит JSON и считает mean_f1_all / mean_f1_good и пр.
Запускается НЕЗАВИСИМО от обучения — можно оценить любой чекпойнт когда угодно:

  # лучший адаптер из конфига на ./data/test_source.csv:
  python scripts/evaluate_model.py --config configs/tier0_E0.yaml --label E0

  # конкретный чекпойнт, своя выборка, ограничение строк (smoke):
  python scripts/evaluate_model.py --config configs/tier0_E0.yaml --label E0@ckpt500 \
      --adapter-path adapters/tier0_E0/checkpoint-500 --input data/valid_source.csv --limit 32

Параметры модели/генерации (model_id, quantization, attn, prompt_style, gen_batch_size,
val_max_new_tokens, adapter_path, data_dir) берутся из того же YAML-конфига, что и обучение.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from peft import PeftModel

from generation import load_base_model, predict_dataframe
from metrics import append_results_csv, compute_metrics, format_summary, summarize
from taxonomy_config import get_system_prompt
from train_qlora_cuda import TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")


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
    ap = argparse.ArgumentParser(description="Генеративная оценка адаптера на тесте.")
    ap.add_argument("--config", type=Path, required=True, help="YAML эксперимента (как при обучении)")
    ap.add_argument("--label", type=str, required=True, help="метка для results.csv, напр. E0")
    ap.add_argument("--adapter-path", type=str, default=None, help="по умолчанию adapter_path из конфига")
    ap.add_argument("--input", type=Path, default=None, help="по умолчанию <data_dir>/test_source.csv")
    ap.add_argument("--output", type=Path, default=None, help="CSV предсказаний; по умолчанию predictions/test_pred_<label>.csv")
    ap.add_argument("--metrics-json", type=Path, default=None, help="по умолчанию predictions/metrics_<label>.json")
    ap.add_argument("--results-csv", type=Path, default=Path("experiments/results.csv"))
    ap.add_argument("--limit", type=int, default=0, help="0 = все строки; >0 — для smoke/ускорения")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    adapter_path = args.adapter_path or cfg.adapter_path
    input_path = args.input or (cfg.data_dir / "test_source.csv")
    safe = args.label.replace("/", "_").replace("@", "_")
    output = args.output or Path(f"./predictions/test_pred_{safe}.csv")
    metrics_json = args.metrics_json or Path(f"./predictions/metrics_{safe}.json")

    df = read_table(input_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    logger.info("Оценка [%s]: адаптер=%s  вход=%s (%d строк)  quant=%s attn=%s",
                args.label, adapter_path, input_path, len(df), cfg.quantization, cfg.attn_implementation)
    base, tokenizer = load_base_model(cfg.model_id, quantization=cfg.quantization, attn=cfg.attn_implementation)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    pred = predict_dataframe(
        model, tokenizer, df, get_system_prompt(cfg.prompt_style),
        max_new_tokens=cfg.val_max_new_tokens, batch_size=cfg.gen_batch_size,
        show_progress=True, desc=f"eval[{args.label}]",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output, index=False)

    metrics = compute_metrics(pred)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarize(metrics)
    print(format_summary(args.label, summary))
    append_results_csv(args.results_csv, args.label, summary)
    logger.info("Предсказания: %s | метрики: %s | строка дописана в %s", output, metrics_json, args.results_csv)


if __name__ == "__main__":
    main()
