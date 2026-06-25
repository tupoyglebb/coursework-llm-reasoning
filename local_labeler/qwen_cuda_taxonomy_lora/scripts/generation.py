#!/usr/bin/env python3
"""Единый источник правды для загрузки модели и генерации предсказаний.

И test-инференс (infer_cuda.py), и валидация внутри обучения (train_qlora_cuda.py) зовут
predict_dataframe(), поэтому считаются одинаково — train-time и test-time не расходятся.

quantization:
  none -> чистый bf16 (на H200 8B ≈ 16-20 ГБ из 143 — самый быстрый путь);
  nf4  -> bitsandbytes NF4 + double-quant (паритет с MLX 4-bit, медленнее);
  int8 -> bitsandbytes 8-bit.
"""
from __future__ import annotations

import logging

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from taxonomy_config import (
    OUTPUT_FIELDS,
    build_user_prompt,
    extract_json_object,
    normalize_annotation_dict,
)

logger = logging.getLogger(__name__)


def chat_template_ids(tokenizer, messages, add_generation_prompt: bool, return_tensors=None):
    """apply_chat_template -> token ids. enable_thinking=False (как на инференсе), с фолбэком.

    transformers >=5 возвращает BatchEncoding (dict-подобный), а не list[int]/Tensor — достаём input_ids.
    """
    kwargs = dict(tokenize=True, add_generation_prompt=add_generation_prompt, return_tensors=return_tensors)
    try:
        out = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        out = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    return out


def load_base_model(model_id: str, quantization: str = "none", attn: str = "sdpa"):
    """Грузит базовую модель + токенизатор. Возвращает (model, tokenizer)."""
    quantization = (quantization or "none").strip().lower()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(dtype=torch.bfloat16, device_map="auto", attn_implementation=attn)
    if quantization == "nf4":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "int8":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization != "none":
        raise ValueError(f"quantization must be one of none|nf4|int8, got {quantization!r}")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except (ValueError, ImportError, RuntimeError) as e:
        # Флэш-внимание может быть не собрано под эту среду — откатываемся на sdpa.
        if attn != "sdpa":
            logger.warning("attn_implementation=%s недоступно (%s); откат на sdpa", attn, e)
            load_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        else:
            raise
    return model, tokenizer


def _device_of(model) -> torch.device:
    try:
        return model.device
    except (AttributeError, RuntimeError):
        return next(model.parameters()).device


@torch.no_grad()
def predict_dataframe(
    model,
    tokenizer,
    df: pd.DataFrame,
    system_prompt: str,
    max_new_tokens: int = 900,
    batch_size: int = 16,
    show_progress: bool = True,
    desc: str = "Predicting",
) -> pd.DataFrame:
    """Батч-генерация предсказаний по строкам df. Возвращает df с колонками pred_<field>, prediction_ok/...

    Левый паддинг батча промптов -> один generate на батч -> срез продолжения по общей длине промпта.
    """
    device = _device_of(model)
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # обязательно для генерации декодером
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    rows = list(df.itertuples(index=False))
    columns = list(df.columns)
    records: list[dict] = []

    batches = range(0, len(rows), batch_size)
    if show_progress:
        batches = tqdm(batches, desc=desc, total=(len(rows) + batch_size - 1) // batch_size)

    try:
        for start in batches:
            chunk = rows[start : start + batch_size]
            chunk_dicts = [dict(zip(columns, r)) for r in chunk]

            prompts = []
            for rec in chunk_dicts:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_prompt(rec)},
                ]
                prompts.append({"input_ids": chat_template_ids(tokenizer, messages, add_generation_prompt=True)})

            batch = tokenizer.pad(prompts, padding=True, return_tensors="pt").to(device)
            out = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
            )
            prompt_len = batch["input_ids"].shape[1]  # левый паддинг => одинаковая длина у всех
            gen = out[:, prompt_len:]

            for rec, gen_ids in zip(chunk_dicts, gen):
                raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
                try:
                    ann = normalize_annotation_dict(extract_json_object(raw), row=rec)
                    ok, err = True, None
                except Exception as e:  # noqa: BLE001
                    ann = {field: 0 for field in OUTPUT_FIELDS}
                    ok, err = False, str(e)

                rec = dict(rec)
                rec["prediction_ok"] = ok
                rec["prediction_error"] = err
                rec["raw_prediction"] = raw
                for field in OUTPUT_FIELDS:
                    rec[f"pred_{field}"] = ann[field]
                records.append(rec)
    finally:
        tokenizer.padding_side = prev_side

    return pd.DataFrame(records)
