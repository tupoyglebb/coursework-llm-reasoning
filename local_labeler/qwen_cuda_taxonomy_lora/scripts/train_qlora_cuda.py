#!/usr/bin/env python3
"""QLoRA-дообучение Qwen3-8B на NVIDIA GPU (transformers + peft + bitsandbytes).

CUDA-порт MLX-стека. Воспроизводит те же гиперпараметры (маппинг — в HANDOFF_CUDA.md):
  - MLX lora.scale       -> peft lora_alpha = scale * rank   (в MLX scale — прямой множитель,
                            в peft множитель = lora_alpha / r, поэтому alpha = scale * r)
  - MLX num_layers (N)   -> peft layers_to_transform = последние N индексов слоёв
  - целевые модули       -> q/k/v/o/gate/up/down_proj (как в MLX)
  - mask_prompt=true     -> лосс только по ответу ассистента (промпт маскируется -100 в labels)
  - 4-bit                -> bitsandbytes NF4 + double quant (в MLX был affine; численно не идентично)

Конфиг — YAML (configs/tier0_E*.yaml). Запуск:
  python scripts/train_qlora_cuda.py --config ./configs/tier0_E1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Те же 7 проекций, что адаптировал MLX-стек.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chat_ids(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    """apply_chat_template -> token ids. enable_thinking=False (как на инференсе), с фолбэком."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt
        )


def build_dataset(jsonl_path: Path, tokenizer, max_len: int, mask_prompt: bool = True) -> Dataset:
    rows = []
    for rec in read_jsonl(jsonl_path):
        msgs = rec["messages"]
        full = chat_ids(tokenizer, msgs, add_generation_prompt=False)
        labels = list(full)
        if mask_prompt:
            # промпт = system+user + заголовок ассистента; маскируем его в labels.
            prompt = chat_ids(tokenizer, msgs[:-1], add_generation_prompt=True)
            for i in range(min(len(prompt), len(full))):
                labels[i] = -100
        full = full[:max_len]
        labels = labels[:max_len]
        rows.append({"input_ids": full, "labels": labels, "attention_mask": [1] * len(full)})
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    set_seed(int(cfg.get("seed", 42)))
    model_id = cfg.get("model_id", "Qwen/Qwen3-8B")
    data_dir = Path(cfg["data_dir"])
    max_len = int(cfg.get("max_seq_length", 3072))
    lora = cfg["lora"]
    rank = int(lora["rank"])
    scale = float(lora.get("scale", 20.0))
    dropout = float(lora.get("dropout", 0.05))
    num_layers = int(cfg.get("num_layers", 16))
    adapter_path = str(cfg["adapter_path"])
    grad_ckpt = bool(cfg.get("grad_checkpoint", True))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)

    n_total_layers = model.config.num_hidden_layers
    layers_to_transform = list(range(n_total_layers - num_layers, n_total_layers))
    peft_cfg = LoraConfig(
        r=rank,
        lora_alpha=int(round(scale * rank)),  # <-- маппинг MLX scale -> peft alpha
        lora_dropout=dropout,
        target_modules=TARGET_MODULES,
        layers_to_transform=layers_to_transform,  # <-- маппинг MLX num_layers (последние N)
        layers_pattern="layers",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    print(f"Слои под LoRA: {layers_to_transform}  | alpha={int(round(scale*rank))} (scale={scale} * rank={rank})")
    model.print_trainable_parameters()

    train_ds = build_dataset(data_dir / "train.jsonl", tokenizer, max_len, mask_prompt=bool(cfg.get("mask_prompt", True)))

    targs = TrainingArguments(
        output_dir=adapter_path,
        per_device_train_batch_size=int(cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("grad_accumulation_steps", 8)),
        max_steps=int(cfg.get("iters", 1700)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        lr_scheduler_type=str(cfg.get("lr_scheduler", "cosine")),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        logging_steps=int(cfg.get("steps_per_report", 10)),
        save_steps=int(cfg.get("save_every", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 6)),
        bf16=True,
        optim=str(cfg.get("optim", "paged_adamw_8bit")),
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=int(cfg.get("seed", 42)),
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, padding=True)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, data_collator=collator)
    trainer.train()

    # Финальный адаптер кладём в корень adapter_path (промежуточные — в checkpoint-*/).
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Готово. Адаптер: {adapter_path}")


if __name__ == "__main__":
    main()
