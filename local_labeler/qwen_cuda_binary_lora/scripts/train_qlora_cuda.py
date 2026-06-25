#!/usr/bin/env python3
"""LoRA-дообучение Qwen3-8B на NVIDIA GPU (transformers + peft, опц. bitsandbytes).

Что делает поверх «просто обучить»:
  - валидация в обучении: eval_loss на valid.jsonl каждые eval_every шагов;
  - лучший чекпойнт выбирается по eval_loss (load_best_model_at_end);
  - метрики в TensorBoard (report_to=tensorboard, logging_dir=<adapter>/runs);
  - скорость под H200: bf16 по умолчанию (quantization=none), SDPA/flash-attention,
    TF32, grad-checkpoint выключен по умолчанию.

Генеративная оценка по полям (mean_f1_all) НЕ делается в обучении — она дорогая и вынесена
в отдельный scripts/evaluate_model.py (генерация на тесте). Запуск обучения:
  python scripts/train_qlora_cuda.py --config ./configs/tier0_E0.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from generation import chat_template_ids, load_base_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")

# Те же 7 проекций, что адаптировал MLX-стек.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    data_dir: Path
    adapter_path: str
    model_id: str = "Qwen/Qwen3-8B"
    # LoRA
    rank: int = 8
    lora_scale: float = 20.0          # -> lora_alpha = scale * rank
    lora_dropout: float = 0.05
    num_layers: int = 16              # последние N из 36
    # оптимизация
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0
    batch_size: int = 1
    grad_accumulation_steps: int = 8
    iters: int = 1700
    max_seq_length: int = 3072
    optim: str = "adamw_torch"        # адаптер крошечный (~2.4M) -> 8-bit/paged не нужны на H200
    mask_prompt: bool = True
    seed: int = 42
    # отчётность/сохранение/валидация (eval_loss)
    steps_per_report: int = 10
    save_every: int = 100
    save_total_limit: int = 6
    eval_every: int = 0               # 0 -> = save_every
    # генеративная оценка — НЕ в обучении; эти ключи читает scripts/evaluate_model.py
    prompt_style: str = "compact"
    val_max_new_tokens: int = 900
    gen_batch_size: int = 16
    # производительность / квантизация
    quantization: str = "nf4"         # nf4(QLoRA: адаптер компенсирует квант) | int8 | none(bf16)
    attn_implementation: str = "sdpa"
    grad_checkpoint: bool = False
    # производные пути (заполняются в __post_init__)
    valid_jsonl: Path = field(default=None)
    valid_source_csv: Path = field(default=None)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.valid_jsonl is None:
            self.valid_jsonl = self.data_dir / "valid.jsonl"
        if self.valid_source_csv is None:
            self.valid_source_csv = self.data_dir / "valid_source.csv"
        self.valid_jsonl = Path(self.valid_jsonl)
        self.valid_source_csv = Path(self.valid_source_csv)
        if self.eval_every <= 0:
            self.eval_every = self.save_every
        if self.quantization not in {"none", "nf4", "int8"}:
            raise ValueError(f"quantization must be none|nf4|int8, got {self.quantization!r}")

    @property
    def lora_alpha(self) -> int:
        return int(round(self.lora_scale * self.rank))

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        lora = cfg.get("lora", {}) or {}
        return cls(
            data_dir=cfg["data_dir"],
            adapter_path=str(cfg["adapter_path"]),
            model_id=cfg.get("model_id", "Qwen/Qwen3-8B"),
            rank=int(lora.get("rank", 8)),
            lora_scale=float(lora.get("scale", 20.0)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            num_layers=int(cfg.get("num_layers", 16)),
            learning_rate=float(cfg.get("learning_rate", 1e-4)),
            lr_scheduler=str(cfg.get("lr_scheduler", "cosine")),
            warmup_steps=int(cfg.get("warmup_steps", 0)),
            batch_size=int(cfg.get("batch_size", 1)),
            grad_accumulation_steps=int(cfg.get("grad_accumulation_steps", 8)),
            iters=int(cfg.get("iters", 1700)),
            max_seq_length=int(cfg.get("max_seq_length", 3072)),
            optim=str(cfg.get("optim", "adamw_torch")),
            mask_prompt=bool(cfg.get("mask_prompt", True)),
            seed=int(cfg.get("seed", 42)),
            steps_per_report=int(cfg.get("steps_per_report", 10)),
            save_every=int(cfg.get("save_every", 100)),
            save_total_limit=int(cfg.get("save_total_limit", 6)),
            eval_every=int(cfg.get("eval_every", 0)),
            prompt_style=str(cfg.get("prompt_style", "compact")),
            val_max_new_tokens=int(cfg.get("val_max_new_tokens", 900)),
            quantization=str(cfg.get("quantization", "nf4")),
            attn_implementation=str(cfg.get("attn_implementation", "sdpa")),
            gen_batch_size=int(cfg.get("gen_batch_size", 16)),
            grad_checkpoint=bool(cfg.get("grad_checkpoint", False)),
            valid_jsonl=cfg.get("valid_jsonl"),
            valid_source_csv=cfg.get("valid_source_csv"),
        )


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(jsonl_path: Path, tokenizer, max_len: int, mask_prompt: bool = True) -> Dataset:
    """Чат-jsonl -> input_ids/labels/attention_mask. При mask_prompt промпт маскируется -100."""
    rows = []
    for rec in read_jsonl(jsonl_path):
        msgs = rec["messages"]
        full = list(chat_template_ids(tokenizer, msgs, add_generation_prompt=False))
        labels = list(full)
        if mask_prompt:
            prompt = chat_template_ids(tokenizer, msgs[:-1], add_generation_prompt=True)
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
    cfg = TrainConfig.from_yaml(args.config)

    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger.info("Загрузка модели %s (quant=%s, attn=%s)", cfg.model_id, cfg.quantization, cfg.attn_implementation)
    model, tokenizer = load_base_model(cfg.model_id, quantization=cfg.quantization, attn=cfg.attn_implementation)
    model.config.use_cache = False

    if cfg.quantization in {"nf4", "int8"}:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.grad_checkpoint)
    elif cfg.grad_checkpoint:
        model.enable_input_require_grads()  # нужно для grad-checkpointing с замороженной базой (bf16)

    n_total_layers = model.config.num_hidden_layers
    layers_to_transform = list(range(n_total_layers - cfg.num_layers, n_total_layers))
    peft_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.lora_alpha,            # маппинг MLX scale -> peft alpha
        lora_dropout=cfg.lora_dropout,
        target_modules=TARGET_MODULES,
        layers_to_transform=layers_to_transform,
        layers_pattern="layers",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    logger.info("Слои под LoRA: %s | alpha=%d (scale=%s * rank=%d)",
                layers_to_transform, cfg.lora_alpha, cfg.lora_scale, cfg.rank)
    model.print_trainable_parameters()

    train_ds = build_dataset(cfg.data_dir / "train.jsonl", tokenizer, cfg.max_seq_length, cfg.mask_prompt)
    eval_ds = build_dataset(cfg.valid_jsonl, tokenizer, cfg.max_seq_length, cfg.mask_prompt)

    targs = TrainingArguments(
        output_dir=cfg.adapter_path,
        logging_dir=str(Path(cfg.adapter_path) / "runs"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        max_steps=cfg.iters,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.steps_per_report,
        eval_strategy="steps",
        eval_steps=cfg.eval_every,
        save_strategy="steps",
        save_steps=cfg.eval_every,            # = eval_every (требование load_best_model_at_end)
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim=cfg.optim,
        gradient_checkpointing=cfg.grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.grad_checkpoint else None,
        report_to=["tensorboard"],
        seed=cfg.seed,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, padding=True)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()

    # load_best_model_at_end уже вернул лучший по eval_loss — сохраняем его в корень adapter_path.
    model.save_pretrained(cfg.adapter_path)
    tokenizer.save_pretrained(cfg.adapter_path)
    logger.info("Готово. Лучший eval_loss=%s (checkpoint: %s). Адаптер: %s",
                trainer.state.best_metric, trainer.state.best_model_checkpoint, cfg.adapter_path)
    logger.info("Оценка по полям (mean_f1_all) — отдельно: python scripts/evaluate_model.py --config <cfg> --label <EXP>")


if __name__ == "__main__":
    main()
