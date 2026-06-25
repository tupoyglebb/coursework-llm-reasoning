# Qwen 8B taxonomy distillation on Apple Silicon / MLX QLoRA

This project trains a local Qwen 8B LoRA adapter to imitate the teacher annotations produced in `making(2).ipynb`.

The notebook defines 37 output variables:

- 19 causality fields with values `0/1/2`
- 3 binary fields with values `0/1`
- 15 direction fields with values `-1/0/1`

If you expected 38 categories, check whether you also want to include another target column outside the JSON schema in the notebook. The training code below uses exactly the schema from the attached notebook.

## Expected input file

A CSV/parquet/jsonl file with at least these columns:

```text
experiment_name
group_name
incentive
row_num
stage1_reasoning_text
```

and all 37 label columns listed in `scripts/taxonomy_config.py` as `OUTPUT_FIELDS`.

Rows with `ok != True` are dropped automatically when an `ok` column exists.

## Run

```bash
cd qwen_mps_taxonomy_lora
bash setup_env.sh
bash prepare_data.sh /path/to/teacher_labeled_10000.csv
bash train_qlora.sh
bash infer.sh ./data_mlx/test_source.csv ./predictions/test_predictions.csv
bash evaluate.sh ./predictions/test_predictions.csv
```

## If 32 GB memory is not enough

Try the safer config:

```bash
source .venv/bin/activate
mlx_lm.lora --config ./configs/qwen3_8b_qlora_oom_safe.yaml
```

Also keep `--prompt-style compact` in `prepare_data.sh`; the full coding prompt from the notebook is available, but it is much longer and increases sequence memory.

## Notes

- The target is only the final JSON annotation, not the teacher reasoning.
- `mask_prompt: true` makes the loss apply to the assistant JSON completion only.
- The model is converted to 4-bit MLX first. Training on a quantized model uses QLoRA.
- Inference disables Qwen3 thinking mode when the tokenizer supports `enable_thinking=False`.
