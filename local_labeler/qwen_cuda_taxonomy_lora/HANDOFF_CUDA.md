# Handoff (NVIDIA / CUDA): порт стека разметчика с MLX на transformers + peft + bitsandbytes

Самодостаточная CUDA-версия проекта. MLX-стек в `../qwen_mps_taxonomy_lora/` **не тронут** — это отдельная папка,
её можно целиком скопировать на GPU-машину. Логика экспериментов та же; см. планы в
`../qwen_mps_taxonomy_lora/experiments/TIER0_PLAN.md` и `TIER1_PLAN.md` (они стек-агностичны).

## TL;DR
```bash
# 0) окружение (на GPU-машине)
pip install torch --index-url https://download.pytorch.org/whl/cu124   # под свою CUDA
pip install -r requirements_cuda.txt

# 1) данные: положить чат-jsonl и source-csv в ./data
python scripts/prepare_data_cuda.py --input <teacher>.csv --out-dir ./data --n 10000 \
       --train-frac 0.90 --valid-frac 0.05 --prompt-style compact --seed 42

# 2) запуск (одна команда на эксперимент)
bash run_experiment.sh E0      # затем E1 E2 E3 ; потом E4 E5
bash select_best_checkpoint.sh E1   # отбор лучшего чекпойнта по valid + число на test
```

---

## Чем отличается от MLX-стека (важно для воспроизводимости плана)

| Параметр в плане (MLX) | Как реализовано здесь (CUDA) |
|---|---|
| 4-bit affine (MLX) | **NF4** + double quant (bitsandbytes). Численно не идентично — сравнивай в пределах одного стека |
| `lora.scale = 20` | `lora_alpha = scale * rank` (в peft множитель = alpha/r). rank8→alpha160, rank32→alpha640 |
| `num_layers: N` (последние N) | `layers_to_transform = [L-N .. L-1]` (L=36). E0:4→[32..35], E4:16→[20..35], E5:24→[12..35] |
| целевые модули | `q,k,v,o,gate,up,down_proj` — те же |
| `mask_prompt: true` | лосс только по ответу: промпт-токены → `-100` в labels (в `train_qlora_cuda.py`) |
| `optimizer: adamw` | `paged_adamw_8bit` (экономнее по памяти; адаптер крошечный, разница невелика) |
| cosine + warmup | `lr_scheduler: cosine`, `warmup_steps` |
| MLX-квант модель | грузим оригинал `Qwen/Qwen3-8B` с HF + bnb на лету |

Адаптеры MLX сюда **не переносятся** (другой формат) — модели переобучаются на этом стеке.

---

## Что нужно проверить/решить (минимум)

1. **PyTorch под свою CUDA** — поставить ПЕРВЫМ (строка-пример в `requirements_cuda.txt`). Без этого bitsandbytes не заведётся.
2. **Данные в `./data`.** Либо сгенерировать `prepare_data_cuda.py` (даст `train/valid/test.jsonl` + `*_source.csv`),
   либо скопировать готовые из `../qwen_mps_taxonomy_lora/data_mlx/` (нужны `train.jsonl`, `valid.jsonl`,
   `valid_source.csv`, `test_source.csv`). Канонический сплит — 6767/376/376 (см. план); тогда `iters: 1700`.
3. **Дымовой тест** перед полным прогоном:
   ```bash
   python scripts/infer_cuda.py --input ./data/test_source.csv --output /tmp/smoke.csv \
       --adapter-path Qwen/Qwen3-8B --limit 2   # упадёт на адаптере — это ок, проверяем загрузку модели/токенизатора
   ```
   Полезнее — запустить обучение на 2 шага: временно `iters: 2` в `configs/tier0_E1.yaml`.
4. **Версии.** Qwen3 требует `transformers >= 4.51`. Если peft не примет `layers_to_transform`/`layers_pattern`
   на твоей версии — обнови peft (`>=0.13`).

---

## Порядок запуска (тот же, что в плане)

### Tier 0 (lr → rank → данные)
```bash
bash run_experiment.sh E0          # контроль
bash run_experiment.sh E1          # + lr 1e-4 + cosine
bash run_experiment.sh E2          # + rank 32
python scripts/oversample_train.py --in-dir ./data --out-dir ./data_balanced --inflate 1.5 --seed 42
bash run_experiment.sh E3          # + oversampling (data_dir уже = ./data_balanced в конфиге)
# лучший чекпойнт по valid + число на test (метка E*-best):
bash select_best_checkpoint.sh E1  # 3 последних checkpoint-* + финал ; третий аргумент = лимит valid
```
Победитель T0\* = max `mean_f1_all` при не-просевшем `mean_f1_good`.

### Tier 1 (глубина; на NVIDIA дешевле, чем на Mac)
Открой `configs/tier1_E4.yaml` / `E5.yaml` — они = конфиг T0\* с изменённым `num_layers`.
Если победитель Tier 0 другой (например, E3 с балансировкой) — поправь `data_dir`/`iters`/`rank` под него.
```bash
bash run_experiment.sh E4          # num_layers 16
bash run_experiment.sh E5          # num_layers 24 (при OOM см. шапку tier1_E5.yaml)
```

---

## Метрики и отбор — без изменений
Те же `scripts/evaluate_predictions.py` и `scripts/aggregate_metrics.py`, что в MLX-стеке (чистый pandas/sklearn).
Критерий — `mean_f1_all` (база prod 0.667) при guardrail `mean_f1_good`; `flat_macro_f1` не используем.
Результаты копятся в `experiments/results.csv`.

## Инвентарь
| Файл | Что это | Откуда |
|---|---|---|
| `scripts/train_qlora_cuda.py` | обучение QLoRA (transformers+peft+bnb) | новый |
| `scripts/infer_cuda.py` | инференс → CSV в формате MLX-стека | новый |
| `scripts/prepare_data_cuda.py` | подготовка чат-jsonl (= MLX prepare) | копия |
| `scripts/taxonomy_config.py`, `evaluate_predictions.py`, `aggregate_metrics.py`, `oversample_train.py` | промпты/нормализация, метрики, баланс | копии (стек-агностичны) |
| `configs/tier0_E0..E3.yaml`, `tier1_E4..E5.yaml` | конфиги экспериментов (CUDA) | новые |
| `run_experiment.sh`, `select_best_checkpoint.sh` | запускаторы | новые |
| `requirements_cuda.txt` | зависимости | новый |
