# Handoff (NVIDIA / CUDA): порт стека разметчика с MLX на transformers + peft + bitsandbytes

Самодостаточная CUDA-версия проекта. MLX-стек в `../qwen_mps_taxonomy_lora/` **не тронут** — это отдельная папка,
её можно целиком скопировать на GPU-машину. Логика экспериментов та же; см. планы в
`../qwen_mps_taxonomy_lora/experiments/TIER0_PLAN.md` и `TIER1_PLAN.md` (они стек-агностичны).

## TL;DR
```bash
# 0) окружение (на GPU-машине)
pip install torch --index-url https://download.pytorch.org/whl/cu124   # под свою CUDA
pip install -r requirements_cuda.txt
# (опц.) flash-attention — ОБЯЗАТЕЛЬНО ограничить потоки сборки, иначе OOM:
#   MAX_JOBS=32 pip install flash-attn --no-build-isolation

# 1) данные: положить чат-jsonl и source-csv в ./data
python scripts/prepare_data_cuda.py --input <teacher>.csv --out-dir ./data --n 10000 \
       --train-frac 0.90 --valid-frac 0.05 --prompt-style compact --seed 42

# 2) запуск (одна команда): обучение (лучший чекпойнт по eval_loss) + генеративная оценка на тесте
CUDA_VISIBLE_DEVICES=1 bash run_experiment.sh E0      # затем E1 E2 E3 ; потом E4 E5

# 2b) ту же оценку можно запускать НЕЗАВИСИМО (любой чекпойнт/выборка):
python scripts/evaluate_model.py --config configs/tier0_E0.yaml --label E0          # лучший адаптер на тесте
python scripts/evaluate_model.py --config configs/tier0_E0.yaml --label E0@ck500 \
       --adapter-path adapters/tier0_E0/checkpoint-500 --input data/valid_source.csv --limit 64

# 3) (опц.) живые кривые eval_loss/train_loss
tensorboard --logdir adapters/tier0_E0/runs
```

Лучший чекпойнт выбирается ВНУТРИ обучения по `eval_loss` (HF `load_best_model_at_end`).
Генеративная оценка по полям (`mean_f1_all`) — дорогая, поэтому вынесена в отдельный
`scripts/evaluate_model.py` и гоняется только на тесте (не в обучении). `select_best_checkpoint.sh` удалён.

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
```
Каждый прогон сам валидируется на `valid_source.csv`, выбирает лучший чекпойнт по `eval_mean_f1_all`
и пишет его число на тесте в `experiments/results.csv` (метка `E*`).
Победитель T0\* = max `mean_f1_all` при не-просевшем `mean_f1_good`.

### Tier 1 (глубина; на NVIDIA дешевле, чем на Mac)
Открой `configs/tier1_E4.yaml` / `E5.yaml` — они = конфиг T0\* с изменённым `num_layers`.
Если победитель Tier 0 другой (например, E3 с балансировкой) — поправь `data_dir`/`iters`/`rank` под него.
```bash
bash run_experiment.sh E4          # num_layers 16
bash run_experiment.sh E5          # num_layers 24 (при OOM см. шапку tier1_E5.yaml)
```

---

## Метрики и отбор
Отбор лучшего чекпойнта в обучении — по `eval_loss` (дёшево, teacher-forcing на `valid.jsonl`).
Генеративная оценка по полям — `scripts/evaluate_model.py` (генерит модель, парсит JSON, считает F1):
критерий `mean_f1_all` (база prod 0.667) при guardrail `mean_f1_good`; `flat_macro_f1` не используем.
Логика метрик — в `scripts/metrics.py` (чистый pandas/sklearn); `evaluate_model.py`,
`evaluate_predictions.py`, `aggregate_metrics.py` зовут одни и те же функции.
Результаты копятся в `experiments/results.csv`; живые кривые eval_loss/train_loss — в TensorBoard.

## Новые ключи конфига (отбор по loss / скорость / оценка; у E1..E5 берутся из дефолтов)
| Ключ | По умолчанию | Смысл |
|---|---|---|
| `quantization` | `nf4` | QLoRA: учим адаптер на 4-бит базе -> он компенсирует квант; инференс тоже nf4 (train/serve совпадают). `none`=bf16 быстрее, но только если сервить не квантованно. `int8` |
| `attn_implementation` | `sdpa` | `sdpa` без зависимостей; `flash_attention_2` если собран |
| `eval_every` | = `save_every` | каждые N шагов: eval_loss + сохранение (save=eval для отбора лучшего) |
| `gen_batch_size` | `16` | батч генерации в `evaluate_model.py` / `infer_cuda.py` |
| `grad_checkpoint` | `false` | для bf16 не нужен; включить при OOM (nf4/большой батч) |
| `prompt_style`, `val_max_new_tokens` | `compact`, `900` | читает `evaluate_model.py` (в обучении не нужны) |

## Инвентарь
| Файл | Что это | Откуда |
|---|---|---|
| `scripts/train_qlora_cuda.py` | обучение LoRA, отбор лучшего по eval_loss (TensorBoard) | новый |
| `scripts/evaluate_model.py` | генеративная оценка адаптера на тесте (независимый запуск) | новый |
| `scripts/generation.py` | загрузка модели + батч-генерация (общий код eval/infer) | новый |
| `scripts/metrics.py` | метрики (общий код evaluate_model/evaluate/aggregate) | новый |
| `scripts/infer_cuda.py` | разметка данных БЕЗ gold → CSV (продакшн-инференс) | новый |
| `scripts/prepare_data_cuda.py` | подготовка чат-jsonl (= MLX prepare) | копия |
| `scripts/taxonomy_config.py`, `oversample_train.py` | промпты/нормализация, баланс | копии (стек-агностичны) |
| `scripts/evaluate_predictions.py`, `aggregate_metrics.py` | CLI-обёртки над `metrics.py` (рескоринг CSV без GPU) | новые |
| `configs/tier0_E0..E3.yaml`, `tier1_E4..E5.yaml` | конфиги экспериментов (CUDA) | новые |
| `run_experiment.sh` | запускатор (train → evaluate_model на тесте) | новый |
| `requirements_cuda.txt` | зависимости | новый |
