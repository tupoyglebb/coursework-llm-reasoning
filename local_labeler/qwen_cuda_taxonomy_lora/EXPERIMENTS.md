# Эксперименты по дообучению Qwen3-8B (CUDA / QLoRA)

Стек: `transformers + peft + bitsandbytes` на NVIDIA H200. Порт MLX-стека (`../qwen_mps_taxonomy_lora/`), адаптеры между стеками несовместимы — модели переобучались заново.

---

## Задача и данные

**Задача.** Автоматическая разметка трассировок рассуждений LLM для социально-научного эксперимента. Модель получает метаданные (название эксперимента, группа/treatment, incentive) и текст рассуждения, и возвращает JSON с 37 переменными.

**Схема разметки (37 полей):**

| Тип | Полей | Значения | Примеры |
|---|---|---|---|
| Causality | 19 | 0 / 1 / 2 | `self_enhancement_causality`, `fear_negative_evaluation_causality` |
| Binary | 3 | 0 / 1 | `possibility_of_different_opinions`, `moral_style_reasoning_present`, `rational_style_reasoning_present` |
| Direction | 15 | −1 / 0 / 1 | `treatment_effect_direction`, `emotional_reasoning_direction`, `cruelty_direction` |

Causality кодирует силу аргумента (0 = отсутствует, 1 = упомянут, 2 = ключевой). Direction — направление влияния аргумента на итоговую оценку.

**Данные.** ~10 000 строк, стратифицированный сплит по `experiment_name × group_name × incentive`.  
Канонический размер: **6 767 train / 376 valid / 376 test**.  
При 1 700 шагах (`iters`) это ≈ 2 эпохи на train.

---

## Базовая конфигурация (одинакова во всех экспериментах)

| Параметр | Значение |
|---|---|
| Модель | `Qwen/Qwen3-8B` |
| Квантизация | NF4 + double quant (bitsandbytes) |
| Compute dtype | bfloat16 |
| Attention | Flash Attention 2 (H200) |
| Целевые модули LoRA | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| LoRA scale | 20.0 → `lora_alpha = 20 × rank` |
| Оптимизатор | `adamw_torch` |
| Маскировка промпта | да — лосс только по ответу ассистента |
| Эффективный batch | 8 (разные комбинации bs × grad_accum) |
| Max sequence length | 3 072 токенов |
| Iters | 1 700 (~2 эпохи) |
| Seed | 42 |
| Отбор чекпойнта | по `eval_loss` внутри обучения (`load_best_model_at_end`) |
| Grad checkpoint | нет (H200 достаточно памяти) |
| Промпт-стиль | compact |

---

## Структура экспериментов

Все эксперименты в **одном уровне (Tier 0)**. Каждый меняет **ровно один фактор** относительно предыдущего:

```
E0 — контроль (4 слоя, lr=1e-5, constant, rank=8)
 ├─ E1 — глубина: num_layers 4 → 36 (все слои)
 ├─ E2 — ранг: rank 8 → 4 (уменьшение ёмкости адаптера)
 ├─ E3 — learning rate: 1e-5 → 1e-4 + cosine
 ├─ E4 — learning rate: 1e-4 → 3e-4 + cosine
 └─ E5 — dropout: 0.05 → 0.01 (при lr=3e-4)

no_data__E3 — oversampling редких классов (отдельный конфиг, rank=32)
```

---

## Эксперименты

### E0 — Контроль

Воспроизводит гиперпараметры продакшн-модели на новом CUDA-стеке. Служит нижней границей.

| Параметр | Значение |
|---|---|
| Learning rate | 1e-5 |
| LR scheduler | constant (без расписания) |
| Warmup steps | 0 |
| LoRA rank | 8 |
| lora_alpha | 160 |
| num_layers | **4** (слои 32–35 из 36) |
| batch_size × grad_accum | 8 × 1 |
| Quantization | nf4 |
| Attn | flash_attention_2 |

---

### E1 — Глубина: все слои

Единственное изменение: LoRA накрывает все трансформерные блоки вместо последних четырёх.

| Параметр | E0 → E1 |
|---|---|
| num_layers | 4 → **36** (слои 0–35, все) |
| batch_size × grad_accum | 8 × 1 → **2 × 4** |

---

### E2 — Ранг: меньше параметров

Единственное изменение: уменьшение ранга адаптера вдвое (снижение ёмкости, меньше риск переобучения).

| Параметр | E0 → E2 |
|---|---|
| LoRA rank | 8 → **4** |
| lora_alpha | 160 → **80** |
| num_layers | 4 → **36** |
| batch_size × grad_accum | 8 × 1 → **2 × 4** |

---

### E3 — Learning rate: 1e-4 + cosine

Единственное изменение: повышение lr и добавление cosine-расписания.

| Параметр | E0 → E3 |
|---|---|
| Learning rate | 1e-5 → **1e-4** |
| LR scheduler | constant → **cosine** |
| num_layers | 4 → **36** |
| batch_size × grad_accum | 8 × 1 → **2 × 4** |

---

### E4 — Learning rate: 3e-4 + cosine

Единственное изменение относительно E3: более агрессивный lr.

| Параметр | E3 → E4 |
|---|---|
| Learning rate | 1e-4 → **3e-4** |

Остальное как в E3 (cosine, rank=8, num_layers=36).

---

### E5 — Dropout

Единственное изменение относительно E4: снижение регуляризации.

| Параметр | E4 → E5 |
|---|---|
| LoRA dropout | 0.05 → **0.01** |

Остальное как в E4 (lr=3e-4, cosine, rank=8, num_layers=36).

---

### no_data\_\_tier0\_E3 — Oversampling редких классов

Отдельный эксперимент с балансировкой обучающей выборки. Valid и test не меняются.

| Параметр | Значение |
|---|---|
| Data dir | `./data_balanced` (inflate=1.5, ~10 150 строк) |
| Iters | **2 500** (~2 эпохи на раздутом наборе) |
| LoRA rank | **32** (alpha=640) |
| num_layers | 4 (слои 32–35) |
| Learning rate | 1e-4, cosine, warmup=50 |
| batch_size × grad_accum | 8 × 1 |

`data_balanced` создаётся: `python scripts/oversample_train.py --in-dir ./data --out-dir ./data_balanced --inflate 1.5 --seed 42`

---

## Сводная таблица

| Эксперимент | Изучаемый фактор | lr | Scheduler | rank | alpha | num_layers | Слои | bs × accum | Data | Iters |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | контроль | 1e-5 | constant | 8 | 160 | 4 | 32–35 | 8 × 1 | data | 1700 |
| E1 | **глубина** | 1e-5 | constant | 8 | 160 | **36** | **0–35** | 2 × 4 | data | 1700 |
| E2 | **ранг ↓** | 1e-5 | constant | **4** | **80** | 36 | 0–35 | 2 × 4 | data | 1700 |
| E3 | **lr + cosine** | **1e-4** | **cosine** | 8 | 160 | 36 | 0–35 | 2 × 4 | data | 1700 |
| E4 | **lr ↑↑** | **3e-4** | cosine | 8 | 160 | 36 | 0–35 | 2 × 4 | data | 1700 |
| E5 | **dropout ↓** | 3e-4 | cosine | 8 | 160 | 36 | 0–35 | 2 × 4 | data | 1700 |
| no_data\_\_E3 | **oversampling** | 1e-4 | cosine | 32 | 640 | 4 | 32–35 | 8 × 1 | **data_balanced** | **2500** |

---

## Отбор чекпойнта и метрики

**Отбор внутри обучения** — по `eval_loss` (teacher-forcing на `valid.jsonl`) каждые 100 шагов. `load_best_model_at_end=True`. Кривые train_loss/eval_loss — в TensorBoard (`tensorboard --logdir adapters/tier0_E*/runs`).

**Генеративная оценка по полям** — запускается отдельно после обучения через `scripts/evaluate_model.py`:

| Метрика | Поля | Роль | Прод-baseline |
|---|---|---|---|
| `mean_f1_all` | все 37 | **PRIMARY** | 0.667 |
| `mean_f1_collapsed` | 14 «проблемных» | Secondary | 0.480 |
| `mean_f1_good` | 10 «хороших» | **Guardrail** | 0.943 |

Критерий победителя: max `mean_f1_all` при условии `mean_f1_good` не ниже 0.943. `flat_macro_f1` для отбора не используется.

---

## Пайплайн запуска

```bash
# Базовые эксперименты
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E0
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E1
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E2
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E3
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E4
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E5

# Oversampling (предварительно создать data_balanced)
python scripts/oversample_train.py --in-dir ./data --out-dir ./data_balanced --inflate 1.5 --seed 42
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E3  # с конфигом no_data__tier0_E3.yaml

# Независимая оценка любого чекпойнта
python scripts/evaluate_model.py --config configs/tier0_E3.yaml --label E3
python scripts/evaluate_model.py --config configs/tier0_E0.yaml --label E0@ck500 \
       --adapter-path adapters/tier0_E0/checkpoint-500 --input data/valid_source.csv --limit 64
```

Каждый вызов `run_experiment.sh` запускает: обучение → `evaluate_model.py` на тесте → дозапись в `experiments/results.csv`.

---

## Инвентарь

| Файл | Что это |
|---|---|
| `scripts/train_qlora_cuda.py` | обучение LoRA, отбор лучшего по eval_loss, TensorBoard |
| `scripts/evaluate_model.py` | генеративная оценка адаптера на тесте (независимый запуск) |
| `scripts/generation.py` | загрузка модели + батч-генерация (общий код eval/infer) |
| `scripts/metrics.py` | метрики (общий код для evaluate_model/evaluate/aggregate) |
| `scripts/infer_cuda.py` | разметка данных без gold → CSV (продакшн-инференс) |
| `scripts/prepare_data_cuda.py` | подготовка чат-jsonl |
| `scripts/taxonomy_config.py` | промпты и нормализация аннотаций |
| `scripts/oversample_train.py` | oversampling редких классов |
| `scripts/evaluate_predictions.py`, `aggregate_metrics.py` | CLI-обёртки над `metrics.py` для рескоринга CSV без GPU |
| `configs/tier0_E0..E5.yaml` | конфиги 6 экспериментов |
| `configs/no_data__tier0_E3.yaml` | конфиг oversampling-эксперимента |
| `run_experiment.sh` | запускатор (train → evaluate_model) |
| `prepare.sh` | подготовка окружения и данных |

---

## Отличия от MLX-стека

| Параметр (MLX) | Реализация здесь (CUDA / H200) |
|---|---|
| 4-bit affine квантизация | NF4 + double quant — численно не идентично |
| `lora.scale = 20` | `lora_alpha = scale × rank` |
| `num_layers: N` (последние N) | `layers_to_transform = [36−N .. 35]` |
| Отбор лучшего чекпойнта | `select_best_checkpoint.sh` удалён — отбор по `eval_loss` внутри обучения |
| `optimizer: adamw` | `adamw_torch` (H200 не требует экономии памяти) |
| batch_size=1, grad_accum=8 | batch_size=8 (или 2×4) — эффективный батч тот же |
| Адаптеры MLX | **Несовместимы** — переобучение с нуля |
