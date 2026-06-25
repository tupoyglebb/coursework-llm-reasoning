# qwen_cuda_binary_lora

QLoRA-дообучение Qwen3-8B для автоматической разметки рассуждений LLM.  
Продолжение `../qwen_cuda_taxonomy_lora/` с переходом causality-переменных с 3 классов на 2.

Подробнее об экспериментах — в `EXPERIMENTS.md`.  
Подробнее об изменениях стека (CUDA vs MLX) — в `HANDOFF_CUDA.md`.

---

## Что лежит в папке

```
configs/          конфиги экспериментов (E0–E8, no_data__E3)
data_binary/      данные для большинства экспериментов (train 6767 / valid 376 / test 376)
data_field_balanced/  данные для E7 (field-aware oversampling, train 8168)
data_balanced_binary/ данные для no_data__E3 (condition oversampling, train 10151)
scripts/          весь код: обучение, оценка, подготовка данных
adapters/         создаётся при запуске — адаптеры LoRA по экспериментам
predictions/      создаётся при запуске — предсказания на тесте (CSV + JSON метрики)
experiments/      создаётся при запуске — results.csv со сводкой всех экспериментов
EXPERIMENTS.md    описание всех экспериментов и мотивация
HANDOFF_CUDA.md   детали CUDA-стека, отличия от MLX, описание ключей конфига
requirements_cuda.txt  зависимости Python
```

**Данные уже сгенерированы локально** и готовы к копированию на сервер.  
Сырой источник: `teacher_labeled_10000_clean.csv` (7519 строк после фильтрации). Команды воспроизведения — в разделе «Если нужно перегенерировать данные» ниже.

---

## Перед первым запуском на сервере

### 1. Скопировать папку на сервер

```bash
# Пример через rsync (с локальной машины):
rsync -av --exclude='*.venv' --exclude='__pycache__' \
    qwen_cuda_binary_lora/ user@server:/path/to/qwen_cuda_binary_lora/
```

Убедиться, что скопировались папки `data_binary/`, `data_field_balanced/`, `data_balanced_binary/`.

### 2. Установить окружение

```bash
# Сначала PyTorch под вашу CUDA (пример для CUDA 12.4):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Затем остальные зависимости:
pip install -r requirements_cuda.txt

# Flash Attention 2 (опционально, но рекомендуется для H200):
MAX_JOBS=32 pip install flash-attn --no-build-isolation
```

Минимальные версии: `transformers >= 4.51` (Qwen3), `peft >= 0.13`.

### 3. Проверить доступность GPU

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 4. Дымовой тест (2 шага обучения)

Временно установить в `configs/tier0_E0.yaml`:
```yaml
iters: 2
eval_every: 1
save_every: 1
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora_cuda.py --config configs/tier0_E0.yaml
```

Если обучение прошло 2 шага без ошибок — вернуть оригинальные значения и запускать эксперименты.

---

## Запуск экспериментов

Каждый эксперимент:
1. Обучает адаптер (сохраняет лучший чекпойнт по `eval_loss`)
2. Запускает генеративную оценку на `data_binary/test_source.csv`
3. Дописывает строку в `experiments/results.csv`

```bash
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E0
```

### Рекомендуемый порядок

```bash
# --- Базовые эксперименты (воспроизведение предыдущей серии на бинарных данных) ---
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E0   # контроль: rank=8, lr=1e-5, 4 слоя
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E1   # + глубина: 36 слоёв
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E2   # + rank=4
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E3   # + lr=1e-4 + cosine (ожидаемый лидер)
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E4   # + lr=3e-4
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E5   # + dropout=0.01

# --- Новые эксперименты ---
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E6          # + rank=16
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E7          # + field-aware oversampling
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E8          # + warmup 170 шагов

# --- Oversampling по условиям (вариант из предыдущей серии, теперь на бинарных данных) ---
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh no_data__E3
```

E0–E5 можно запускать параллельно если есть несколько GPU:
```bash
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E0 &
CUDA_VISIBLE_DEVICES=1 bash run_experiment.sh E1 &
wait
```

### Только оценка (без переобучения)

```bash
# Переоценить готовый адаптер на тесте:
python scripts/evaluate_model.py --config configs/tier0_E3.yaml --label E3

# Оценить конкретный чекпойнт на validation, первые 64 строки (smoke):
python scripts/evaluate_model.py --config configs/tier0_E3.yaml --label E3@ck500 \
    --adapter-path adapters/tier0_E3/checkpoint-500 \
    --input data_binary/valid_source.csv --limit 64
```

### Пересчёт метрик без GPU (если предсказания уже есть)

```bash
python scripts/evaluate_predictions.py \
    --pred predictions/test_pred_E3.csv \
    --label E3_rescore
```

---

## Метрики

| Метрика | Смысл | Baseline (предыдущая серия) |
|---|---|---|
| `mean_f1_all` | macro-F1, усреднённый по всем 37 полям | 0.667 |
| `mean_f1_collapsed` | 14 «проблемных» полей (мера прогресса) | 0.480 |
| `mean_f1_good` | 10 «хороших» полей (guardrail, не должен падать) | 0.943 |

**Критерий победителя:** максимальный `mean_f1_all` при `mean_f1_good` ≥ 0.943.

Лучшие результаты предыдущей серии (3-классовая схема):

| Эксп. | mean_f1_all | mean_f1_collapsed | mean_f1_good |
|---|---|---|---|
| E3 (лидер) | 0.8124 | 0.6852 | 0.9806 |
| E4 | 0.8084 | 0.6834 | 0.9807 |
| E5 | 0.8077 | 0.6820 | 0.9809 |

---

## Если нужно перегенерировать данные

```bash
cd scripts

# Шаг 1: основные данные (бинарный промпт + нормализация 2→1)
python prepare_data_cuda.py \
    --input ../../qwen_mps_taxonomy_lora/teacher_labeled_10000_clean.csv \
    --out-dir ../data_binary \
    --n 10000 --seed 42 --train-frac 0.90 --valid-frac 0.05 --prompt-style compact

# Шаг 2: field-aware oversampling (для E7)
python oversample_fields.py \
    --in-dir ../data_binary --out-dir ../data_field_balanced \
    --max-ratio 10.0 --seed 42

# Шаг 3: oversampling по условиям (для no_data__E3)
python oversample_train.py \
    --in-dir ../data_binary --out-dir ../data_balanced_binary \
    --inflate 1.5 --seed 42
```

---

## Описание скриптов

| Скрипт | Что делает |
|---|---|
| `train_qlora_cuda.py` | QLoRA-обучение; отбор лучшего чекпойнта по `eval_loss`; логи в TensorBoard |
| `evaluate_model.py` | Генеративная оценка адаптера: генерация → парсинг JSON → F1; пишет в `results.csv` |
| `generation.py` | Загрузка модели + батч-генерация (общий код для evaluate/infer) |
| `metrics.py` | Вычисление метрик (macro-F1 по полям, агрегация) |
| `infer_cuda.py` | Разметка данных без gold-меток → CSV (продакшн-инференс) |
| `prepare_data_cuda.py` | Подготовка chat-jsonl из сырого CSV |
| `taxonomy_config.py` | Промпты, список полей, нормализация меток (бинарная схема) |
| `remap_to_binary.py` | Перемаппинг готовых jsonl/csv: gold=2 → 1 (вспомогательный) |
| `oversample_fields.py` | Field-aware oversampling: балансировка на уровне отдельных полей |
| `oversample_train.py` | Oversampling по условиям эксперимента (experiment × group × incentive) |
| `evaluate_predictions.py` | Пересчёт метрик из готового CSV предсказаний (без GPU) |
| `aggregate_metrics.py` | Агрегация нескольких CSV с метриками |

---

## Живые кривые обучения

```bash
tensorboard --logdir adapters/tier0_E3/runs
```
