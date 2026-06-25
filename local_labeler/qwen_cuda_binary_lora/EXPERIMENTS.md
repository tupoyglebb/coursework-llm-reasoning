# Эксперименты: бинарная схема causality (CUDA / QLoRA)

Продолжение `../qwen_cuda_taxonomy_lora/`. Единственное принципиальное изменение схемы — переход causality-переменных с трёх классов на два.

---

## Мотивация: зачем бинарная схема

В предыдущей серии (`qwen_cuda_taxonomy_lora`) causality-переменные имели три значения:
- `0` — категория отсутствует
- `1` — упомянута, но не центральная
- `2` — ключевой аргумент

Анализ тестового набора (376 наблюдений) показал, что **класс 2 отсутствует во всех 19 causality-полях**. Метрика вычислялась по `labels = union(unique(gold), unique(pred)) = {0, 1}`, то есть де-факто была бинарной. При этом модель обучалась на трёхклассовых таргетах: примеры с gold=2 учили модель выдавать `2`, тогда как корректный ответ в тестовом распределении — `1`. Это вносило шум в обучение без возможности проверить его эффект.

**Переход к явной бинарной схеме** устраняет несоответствие: обучение и оценка теперь работают с одним и тем же пространством меток.

---

## Изменения относительно предыдущей серии

### Схема разметки

| Тип | Было | Стало |
|---|---|---|
| Causality (19 полей) | 0 / 1 / 2 | **0 / 1** (2 → 1) |
| Binary (3 поля) | 0 / 1 | без изменений |
| Direction (15 полей) | −1 / 0 / 1 | без изменений |

### Промпт

Causality-инструкция упрощена:
- Было: `0 absent; 1 mentioned but not central; 2 key argument`
- Стало: `0 absent; 1 present (regardless of centrality)`

### Нормализация (`taxonomy_config.py`)

```python
# Было:
clean[col] = normalize_int_label(ann.get(col, 0), allowed={0, 1, 2}, default=0)

# Стало:
raw = normalize_int_label(ann.get(col, 0), allowed={0, 1, 2}, default=0)
clean[col] = min(1, raw)  # 2 → 1
```

`min(1, raw)` сохраняет обратную совместимость: если учительская модель всё ещё генерирует `2`, оно корректно схлопывается в `1` при нормализации.

---

## Шаг 1: подготовка данных

Перемаппировать существующие данные из `qwen_cuda_taxonomy_lora/data/`:

```bash
python scripts/remap_to_binary.py \
    --in-dir ../qwen_cuda_taxonomy_lora/data \
    --out-dir ./data_binary

# Если используется балансировка:
python scripts/remap_to_binary.py \
    --in-dir ../qwen_cuda_taxonomy_lora/data_balanced \
    --out-dir ./data_balanced_binary
```

Скрипт обрабатывает:
- `train.jsonl`, `valid.jsonl`, `test.jsonl` — меняет значение `2 → 1` в JSON ответа ассистента
- `train_source.csv`, `valid_source.csv`, `test_source.csv` — меняет значение `2 → 1` в gold-колонках

После перемаппинга обновить `data_dir` в конфигах на `./data_binary`.

---

## Конфиги экспериментов

### Перенесённые из предыдущей серии (E0–E5 + no_data\_\_E3)

Гиперпараметры не меняются — изменилась только схема данных (`data_binary`).

| Эксп. | Изучаемый фактор | lr | rank | num_layers | Data |
|---|---|---|---|---|---|
| E0 | контроль | 1e-5 | 8 | 4 | data_binary |
| E1 | глубина: 4→36 слоёв | 1e-5 | 8 | 36 | data_binary |
| E2 | ранг: 8→4 | 1e-5 | 4 | 36 | data_binary |
| E3 | lr: 1e-5→1e-4 + cosine | 1e-4 | 8 | 36 | data_binary |
| E4 | lr: 1e-4→3e-4 | 3e-4 | 8 | 36 | data_binary |
| E5 | dropout: 0.05→0.01 | 3e-4 | 8 | 36 | data_binary |
| no_data\_\_E3 | oversampling по условиям | 1e-4 | 32 | 4 | data_balanced_binary |

### Новые эксперименты (E6–E8)

| Эксп. | Изучаемый фактор | lr | rank | num_layers | warmup | Data |
|---|---|---|---|---|---|---|
| E6 | **ранг: 8→16** | 1e-4 | **16** | 36 | 0 | data_binary |
| E7 | **field-aware oversampling** | 1e-4 | 8 | 36 | 0 | data_field_balanced |
| E8 | **warmup: 0→170 шагов** | 1e-4 | 8 | 36 | **170** | data_binary |

---

## Описание новых экспериментов

### E6 — Ранг 16 (расширение ёмкости адаптера)

Предыдущая серия показала, что rank=4 почти не уступает rank=8 (−0.003). Эксперимент проверяет другую сторону: помогает ли бо́льшая ёмкость при бинарной схеме, которая проще трёхклассовой. lora_alpha = 20 × 16 = 320.

### E7 — Field-aware oversampling

Стандартный оверсамплинг (`oversample_train.py`) балансирует по условиям эксперимента (experiment × group × incentive), но не затрагивает конкретные поля разметки. В тестовом наборе обнаружены поля с крайним дисбалансом:

| Поле | 0 | 1 | F1 в E3 | Проблема |
|---|---|---|---|---|
| `self_deception_causality` | 372 | 4 | 0.497 | Модель никогда не предсказывает 1 |
| `common_interests_prioritization_causality` | 1 | 375 | 0.499 | Модель никогда не предсказывает 0 |

`scripts/oversample_fields.py` дублирует строки, в которых **редкий класс присутствует** в любом causality-поле, до тех пор, пока соотношение majority/minority не опустится до `--max-ratio` (default 10). Это целевое вмешательство, которое не затрагивает сбалансированные поля.

Подготовка данных:
```bash
python scripts/oversample_fields.py \
    --in-dir ./data_binary --out-dir ./data_field_balanced \
    --max-ratio 10.0 --seed 42
```

### E8 — Warmup

E3 (победитель) обучает LoRA на всех 36 слоях при lr=1e-4 без warmup. 36 слоёв — это 9× больше параметров адаптера, чем у E0 (4 слоя). Первые шаги с полным lr по всем слоям одновременно могут давать нестабильные градиенты. Warmup 170 шагов (~10% от 1700) — стандартная практика для глубоких адаптеров и является единственным отличием от E3.

---

## Порядок запуска

```bash
# Шаг 1: перемаппинг данных
python scripts/remap_to_binary.py \
    --in-dir ../qwen_cuda_taxonomy_lora/data \
    --out-dir ./data_binary

# Шаг 2: перенесённые эксперименты (E0–E5)
bash run_experiment.sh E0
bash run_experiment.sh E1
bash run_experiment.sh E2
bash run_experiment.sh E3   # ожидаемый победитель (= E3 из предыдущей серии + binary)
bash run_experiment.sh E4
bash run_experiment.sh E5

# Шаг 3: новые эксперименты
bash run_experiment.sh E6   # rank=16

python scripts/oversample_fields.py \
    --in-dir ./data_binary --out-dir ./data_field_balanced --max-ratio 10.0 --seed 42
bash run_experiment.sh E7   # field-aware oversampling

bash run_experiment.sh E8   # warmup

# Шаг 4: оверсамплинг по условиям (если нужен)
python scripts/oversample_train.py \
    --in-dir ./data_binary --out-dir ./data_balanced_binary --inflate 1.5 --seed 42
bash run_experiment.sh no_data__E3
```

---

## Ожидаемые эффекты

Поля, которые были «заморожены» в предыдущей серии из-за дисбаланса 0/1, после перемаппинга могут улучшиться — данные теперь согласованы с оценкой. Особенно интересно наблюдать за:

- `self_deception_causality` (gold: 372×0, 4×1 — крайний дисбаланс, не связан с классом 2)
- `common_interests_prioritization_causality` (gold: 1×0, 375×1 — обратный дисбаланс)
- `ratio_of_goals_and_means_causality` (в предыдущей серии лучший прирост от lr-изменения)

---

## Базовые метрики для сравнения (из `qwen_cuda_taxonomy_lora`)

| Эксп. | mean_f1_all | mean_f1_collapsed | mean_f1_good |
|---|---|---|---|
| E0 | 0.7585 | 0.5986 | 0.9733 |
| E1 | 0.7963 | 0.6678 | 0.9782 |
| E2 | 0.7931 | 0.6664 | 0.9781 |
| **E3** | **0.8124** | **0.6852** | 0.9806 |
| E4 | 0.8084 | 0.6834 | 0.9807 |
| E5 | 0.8077 | 0.6820 | **0.9809** |
| Прод-baseline | 0.667 | 0.480 | 0.943 |
