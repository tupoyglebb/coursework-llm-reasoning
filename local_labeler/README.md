# MoralLens — LLM-разметчик reasoning-трейсов

Дообучение **Qwen3-8B (QLoRA)** для автоматической разметки reasoning-трейсов по таксономии
из социально-научного эксперимента. Модель имитирует разметку «учителя» (GPT-кодера из
[`../notebooks/04_markup_labeling.ipynb`](../notebooks/04_markup_labeling.ipynb)) — на вход трейс,
на выход один JSON-объект из **37 полей**.

Цель — заменить платный API-кодер локальной моделью, не потеряв в качестве разметки.

---

## Три папки: что где

| Папка | Статус | Железо / стек | Схема causality | С чего начать |
|---|---|---|---|---|
| [`qwen_cuda_binary_lora/`](qwen_cuda_binary_lora/) | **актуальная серия** | NVIDIA GPU, `transformers`+`peft`+`bitsandbytes` | бинарная `0/1` | [`README.md`](qwen_cuda_binary_lora/README.md) |
| [`qwen_cuda_taxonomy_lora/`](qwen_cuda_taxonomy_lora/) | предшественник | NVIDIA GPU, тот же стек | 3 класса `0/1/2` | [`EXPERIMENTS.md`](qwen_cuda_taxonomy_lora/EXPERIMENTS.md) |
| [`qwen_mps_taxonomy_lora/`](qwen_mps_taxonomy_lora/) | первая версия | Apple Silicon, MLX / `mlx_lm` | 3 класса `0/1/2` | [`experiments/HANDOFF.md`](qwen_mps_taxonomy_lora/experiments/HANDOFF.md) |

Порядок разработки был такой: прототип на Mac (MLX) → порт на CUDA-сервер для полноценных
прогонов → переход на бинарную схему causality после анализа тестового распределения.

CUDA-версия — самодостаточный порт MLX-версии: то же ядро (промпты, нормализация, метрики),
переписан только движок обучения и инференса. Маппинг гиперпараметров MLX↔CUDA —
в [`qwen_cuda_taxonomy_lora/HANDOFF_CUDA.md`](qwen_cuda_taxonomy_lora/HANDOFF_CUDA.md).

> **Почему код продублирован между CUDA-папками.** 10 из 12 файлов в `scripts/` у обеих
> CUDA-папок идентичны — различается только `taxonomy_config.py` (схема меток и промпт).
> Это сделано намеренно: каждая папка целиком копируется на GPU-сервер через `rsync` и должна
> запускаться там без внешних зависимостей, а старая серия должна оставаться воспроизводимой
> ровно в том виде, в каком её прогоняли. Общий пакет сломал бы оба свойства.

---

## Схема разметки

37 полей на трейс:

| Тип | Кол-во | Значения |
|---|---|---|
| causality | 19 | `0` отсутствует / `1` присутствует (в старых сериях — ещё `2` «ключевой аргумент») |
| binary | 3 | `0` / `1` |
| direction | 15 | `-1` / `0` / `1` |

Переход causality с трёх классов на два мотивирован тем, что класс `2` не встретился ни в одном
из 19 полей тестового набора, — то есть модель обучалась на метках, которых нет в оценке.
Разбор — в [`qwen_cuda_binary_lora/EXPERIMENTS.md`](qwen_cuda_binary_lora/EXPERIMENTS.md).

---

## Дизайн экспериментов

Серия контролируемых прогонов **по одному фактору за раз**:
lr (E1) → rank (E2) → learning rate / schedule (E3, E4) → dropout (E5) → ёмкость адаптера (E6) →
стратегия сэмплирования данных (E7) → warmup (E8).

**Метрики:**

| Метрика | Смысл |
|---|---|
| `mean_f1_all` | macro-F1, усреднённый по всем 37 полям — **основной критерий** |
| `mean_f1_collapsed` | 14 «проблемных» полей, схлопнувшихся в мажоритарный класс — мера прогресса |
| `mean_f1_good` | 10 «хороших» полей — guardrail, не должен падать |
| `flat_macro_f1` | для сравнения **не используется**: завышен, маскирует схлопнувшиеся поля |

**Критерий победителя:** максимальный `mean_f1_all` при `mean_f1_good ≥ 0.943`.

Baseline (прод-разметчик предыдущего поколения): `mean_f1_all = 0.667`, `mean_f1_good = 0.943`.

---

## Результаты актуальной серии (бинарная схема, E0–E7)

| Эксп. | Изучаемый фактор | `mean_f1_all` | `mean_f1_collapsed` | `mean_f1_good` |
|---|---|---|---|---|
| E0 | контроль (rank=8, lr=1e-5, 4 слоя) | 0.7530 | 0.6045 | 0.9732 |
| E1 | глубина адаптера: 4→36 слоёв | 0.7976 | 0.6715 | 0.9790 |
| E2 | ранг: 8→4 | 0.7892 | 0.6549 | 0.9783 |
| E3 | lr=1e-4 + cosine | 0.8071 | 0.6858 | 0.9814 |
| E4 | lr=3e-4 | 0.8032 | 0.6775 | 0.9798 |
| E5 | dropout=0.01 | 0.8115 | 0.6923 | 0.9790 |
| E6 | ранг: 8→16 | 0.8178 | 0.6954 | 0.9822 |
| **E7** | **field-aware oversampling** | **0.8204** | **0.7081** | **0.9825** |
| — | *baseline (прод-разметчик)* | *0.667* | *0.480* | *0.943* |

**Победитель — E7:** `mean_f1_all` 0.667 → **0.820** относительно baseline. Ключевой вклад —
целевой оверсэмплинг по редким классам отдельных полей, который «разморозил»
`common_interests_prioritization_causality` (F1 0.499 → 0.698); остальные методы это поле не сдвигали.

E8, `no_data__E3` и Tier-1 конфиги не завершены — прогон на сервере остановился раньше.
Честный статус каждого эксперимента и разбор оставшихся «замороженных» полей —
в [`qwen_cuda_binary_lora/EXPERIMENTS.md`](qwen_cuda_binary_lora/EXPERIMENTS.md).

---

## Как читать код

Основное — в `qwen_cuda_binary_lora/scripts/`:

| Файл | Что делает |
|---|---|
| `taxonomy_config.py` | схема 37 полей, сборка промпта, нормализация и валидация JSON-ответа модели |
| `prepare_data_cuda.py` | сырой CSV учительской разметки → `train/valid/test.jsonl` + source-CSV |
| `remap_to_binary.py` | перемаппинг 3-классовой схемы в бинарную (`2 → 1`) |
| `oversample_fields.py` | field-aware oversampling (эксперимент E7) |
| `oversample_train.py` | oversampling по условиям эксперимента (experiment × group × incentive) |
| `train_qlora_cuda.py` | **обучение**: 4-bit квантизация + LoRA, чекпойнты по `eval_loss` |
| `generation.py` / `infer_cuda.py` | генеративный инференс на тесте |
| `metrics.py` / `evaluate_predictions.py` | per-field macro-F1, разбиение на collapsed/good поля |
| `evaluate_model.py` | инференс + метрики одной командой |
| `aggregate_metrics.py` | сводка всех прогонов в `experiments/results.csv` |

Конфиг эксперимента — один YAML в `configs/` (гиперпараметры LoRA, lr/schedule, путь к данным).

---

## Запуск

```bash
cd qwen_cuda_binary_lora
pip install -r requirements_cuda.txt

CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E7   # обучение → инференс → метрики → results.csv
bash select_best_checkpoint.sh E7                  # лучший чекпойнт по valid + число на test
```

Полная инструкция (подготовка сервера, дымовой тест, переоценка без переобучения,
регенерация данных) — в [`qwen_cuda_binary_lora/README.md`](qwen_cuda_binary_lora/README.md).

---

## Что не лежит в репозитории

Код, конфиги, обучающие данные и файлы метрик — в репозитории и достаточны для воспроизведения.

Исключены через `.gitignore`: `.venv/`, базовая модель (`models/`, качается с HuggingFace),
обученные LoRA-адаптеры и `*.safetensors`, предсказания на тесте (`test_pred_*.csv`), архивы
прогонов. Как их восстановить — в соответствующем HANDOFF.
