# Thinking Exposition in LLMs — Coursework Analysis

Курсовая работа (ВШЭ, 2025–2026). Исследование того, как reasoning-модели реагируют на социально-психологические виньетки и как prompt-based стимулы влияют на ответы и процесс рассуждения.

## Что измеряется

Модели симулируют «правдоподобного респондента опроса» в двух репликациях классических социальных экспериментов:

| Эксперимент | Тема |
|---|---|
| Hassner 2023 | Поддержка sleep deprivation при допросе (4 тритмента) |
| Savary & Goldsmith 2020 | Готовность пожертвовать $5 (3 тритмента) |

Манипуляция — **incentive**: дополнительный абзац в system prompt, варьирующий условия реализма/целей опроса (ключи A, C, H, I, J, K, L, M, X, `no_incentive`).

**Модели:** `moonshotai/kimi-k2-thinking`, `z-ai/glm-4.6`, `qwen/qwen3.5-9b`, `minimax/minimax-m2.5` (через OpenRouter API) + `Qwen3-8B` локально (скрытые состояния).

---

## Структура репозитория

```
├── notebooks/          ← основное: ноутбуки в порядке пайплайна
├── configs/            ← конфиги экспериментов и промптов
├── annotation/         ← оценка LLM-кодера + результаты
├── paper/              ← черновик статьи (LaTeX + PDF) и фигуры
├── docs/               ← интерпретация результатов
├── scripts/            ← служебные скрипты
└── local_labeler/      ← локальный QLoRA-разметчик reasoning (Qwen3-8B): MLX + CUDA
```

---

## Ноутбуки

> Ноутбуки содержат готовые выходные данные — можно просматривать без запуска.
> Для воспроизведения нужны данные (не включены в репозиторий, исключены `.gitignore`).

### Сбор данных

| Ноутбук | Описание |
|---|---|
| [`01_running_experiment.ipynb`](notebooks/01_running_experiment.ipynb) | Запуск API-прогонов (AsyncOpenAI + OpenRouter): батчи ответов с reasoning trace |
| [`02_additional_run.ipynb`](notebooks/02_additional_run.ipynb) | Добор строк без финального ответа/reasoning |
| [`03_constructing_master_table.ipynb`](notebooks/03_constructing_master_table.ipynb) | Склейка всех прогонов → `master_table.parquet` (~15.7k строк) |

### Разметка reasoning

| Ноутбук | Описание |
|---|---|
| [`04_markup_labeling.ipynb`](notebooks/04_markup_labeling.ipynb) | GPT-кодирование reasoning traces по схеме категорий (treatment awareness, social desirability, epistemic certainty и др.) |

### Анализ

| Ноутбук | Описание |
|---|---|
| [`05_statistical_analysis.ipynb`](notebooks/05_statistical_analysis.ipynb) | Распределения финальных ответов из master table: сетка experiment × model, сравнение температур |
| [`06_main_analysis.ipynb`](notebooks/06_main_analysis.ipynb) | **Главный аналитический ноутбук:** описательная статистика, сравнение incentive vs no_incentive, классификаторы, SHAP/LIME, тематическое моделирование |
| [`07_markup_analysis.ipynb`](notebooks/07_markup_analysis.ipynb) | Анализ LLM-разметки reasoning traces: частотность категорий, logit-регрессии (treatment/incentive → категория → ответ) |
| [`08_hidden_states_analysis.ipynb`](notebooks/08_hidden_states_analysis.ipynb) | Сравнение скрытых состояний Qwen3-8B: L2-норма сдвига по слоям трансформера при разных incentive |

---

## Данные

Сырые данные не включены в репозиторий (размер > 100 MB). Ноутбуки содержат все результаты в выходных ячейках.

**Что исключено:**
- `attitude/master_table.parquet` (~146 MB) — основная аналитическая таблица
- `attitude/runs/` (~428 MB) — сырые ответы API по прогонам
- `hidden_states/outputs_qwen3_8b_instruct/` (~168 MB) — тензоры скрытых состояний
- `input_for_gpt_analysis.csv` (~52 MB) — экспорт для GPT-кодера

---

## Конфиги ([`configs/`](configs/))

| Файл | Содержание |
|---|---|
| `two_experiments_hassner_savary_goldsmith.json` | Виньетки, вопросы, шкалы ответов по тритментам |
| `two_experiments_hassner_savary_goldsmith.md` | Читаемое описание экспериментов |
| `incentives.json` | Тексты prompt-based стимулов (ключ → абзац) |
| `prompts_attitude.json` | Шаблоны one-stage / two-step промптов |

---

## Оценка разметки ([`annotation/`](annotation/))

Оценка согласованности GPT-кодера с ручной разметкой на golden set.

- `goldenset.csv` — золотой стандарт (ручная разметка)
- `evaluate_openai_coders.py` — скрипт сравнения моделей-кодеров
- `results/` — метрики по полям и моделям, предсказания

---

## Локальный разметчик ([`local_labeler/`](local_labeler/))

Дообучение **Qwen3-8B (QLoRA)** для воспроизведения разметки reasoning traces локально — альтернатива GPT-кодеру из [`04_markup_labeling.ipynb`](notebooks/04_markup_labeling.ipynb). Модель выдаёт один JSON-объект из 37 полей таксономии (19 causality `0/1/2`, 3 binary `0/1`, 15 direction `-1/0/1`).

- Два стека: **MLX** (Apple Silicon) и **CUDA** (NVIDIA, `transformers`+`peft`+`bitsandbytes`).
- План экспериментов **E0–E5** (по одному фактору за прогон: lr → rank → данные → глубина адаптера).
- Метрика отбора — mean per-field macro-F1 (база 0.667), а не flat (0.81).
- С чего начать: [`local_labeler/README.md`](local_labeler/README.md) → хэндофы `qwen_mps_taxonomy_lora/experiments/HANDOFF.md` (MLX) и `qwen_cuda_taxonomy_lora/HANDOFF_CUDA.md` (NVIDIA).

---

## Статья ([`paper/`](paper/))

Черновик статьи по результатам исследования (LaTeX + PDF).
