# thinking_expos — обзор папки

Проект измеряет, как модели отвечают на виньетки из двух социальных экспериментов, когда их просят имитировать «правдоподобного респондента опроса». Есть **два параллельных трека**:

1. `**attitude/`** — API-прогоны (OpenRouter, thinking-модели): финальный ответ, **reasoning trace**, разметка reasoning.
2. `**hidden_states/`** — локальный прогон **Qwen3-8B**: снятие **скрытых состояний** по слоям трансформера при тех же виньетках и стимулах.

Ниже — где лежат данные, конфиги и ноутбуки, и в каком порядке ими пользоваться.

---

## Логика проекта (коротко)

```text
Конфиги (attitude/*.json) — общие виньетки и тексты стимулов
        │
        ├─► running_think_expos.ipynb  →  attitude/runs/<run_dir_name>/
        │         ↓
        │   constructing_master_table.ipynb  →  master_table.parquet
        │         ↓
        │   markup.ipynb / main_analysis_p1.ipynb / statistical_analysis.ipynb / reasoning_review_*
        │
        └─► [вне репозитория: извлечение hidden states]  →  hidden_states/outputs_qwen3_8b_instruct/
                  ↓
            hidden_states/hs_analysis.ipynb
```

**Два эксперимента** (см. `attitude/two_experiments_hassner_savary_goldsmith.json` и `.md`):


| Эксперимент                                                       | Тема                                    | `group_name` (тритменты)                                                                                                             |
| ----------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `Hassner_2023_Persuasive_and_Unpersuasive_Critiques_of_Torture`   | поддержка sleep deprivation при допросе | `control_base_case`, `treatment_modest_intelligence`, `treatment_protracted_duration`, `treatment_duration_plus_modest_intelligence` |
| `Savary_Goldsmith_2020_Unobserved_Altruism_Willingness_to_Donate` | готовность пожертвовать $5              | `private_donation_control`, `public_recognition_treatment`, `public_recognition_with_social_benefit`                                 |


**Стимулы (incentive)** — дополнительные абзацы в system prompt из `attitude/incentives.json` (ключи `A_…`, `C_…`, `H_…`, …, `no_incentive`). В таблице каноническое поле — столбец `**incentive`**.

**Модели** (в master table): `moonshotai/kimi-k2-thinking`, `z-ai/glm-4.6`, `qwen/qwen3.5-9b`, `minimax/minimax-m2.5`.

---

## Структура каталогов

```text
thinking_expos/
├── README.md                          ← этот файл
│
├── attitude/                          ← ядро: конфиги, прогоны, master table
│   ├── master_table.parquet           ← все прогоны в одной таблице (~15.7k строк)
│   ├── constructing_master_table.ipynb
│   ├── incentives.json
│   ├── prompts_attitude.json
│   ├── two_experiments_hassner_savary_goldsmith.json
│   ├── two_experiments_hassner_savary_goldsmith.md
│   ├── twin2k_sample_100_with_demographics.parquet   ← профили (опционально в промпте)
│   └── runs/                          ← сырые результаты по прогонам
│       └── <run_dir_name>/
│           ├── run_metadata.json
│           ├── responses_final*.parquet / .csv
│           ├── responses_final*_markup.parquet      ← если есть разметка
│           ├── progress.parquet
│           ├── token_usage_summary.parquet
│           └── parts/                               ← чекпоинты батча
│
├── running_think_expos.ipynb          ← запуск новых API-прогонов
├── additional_run_thinkig_expos.ipynb ← добор пропущенных ответов
├── additional_rerun_20260423*/        ← артефакты дозапусков (csv/parquet)
├── additional_rerun_20260425T201559Z/
│
├── markup.ipynb                       ← подготовка CSV + разметка reasoning (GPT)
├── main_analysis_p1.ipynb             ← основной анализ (разметка + ответы + ML)
├── statistical_analysis.ipynb         ← распределения финальных ответов
├── main_markup_run.ipynb              ← черновик/фрагмент (загрузка разметки)
│
├── reasoning_review_package/          ← просмотр reasoning по иерархии
│   ├── reasoning_review_tools.ipynb
│   ├── generate_reasoning_outline_cells.py
│   └── Liza_*_reasoning.xlsx          ← ручная разметка (6 файлов)
│
├── hidden_states/                     ← скрытые состояния Qwen3-8B (локально)
│   ├── hs_analysis.ipynb
│   └── outputs_qwen3_8b_instruct/
│       ├── features.npz               ← матрицы (37 слоёв × 4096), ~175 MB
│       ├── features_index.csv         ← связь feature_key ↔ run_id ↔ тип
│       └── runs_metadata.jsonl        ← условия и промпты по каждому run_id
│
├── input_for_gpt_analysis.csv       ← экспорт из markup.ipynb для кодера
├── missing_rows_for_additional_run.*  ← строки без валидного ответа
├── responses_final_*_markup.parquet   ← копии разметки в корне (дубликаты run-путей)
├── prompt_v2.txt                      ← черновик/вариант промпта
└── _patch_notebook_requests.py        ← служебный патч ноутбука
```

---

## Главные данные

### `attitude/master_table.parquet`

**Единая точка входа для анализа.** Собирается ноутбуком `attitude/constructing_master_table.ipynb` из всех `attitude/runs/*/responses_final*`.


| Что важно            | Столбцы                                                                |
| -------------------- | ---------------------------------------------------------------------- |
| Условие эксперимента | `experiment_name`, `group_name`                                        |
| Стимул в промпте     | `incentive`, `incentive_key`                                           |
| Модель               | `model`                                                                |
| Прогон               | `run_dir_name`, `run_dir_path`, `run_meta__*`                          |
| Ответ                | `final_answer_text`, `raw_response_text`, `parse_status`               |
| Reasoning            | `reasoning_trace_combined`, `stage1_reasoning_*`, `stage2_reasoning_*` |
| Идентификаторы       | `job_id`, `seed`, `respondent_id`                                      |


**Уникальный ключ строки:** `job_id` **один не годится** (повторяется в каждом из 19 прогонов). Используйте:

- `**(run_dir_name, job_id)`** — минимальный ключ по всей таблице;
- или `**(run_dir_name, model, job_id)**` — то же по смыслу (модель однозначна при фиксированных `run_dir_name` + `job_id`).

Внутри **одного** `run_dir_name`: 1400 строк = 4 модели × 350 заданий (50 респондентов × 7 комбинаций experiment×group).

### `attitude/runs/<run_dir_name>/`

Один каталог = один batch-прогон (часто несколько моделей в одном parquet). Имя папки совпадает с `run_dir_name` в master table.

Примеры значений `run_dir_name` (всего **19**):


| `run_dir_name`                                         | `incentive`  | Примечание                      |
| ------------------------------------------------------ | ------------ | ------------------------------- |
| `incentive_A_original_core_realism_50_reasoning_trace` | A            | 4 модели, 1400 строк            |
| `incentive_C_balanced_realism_50_reasoning_trace`      | C            | основной + `_p2`                |
| `incentive_H_…_glm_kimi_t001`                          | H            | kimi + glm; есть `__kimi_rerun` |
| `incentive_I_…`, `J_…`, `K_…`                          | I, J, K      | аналогично                      |
| `incentive_L_…`, `incentive_M_…`                       | L, M         | L: отдельный `__glm_only_patch` |
| `no_incentive_50_reasoning_trace`                      | no_incentive | + `_p2`, `_t1` (T=1.0)          |
| `no_incentive_zero_temp_2_expos*`                      | no_incentive | урезанные прогоны (840 строк)   |


В каждой папке:

- `**run_metadata.json`** — температура, пути к конфигам, `responses_final_path`, статистика токенов;
- `**responses_final*.parquet**` — сырые ответы API;
- `***_markup.parquet**` — если прогон проходил через пайплайн разметки из `markup.ipynb` / `main_analysis_p1.ipynb`.

### Разметка reasoning

Не во всех прогонах есть markup-файлы. Основные пути с разметкой (используются в `main_analysis_p1.ipynb`):

- `attitude/runs/incentive_C_balanced_realism_50_reasoning_trace/responses_final_*_markup.parquet`
- `attitude/runs/no_incentive_50_reasoning_trace_p2/...`
- `attitude/runs/no_incentive_50_reasoning_trace_t1/...` (также `.csv`)
- `attitude/runs/incentive_X/responses_final_X_markup.parquet`

Поля разметки — ordinal/binary категории (treatment awareness, social desirability, epistemic certainty и т.д.), задаются в `markup.ipynb`.

### Вспомогательные CSV в корне


| Файл                                                      | Назначение                                                         |
| --------------------------------------------------------- | ------------------------------------------------------------------ |
| `input_for_gpt_analysis.csv`                              | Плоский экспорт промптов + reasoning для внешнего кодера           |
| `missing_rows_for_additional_run.csv` / `.parquet`        | Строки без валидного ответа → `additional_run_thinkig_expos.ipynb` |
| `missing_rows_stats_by_exp_treatment_incentive_model.csv` | Сводка пропусков по условиям                                       |


---

## `hidden_states/` — скрытые состояния (Qwen3-8B)

Отдельный экспериментальный трек: не OpenRouter и **не** попадает в `master_table.parquet`. Те же **два эксперимента** и `**group_name`**, что в `attitude/two_experiments_hassner_savary_goldsmith.json`, но другая постановка прогона.

### Что измеряется

При forward-pass локальной модели `**Qwen/Qwen3-8B**` сохраняются активации скрытых слоёв в момент ответа на промпт. В анализе по умолчанию используется репрезентация `**last_token**`: для каждого прогона матрица `**(37, 4096)**` — 37 позиций по глубине трансформера, размерность hidden size 4096.

Дополнительно в дампе есть `**options_mean**` (усреднение по токенам вариантов ответа) — 196 записей из 490 в индексе; основной анализ в `hs_analysis.ipynb` опирается на `**last_token**`.

### Файлы в `outputs_qwen3_8b_instruct/`


| Файл                      | Назначение                                                                                                                                                                               |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**features.npz**`        | Словарь `feature_XXXXXXXX` → ndarray `(37, 4096)`. Ключи перечислены в `features_index.csv`.                                                                                             |
| `**features_index.csv**`  | Столбцы: `feature_key`, `run_id`, `representation_type`, `shape`.                                                                                                                        |
| `**runs_metadata.jsonl**` | Одна JSON-строка на прогон: `run_id`, `experiment_name`, `group_name`, `incentive_condition`, `prompt_mode`, `stage`, `context_condition`, тексты промпта, logprobs по кандидатам и т.д. |


**Масштаб дампа:** 294 прогона (`run_000000` …), 490 feature-записей в npz.

### Дизайн прогонов (факторы)


| Поле в `runs_metadata.jsonl` | Значения                                                                                                                                                             | Смысл                                                                                                            |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `incentive_condition`        | `none`, `A_original_core_realism`, `B_softened_realism`, `C_balanced_realism`, `D_research_purpose_realism`, `F_stronger_stakes_realism`, `X_measurement_neutrality` | Подмножество ключей из `attitude/incentives.json` (6 стимулов + база без стимула). По **42** прогона на условие. |
| `group_name`                 | все 7 тритментов (Hassner ×4 + Savary ×3)                                                                                                                            | То же, что в attitude                                                                                            |
| `context_condition`          | `with_vignette` / `no_vignette`                                                                                                                                      | С виньеткой или только вопрос                                                                                    |
| `prompt_mode`                | `one_stage` / `two_step`                                                                                                                                             | Один этап (сразу ответ) или два (суждение → выбор опции)                                                         |
| `stage`                      | `answer`, `stage1_judgment`, `stage2_answer`                                                                                                                         | Стадия в two-step                                                                                                |


Сетка: **7 групп × 2 контекста × 3 «режима»** (one_stage + 2 стадии two_step) = **42** прогона на каждый `incentive_condition`.

### Уникальный ключ

- `**run_id`** — уникален внутри `runs_metadata.jsonl` и связан с `features_index.csv`.
- Для загрузки эмбеддинга: `feature_key` из индекса или пара `**(run_id, representation_type)**`.

### Ноутбук анализа


| Ноутбук                               | Роль                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**hidden_states/hs_analysis.ipynb**` | Загружает npz + jsonl + csv. Группирует по `incentive_condition`. Для каждого стимула (кроме `none`) строит **пары** с базой `none` при одинаковом контексте (эксперимент, тритмент, стадия, режим промпта) и считает **среднюю L2-норму разности** hidden states по слоям. Итог — линейные графики «слой → величина сдвига» (по одному на стимул + сводный). |


Запуск: из папки `hidden_states/` или из корня `thinking_expos/` (ноутбук сам ищет `outputs_qwen3_8b_instruct/`).

### Связь с `attitude/`


|             | `attitude/`                                             | `hidden_states/`                                                                       |
| ----------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Модели      | kimi, glm, qwen3.5-9b, minimax (API)                    | Qwen3-8B (локально)                                                                    |
| Выход       | текст, reasoning trace                                  | тензоры слоёв                                                                          |
| Стимулы     | полный набор в прогонах (A, C, H, I, …, `no_incentive`) | фиксированные A, B, C, D, F, X + `none`                                                |
| Респонденты | 50 профилей / job grid                                  | нет отдельных профилей — факториальный дизайн по условиям                              |
| Склейка     | `master_table.parquet`                                  | только вручную по `experiment_name`, `group_name`, `incentive` ↔ `incentive_condition` |


Скрипт **извлечения** hidden states в репозитории не лежит — в папке только **готовый дамп** и **анализ**. При повторном сборе нужно воспроизвести тот же формат трёх файлов в `outputs_qwen3_8b_instruct/`.

---

## Конфиги (`attitude/`)


| Файл                                            | Содержание                                                               |
| ----------------------------------------------- | ------------------------------------------------------------------------ |
| `two_experiments_hassner_savary_goldsmith.json` | Виньетки, вопросы, шкалы ответов по `group_name`                         |
| `incentives.json`                               | Тексты prompt-based стимулов (ключ → абзац в system prompt)              |
| `prompts_attitude.json`                         | Шаблоны `one_stage_attitude_expression` / `two_step_attitude_expression` |
| `twin2k_sample_100_with_demographics.parquet`   | Профили респондентов (если `include_profile_in_prompt`)                  |


---

## Ноутбуки: что для чего

### Сбор данных


| Ноутбук                                        | Роль                                                                                                                                               |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**running_think_expos.ipynb`**                | Основной раннер: AsyncOpenAI + OpenRouter, батчи в `attitude/runs/`, reasoning trace, чекпоинты. Задаются модели, `RUN_ID`, инсентив, температура. |
| `**additional_run_thinkig_expos.ipynb**`       | Добор строк без финального ответа/reasoning; пишет в `additional_rerun_*` и отчёты merge.                                                          |
| `**attitude/constructing_master_table.ipynb**` | Склеивает все `runs/*/responses_final*` + метаданные → `**master_table.parquet**`. После новых прогонов запускать заново.                          |


### Разметка


| Ноутбук                     | Роль                                                                                                                                                   |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `**markup.ipynb**`          | Собирает контекст (vignette, treatment summary, incentive), готовит `input_for_gpt_analysis.csv`, вызывает GPT-кодер reasoning (JSON-схема категорий). |
| `**main_markup_run.ipynb**` | Незавершённый фрагмент; для работы ориентируйтесь на `markup.ipynb` и ядро в `main_analysis_p1.ipynb`.                                                 |


### Анализ


| Ноутбук                                                            | Роль                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**main_analysis_p1.ipynb**`                                       | **Главный аналитический ноутбук:** загрузка нескольких `*_markup.parquet`, объединение в `df`, описательная статистика, сравнение incentive vs no_incentive, парные сравнения, классификаторы, SHAP/LIME, тематическое моделирование, QA покрытия. В начале — оглавление по ячейкам. Экспортирует `fig_*.png`, CSV. |
| `**statistical_analysis.ipynb`**                                   | Распределения **финальных ответов** из `master_table.parquet` (сетка эксперимент × модель), сравнение T=0.01 vs T=1.0 для no_incentive.                                                                                                                                                                             |
| `**markup_analysis.ipynb**`                                        | **Анализ LLM-разметки reasoning traces** (`markup-1/predictions_part_1_checkpoint_3300.csv`). Источник: n=2880 строк (incentive A+C+no_incentive, 4 модели). Разделы: §0 препроцесс (бинаризация causality, signed_presence), §1 baseline behavioral model (treatment+incentive→answer), §2 частотность категорий, §3 категории по treatment×incentive, §4 treatment/incentive→category (logit, FDR), §5 categories→final answer (logit Hassner / OLS Savary). |
| `**reasoning_review_package/reasoning_review_tools.ipynb`**        | Интерактивный просмотр `reasoning_trace_combined` с фильтрами (`model`, `experiment_name`, `group_name`, `incentive_type`, temperature). Иерархия в Outline генерируется скриптом ниже.                                                                                                                             |
| `**reasoning_review_package/generate_reasoning_outline_cells.py**` | Пересборка markdown-оглавления в `reasoning_review_tools.ipynb` по актуальному `master_table.parquet`.                                                                                                                                                                                                              |
| `**hidden_states/hs_analysis.ipynb**`                              | Сравнение hidden states: пары `none` vs каждый стимул, L2 по слоям, графики. См. раздел [hidden_states](#hidden_states--скрытые-состояния-qwen3-8b).                                                                                                                                                                |


### Служебное


| Файл                          | Роль                                                                |
| ----------------------------- | ------------------------------------------------------------------- |
| `_patch_notebook_requests.py` | Патч зависимостей/запросов в ноутбуке раннера                       |
| `prompt_v2.txt`               | Текстовый вариант инструкций (не подключён автоматически к раннеру) |


---

## Рекомендуемый порядок работы

**Трек attitude (API + reasoning):**

1. Анализ → `main_analysis_p1.ipynb`, распределения ответов → `statistical_analysis.ipynb`.
2. Просмотр traces → `reasoning_review_tools.ipynb`.

**Трек hidden states:**

1. Убедиться, что есть `hidden_states/outputs_qwen3_8b_instruct/features.npz` (+ index + jsonl).
2. Анализ сдвига по стимулам → `hidden_states/hs_analysis.ipynb`.
3. При сопоставлении с текстовыми ответами — джойн по `experiment_name`, `group_name` и ключу стимула (`incentive` ↔ `incentive_condition`).

---

## Связь с курсовой разметкой (ручная)

Для привязки текста reasoning к строке master table используйте:

```text
(run_dir_name, job_id)   или   (model, experiment_name, group_name, incentive, respondent_id, seed)
```

Примеры из PDF-разметки (6 reasoning) лежат в разных комбинациях `group_name` + `incentive`; см. отдельную сводку в чате или фильтруйте по уникальным фразам в `reasoning_trace_combined`.

---

## Замечания

- Пути в `run_metadata.json` могут указывать на машину, где гоняли прогон; для анализа опирайтесь на **относительные** пути внутри `attitude/runs/`.
- Дубликаты прогонов (`_p2`, `__kimi_rerun`) — осознанные повторы; при анализе их нужно **мержить или фильтровать** явно (логика есть в `statistical_analysis.ipynb` и `main_analysis_p1.ipynb`).
- В `markup.ipynb` не храните API-ключи в репозитории — используйте переменные окружения.

