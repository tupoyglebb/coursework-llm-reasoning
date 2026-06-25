# MoralLens — LLM-разметчик reasoning-трейсов

Дообучение Qwen3-8B (QLoRA) для автоматической разметки reasoning-трейсов по таксономии
из социально-научного эксперимента: **37 полей** (19 causality `0/1/2`, 3 binary `0/1`, 15 direction `-1/0/1`).
Модель имитирует разметку «учителя» — выдаёт один JSON-объект на трейс.

## Две ветки стека

| Папка | Железо | Стек | С чего начать |
|---|---|---|---|
| `qwen_mps_taxonomy_lora/` | Apple Silicon (Mac) | MLX / `mlx_lm` | `experiments/HANDOFF.md` |
| `qwen_cuda_taxonomy_lora/` | NVIDIA GPU | `transformers`+`peft`+`bitsandbytes` | `HANDOFF_CUDA.md` |

CUDA-папка — самодостаточный порт MLX-версии (то же ядро: промпты, нормализация, метрики; переписан только движок
обучения/инференса). Маппинг гиперпараметров MLX↔CUDA — в `qwen_cuda_taxonomy_lora/HANDOFF_CUDA.md`.

## План экспериментов
`qwen_mps_taxonomy_lora/experiments/TIER0_PLAN.md` и `TIER1_PLAN.md`.
Серия контролируемых прогонов **по одному фактору за раз**: lr (E1) → rank (E2) → данные/oversampling (E3) → глубина адаптера `num_layers` (E4, E5).

**Метрика отбора:** mean per-field macro-F1 (база prod = **0.667**) при guardrail на «хороших» полях.
`flat_macro_f1` (0.813) для сравнения не используется — он завышен и маскирует то, что 14 из 37 полей
схлопнулись в мажоритарный класс.

## Запуск (кратко)
```bash
bash run_experiment.sh E1          # обучение -> инференс -> метрики -> строка в experiments/results.csv
bash select_best_checkpoint.sh E1  # отбор лучшего чекпойнта по valid + число на test
```

## Что не лежит в репозитории
`.venv/`, базовая модель (`models/`, качается с HF), обученные адаптеры, предсказания и сырые данные
исключены через `.gitignore` (см. соответствующий HANDOFF, как восстановить данные и окружение).
