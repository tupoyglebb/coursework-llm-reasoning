# Подготовка финальной пересборки — 15 октября 2026

Аудит и PDF опубликованы обычным fast-forward push: `45bce3f0f1500dbeb4fe8c4f461ac93243ed1c77`, технический коммит `2ec29d3`. Удалённая ветка перед публикацией указывала на `2b5366101eb8bd33e3e2856c74fb29f89a8d83ca`; конфликтов не было. Научный текст и библиографические записи в этой итерации не менялись. Графики не вставлены, оценки не пересчитывались.

## Три обязательные визуализации: кандидаты

**Готового сопоставимого PDF behavioral effects по всем четырём экспериментам среди 37 файлов нет.** Для первого пункта есть только два отдельных кандидата ниже. Они не представляют один и тот же estimand: первый показывает contrasts, второй — levels. Не объединять их значения или оси и не называть второй график differences vs N0. Если требуется ровно один общий график effects по четырём случаям, этот пункт пока не готов; новых рисунков и оценок не создавали.

Все пути ниже относительны исходной локальной рабочей папке `thinking_expos`. Сами analysis PDF пока отсутствуют в GitHub; их добавление в рукопись ожидает выбора автора.

| Визуализация / путь | Содержание | Источник данных | Место в Results | Baseline, интервалы и ограничения |
|---|---|---|---|---|
| 1а. `analysis/old_experiments_n_behavioral/figures/02_primary_stimulus_contrasts.pdf` | Hassner/Savary, Kimi и GLM отдельно; N0−none, N−−N0, N+−N0, N+−N− | В том же каталоге: `ANALYTICAL_DATASET.csv`, `tables/primary_stimulus.csv`, `CELL_SUMMARY.csv`; два новых сохранённых запуска none/N-series | §6.3, после сравнения N−/N+ по моделям | Равные веса original treatments. Hassner: процентные пункты и Wilson-MOVER 95%; Savary: баллы 1–5 и stratified percentile bootstrap 95%. Поточечные интервалы, не family-adjusted. N0 — neutral realism; none — новый отдельный запуск, не historical no_incentive. N0−none смешан с временем/provider routing. Нет старых A/C. |
| 1б. `analysis/new_experiments_behavioral_20261015/figures/05_balanced_stimulus_levels.pdf` | Bolsen/Corbacho, GLM/Kimi отдельно, уровни N0/A/C/N−/N+ | В том же каталоге: `ANALYTICAL_DATASET.csv`, `tables/balanced_marginals.csv`; финальные responses_final, 6 000 valid из 6 034 raw | §6.4, после обсуждения stimulus shifts | Равные веса treatments; Bolsen — шкала 1–7, Corbacho — P(True). Pointwise block-10 conditional reference intervals; не гарантируют независимость процесса. N0 — neutral realism. Это уровни, а не график эффектов относительно N0; False/refusal здесь не показаны. |
| 2. `analysis/new_experiments_behavioral_20261015/figures/04_corbacho_distributions.pdf` | Все три исхода True / False / refusal по treatment × stimulus, отдельно GLM/Kimi, с valid n | В том же каталоге: `BEHAVIORAL_CELL_SUMMARY.csv`, финальный behavioral dataset | §6.4, после «Corbacho: three response outcomes and placebo sensitivity» | Описательные доли, без интервалов. Показаны control, corruption и inefficiency placebo. N0 — reference instruction, не отсутствие инструкции; refusal не объединять с False. Это лучший прямой кандидат для второго обязательного пункта. |
| 3. `analysis/reasoning_2new_p1_20261015/figures/02_bolsen_stimulus_contrasts.pdf` | Adjusted presence RD A/C/N−/N+ относительно N0; девять категорий | В том же каталоге: `tables/stimulus_regressions.csv`, snapshot `reasoning_labeled_2new_p1.parquet`, linked responses_final; Bolsen n=799 | §6.7, после абзаца о prevalence shifts | LPM adjusted for treatment/model/annotation_version; pointwise 95% cell-cluster t CI, 40 cells. Девять категорий выбраны скриптом по максимальному абсолютному эффекту; BH q — в полной CSV, не simultaneous CI. Только 4/9 Bolsen treatments; первая партия, не полный корпус, human-gold validation нет. Оценки по моделям здесь скорректированы, а не разнесены отдельными сериями. |

Для третьего пункта Corbacho-аналог `analysis/reasoning_2new_p1_20261015/figures/02_corbacho_stimulus_contrasts.pdf` возможен как дополнительный отдельный график, если автор хочет оба новых случая; n=599, 30 cells, те же определения. Не добавлять без согласования.

Проверка актуальности: PDF и source CSV двух behavioral-кандидатов и Corbacho distributions совпали с сохранёнными SHA256-манифестами. Проверены существующие plotting code, figure index и методы; никаких скриптов анализа не запускали. Числа Corbacho Kimi/A (balanced True=0.697959…, refusal=0.282041…) соответствуют §6.4; reasoning Bolsen overcorrection A=0.506647, C=0.513294 соответствует 50.7/51.3 pp в §6.7. Reasoning-партия 799+599=1 398 соответствует текущей рукописи, но источники прямо называют её предварительной неполной партией. Подтверждать её как окончательный полный корпус нельзя; нужна фиксация этого состава научным редактором.

## Точные редакционные формулировки

Номера строк относятся к опубликованному `COMPETITION_MANUSCRIPT_ASSEMBLED.tex` коммита `45bce3f`, до последующей научной редактуры.

**Строка 145, противоречие покрытия reasoning:**

```text
The behavioral evidence covers four cases, whereas the reasoning-annotation and local hidden-state findings reported here remain limited to the original studies. We analyze behavioral and reasoning evidence across all four cases.
```

**§3.2, строка 195, редакционное требование, оставшееся в научном тексте:**

```text
The exact template and verbatim stimulus strings must be preserved from the generation configurations before the final manuscript is frozen.
```

**§3.3, строка 205, предложения о статусе исторических чисел и предстоящей сверке:**

```text
The existing manuscript reports 15{,}680 raw runs, 12{,}434 retained rows after preprocessing, and 11{,}040 observations in the main $T=0.01$ analysis. These are \emph{reported historical totals}, not figures independently recomputed for this editorial draft. Final counts, exclusion rules, rerun handling, and cell-level coverage must be reconciled against raw data and analysis scripts before submission.
```

Эти фразы выписаны буквально из LaTeX; исправления оставлены научному редактору. Позже в §6.1 уже есть historical-data reconciliation, поэтому статус прежних оговорок нужно согласовать.

## Готовность и остановка

- Техническая сборка готова, опубликован PDF на 32 страницах; все 25 источников найдены, undefined/overfull 0.
- Финальная пересборка **ожидает** готовую сокращённую рукопись, исправление охвата reasoning и выбор конкретных графиков. Научный текст не сокращать автоматически.
- Текущий объём: 61 866 (основной), 62 162 (layout). Для желаемого запаса 500 нужно не более 59 500 обоими методами; текущая layout-оценка требует уменьшения минимум на 2 662 **ещё до добавления** графиков и подписей. Изменения содержания/вёрстки могут изменить оба счётчика.
- После согласования: вставить только утверждённые PDF, согласовать компактные содержательные подписи, выполнить полный Biber build, проверить ссылки/нумерацию/TOC/переполнения и обновить подсчёты и аудит.
- При подсчёте включить текст внутри самих графиков, а не только подписи LaTeX: после вставки проверить извлечение labels из каждого PDF. При неоднозначном порядке/пробелах показать альтернативную оценку, как в CHARACTER_COUNT.txt.
