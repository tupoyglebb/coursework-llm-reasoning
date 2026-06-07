Ниже — цельная версия инструкции, куда встроены все рекомендации. Ее можно копировать как готовый промпт/задание для анализа.

---

Ты — специалист по анализу данных в социальных науках, survey experiments, social desirability bias и LLM-as-a-respondent. В `README` находится краткая информация о структуре проекта, дизайне экспериментов, переменных и файлах с данными.

Мне нужно провести анализ LLM-ответов с позиции синтетического респондента. На этом этапе **не анализируй reasoning traces, их разметку, topic modeling или STM**. STM, Structural Topic Model и анализ текстов reasoning traces будут отдельным этапом исследования.

## Общая цель анализа

Проект изучает, способны ли LLM воспроизводить эффекты, обнаруженные в реальных survey/vignette experiments по social desirability bias, и как дополнительные prompt-based стимулы/incentives меняют ответы модели.

Меня интересуют два основных исследовательских вопроса:

1. **Как incentive влияет на финальные ответы модели?**
2. **Как incentive влияет на формирование ответа в скрытых состояниях модели?**

Особенно важно не только проверить отдельный эффект incentive, но и понять, **модерирует ли incentive исходный treatment-effect**. То есть меняет ли incentive чувствительность модели к экспериментальному воздействию из оригинальной статьи.

---

# Часть 1. Statistical analysis финальных ответов

Проведи статистический и модельный анализ в файле `statistical analysis`.

## 1. Data validation and preprocessing

Перед основным анализом проверь структуру данных.

Нужно проверить:

* количество наблюдений по каждому experiment;
* количество наблюдений по treatment condition;
* количество наблюдений по incentive condition;
* количество наблюдений по model;
* количество наблюдений по prompt template;
* количество наблюдений по respondent profile, если такие профили есть;
* наличие пропусков;
* наличие дубликатов;
* наличие невалидных или некорректно распарсенных ответов;
* баланс групп;
* соответствие типов переменных ожидаемой структуре.

Отдельно проверь, что outcome-переменная приведена к единому формату внутри каждого эксперимента:

* бинарный outcome;
* ordinal outcome;
* шкальный outcome;
* агрегированный или бинаризованный outcome, если это необходимо для сопоставления с оригинальной статьей.

Если в данных есть нерелевантные, пустые, нераспарсенные или отказные ответы модели, отдельно опиши, как они были обработаны.

---

## 2. Descriptive statistics and visualizations

Построй описательную статистику и графики, показывающие распределение ответов модели.

Нужны как минимум:

* гистограммы распределения ответов;
* bar plots / count plots по категориям ответа;
* boxplots или violin plots для шкальных ответов;
* доли ответов по treatment condition;
* доли ответов по incentive condition;
* распределения ответов внутри каждой комбинации `treatment × incentive`;
* распределения ответов отдельно по experiment;
* распределения ответов отдельно по model и prompt template, если эти переменные есть.

Для каждого эксперимента покажи:

* среднее значение outcome, если outcome шкальный;
* долю положительного/социально желательного ответа, если outcome бинарный;
* распределение по категориям, если outcome ordinal или categorical;
* confidence intervals для ключевых долей или средних.

---

## 3. Анализ категоричности, разнородности и сбалансированности ответов

Отдельно оцени, насколько ответы модели становятся более или менее категоричными, однородными или сбалансированными в зависимости от incentive.

Для этого рассчитай:

* entropy распределения ответов по условиям;
* variance / standard deviation шкальных ответов;
* долю крайних ответов на шкалах;
* concentration index — долю самой частой категории ответа;
* balance score — насколько равномерно распределены ответы между категориями;
* agreement rate внутри одинаковых экспериментальных условий.

Сравни эти метрики между:

* incentive vs no incentive;
* разными типами incentive;
* treatment vs control внутри фиксированного incentive;
* experiment-level conditions.

Главный вопрос этого блока: делает ли incentive ответы модели более однородными, более категоричными, более осторожными или более сбалансированными?

---

## 4. Statistical tests

Проведи статистические тесты для проверки эффектов.

Меня интересуют следующие сравнения:

### 4.1. Эффект treatment внутри фиксированного incentive

Проверь, влияет ли treatment на ответ модели отдельно внутри каждого incentive condition.

Например:

* treatment effect при no incentive;
* treatment effect при incentive A;
* treatment effect при incentive B.

Используй подходящие тесты в зависимости от типа outcome:

* t-test или Mann–Whitney U test для шкальных переменных;
* chi-square test или Fisher’s exact test для категориальных/бинарных outcome;
* ordinal tests для ordinal outcome, если применимо.

### 4.2. Эффект incentive на ответ

Проверь, влияет ли incentive на ответ модели независимо от treatment.

Сравни:

* no incentive vs incentive;
* разные incentive conditions между собой;
* incentive effects внутри каждого treatment condition.

### 4.3. Interaction treatment × incentive

Главный статистический интерес — interaction effect:

> Меняет ли incentive силу или направление treatment effect?

Проверь, отличается ли treatment effect в условиях с incentive от treatment effect в условиях без incentive.

---

## 5. Effect sizes and confidence intervals

Не ограничивайся p-values. Для всех ключевых сравнений рассчитай effect sizes и confidence intervals.

Используй:

* difference in proportions для бинарных outcome;
* odds ratios для логистических моделей;
* Cohen’s d или standardized mean difference для шкальных outcome;
* Cramér’s V для таблиц сопряженности;
* marginal effects для регрессионных моделей;
* 95% confidence intervals для всех ключевых оценок.

Важно отдельно интерпретировать не только статистическую значимость, но и substantive significance: насколько эффект содержательно велик.

---

## 6. Replication-oriented analysis

Для каждого эксперимента оцени, воспроизводит ли LLM эффект оригинальной статьи.

Проверь:

* совпадает ли знак эффекта с оригинальной статьей;
* совпадает ли направление различий между группами;
* воспроизводится ли ранжирование условий;
* насколько LLM-effect size близок к human-effect size, если в данных есть информация из оригинальной статьи;
* усиливает или ослабляет incentive исходный treatment-effect;
* появляются ли условия, где LLM дает противоположный эффект.

Итог этого блока должен отвечать на вопрос:

> Можно ли считать, что модель воспроизводит экспериментальный эффект на уровне observable outcome?

---

## 7. Regression and model-based analysis

Построй модели, оценивающие эффект treatment, incentive и их взаимодействия.

Базовая спецификация:

```text
outcome ~ treatment * incentive
```

Расширенная спецификация:

```text
outcome ~ treatment * incentive + experiment + model + prompt_template + respondent_profile
```

Если данных достаточно, также проверь:

```text
outcome ~ treatment * incentive * prompt_template
```

или

```text
outcome ~ treatment * incentive * model
```

Главный параметр интереса — interaction `treatment * incentive`.

---

## 8. Выбор модели

Выбери модель в зависимости от типа outcome.

### Если outcome бинарный

Используй:

* logistic regression;
* mixed-effects logistic regression / GLMM, если есть вложенная структура данных;
* marginal effects для интерпретации.

### Если outcome ordinal

Используй:

* ordinal logistic regression;
* при необходимости сравни с линейной моделью как robustness check.

### Если outcome шкальный

Используй:

* OLS;
* linear mixed model;
* robust standard errors, если есть основания ожидать гетероскедастичность.

### Если outcome categorical с несколькими неупорядоченными категориями

Используй:

* multinomial logistic regression;
* chi-square tests как descriptive/statistical baseline.

---

## 9. GLMM: когда использовать

Проанализируй, уместны ли GLMM / mixed-effects models в этой задаче.

GLMM уместна, если наблюдения вложены в:

* experiments;
* prompt templates;
* models;
* respondent profiles;
* seeds / repeated runs;
* vignette versions.

Пример возможной модели:

```text
outcome ~ treatment * incentive + (1 | experiment) + (1 | prompt_template) + (1 | model)
```

Если количество уровней random effect слишком маленькое, не используй random effects механически. В таком случае лучше использовать fixed effects:

```text
outcome ~ treatment * incentive + experiment + prompt_template + model
```

Сравни несколько спецификаций и выбери наиболее устойчивую и интерпретируемую.

---

## 10. Robustness checks

Проведи robustness checks.

Минимально нужно проверить:

* сохраняются ли результаты отдельно по каждому experiment;
* сохраняются ли результаты отдельно по каждой model;
* сохраняются ли результаты отдельно по каждому prompt template;
* сохраняются ли результаты при исключении невалидных/неполных ответов;
* сохраняются ли результаты при разных способах кодирования outcome;
* совпадают ли выводы для исходной шкалы и бинаризованного outcome;
* чувствительны ли результаты к размеру групп или дисбалансу наблюдений.

Если выводы нестабильны, явно укажи, где именно они меняются.

---

## 11. Stability analysis повторных генераций

Оцени устойчивость ответов модели при повторных генерациях внутри одинаковых условий.

Рассчитай:

* agreement rate внутри одного condition;
* variance across runs;
* долю случаев, где модель меняет ответ при одинаковом prompt setup;
* ICC для шкальных outcomes, если применимо;
* Fleiss’ kappa / Krippendorff’s alpha для категориальных ответов, если применимо.

Главный вопрос:

> Насколько поведение модели стабильно внутри одного экспериментального условия, и влияет ли incentive на эту стабильность?

---

## 12. Prompt sensitivity analysis

Если в данных есть разные prompt templates, отдельно проверь, насколько результаты зависят от prompt design.

Проверь:

* treatment effect внутри каждого prompt template;
* incentive effect внутри каждого prompt template;
* interaction `treatment * incentive * prompt_template`, если данных достаточно;
* меняется ли направление эффекта при смене prompt template.

Важно понять, является ли обнаруженный эффект свойством experimental condition или артефактом конкретного prompt format.

---

# Часть 2. Hidden state analysis

В файле `hidden state analysis` проведи анализ скрытых состояний модели.

Меня интересует, как incentive влияет на hidden states и можно ли обнаружить слои, на которых различие между условиями становится особенно сильным.

---

## 1. Layer-wise comparison incentive vs no incentive

Проведи послойное сравнение hidden states между условиями:

* no incentive;
* incentive;
* разные типы incentive, если их несколько.

Для каждого слоя рассчитай расстояния между hidden-state representations.

Используй:

* cosine distance;
* Euclidean distance;
* centered kernel alignment, CKA, если применимо;
* SVCCA, если применимо;
* mean representation difference между условиями;
* within-condition vs between-condition distances.

Главный вопрос:

> На каких слоях hidden states начинают сильнее различаться между incentive и no incentive?

---

## 2. Detection of separation point

Попробуй определить слой или диапазон слоев, где различие между условиями становится максимальным.

Для этого можно использовать:

* layer-wise distance curves;
* separation score;
* отношение between-condition distance к within-condition distance;
* максимум classification accuracy в linear probing;
* изменение расстояний между соседними слоями.

Итогом должен быть график или таблица, показывающая, на каких слоях различие между incentive и no incentive наиболее выражено.

---

## 3. Linear probing

Проведи linear probing по hidden states.

Проверь, можно ли по hidden states предсказать:

* наличие incentive;
* тип incentive;
* treatment condition;
* итоговый ответ модели;
* изменение ответа под воздействием incentive.

Для каждого слоя обучи простой линейный классификатор, например logistic regression или linear SVM, и оцени качество классификации.

Метрики:

* accuracy;
* balanced accuracy;
* ROC-AUC для бинарных задач;
* F1-score;
* cross-validation score.

Главный вопрос:

> На каких слоях информация об incentive становится линейно декодируемой?

---

## 4. Dimensionality reduction and visualization

Построй визуализации hidden states.

Используй:

* PCA;
* UMAP;
* t-SNE, если уместно, но не как основной метод.

Визуализируй hidden states с цветовой разметкой по:

* incentive condition;
* treatment condition;
* experiment;
* final answer;
* model, если моделей несколько.

Построй визуализации отдельно для нескольких ключевых слоев:

* ранний слой;
* средний слой;
* поздний слой;
* слой с максимальной separation score.

---

## 5. Hidden-state transformations across layers

Оцени, насколько сильные преобразования происходят со скрытыми состояниями при прохождении через слои модели.

Для этого сравни:

* расстояние между hidden states соседних слоев;
* cumulative transformation от раннего к позднему слою;
* trajectory length hidden state representation;
* различия trajectory length между incentive и no incentive.

Главный вопрос:

> Сильнее ли преобразуются hidden states в условиях с incentive, чем в условиях без incentive?

---

## 6. Сравнение within-condition и between-condition dynamics

Сравни близость hidden states:

* для одного и того же experiment в no incentive condition;
* для того же experiment в incentive condition;
* между incentive и no incentive;
* внутри одного treatment;
* между treatment и control.

Можно рассчитать:

```text
distance_same_condition
distance_between_incentive_conditions
distance_between_treatment_conditions
```

Затем сравнить, какие расстояния больше.

Особенно важно проверить:

> Является ли эффект incentive на hidden states сильнее, чем обычная вариативность hidden states внутри одного условия?

---

## 7. Связь hidden states с финальными ответами

Свяжи hidden state analysis с observable outcome.

Сравни hidden-state shifts в двух типах случаев:

1. incentive изменил финальный ответ модели;
2. incentive не изменил финальный ответ модели.

Проверь:

* больше ли hidden-state distance в случаях, где ответ изменился;
* появляются ли сильные shifts на тех слоях, где final answer становится предсказуемым;
* коррелирует ли magnitude hidden-state shift с magnitude behavioral effect;
* отличаются ли hidden states для социально желательных и социально нежелательных ответов.

Главный вопрос:

> Можно ли связать репрезентационные изменения в hidden states с изменением финального ответа модели?

---

## 8. Layer-wise link with behavioral treatment effect

Для каждого experiment сравни behavioral treatment effect и hidden-state separation.

Проверь:

* в каких экспериментах incentive сильнее меняет ответы;
* в этих же экспериментах сильнее ли hidden-state separation;
* есть ли корреляция между размером behavioral effect и размером hidden-state shift.

Если возможно, построй таблицу:

```text
experiment | behavioral_effect_size | hidden_state_distance | max_separation_layer
```

---

# Часть 3. Итоговая структура вывода

В конце анализа нужно дать содержательную интерпретацию.

Ответ должен включать:

1. **Есть ли эффект treatment на ответы модели?**
2. **Есть ли эффект incentive на ответы модели?**
3. **Модерирует ли incentive исходный treatment-effect?**
4. **Воспроизводит ли модель направление эффектов из оригинальных human experiments?**
5. **Делает ли incentive ответы более категоричными, однородными или сбалансированными?**
6. **Насколько устойчивы ответы модели при повторных генерациях?**
7. **Насколько результаты зависят от prompt template и model?**
8. **На каких слоях hidden states сильнее всего различаются между incentive и no incentive?**
9. **Можно ли по hidden states линейно предсказать incentive, treatment или final answer?**
10. **Связаны ли hidden-state shifts с изменением финального ответа?**

---

# Scope restriction

На этом этапе не проводить:

* анализ reasoning traces;
* ручную или LLM-разметку reasoning traces;
* STM;
* topic modeling;
* анализ аргументов внутри reasoning;
* анализ скрытой логики reasoning по тексту.

Эти задачи относятся к отдельному этапу исследования reasoning traces.

На текущем этапе анализируются только:

1. финальные ответы модели как synthetic respondent outcomes;
2. статистические эффекты treatment, incentive и их interaction;
3. hidden states как внутренние репрезентации модели.
