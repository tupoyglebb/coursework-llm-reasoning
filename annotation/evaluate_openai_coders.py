# pip install openai pandas numpy scikit-learn tqdm

import os
import re
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# =========================
# CONFIG
# =========================

INPUT_CSV = "goldenset.csv"

# Замени / расширь список на модели, которые хочешь сравнить.
# Если какая-то модель недоступна на твоем аккаунте, скрипт запишет ошибку и пойдет дальше.
MODELS = [
    "openai/gpt-4.5",   # замени на нужное имя модели с openrouter.ai/models
]

RESULTS_DIR = Path("openai_annotation_eval")
RESULTS_DIR.mkdir(exist_ok=True)

CACHE_PATH = RESULTS_DIR / "raw_predictions.jsonl"
PREDICTIONS_CSV = RESULTS_DIR / "predictions_all_models.csv"
METRICS_BY_MODEL_CSV = RESULTS_DIR / "metrics_by_model.csv"
METRICS_BY_FIELD_CSV = RESULTS_DIR / "metrics_by_field.csv"

# Для теста можно поставить, например, 5.
# Для полного прогона оставь None.
SAMPLE_N = None

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


# =========================
# LABEL COLUMNS
# =========================

CAUSALITY_COLS = [
    "overcorrection_concern_causality",
    "treatment_awareness_causality",
    "incentive_awareness_causality",
    "self_enhancement_causality",
    "fear_negative_evaluation_causality",
    "self_deception_causality",
    "majority_reliance_causality",
    "source_reliance_causality",
    "common_interests_prioritization_causality",
    "private_interests_prioritization_causality",
    "personal_cost_burden_causality",
    "ratio_of_goals_and_means_causality",
    "emotional_reasoning_causality",
    "public_recognition_causality",
    "privacy_factor_causality",
    "social_reward_causality",
    "emergency_urgency_causality",
    "legality_assessment_causality",
    "cruelty_causality",
]

BINARY_COLS = [
    "possibility_of_different_opinions",
]

DIRECTION_COLS = [
    "treatment_effect_direction",
    "incentive_effect_direction",
    "majority_reliance_direction",
    "source_reliance_direction",
    "common_interests_prioritization_direction",
    "private_interests_prioritization_direction",
    "personal_cost_burden_direction",
    "ratio_of_goals_and_means_direction",
    "emotional_reasoning_direction",
    "public_recognition_direction",
    "privacy_factor_direction",
    "social_reward_direction",
    "emergency_urgency_direction",
    "legality_assessment_direction",
    "cruelty_direction",
]

STYLE_COL = "reasoning_style"

OUTPUT_FIELDS = (
    CAUSALITY_COLS
    + BINARY_COLS
    + DIRECTION_COLS
    + [STYLE_COL]
)

DIRECTION_BY_CAUSALITY = {
    "treatment_effect_direction": "treatment_awareness_causality",
    "incentive_effect_direction": "incentive_awareness_causality",
    "majority_reliance_direction": "majority_reliance_causality",
    "source_reliance_direction": "source_reliance_causality",
    "common_interests_prioritization_direction": "common_interests_prioritization_causality",
    "private_interests_prioritization_direction": "private_interests_prioritization_causality",
    "personal_cost_burden_direction": "personal_cost_burden_causality",
    "ratio_of_goals_and_means_direction": "ratio_of_goals_and_means_causality",
    "emotional_reasoning_direction": "emotional_reasoning_causality",
    "public_recognition_direction": "public_recognition_causality",
    "privacy_factor_direction": "privacy_factor_causality",
    "social_reward_direction": "social_reward_causality",
    "emergency_urgency_direction": "emergency_urgency_causality",
    "legality_assessment_direction": "legality_assessment_causality",
    "cruelty_direction": "cruelty_causality",
}


# =========================
# JSON SCHEMA FOR MODEL OUTPUT
# =========================

def build_annotation_schema():
    properties = {}

    for col in CAUSALITY_COLS:
        properties[col] = {
            "type": "integer",
            "enum": [0, 1],
            "description": "0 absent, 1 present (mentioned or used as argument)",
        }

    for col in BINARY_COLS:
        properties[col] = {
            "type": "integer",
            "enum": [0, 1],
            "description": "0 absent, 1 present",
        }

    for col in DIRECTION_COLS:
        properties[col] = {
            "type": "integer",
            "enum": [-1, 0, 1],
            "description": "-1 lowers support, 1 increases support, 0 means NA/unclear/not applicable",
        }

    properties[STYLE_COL] = {
        "type": "string",
        "enum": ["utilitarian", "moral", "mixed", "unclear"],
    }

    return {
        "type": "object",
        "properties": properties,
        "required": OUTPUT_FIELDS,
        "additionalProperties": False,
    }


ANNOTATION_SCHEMA = build_annotation_schema()


# =========================
# PROMPT
# =========================

CODING_PROMPT = """
You are an expert annotator coding LLM reasoning traces for a social-science experiment.

You must code the reasoning trace itself, not your own opinion and not the final answer alone.

Return only the JSON object required by the schema.

GENERAL RULES

Causality variables:
0 = category absent — the concept is not mentioned or used in the reasoning.
1 = category present — the concept is explicitly mentioned or used as an argument, regardless of how central it is.

The category must be explicitly present in the text, not merely implied or inferable.

Direction variables:
-1 = the local argument pushes toward lower support for the described action, a more negative evaluation, or a less supportive answer.
1 = the local argument pushes toward higher support for the described action, a more positive evaluation, or a more supportive answer.
0 = NA: unclear, balanced, purely descriptive, or the corresponding causality variable is 0.

Direction codes the local influence of the argument, not necessarily the final answer.

IMPORTANT TREATMENT RULE

You receive group_name as the treatment name.

If group_name contains "control", treat it as a control condition:
- treatment_awareness_causality must be 0
- treatment_effect_direction must be 0

Do not fill treatment fields for control conditions, even if the base vignette mentions terrorism, torture, charity, donation, sleep deprivation, a suspect, helping families, or similar base-scenario content.

Only code treatment_awareness_causality when the reasoning uses non-control treatment-specific content:
- treatment_modest_intelligence: sleep deprivation is unlikely to provide new information, but may confirm existing CIA suspicions.
- treatment_protracted_duration: interrogation or sleep deprivation may last about a month.
- treatment_duration_plus_modest_intelligence: interrogation may last about a month and has limited informational value.
- public_recognition_treatment: the respondent's donor name will be publicly recognized.
- public_recognition_with_social_benefit: public donor recognition likely brings social benefits, approval, appreciation, or positive visibility.

IMPORTANT INCENTIVE RULE

You receive incentive as the prompt-based intervention name.

If incentive is "no_incentive", empty, missing, or null:
- incentive_awareness_causality must be 0
- incentive_effect_direction must be 0

Code incentive_awareness_causality only if the reasoning refers to consequences of realistic/unrealistic answers for measurement, research, money, science, public understanding, product quality, company value, or product usefulness.

Do not code incentive awareness for generic statements like:
- answer realistically
- avoid moralizing
- avoid socially desirable answers
- answer as a plausible respondent
- avoid the safest answer
- this is a measurement task
unless the reasoning also states a consequence for measurement, research, science, product, company, money, or usefulness.

VARIABLE DEFINITIONS

overcorrection_concern_causality:
Concern about giving an overly sterile, sanitized, safe, socially expected, or model-perfect answer.
Do not code merely because the model says it should answer as a plausible human.
0 = absent.
1 = present: concern about giving a sanitized or overly cautious answer is explicitly mentioned or used.

treatment_awareness_causality:
Use of treatment-specific vignette content, subject to the control rule above.
0 = absent (or control condition).
1 = present: treatment-specific content is explicitly mentioned or used in the reasoning.

treatment_effect_direction:
Direction of the treatment-specific argument.
-1 = treatment content argues against the action.
1 = treatment content argues in favour of the action.
0 = NA (control condition or treatment not used).

incentive_awareness_causality:
Use of incentive-specific consequences, subject to the no_incentive rule above.
0 = absent (or incentive is no_incentive).
1 = present: incentive-specific consequences are explicitly mentioned or used in the reasoning.

incentive_effect_direction:
Direction of the incentive-specific argument.
-1 = incentive awareness lowers willingness to give a realistic/extreme answer.
1 = incentive awareness increases willingness to give a realistic/extreme answer.
0 = NA.

self_enhancement_causality:
People want to present themselves as better, kinder, more generous, more moral, or more principled.
0 = absent.
1 = present: self-enhancement motivation is explicitly mentioned or used.

fear_negative_evaluation_causality:
People avoid answers that make them look selfish, cruel, aggressive, indifferent, extreme, or socially unacceptable.
0 = absent.
1 = present: fear of negative evaluation is explicitly mentioned or used.

self_deception_causality:
The model attributes the respondent's answer to respondent-level self-deception or internalized moral self-image: the respondent may sincerely choose a socially desirable or morally favorable answer not mainly to impress others, but because it fits their self-image as a good, honest, fair, compassionate, responsible, principled, or non-prejudiced person.

Code this when the reasoning says or clearly implies that:
- the respondent sincerely believes the socially desirable / morally favorable answer reflects who they are;
- the respondent's answer follows from a positive self-concept, such as "I am a good person", "I am fair", "I am compassionate", "I am not cruel", "I am not selfish", "I am principled";
- the respondent may internalize a socially approved norm as their own genuine belief;
- the respondent is not consciously trying to impress others, but is guided by a favorable moral self-image.

Do NOT code self_deception_causality when:
- the reasoning is about the model itself claiming to answer realistically or as a plausible human;
- the reasoning says people consciously want to look better to others — that is self_enhancement_causality;
- the reasoning says people avoid looking bad, cruel, selfish, aggressive, or socially unacceptable — that is fear_negative_evaluation_causality;
- the reasoning only gives a moral evaluation of the action, without saying that the respondent's answer reflects their self-image or sincere internalized belief;
- the reasoning only says "people sincerely believe this is right/wrong" without linking it to positive self-image, self-concept, internalized moral identity, or self-perception;
- the reasoning is merely about avoiding a sterile / model-perfect answer — that belongs to overcorrection_concern_causality if applicable.

Important distinction:
- self_enhancement_causality = the respondent wants to appear better to others.
- fear_negative_evaluation_causality = the respondent wants to avoid appearing bad to others.
- self_deception_causality = the respondent sincerely sees the socially desirable / morally favorable answer as consistent with their own positive self-image.

Code self_deception_causality:
- "A respondent may genuinely see themselves as a compassionate person, so saying they would donate feels like the truthful answer."
- "People may sincerely reject torture because they view themselves as humane and principled."
- "The socially approved answer may be internalized as part of the respondent's self-concept."
- "They may not be trying to impress anyone; they may simply believe that a good person would answer this way."

Do not code self_deception_causality:
- "I should answer realistically, not as a model citizen, so I will choose the cautious answer." → this is about the LLM, not the respondent.
- "People may say they would donate to look generous." → self_enhancement_causality.
- "Respondents may avoid supporting torture because they don't want to seem cruel." → fear_negative_evaluation_causality.
- "Torture is morally wrong." → moral reasoning / cruelty / legality depending on context, but not self_deception unless linked to respondent self-image.

0 = absent: respondent-level self-deception / internalized moral self-image is not mentioned.
1 = present: respondent-level self-deception / internalized moral self-image is explicitly mentioned or used as an argument.

possibility_of_different_opinions:
The reasoning mentions that different opinions, positions, moral intuitions, or social evaluations may exist.
0 = absent.
1 = different opinions are explicitly acknowledged (e.g., "some people would say X, others Y").

majority_reliance_causality:
The answer is based on presumed majority opinion or an averaged human-response strategy.
0 = absent.
1 = present: majority opinion or typical human response is explicitly referenced as a basis for the answer.

majority_reliance_direction:
Whether presumed majority opinion increases or decreases support for the action.
-1 = the assumed majority view argues against the action.
1 = the assumed majority view argues in favour of the action.
0 = NA.

source_reliance_causality:
The model refers to research, articles, empirical evidence, expert sources, international standards, legal frameworks, or common factual knowledge as a basis for the answer.
0 = absent.
1 = present: an external source, fact, or standard is explicitly cited or referenced.

source_reliance_direction:
Whether sources support a positive or negative evaluation of the action.
-1 = cited sources argue against the action.
1 = cited sources argue in favour of the action.
0 = NA.

common_interests_prioritization_causality:
Priority of public, collective, group, or other people's interests: public safety, helping others, preventing harm, community benefit, social good.
0 = absent.
1 = present: collective or other-regarding interests are explicitly mentioned or used as an argument.

NOTE: In vignettes involving terrorism or charity, public safety / helping others is nearly always mentioned. Code 1 whenever it appears explicitly, even as background context.

common_interests_prioritization_direction:
Whether collective/other-regarding interests increase or decrease support.
-1 = collective interests argue against the action.
1 = collective interests argue in favour of the action.
0 = NA.

private_interests_prioritization_causality:
Priority of the survey respondent's own private interests: personal benefit, safety of self or family, resources, comfort, reputation, property, time, money.
Do not code interests of the suspect, defendant, charity recipient, or another vignette character.
0 = absent.
1 = present: respondent's own private interests are explicitly mentioned or used.

private_interests_prioritization_direction:
Whether respondent private interests increase or decrease support.
-1 = private interests argue against the action.
1 = private interests argue in favour of the action.
0 = NA.

personal_cost_burden_causality:
Whether the action is costly, burdensome, accessible, easy, risky, expensive, or minor for the respondent's own resources.
0 = absent.
1 = present: the cost, ease, or burden of the action for the respondent is explicitly mentioned.

personal_cost_burden_direction:
-1 = costly, burdensome, or risky — lowers support.
1 = low-cost, easy, or acceptable — increases support.
0 = NA.

ratio_of_goals_and_means_causality:
Whether the action is effective, meaningful, impactful, consequential, or capable of affecting the situation given the scale of the action.
0 = absent.
1 = present: the effectiveness, meaningfulness, or proportionality of the action is explicitly mentioned.

ratio_of_goals_and_means_direction:
-1 = action is insignificant, ineffective, symbolic, or useless — lowers support.
1 = action is meaningful, effective, or consequential — increases support.
0 = NA.

reasoning_style:
Classify the DOMINANT mode of argumentation using this decision procedure:

Step 1: Identify the core arguments that directly support the conclusion (not just listed considerations).
Step 2: Are those arguments mainly about consequences, outcomes, effectiveness, cost-benefit, risks, practical results, or harms? → "utilitarian"
Step 3: Are those arguments mainly about rights, duties, dignity, prohibitions, moral permissibility, justice, or good/evil as principles? → "moral"
Step 4: Are BOTH utilitarian AND moral arguments clearly developed and roughly equally present in the core reasoning? → "mixed"
Step 5: No substantive argumentation — only vignette summary, a list of considerations without weighing, or a very brief/circular text? → "unclear"

KEY RULES:
- A brief mention of moral concerns inside a mainly consequentialist argument does NOT make it "mixed". Use "mixed" only when both modes are substantively developed.
- "unclear" is for texts that enumerate without arguing, not for texts that are hard to classify.
- When unsure between "mixed" and a single style, look at the final justification sentence: which mode does it use?
- "unclear" is NOT a fallback for ambiguous cases — if there is developed argumentation, classify it as utilitarian, moral, or mixed.

emotional_reasoning_causality:
The reasoning explicitly invokes emotional reactions — feelings, gut responses, empathy, fear, outrage, sympathy, or emotional aversion — as a factor supporting the answer.

IMPORTANT: In LLM reasoning traces, emotional reasoning rarely appears as a direct "I feel sad." Instead look for:
- References to gut reaction or intuition as a decision factor: "gut reaction says no", "instinctive aversion", "this feels deeply wrong"
- The scenario's emotional pull used to justify an answer: "the ticking-bomb scenario is emotionally powerful and pushes toward Yes", "many people would be appalled", "the cruelty creates revulsion"
- Empathy-based reasoning: "thinking about the victims creates a strong pull toward action"

Do NOT code if emotions are only described as a general observation about human psychology without being used to justify this specific answer ("terror scenarios trigger emotional responses" alone, without connecting it to the conclusion, is not sufficient).
0 = absent.
1 = present: emotional reasoning is explicitly invoked as a factor supporting the answer.

emotional_reasoning_direction:
Whether emotional reasoning increases or decreases support.
-1 = the emotional reaction (e.g., revulsion, horror, empathy with the victim) lowers support.
1 = the emotional reaction (e.g., fear of attack, anger at suspect, empathy with potential victims) increases support.
0 = NA.

public_recognition_causality:
Public recognition, disclosure of donor name, reputation, or public visibility of charity.
0 = absent.
1 = present: public recognition or visibility of the donation is explicitly mentioned.

public_recognition_direction:
Whether public recognition increases or decreases willingness to donate.
-1 = public recognition lowers willingness (e.g., feels performative, privacy concern).
1 = public recognition increases willingness (e.g., social reward, motivation to be seen as generous).
0 = NA.

privacy_factor_causality:
Privacy, anonymity, modesty, or non-public charity as a factor.
0 = absent.
1 = present: privacy or anonymity of the action is explicitly mentioned.

privacy_factor_direction:
Whether privacy increases or decreases willingness to donate.
-1 = lack of privacy lowers willingness.
1 = privacy or anonymity increases willingness.
0 = NA.

social_reward_causality:
Social rewards, praise, approval, appreciation, status, reputational bonus, or positive social reinforcement in charity.
0 = absent.
1 = present: social reward or reputational benefit from the action is explicitly mentioned.

social_reward_direction:
Whether social rewards increase donation or decrease it because it feels performative/insincere.
-1 = social reward framing reduces willingness (feels insincere or transactional).
1 = social reward increases willingness.
0 = NA.

emergency_urgency_causality:
Urgency, emergency, ticking-bomb logic, or need to quickly prevent harm.
0 = absent.
1 = present: urgency or emergency of the situation is explicitly mentioned.

emergency_urgency_direction:
Whether urgency increases or decreases support.
-1 = urgency framing lowers support (e.g., rushed decisions are dangerous).
1 = urgency increases support (e.g., lives are at stake, must act now).
0 = NA.

legality_assessment_causality:
Legality, illegality, legal norms, human rights, rights violations, international law, or legal permissibility.
0 = absent.
1 = present: legal or rights considerations are explicitly mentioned.

legality_assessment_direction:
-1 = legal assessment lowers support (action is illegal or violates rights/norms).
1 = legal assessment increases support (action is legal or permissible under the conditions).
0 = NA.

cruelty_causality:
Harshness, violence, severity, excessiveness, harm, physical/psychological coercion, or permissibility of coercion toward suspect/defendant/offender.
0 = absent.
1 = present: the harshness, severity, or coercive nature of the action is explicitly mentioned.

cruelty_direction:
-1 = harshness, harm, or coercion lowers support (described as unacceptable or excessive).
1 = coercion or harshness is described as justified, necessary, or permissible.
0 = NA.

FINAL CONSISTENCY RULES

If any causality variable is 0, its corresponding direction variable must be 0.
Be conservative. Do not infer categories that are not present.
Return only the JSON object.
""".strip()


def build_item_prompt(row: pd.Series) -> str:
    group_name = str(row.get("group_name", ""))
    incentive = str(row.get("incentive", ""))
    reasoning = str(row.get("stage1_reasoning_text", ""))

    return f"""
{CODING_PROMPT}

Now code the following item.

METADATA

experiment_name:
{row.get("experiment_name", "")}

group_name / treatment_name:
{group_name}

incentive:
{incentive}

REMINDER:
- If group_name contains "control", set treatment_awareness_causality = 0 and treatment_effect_direction = 0.
- If incentive is no_incentive, empty, missing, or null, set incentive_awareness_causality = 0 and incentive_effect_direction = 0.

REASONING TRACE

{reasoning}
""".strip()


# =========================
# NORMALIZATION
# =========================

def is_control_group(value) -> bool:
    return "control" in str(value).lower()


def is_no_incentive(value) -> bool:
    v = str(value).strip().lower()
    return v in {"", "nan", "none", "null", "no_incentive"}


def normalize_style(value) -> str:
    if pd.isna(value):
        return "unclear"

    v = str(value).strip().lower()

    mapping = {
        "mixes": "mixed",
        "mix": "mixed",
        "mixed": "mixed",
        "2": "mixed",          # на случай старого кодирования в goldenset
        "utilitarian": "utilitarian",
        "utility": "utilitarian",
        "moral": "moral",
        "unclear": "unclear",
        "unknown": "unclear",
        "nan": "unclear",
    }

    return mapping.get(v, "unclear")


def normalize_int_label(value, allowed, default=0) -> int:
    if pd.isna(value):
        return default

    if isinstance(value, str):
        v = value.strip().upper()
        if v in {"NA", "N/A", "NONE", "NULL", ""}:
            return default
        value = v

    try:
        x = int(float(value))
    except Exception:
        return default

    return x if x in allowed else default


def normalize_annotation_dict(ann: dict, row: pd.Series | None = None) -> dict:
    clean = {}

    for col in CAUSALITY_COLS:
        clean[col] = normalize_int_label(ann.get(col, 0), allowed={0, 1}, default=0)

    for col in BINARY_COLS:
        clean[col] = normalize_int_label(ann.get(col, 0), allowed={0, 1}, default=0)

    for col in DIRECTION_COLS:
        clean[col] = normalize_int_label(ann.get(col, 0), allowed={-1, 0, 1}, default=0)

    clean[STYLE_COL] = normalize_style(ann.get(STYLE_COL, "unclear"))

    # Direction must be 0 if the corresponding causality variable is 0
    for direction_col, causality_col in DIRECTION_BY_CAUSALITY.items():
        if clean[causality_col] == 0:
            clean[direction_col] = 0

    # Hard rule from your comment: control treatment => no treatment fields
    if row is not None and is_control_group(row.get("group_name", "")):
        clean["treatment_awareness_causality"] = 0
        clean["treatment_effect_direction"] = 0

    # Useful analogous rule: no incentive => no incentive fields
    if row is not None and is_no_incentive(row.get("incentive", "")):
        clean["incentive_awareness_causality"] = 0
        clean["incentive_effect_direction"] = 0

    return clean


def normalize_gold_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in CAUSALITY_COLS:
        # binarize: any positive value (was 1 or 2) → 1
        out[col] = out[col].apply(lambda x: normalize_int_label(x, {0, 1, 2}, 0))
        out[col] = (out[col] > 0).astype(int)

    for col in BINARY_COLS:
        out[col] = out[col].apply(lambda x: normalize_int_label(x, {0, 1}, 0))

    for col in DIRECTION_COLS:
        out[col] = out[col].apply(lambda x: normalize_int_label(x, {-1, 0, 1}, 0))

    out[STYLE_COL] = out[STYLE_COL].apply(normalize_style)

    for idx, row in out.iterrows():
        if is_control_group(row.get("group_name", "")):
            out.at[idx, "treatment_awareness_causality"] = 0
            out.at[idx, "treatment_effect_direction"] = 0

        if is_no_incentive(row.get("incentive", "")):
            out.at[idx, "incentive_awareness_causality"] = 0
            out.at[idx, "incentive_effect_direction"] = 0

        for direction_col, causality_col in DIRECTION_BY_CAUSALITY.items():
            if out.at[idx, causality_col] == 0:
                out.at[idx, direction_col] = 0

    return out


# =========================
# OPENAI CALLS
# =========================

def call_openai_annotation(model_name: str, row: pd.Series, max_retries: int = 5) -> dict:
    prompt = build_item_prompt(row)

    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "reasoning_annotation",
                        "schema": ANNOTATION_SCHEMA,
                        "strict": True,
                    },
                },
            )

            raw_text = response.choices[0].message.content
            ann = json.loads(raw_text)
            return normalize_annotation_dict(ann, row=row)

        except Exception as e:
            last_error = str(e)
            sleep_s = min(60, 2 ** attempt + random.random())
            time.sleep(sleep_s)

    raise RuntimeError(last_error)


# =========================
# CACHE
# =========================

def load_cache(path: Path) -> dict:
    cache = {}

    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            rec = json.loads(line)
            key = (rec["coder_model"], rec["row_id"])

            # Последняя запись по ключу побеждает.
            cache[key] = rec

    return cache


def append_cache_record(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# RUN ANNOTATION
# =========================

def run_annotations(gold: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    cache = load_cache(CACHE_PATH)
    records = []

    if SAMPLE_N is not None:
        gold_iter = gold.head(SAMPLE_N)
    else:
        gold_iter = gold

    for coder_model in models:
        for row_id, row in tqdm(
            gold_iter.iterrows(),
            total=len(gold_iter),
            desc=f"Annotating with {coder_model}",
        ):
            key = (coder_model, int(row_id))

            if key in cache and cache[key].get("ok") is True:
                records.append(cache[key])
                continue

            try:
                ann = call_openai_annotation(coder_model, row)
                rec = {
                    "coder_model": coder_model,
                    "row_id": int(row_id),
                    "ok": True,
                    "error": None,
                    **ann,
                }

            except Exception as e:
                rec = {
                    "coder_model": coder_model,
                    "row_id": int(row_id),
                    "ok": False,
                    "error": str(e),
                }

            append_cache_record(CACHE_PATH, rec)
            records.append(rec)

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    return pred_df


# =========================
# METRICS
# =========================

def labels_for_col(col: str):
    if col in CAUSALITY_COLS:
        return [0, 1]
    if col in BINARY_COLS:
        return [0, 1]
    if col in DIRECTION_COLS:
        return [-1, 0, 1]
    if col == STYLE_COL:
        return ["utilitarian", "moral", "mixed", "unclear"]
    raise ValueError(col)


def compute_metrics(gold: pd.DataFrame, pred_df: pd.DataFrame):
    gold_eval = gold.copy()
    gold_eval["row_id"] = gold_eval.index.astype(int)

    successful = pred_df[pred_df["ok"] == True].copy()

    metrics_by_field = []
    metrics_by_model = []

    for coder_model, model_pred in successful.groupby("coder_model"):
        merged = model_pred.merge(
            gold_eval[["row_id"] + OUTPUT_FIELDS],
            on="row_id",
            how="inner",
            suffixes=("_pred", "_gold"),
        )

        if merged.empty:
            continue

        field_f1s = []
        field_accs = []
        total_correct = 0
        total_cells = 0

        for col in OUTPUT_FIELDS:
            y_true = merged[f"{col}_gold"]
            y_pred = merged[f"{col}_pred"]

            labels = labels_for_col(col)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )

            field_accs.append(acc)
            field_f1s.append(f1)

            total_correct += (y_true.values == y_pred.values).sum()
            total_cells += len(y_true)

            metrics_by_field.append({
                "coder_model": coder_model,
                "field": col,
                "accuracy": acc,
                "macro_f1": f1,
                "n": len(y_true),
            })

        pred_matrix = merged[[f"{c}_pred" for c in OUTPUT_FIELDS]].copy()
        gold_matrix = merged[[f"{c}_gold" for c in OUTPUT_FIELDS]].copy()

        pred_matrix.columns = OUTPUT_FIELDS
        gold_matrix.columns = OUTPUT_FIELDS

        row_exact_match = (pred_matrix.astype(str) == gold_matrix.astype(str)).all(axis=1).mean()

        causality_acc = np.mean([
            accuracy_score(merged[f"{c}_gold"], merged[f"{c}_pred"])
            for c in CAUSALITY_COLS
        ])

        direction_acc = np.mean([
            accuracy_score(merged[f"{c}_gold"], merged[f"{c}_pred"])
            for c in DIRECTION_COLS
        ])

        metrics_by_model.append({
            "coder_model": coder_model,
            "n_rows": len(merged),
            "cell_accuracy": total_correct / total_cells,
            "mean_field_accuracy": float(np.mean(field_accs)),
            "mean_field_macro_f1": float(np.mean(field_f1s)),
            "row_exact_match": row_exact_match,
            "causality_mean_accuracy": causality_acc,
            "direction_mean_accuracy": direction_acc,
            "reasoning_style_accuracy": accuracy_score(
                merged[f"{STYLE_COL}_gold"],
                merged[f"{STYLE_COL}_pred"],
            ),
        })

    metrics_model_df = pd.DataFrame(metrics_by_model).sort_values(
        ["mean_field_macro_f1", "cell_accuracy"],
        ascending=False,
    )

    metrics_field_df = pd.DataFrame(metrics_by_field).sort_values(
        ["coder_model", "macro_f1"],
        ascending=[True, True],
    )

    metrics_model_df.to_csv(METRICS_BY_MODEL_CSV, index=False)
    metrics_field_df.to_csv(METRICS_BY_FIELD_CSV, index=False)

    return metrics_model_df, metrics_field_df


# =========================
# MAIN
# =========================

def main():
    gold_raw = pd.read_csv(INPUT_CSV)

    missing = [c for c in ["stage1_reasoning_text", "group_name", "incentive"] + OUTPUT_FIELDS if c not in gold_raw.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    gold = normalize_gold_df(gold_raw)

    pred_df = run_annotations(gold, MODELS)

    metrics_model_df, metrics_field_df = compute_metrics(gold, pred_df)

    print("\n=== Metrics by model ===")
    print(metrics_model_df.to_string(index=False))

    if not metrics_model_df.empty:
        best = metrics_model_df.iloc[0]
        print("\n=== Best model ===")
        print(f"Best model: {best['coder_model']}")
        print(f"Mean field macro-F1: {best['mean_field_macro_f1']:.4f}")
        print(f"Cell accuracy: {best['cell_accuracy']:.4f}")
        print(f"Row exact match: {best['row_exact_match']:.4f}")

    print(f"\nSaved predictions to: {PREDICTIONS_CSV}")
    print(f"Saved model metrics to: {METRICS_BY_MODEL_CSV}")
    print(f"Saved field metrics to: {METRICS_BY_FIELD_CSV}")
    print(f"Saved raw cache to: {CACHE_PATH}")


if __name__ == "__main__":
    main()