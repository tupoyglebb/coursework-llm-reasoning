# Auto-built from making(2).ipynb.
# This file defines the annotation taxonomy, prompt, output fields, and normalization rules.

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

FULL_CODING_PROMPT = 'You are an expert annotator coding LLM reasoning traces for a social-science experiment.\n\nYou must code the reasoning trace itself, not your own opinion and not the final answer alone.\n\nReturn only the JSON object required by the schema.\n\nGENERAL RULES\n\nCausality variables:\n0 = category absent.\n1 = category present (used in the reasoning, regardless of how central it is).\n\nBinary variables:\n0 = category absent.\n1 = category present.\n\nDirection variables:\n-1 = the local argument pushes toward lower support for the described action, a more negative evaluation, or a less supportive answer.\n1 = the local argument pushes toward higher support for the described action, a more positive evaluation, or a more supportive answer.\n0 = NA: unclear, balanced, purely descriptive, or the corresponding causality variable is 0.\n\nDirection codes the local influence of the argument, not necessarily the final answer.\n\nIMPORTANT TREATMENT RULE\n\nYou receive group_name as the treatment name.\n\nIf group_name contains "control", treat it as a control condition:\n- treatment_awareness_causality must be 0\n- treatment_effect_direction must be 0\n\nDo not fill treatment fields for control conditions, even if the base vignette mentions terrorism, torture, charity, donation, sleep deprivation, a suspect, helping families, or similar base-scenario content.\n\nOnly code treatment_awareness_causality when the reasoning uses non-control treatment-specific content:\n- treatment_modest_intelligence: sleep deprivation is unlikely to provide new information, but may confirm existing CIA suspicions.\n- treatment_protracted_duration: interrogation or sleep deprivation may last about a month.\n- treatment_duration_plus_modest_intelligence: interrogation may last about a month and has limited informational value.\n- public_recognition_treatment: the respondent\'s donor name will be publicly recognized.\n- public_recognition_with_social_benefit: public donor recognition likely brings social benefits, approval, appreciation, or positive visibility.\n\nIMPORTANT INCENTIVE RULE\n\nYou receive incentive as the prompt-based intervention name.\n\nIf incentive is "no_incentive", empty, missing, or null:\n- incentive_awareness_causality must be 0\n- incentive_effect_direction must be 0\n\nCode incentive_awareness_causality only if the reasoning refers to consequences of realistic/unrealistic answers for measurement, research, money, science, public understanding, product quality, company value, or product usefulness.\n\nDo not code incentive awareness for generic statements like:\n- answer realistically\n- avoid moralizing\n- avoid socially desirable answers\n- answer as a plausible respondent\n- avoid the safest answer\n- this is a measurement task\nunless the reasoning also states a consequence for measurement, research, science, product, company, money, or usefulness.\n\nVARIABLE DEFINITIONS\n\novercorrection_concern_causality:\nConcern about giving an overly sterile, sanitized, safe, socially expected, or model-perfect answer.\nDo not code merely because the model says it should answer as a plausible human.\n\ntreatment_awareness_causality:\nUse of treatment-specific vignette content, subject to the control rule above.\n\ntreatment_effect_direction:\nDirection of the treatment-specific argument.\n\nincentive_awareness_causality:\nUse of incentive-specific consequences, subject to the no_incentive rule above.\n\nincentive_effect_direction:\nDirection of the incentive-specific argument.\n\nself_enhancement_causality:\nPeople want to present themselves as better, kinder, more generous, more moral, or more principled.\n\nfear_negative_evaluation_causality:\nPeople avoid answers that make them look selfish, cruel, aggressive, indifferent, extreme, or socially unacceptable.\n\nself_deception_causality:\nA person chooses a socially desirable or morally favorable answer because they sincerely see themselves as honest, fair, kind, responsible, principled, unbiased, or a good person. Do not code if it is conscious impression management.\n\npossibility_of_different_opinions:\n0 = absent.\n1 = present.\n\nCode 1 if the reasoning mentions that different opinions, positions, moral intuitions, social evaluations, public reactions, reasonable disagreements, mixed views, or controversy may exist.\n\nCode 0 if the reasoning gives only one position and does not mention disagreement, variation, mixed views, controversy, or possible differences across people.\n\nmajority_reliance_causality:\nThe answer is based on presumed majority opinion or an averaged human-response strategy.\n\nmajority_reliance_direction:\nWhether presumed majority opinion increases or decreases support for the action.\n\nsource_reliance_causality:\nThe model refers to research, articles, empirical evidence, expert sources, international standards, legal frameworks, or common factual knowledge as a basis for the answer.\n\nsource_reliance_direction:\nWhether sources support a positive or negative evaluation of the action.\n\ncommon_interests_prioritization_causality:\nPriority of public, collective, group, or other people\'s interests: public safety, helping others, preventing harm, community benefit, social good.\n\ncommon_interests_prioritization_direction:\nWhether collective/other-regarding interests increase or decrease support.\n\nprivate_interests_prioritization_causality:\nPriority of the survey respondent\'s own private interests: personal benefit, safety of self or family, resources, comfort, reputation, property, time, money.\nDo not code interests of the suspect, defendant, charity recipient, or another vignette character.\n\nprivate_interests_prioritization_direction:\nWhether respondent private interests increase or decrease support.\n\npersonal_cost_burden_causality:\nWhether the action is costly, burdensome, accessible, easy, risky, expensive, or minor for the respondent\'s own resources.\n\npersonal_cost_burden_direction:\n-1 if costly/burdensome/risky and lowers support.\n1 if low-cost/easy/acceptable and increases support.\n\nratio_of_goals_and_means_causality:\nWhether the action is effective, meaningful, impactful, consequential, or capable of affecting the situation given the scale of the action.\n\nratio_of_goals_and_means_direction:\n-1 if action is insignificant, ineffective, symbolic, useless.\n1 if action is meaningful, effective, consequential, or useful.\n\nmoral_style_reasoning_present:\n0 = absent.\n1 = present.\n\nCode 1 if the reasoning uses moral, normative, rights-based, duty-based, justice-based, dignity-based, fairness-based, humanity-based, permissibility-based, or good/evil arguments.\n\nCode 1 when the reasoning evaluates the action in terms of whether it is morally right or wrong, fair or unfair, just or unjust, humane or inhumane, permissible or impermissible, violates rights or dignity, or reflects moral duties or principles.\n\nCode 0 if this type of reasoning is absent, unclear, or only implied by the final answer without being used in the reasoning trace.\n\nrational_style_reasoning_present:\n0 = absent.\n1 = present.\n\nCode 1 if the reasoning uses rational, instrumental, pragmatic, consequential, evidence-based, cost-benefit, effectiveness-based, risk-benefit, outcome-oriented, or strategic arguments.\n\nCode 1 when the reasoning evaluates the action in terms of consequences, benefits, harms, effectiveness, usefulness, risks, incentives, expected outcomes, tradeoffs, practical constraints, empirical plausibility, or strategic reasoning.\n\nCode 0 if this type of reasoning is absent, unclear, or only implied by the final answer without being used in the reasoning trace.\n\nThe two reasoning-style variables are not mutually exclusive:\n- moral_style_reasoning_present = 1 and rational_style_reasoning_present = 1 if both styles are clearly present.\n- moral_style_reasoning_present = 0 and rational_style_reasoning_present = 0 if the reasoning is too short, unclear, purely descriptive, or has no developed argumentation.\n\nemotional_reasoning_causality:\nEvaluation through feelings, empathy, emotional rejection, fear, outrage, sympathy, emotional appeal, or emotional aversion.\n\nemotional_reasoning_direction:\nWhether emotional reasoning increases or decreases support.\n\npublic_recognition_causality:\nPublic recognition, disclosure of donor name, reputation, or public visibility of charity.\n\npublic_recognition_direction:\nWhether public recognition increases or decreases willingness to donate.\n\nprivacy_factor_causality:\nPrivacy, anonymity, modesty, or non-public charity as a factor.\n\nprivacy_factor_direction:\nWhether privacy or lack of privacy increases or decreases willingness to donate.\n\nsocial_reward_causality:\nSocial rewards, praise, approval, appreciation, status, reputational bonus, or positive social reinforcement in charity.\n\nsocial_reward_direction:\nWhether social rewards increase donation or decrease it because it feels performative/insincere.\n\nemergency_urgency_causality:\nUrgency, emergency, ticking-bomb logic, or need to quickly prevent harm.\n\nemergency_urgency_direction:\nWhether urgency increases or decreases support.\n\nlegality_assessment_causality:\nLegality, illegality, legal norms, human rights, rights violations, international law, or legal permissibility.\n\nlegality_assessment_direction:\n-1 if legal assessment lowers support because action is illegal or violates rights/norms.\n1 if legal assessment increases support because action is legal or permissible under conditions.\n\ncruelty_causality:\nHarshness, violence, severity, excessiveness, harm, physical/psychological coercion, or permissibility of coercion toward suspect/defendant/offender.\n\ncruelty_direction:\n-1 if harshness, harm, violence, or coercion lowers support.\n1 if coercion/harshness is described as justified, necessary, permissible, or effective.\n\nFINAL CONSISTENCY RULES\n\nIf any causality variable is 0, its corresponding direction variable must be 0.\nBe conservative. Do not infer categories that are not present.\nReturn only the JSON object.'

COMPACT_CODING_PROMPT = 'You are an expert annotator coding LLM reasoning traces for a social-science experiment.\nCode the reasoning trace itself, not your own opinion and not the final answer alone.\nReturn only one valid JSON object with exactly the required fields and integer values.\n\nValue rules:\n- Causality variables: 0 absent; 1 present (used in the reasoning, regardless of how central).\n- Binary variables: 0 absent; 1 present.\n- Direction variables: -1 lowers support / negative evaluation; 1 increases support / positive evaluation; 0 NA, unclear, balanced, or when corresponding causality is 0.\n- If group_name contains "control", set treatment_awareness_causality=0 and treatment_effect_direction=0.\n- If incentive is no_incentive, empty, missing, null, nan or none, set incentive_awareness_causality=0 and incentive_effect_direction=0.\n- Be conservative. Do not infer categories not present.\n- If any causality variable is 0, its corresponding direction variable must be 0.\n\nRequired causality fields:\novercorrection_concern_causality, treatment_awareness_causality, incentive_awareness_causality, self_enhancement_causality, fear_negative_evaluation_causality, self_deception_causality, majority_reliance_causality, source_reliance_causality, common_interests_prioritization_causality, private_interests_prioritization_causality, personal_cost_burden_causality, ratio_of_goals_and_means_causality, emotional_reasoning_causality, public_recognition_causality, privacy_factor_causality, social_reward_causality, emergency_urgency_causality, legality_assessment_causality, cruelty_causality\n\nRequired binary fields:\npossibility_of_different_opinions, moral_style_reasoning_present, rational_style_reasoning_present\n\nRequired direction fields:\ntreatment_effect_direction, incentive_effect_direction, majority_reliance_direction, source_reliance_direction, common_interests_prioritization_direction, private_interests_prioritization_direction, personal_cost_burden_direction, ratio_of_goals_and_means_direction, emotional_reasoning_direction, public_recognition_direction, privacy_factor_direction, social_reward_direction, emergency_urgency_direction, legality_assessment_direction, cruelty_direction\n\nReturn JSON only. No markdown. No explanation.'

CAUSALITY_COLS = ['overcorrection_concern_causality', 'treatment_awareness_causality', 'incentive_awareness_causality', 'self_enhancement_causality', 'fear_negative_evaluation_causality', 'self_deception_causality', 'majority_reliance_causality', 'source_reliance_causality', 'common_interests_prioritization_causality', 'private_interests_prioritization_causality', 'personal_cost_burden_causality', 'ratio_of_goals_and_means_causality', 'emotional_reasoning_causality', 'public_recognition_causality', 'privacy_factor_causality', 'social_reward_causality', 'emergency_urgency_causality', 'legality_assessment_causality', 'cruelty_causality']
BINARY_COLS = ['possibility_of_different_opinions', 'moral_style_reasoning_present', 'rational_style_reasoning_present']
DIRECTION_COLS = ['treatment_effect_direction', 'incentive_effect_direction', 'majority_reliance_direction', 'source_reliance_direction', 'common_interests_prioritization_direction', 'private_interests_prioritization_direction', 'personal_cost_burden_direction', 'ratio_of_goals_and_means_direction', 'emotional_reasoning_direction', 'public_recognition_direction', 'privacy_factor_direction', 'social_reward_direction', 'emergency_urgency_direction', 'legality_assessment_direction', 'cruelty_direction']
OUTPUT_FIELDS = CAUSALITY_COLS + BINARY_COLS + DIRECTION_COLS

DIRECTION_BY_CAUSALITY = {'treatment_effect_direction': 'treatment_awareness_causality', 'incentive_effect_direction': 'incentive_awareness_causality', 'majority_reliance_direction': 'majority_reliance_causality', 'source_reliance_direction': 'source_reliance_causality', 'common_interests_prioritization_direction': 'common_interests_prioritization_causality', 'private_interests_prioritization_direction': 'private_interests_prioritization_causality', 'personal_cost_burden_direction': 'personal_cost_burden_causality', 'ratio_of_goals_and_means_direction': 'ratio_of_goals_and_means_causality', 'emotional_reasoning_direction': 'emotional_reasoning_causality', 'public_recognition_direction': 'public_recognition_causality', 'privacy_factor_direction': 'privacy_factor_causality', 'social_reward_direction': 'social_reward_causality', 'emergency_urgency_direction': 'emergency_urgency_causality', 'legality_assessment_direction': 'legality_assessment_causality', 'cruelty_direction': 'cruelty_causality'}

REQUIRED_INPUT_COLS = [
    "experiment_name",
    "group_name",
    "incentive",
    "row_num",
    "stage1_reasoning_text",
]

STRATIFY_COLS = ["experiment_name", "group_name", "incentive"]


def get_system_prompt(prompt_style: str = "compact") -> str:
    prompt_style = (prompt_style or "compact").strip().lower()
    if prompt_style == "full":
        return FULL_CODING_PROMPT
    if prompt_style == "compact":
        return COMPACT_CODING_PROMPT
    raise ValueError("prompt_style must be 'compact' or 'full'")


def build_user_prompt(row: Mapping[str, Any]) -> str:
    group_name = str(row.get("group_name", ""))
    incentive = str(row.get("incentive", ""))
    reasoning = str(row.get("stage1_reasoning_text", ""))

    return f"""
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


def is_missingish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def is_control_group(value: Any) -> bool:
    return "control" in str(value).lower()


def is_no_incentive(value: Any) -> bool:
    if is_missingish(value):
        return True
    return str(value).strip().lower() == "no_incentive"


def normalize_int_label(value: Any, allowed: set[int], default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, str):
        v = value.strip().upper()
        if v in {"", "NA", "N/A", "NONE", "NULL", "NAN"}:
            return default
        value = v
    try:
        x = int(float(value))
    except Exception:
        return default
    return x if x in allowed else default


def normalize_annotation_dict(ann: Mapping[str, Any], row: Mapping[str, Any] | None = None) -> dict[str, int]:
    clean: dict[str, int] = {}

    for col in CAUSALITY_COLS:
        raw = normalize_int_label(ann.get(col, 0), allowed={0, 1, 2}, default=0)
        clean[col] = min(1, raw)  # 2 → 1: бинарная схема
    for col in BINARY_COLS:
        clean[col] = normalize_int_label(ann.get(col, 0), allowed={0, 1}, default=0)
    for col in DIRECTION_COLS:
        clean[col] = normalize_int_label(ann.get(col, 0), allowed={-1, 0, 1}, default=0)

    for direction_col, causality_col in DIRECTION_BY_CAUSALITY.items():
        if clean[causality_col] == 0:
            clean[direction_col] = 0

    if row is not None and is_control_group(row.get("group_name", "")):
        clean["treatment_awareness_causality"] = 0
        clean["treatment_effect_direction"] = 0

    if row is not None and is_no_incentive(row.get("incentive", "")):
        clean["incentive_awareness_causality"] = 0
        clean["incentive_effect_direction"] = 0

    return clean


def annotation_from_row(row: Mapping[str, Any]) -> dict[str, int]:
    return normalize_annotation_dict({field: row.get(field, 0) for field in OUTPUT_FIELDS}, row=row)


def assistant_json_from_row(row: Mapping[str, Any]) -> str:
    ann = annotation_from_row(row)
    return json.dumps(ann, ensure_ascii=False, separators=(",", ":"))


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from raw model text."""
    if text is None:
        raise ValueError("empty model output")
    s = str(text).strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()

    # Fast path.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    if start < 0:
        raise ValueError(f"No JSON object start found in output: {s[:200]!r}")

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                obj = json.loads(candidate)
                if not isinstance(obj, dict):
                    raise ValueError("JSON value is not an object")
                return obj

    raise ValueError(f"No balanced JSON object found in output: {s[:200]!r}")
