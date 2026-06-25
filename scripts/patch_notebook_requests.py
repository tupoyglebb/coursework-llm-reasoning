"""One-off script to refactor running_think_expos.ipynb — dataset + request phases."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "running_think_expos.ipynb"

CELL2_NEW = r'''# Все входные данные эксперимента — в thinking_expos/attitude/
PROMPT_TEMPLATES_PATH = ATTITUDE_DIR / "prompts_attitude.json"
EXPERIMENTS_PATH = ATTITUDE_DIR / "expos_attitude.json"
PROFILES_PARQUET_PATH = ATTITUDE_DIR / "twin2k_sample_100_with_demographics.parquet"

with open(PROMPT_TEMPLATES_PATH, "r", encoding="utf-8") as f:
    prompt_templates: Dict[str, Dict[str, str]] = json.load(f)

with open(EXPERIMENTS_PATH, "r", encoding="utf-8") as f:
    sdb_experiments: List[Dict[str, Any]] = json.load(f)


def flatten_experiment_groups(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for exp in experiments:
        exp_name = exp["experiment_name"]
        groups = exp.get("groups", {})
        for group_name, group_data in groups.items():
            rows.append(
                {
                    "experiment_name": exp_name,
                    "group_name": group_name,
                    "vignette": group_data.get("vignette", "") or "",
                    "question": group_data.get("question", "") or "",
                    "response_options_text": group_data.get("response_options_text", "") or "",
                    "open_rationale_question": group_data.get("open_rationale_question", "") or "",
                    "wording_status": group_data.get("wording_status", "") or "",
                }
            )
    return rows


def load_profiles_from_twin2k_parquet(parquet_path: Path, n_profiles: int = 100) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    if len(df) != n_profiles:
        raise ValueError(
            f"Expected exactly {n_profiles} rows in {parquet_path.name}, got {len(df)}"
        )
    required = {"pid", "persona_summary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {sorted(missing)}")

    profiles: List[Dict[str, Any]] = []
    for i in range(len(df)):
        row = df.iloc[i]
        summary = row["persona_summary"]
        if pd.isna(summary):
            summary = ""
        else:
            summary = str(summary)
        profiles.append(
            {
                "profile_id": int(i),
                "pid": str(row["pid"]),
                "persona_summary": summary,
            }
        )
    return profiles


selected_profiles = load_profiles_from_twin2k_parquet(
    PROFILES_PARQUET_PATH,
    n_profiles=RESPONDENTS_PER_GROUP,
)
print("Loaded respondent profiles from parquet:", len(selected_profiles))
'''

CELL3_PREFIX = '''def is_two_stage_template(template_cfg: Dict[str, str]) -> bool:
'''

# Insert block AFTER normalize_job_output and BEFORE _test_job
CELL3_INSERT = r'''

# Плейсхолдеры для сборки stage2 user до получения ответа stage1 (см. build_requests_dataset)
STAGE1_RAW_PLACEHOLDER = "<<<OR_INSERT_STAGE1_RAW_RESPONSE>>>"
STAGE1_FINAL_ANSWER_PLACEHOLDER = "<<<OR_INSERT_STAGE1_FINAL_ANSWER>>>"


def build_requests_dataset(
    models_list: List[str],
    templates_map: Dict[str, Dict[str, str]],
    experiments: List[Dict[str, Any]],
    respondent_profiles: List[Dict[str, Any]],
    num_respondents: int,
    base_seed: int = 42,
) -> pd.DataFrame:
    """
    Одна строка = один HTTP-запрос к API.
    Для two-stage — две строки с общим logical_job_id (stage_index 1 и 2);
    во второй user_prompt вставлены плейсхолдеры под ответ первого шага.
    """
    if num_respondents < 1:
        raise ValueError("num_respondents must be >= 1")
    if len(respondent_profiles) < num_respondents:
        raise ValueError(
            f"need at least {num_respondents} profiles, got {len(respondent_profiles)}"
        )

    group_rows = flatten_experiment_groups(experiments)
    template_names = sorted(templates_map.keys())
    respondent_profile_map = {i + 1: respondent_profiles[i] for i in range(num_respondents)}

    rows_out: List[Dict[str, Any]] = []
    logical_job_id = 0
    request_id = 0

    for model_name in models_list:
        provider_route_id = PROVIDER_ROUTE_BY_MODEL.get(model_name, "")
        for template_name in template_names:
            template_cfg = templates_map[template_name]
            for group in group_rows:
                for respondent_id in range(1, num_respondents + 1):
                    logical_job_id += 1
                    profile = respondent_profile_map[respondent_id]
                    run_seed = base_seed + logical_job_id
                    job: Dict[str, Any] = {
                        "job_id": logical_job_id,
                        "seed": run_seed,
                        "model": model_name,
                        "template_name": template_name,
                        "experiment_name": group["experiment_name"],
                        "group_name": group["group_name"],
                        "respondent_id": respondent_id,
                        "profile_id": int(profile["profile_id"]),
                        "profile_pid": profile.get("pid"),
                        "profile_summary": profile.get("persona_summary", "") or "",
                        "vignette": group["vignette"],
                        "question": group["question"],
                        "response_options_text": group["response_options_text"],
                        "open_rationale_question": group["open_rationale_question"],
                        "wording_status": group["wording_status"],
                    }
                    rendered = render_prompts_for_job(template_name, template_cfg, job)
                    collect_rt = bool(REASONING_EXTRA_BODY.get("exclude") is not True)

                    base_meta = {
                        "logical_job_id": logical_job_id,
                        "model": model_name,
                        "provider_route_id": provider_route_id,
                        "template_name": template_name,
                        "template_type": rendered["template_type"],
                        "experiment_name": group["experiment_name"],
                        "group_name": group["group_name"],
                        "treatment_name": group["group_name"],
                        "respondent_id": respondent_id,
                        "profile_id": int(profile["profile_id"]),
                        "profile_pid": profile.get("pid"),
                        "profile_summary": job["profile_summary"],
                        "incentive_key": INCENTIVE_KEY,
                        "seed": run_seed,
                        "vignette": group["vignette"],
                        "question": group["question"],
                        "response_options_text": group["response_options_text"],
                        "open_rationale_question": group["open_rationale_question"],
                        "wording_status": group["wording_status"],
                        "collect_reasoning_trace": collect_rt,
                        "reasoning_extra_body_json": json.dumps(REASONING_EXTRA_BODY, ensure_ascii=False),
                        "include_profile_in_prompt": INCLUDE_PROFILE_IN_PROMPT,
                    }

                    if rendered["template_type"] == "single_stage":
                        request_id += 1
                        rows_out.append(
                            {
                                **base_meta,
                                "request_id": request_id,
                                "stage_index": 1,
                                "system_prompt": rendered["stage1"]["system"],
                                "user_prompt": rendered["stage1"]["user"],
                                "user_prompt_needs_stage1_fill": False,
                            }
                        )
                    else:
                        request_id += 1
                        rows_out.append(
                            {
                                **base_meta,
                                "request_id": request_id,
                                "stage_index": 1,
                                "system_prompt": rendered["stage1"]["system"],
                                "user_prompt": rendered["stage1"]["user"],
                                "user_prompt_needs_stage1_fill": False,
                            }
                        )
                        common = dict(rendered["stage2"]["common_values"])
                        pv = dict(common)
                        pv["DELIBERATION_TEXT"] = STAGE1_RAW_PLACEHOLDER
                        pv["RATIONALE_TEXT"] = STAGE1_RAW_PLACEHOLDER
                        pv["SELECTED_ANSWER"] = STAGE1_FINAL_ANSWER_PLACEHOLDER
                        u2 = _safe_format(rendered["stage2"]["user_template"], pv)
                        request_id += 1
                        rows_out.append(
                            {
                                **base_meta,
                                "request_id": request_id,
                                "stage_index": 2,
                                "system_prompt": rendered["stage2"]["system"],
                                "user_prompt": u2,
                                "user_prompt_needs_stage1_fill": True,
                            }
                        )

    df = pd.DataFrame(rows_out)
    return df


requests_df = build_requests_dataset(
    models_list=models,
    templates_map=prompt_templates,
    experiments=sdb_experiments,
    respondent_profiles=selected_profiles,
    num_respondents=RESPONDENTS_PER_GROUP,
    base_seed=RANDOM_SEED,
)

_mapping_variants = (
    requests_df[["respondent_id", "profile_id"]]
    .drop_duplicates()
    .groupby("respondent_id")["profile_id"]
    .nunique()
)
assert (_mapping_variants == 1).all(), "Profile mapping is not stable across groups"

print("requests_df rows (1 row = 1 API call):", len(requests_df))
print("logical jobs:", requests_df["logical_job_id"].nunique())
requests_df[
    ["request_id", "logical_job_id", "stage_index", "model", "template_name", "experiment_name", "group_name"]
].head(6)
'''

CELL4_REPLACE_START = "async def _chat_completion_with_retry("
CELL4_NEW_CHAT = '''async def _chat_completion_with_retry(
    model: str,
    system_text: str,
    user_text: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    reasoning_body: Optional[Dict[str, Any]] = None,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    reasoning_body = reasoning_body if reasoning_body is not None else REASONING_EXTRA_BODY
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=temperature,
                    extra_body=build_openrouter_extra_body_for_model(model, reasoning_body),
                ),
                timeout=timeout_seconds,
            )

            choice = completion.choices[0] if completion.choices else None
            msg_obj = _safe_get(choice, "message", None)
            response_text = _safe_get(msg_obj, "content", "") or ""
            usage_dict = _usage_to_dict(_safe_get(completion, "usage", None))
            rfields = extract_reasoning_trace_from_message(msg_obj)

            return {
                "ok": True,
                "response_text": response_text,
                "usage": usage_dict,
                "error_message": None,
                "attempt": attempt,
                **rfields,
            }
        except Exception as exc:
            last_err = exc
            if attempt >= max_retries or not _is_retryable_error(exc):
                break
            sleep_s = min(20, (2**attempt) + random.random())
            await asyncio.sleep(sleep_s)

    er = _empty_reasoning_fields()
    return {
        "ok": False,
        "response_text": "",
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "usage_missing": True,
        },
        "error_message": str(last_err),
        "attempt": max_retries,
        **er,
    }


def _reasoning_body_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("reasoning_extra_body_json") or ""
    if not raw:
        return dict(REASONING_EXTRA_BODY)
    try:
        return json.loads(raw)
    except Exception:
        return dict(REASONING_EXTRA_BODY)


def _fill_stage2_user_prompt(user_prompt: str, stage1_response_text: str) -> str:
    fa = extract_final_answer(stage1_response_text)
    return (
        user_prompt.replace(STAGE1_RAW_PLACEHOLDER, stage1_response_text)
        .replace(STAGE1_FINAL_ANSWER_PLACEHOLDER, fa)
    )


async def _run_one_logical_job_async(
    job_rows: List[Dict[str, Any]],
    temperature: float,
) -> Dict[str, Any]:
    """job_rows — все строки requests_df с одним logical_job_id, по возрастанию stage_index."""
    job_rows = sorted(job_rows, key=lambda r: int(r["stage_index"]))
    first = job_rows[0]
    base_result = {
        "job_id": int(first["logical_job_id"]),
        "seed": first["seed"],
        "model": first["model"],
        "template_name": first["template_name"],
        "template_type": first["template_type"],
        "experiment_name": first["experiment_name"],
        "group_name": first["group_name"],
        "respondent_id": int(first["respondent_id"]),
        "profile_id": int(first["profile_id"]) if first.get("profile_id") is not None else None,
        "profile_pid": first.get("profile_pid"),
        "incentive_key": INCENTIVE_KEY,
        "wording_status": first.get("wording_status", ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    rb = _reasoning_body_from_row(first)
    r1row = job_rows[0]
    stage1 = await _chat_completion_with_retry(
        model=str(r1row["model"]),
        system_text=str(r1row["system_prompt"]),
        user_text=str(r1row["user_prompt"]),
        reasoning_body=rb,
        temperature=temperature,
    )

    def _stage_reasoning(prefix: str, st: Dict[str, Any]) -> Dict[str, Any]:
        return {
            f"{prefix}_reasoning_text": st.get("reasoning_text", ""),
            f"{prefix}_reasoning_details_json": st.get("reasoning_details_json"),
            f"{prefix}_reasoning_trace": st.get("reasoning_trace", ""),
        }

    r1 = _stage_reasoning("stage1", stage1)

    if len(job_rows) == 1:
        parsed = normalize_job_output(stage1["response_text"], stage1["error_message"])
        return {
            **base_result,
            **parsed,
            "stage1_response_text": stage1["response_text"],
            "stage2_response_text": None,
            "stage1_prompt_tokens": stage1["usage"]["prompt_tokens"],
            "stage1_completion_tokens": stage1["usage"]["completion_tokens"],
            "stage1_total_tokens": stage1["usage"]["total_tokens"],
            "stage2_prompt_tokens": None,
            "stage2_completion_tokens": None,
            "stage2_total_tokens": None,
            "run_total_tokens": stage1["usage"]["total_tokens"],
            "usage_missing": stage1["usage"]["usage_missing"],
            "retry_attempts_stage1": stage1["attempt"],
            "retry_attempts_stage2": None,
            **r1,
            **_stage_reasoning("stage2", _empty_reasoning_fields()),
            "reasoning_trace_combined": r1["stage1_reasoning_trace"],
        }

    r2row = job_rows[1]
    u2 = (
        _fill_stage2_user_prompt(str(r2row["user_prompt"]), stage1["response_text"])
        if r2row.get("user_prompt_needs_stage1_fill")
        else str(r2row["user_prompt"])
    )
    rb2 = _reasoning_body_from_row(r2row)
    stage2 = await _chat_completion_with_retry(
        model=str(r2row["model"]),
        system_text=str(r2row["system_prompt"]),
        user_text=u2,
        reasoning_body=rb2,
        temperature=temperature,
    )

    final_error = stage1["error_message"] or stage2["error_message"]
    parsed = normalize_job_output(stage2["response_text"], final_error)

    st1_total = stage1["usage"]["total_tokens"]
    st2_total = stage2["usage"]["total_tokens"]
    run_total = (
        None
        if (st1_total is None and st2_total is None)
        else (st1_total or 0) + (st2_total or 0)
    )

    r2 = _stage_reasoning("stage2", stage2)
    _parts: List[str] = []
    if (r1.get("stage1_reasoning_trace") or "").strip():
        _parts.append("=== STAGE 1 reasoning ===\\n" + (r1["stage1_reasoning_trace"] or ""))
    if (r2.get("stage2_reasoning_trace") or "").strip():
        _parts.append("=== STAGE 2 reasoning ===\\n" + (r2["stage2_reasoning_trace"] or ""))
    combined = "\\n\\n".join(_parts)

    return {
        **base_result,
        **parsed,
        "stage1_response_text": stage1["response_text"],
        "stage2_response_text": stage2["response_text"],
        "stage1_prompt_tokens": stage1["usage"]["prompt_tokens"],
        "stage1_completion_tokens": stage1["usage"]["completion_tokens"],
        "stage1_total_tokens": st1_total,
        "stage2_prompt_tokens": stage2["usage"]["prompt_tokens"],
        "stage2_completion_tokens": stage2["usage"]["completion_tokens"],
        "stage2_total_tokens": st2_total,
        "run_total_tokens": run_total,
        "usage_missing": stage1["usage"]["usage_missing"] or stage2["usage"]["usage_missing"],
        "retry_attempts_stage1": stage1["attempt"],
        "retry_attempts_stage2": stage2["attempt"],
        **r1,
        **r2,
        "reasoning_trace_combined": combined,
    }


async def openrouter_chat_with_reasoning(
    model: str,
    system_text: str,
    user_text: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Один async-запрос к OpenRouter; в ответе — `response_text`, поля reasoning и usage (см. `_chat_completion_with_retry`)."""
    return await _chat_completion_with_retry(model, system_text, user_text, **kwargs)
'''


def main():
    with open(NB) as f:
        nb = json.load(f)

    # --- cell 2 ---
    nb["cells"][2]["source"] = [line + "\n" for line in CELL2_NEW.split("\n")]

    # --- cell 3: insert build_requests_dataset before _test_job ---
    c3 = "".join(nb["cells"][3]["source"])
    mark = "_test_job = run_plan_df.iloc[0].to_dict()"
    if "run_plan_df" not in c3:
        mark = "_test_job = requests_df.iloc[0].to_dict()"
    if mark not in c3 and "_test_job =" in c3:
        # find _test_job line
        for line in c3.split("\n"):
            if "_test_job =" in line:
                mark = line.strip()
                break
    insert_at = c3.find(mark)
    if insert_at == -1:
        raise RuntimeError("Could not find _test_job anchor in cell 3")
    new_c3 = c3[:insert_at] + CELL3_INSERT + "\n" + c3[insert_at:]
    # Remove old run_plan_df test if present
    new_c3 = new_c3.replace("run_plan_df.iloc[0]", "requests_df.iloc[0]")
    nb["cells"][3]["source"] = [line + "\n" for line in new_c3.split("\n")]

    # --- cell 4: replace from async def _chat_completion through end of _run_one_job_async (before _parts_dir cell) ---
    c4 = "".join(nb["cells"][4]["source"])
    start = c4.find("async def _chat_completion_with_retry(")
    end = c4.find("async def openrouter_chat_with_reasoning(")
    if start == -1 or end == -1:
        raise RuntimeError("cell4 structure unexpected")
    # find end of _run_one_job_async — next double newline after "reasoning_trace_combined" closing the return
    end_block = c4.find('\n\nasync def openrouter_chat_with_reasoning(')
    if end_block == -1:
        end_block = end
    # Remove old _chat_completion, _openrouter, _run_one_job; keep from openrouter_chat if we replaced wrong
    # Simpler: find from _chat_completion to exclusive start of def _parts_dir - that's next cell
    # Our cell 4 only contains API code - find trailing of _run_one_job
    idx_end = c4.rfind("        \"reasoning_trace_combined\": combined,\n    }")
    if idx_end == -1:
        raise RuntimeError("end of _run_one_job not found")
    idx_end = c4.find("\n", idx_end + 50)  # after closing brace line
    # extend to include full return block
    tail_from_openrouter = c4[c4.find("async def openrouter_chat_with_reasoning("):]
    # rebuild: CELL4_NEW_CHAT + tail_from_openrouter from openrouter_chat onward BUT openrouter is duplicated in CELL4_NEW_CHAT
    ostart = c4.find("async def openrouter_chat_with_reasoning(")
    old_openrouter_to_end = c4[ostart:]
    new_c4 = c4[:start] + CELL4_NEW_CHAT + "\n" + old_openrouter_to_end
    # Remove duplicate openrouter - CELL4_NEW_CHAT ends with openrouter_chat - strip old
    # CELL4_NEW_CHAT already includes openrouter_chat - so take c4[:start] + CELL4_NEW_CHAT only, drop old openrouter and _run_one_job
    o2 = CELL4_NEW_CHAT.find("async def openrouter_chat_with_reasoning(")
    if o2 == -1:
        raise RuntimeError("new chat block missing openrouter")
    # CELL4_NEW_CHAT is complete through end of file for cell4 - need to remove duplicate from original
    new_c4 = c4[:start] + CELL4_NEW_CHAT
    nb["cells"][4]["source"] = [line + "\n" for line in new_c4.split("\n")]

    # --- cell 6: orchestrator ---
    c6 = "".join(nb["cells"][6]["source"])
    c6 = c6.replace(
        "async def run_sdb_experiments_async(\n    run_plan: pd.DataFrame,\n    templates_map: Dict[str, Dict[str, str]],",
        "async def run_sdb_experiments_async(\n    requests_df: pd.DataFrame,\n    temperature: float = 0.7,",
    )
    c6 = c6.replace("templates_map: Dict[str, Dict[str, str]],\n", "", 1)  # might break - do manual

    with open(NB, "w") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

    print("Partial patch applied — manual fix may be needed for cell 4/6")


if __name__ == "__main__":
    main()
