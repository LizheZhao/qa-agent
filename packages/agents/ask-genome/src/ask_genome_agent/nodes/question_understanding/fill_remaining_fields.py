"""Model: fills the questionnaire fields BERT didn't cover or wasn't confident about, one
dimension at a time, through with_structured_output(method="function_calling"), the same typed
tool-calling used elsewhere in this graph. It replaces ask-genome-core's two ad hoc HTTP paths: an
internal grammar-constrained call, and a gateway call with hand-parsed JSON.

Retries each dimension up to 3 times, matching SimpleFilterExtractor. The source logs "adding to
clarification list" on final failure and then re-raises anyway, crashing the request; here
exhausting retries actually degrades to clarification.
"""

import json
import logging
from collections.abc import Collection, Sequence
from typing import Any

import pandas as pd
from orchestration_core import AgentDependencies
from pydantic import BaseModel

from ask_genome_agent.config import NerData, load_ner_data
from ask_genome_agent.contracts import build_field_extraction_model, sanitize_field_name
from ask_genome_agent.nodes.question_understanding.support.time_periods import (
    format_today,
    get_few_shot_learning_examples_for_time,
)
from ask_genome_agent.prompts import (
    DYNAMIC_TIME_PROMPT,
    FIELD_EXTRACTION_PROMPT,
    HIERARCHY_FIELD_PROMPT,
)
from ask_genome_agent.state import AskGenomeState

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS_PER_DIMENSION = 3
_CONFIDENCE_THRESHOLD = 0.95
_TIME_DIMENSION = "time"


def _normalize_extracted_value(value: Any) -> Any:
    """Decode a JSON-encoded list the model returned as a string.

    Each field accepts `list[str] | str`, and that ambiguity sometimes makes the model encode a
    list as a string, such as `{"start": "[\\"202504\\"]"}`. Left alone, time filtering
    downstream gets the literal characters instead of a date.
    """

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return value
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    if isinstance(decoded, list) and all(isinstance(item, str | int | float) for item in decoded):
        return [str(item) for item in decoded]
    return value


def _fields_needing_llm(
    ner_data: NerData, predictions: dict[str, Any], probabilities: dict[str, Any]
) -> set[str]:
    all_fields = [f for spec in ner_data.questionnaire.values() for f in spec["field"]]
    uncovered = [f for f in all_fields if f not in predictions]
    low_score = [
        f
        for f in all_fields
        if f in probabilities and max(probabilities[f].values()) <= _CONFIDENCE_THRESHOLD
    ]
    return set(uncovered) | set(low_score)


def get_answer_format(
    dimension: str, fields: Sequence[str], df: pd.DataFrame, custom_cols: Collection[str] = ()
) -> str:
    """Ported from ask-genome-core/src/model/filter_generator.py:get_answer_format.

    Fills the `{answer_format}` placeholder in the questionnaire prompts. It looks redundant next
    to the tool schema, but it isn't: the templates read "...with the following structure:
    {answer_format}", so leaving it blank strands the instruction, and it's also where
    'irrelevant'/'all' are named as legal answers.
    """

    if dimension == "classification":
        return json.dumps(
            {"intention": "one of the relevant values from predefined list or 'irrelevant'"}
        )
    if dimension == "tag":
        return json.dumps({name: "'relevant' or 'irrelevant'" for name in fields})
    if dimension == "trend":
        return json.dumps(
            {
                "trend": "'yes' or 'no'",
                "rank": "'top' or 'bottom' or 'na'",
                "how_many": "number or 'na'",
            }
        )
    if set(fields) & set(custom_cols):
        return json.dumps({name: "'irrelevant' or 'all' or 'specific'" for name in fields})
    selections = {name: df[name].dropna().unique() for name in fields if name in df.columns}
    return json.dumps(
        {
            name: f"'irrelevant' or 'all' or list of relevant entries from {list(values)}"
            for name, values in selections.items()
        }
    )


def _render_questionnaire_prompt(
    spec: dict[str, Any],
    dimension: str,
    fields: Sequence[str],
    query: str,
    predefined_list: str,
    df_bi: pd.DataFrame,
    custom_cols: Collection[str],
) -> str:
    """Substitute the questionnaire prompt's placeholders instead of passing them through.

    The prompt column in questionnaire.csv is a template, not prose. ask-genome-core runs every
    dimension through construct_prompt to fill it. Concatenating it raw sent the model literal
    text like "Today is {date}." Braces are doubled in the client data, so .format() unescapes the
    embedded JSON rather than choking on it.

    `example` is the one placeholder not genuinely filled: picking few-shot examples needs
    construct_learning_examples and the embedding-similarity scoring, neither ported yet. It says
    "none available" rather than rendering empty, so the prompt doesn't promise examples and then
    show none.
    """

    template = spec.get("prompt") or ""
    if not template:
        return f"Question: {query}"
    try:
        rendered = template.format(
            cols=", ".join(fields),
            predefined_list=predefined_list,
            question=query,
            example="(no examples available)",
            answer_format=get_answer_format(dimension, fields, df_bi, custom_cols),
            context=spec.get("context") or "",
            date=format_today(),
        )
    except (KeyError, IndexError, ValueError) as exc:
        # A bad placeholder in one client's questionnaire shouldn't fail the request; fall back
        # to raw text for that dimension only.
        logger.warning(
            "could not render questionnaire prompt template: %s: %s", type(exc).__name__, exc
        )
        rendered = template
    # The template may already embed {question}; only append it when it does not.
    return rendered if "{question}" in template else f"{rendered}\n\nQuestion: {query}"


def get_predefined_list(fields: Sequence[str], df: pd.DataFrame) -> str:
    """Ported from ask-genome-core/src/model/filter_generator.py:get_predefined_list.

    The valid values for each field, read off the real data. Without it the model guesses blind,
    with no way to know 'lms' is a real business unit and 'marketing' isn't.
    """

    lines = []
    for name in fields:
        if name not in df.columns:
            continue
        values = [v for v in df[name].dropna().unique() if v and v != "overall"]
        if values:
            lines.append(f"{name}: {sorted(str(v) for v in values)}")
    return "\n".join(lines)


async def _fill_dimension(
    dimension: str,
    fields: list[str],
    spec: dict[str, Any],
    query: str,
    dependencies: AgentDependencies,
    hierarchy_fields: Collection[str],
    df_bi: pd.DataFrame,
) -> dict[str, Any] | None:
    is_custom = bool(set(fields) & set(hierarchy_fields))
    schema = build_field_extraction_model(
        fields, hierarchy_fields=hierarchy_fields if is_custom else ()
    )
    structured_model = dependencies.model.with_structured_output(schema, method="function_calling")
    predefined_list = get_predefined_list(fields, df_bi) or "(no values available)"

    if dimension == _TIME_DIMENSION:
        # construct_prompt overrides the questionnaire's own prompt for time. Relative periods
        # are unanswerable without today's date.
        system_prompt = DYNAMIC_TIME_PROMPT.format(
            date=format_today(), example=get_few_shot_learning_examples_for_time()
        )
        user_message = f"Question: {query}"
    elif is_custom:
        # A classification, not an extraction. See HIERARCHY_FIELD_PROMPT's docstring.
        system_prompt = HIERARCHY_FIELD_PROMPT.format(
            context=spec.get("context") or "",
            cols=", ".join(fields),
            predefined_list=predefined_list,
        )
        user_message = f"Question: {query}"
    else:
        system_prompt = FIELD_EXTRACTION_PROMPT.format(
            dimension=dimension,
            context=spec.get("context") or "",
            predefined_list=predefined_list,
        )
        user_message = _render_questionnaire_prompt(
            spec, dimension, fields, query, predefined_list, df_bi, hierarchy_fields
        )

    # The schema is keyed by sanitized names, because tool params must be identifier-safe, so
    # map the answer back to the real field names.
    original_by_sanitized = {sanitize_field_name(name): name for name in fields}

    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS_PER_DIMENSION + 1):
        try:
            messages = [("system", system_prompt), ("user", user_message)]
            result = await structured_model.ainvoke(messages)
            if not isinstance(result, BaseModel):
                raise TypeError("field extraction expected a structured model result")
            answer = {
                original_by_sanitized.get(key, key): _normalize_extracted_value(value)
                for key, value in result.model_dump().items()
                if value is not None
            }
            if answer:
                return answer
        except Exception as exc:
            last_error = exc
            logger.warning(
                "field extraction failed for dimension %r (attempt %d/%d): %s: %s",
                dimension,
                attempt,
                _MAX_ATTEMPTS_PER_DIMENSION,
                type(exc).__name__,
                exc,
            )
            continue

    # Degrade to clarification rather than raising, but log the cause. Swallowing it once hid a
    # gateway 500 caused by a field name containing spaces.
    logger.error(
        "field extraction exhausted %d attempts for dimension %r; deferring %s to clarification. "
        "last error: %s",
        _MAX_ATTEMPTS_PER_DIMENSION,
        dimension,
        fields,
        last_error,
    )
    return None


async def fill_remaining_fields(
    state: AskGenomeState, dependencies: AgentDependencies
) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    ner_data = load_ner_data()
    ner_state = dict(state.get("ner_state", {}))
    entry = dict(ner_state[subquery.id])

    predictions: dict[str, Any] = dict(entry["predictions"])
    probabilities: dict[str, Any] = entry["probabilities"]
    # What the user told us directly. Never re-ask the LLM about it: they're the authority, and
    # asking again invites a contradiction, which is what kept the loop from converging.
    user_confirmed: dict[str, Any] = dict(entry.get("user_confirmed", {}))
    llm_fields = _fields_needing_llm(ner_data, predictions, probabilities) - set(user_confirmed)

    # Seed with every BERT prediction and let the LLM override, matching the source's merge order
    # (filter_generator_refactor.py:435-437). Excluding LLM-bound fields up front meant a failed
    # call lost BERT's fallback value and the field vanished from the filter.
    ner_results: dict[str, Any] = dict(predictions)
    field_status: dict[str, str] = dict.fromkeys(ner_results, "bert")
    clarification_list: list[str] = list(entry.get("clarification_list", []))
    custom_cols = ner_data.custom_cols

    for dimension, spec in ner_data.questionnaire.items():
        dim_fields = [f for f in spec["field"] if f in llm_fields]
        if not dim_fields:
            continue
        answer = await _fill_dimension(
            dimension,
            dim_fields,
            spec,
            entry["extended_query"],
            dependencies,
            hierarchy_fields=custom_cols,
            df_bi=ner_data.df_bi,
        )
        if answer is None:
            clarification_list.extend(dim_fields)
            continue
        ner_results.update(answer)
        field_status.update(dict.fromkeys(answer, "llm"))
        missing = [f for f in dim_fields if f not in answer]
        clarification_list.extend(missing)

    # Applied last so the user's own answers outrank both BERT's prediction and the LLM's guess.
    ner_results.update(user_confirmed)
    field_status.update(dict.fromkeys(user_confirmed, "user-confirmed"))
    clarification_list = [f for f in clarification_list if f not in user_confirmed]

    entry["ner_results"] = ner_results
    entry["field_status"] = field_status
    entry["clarification_list"] = list(set(clarification_list))
    ner_state[subquery.id] = entry
    return {"ner_state": ner_state}
