"""Prompts owned and versioned by the Ask Genome agent. Copied from ask-genome-core/src/prompts.py,
but only includes prompts needed by the current planning scope.
"""

from pathlib import Path

PLANNER_PROMPT = """<context>
You are a query planner for a marketing analytics assistant.
Analytical intention classes for this client: INTENT_CATALOG_PLACEHOLDER
</context>

<task_instruction>
Given a user query, decide how to plan it:
1. rephrased_query: a clear, professional rewrite of the whole query.
2. is_multi: true if the query contains more than one distinct question/intention.
3. subqueries: one entry per atomic question, in dependency order. Always emit at least one (a
   single-intent query yields exactly one sub-query with id 0). Fill shared info (channel, time,
   etc.) into each sub-query so it stands alone.
</task_instruction>"""

RESOLVE_FROM_CONTEXT_PROMPT = """<context>
You finalize a follow-up sub-query using the data context produced by a previous sub-query.
</context>

<task_instruction>
Given the previous sub-query's data context/answer and the current follow-up query, produce the
concrete, self-contained sub-queries to run next. Decide among three outcomes:
- REJECT: if the context does not identify what the follow-up refers to (the entity, item or value
  its pronoun or reference points at), return an empty "subqueries" list with "resolved": false
  and a brief "reason". The follow-up's own figures are fetched after this step, so do not reject
  because the context lacks the metric the follow-up asks about.
- REPHRASE (one): if the follow-up resolves to a single question, return exactly one self-contained
  sub-query (fill in the specific entity/value from the context, e.g. the named or top-ranked item).
- SPLIT (many): if the context implies multiple items the follow-up applies to (for example several
  metrics or values), return one self-contained sub-query per item.
Each returned sub-query must stand alone (no pronouns, scope filled in from the context).
</task_instruction>"""
"""Ported from ask-genome-core's RESOLVE_FROM_CONTEXT_PROMPT (config/prompt.yaml).

One change: REJECT only when the reference can't be identified. Upstream's wording got "What was
its ROI?" refused even though the context named OTT/CTV. The ROI is fetched after the rewrite.
"""

HIERARCHY_FIELD_PROMPT = """<context>
You are a natural language processing engine, specialized in marketing data analysis.
{context}
</context>

<task_instruction>
Classify whether the question is relevant to {cols}, following the question and examples closely.
Do not imply or assume additional information. Choose exactly one of:
- "irrelevant": the question does not mention {cols} at all.
- "all": the question refers to {cols} in general -- typically "by", "each", "for all",
  "different", or "all other" followed by {cols}. Also choose "all" if the question compares one
  specific {cols} against the rest.
- "specific": the question names particular {cols} values that appear in the list below.

Values that exist for these fields:
{predefined_list}
</task_instruction>"""
"""Ported from ask-genome-core's CUSTOM_FIELD_BASE_PROMPT (config/prompt.yaml).

Hierarchy and tagging fields are a three-way classification in the source, not value extraction:
the LLM returns 'irrelevant', 'all' or 'specific', and _get_specific_values later turns 'specific'
into values from the spaCy captures. Asking the LLM for the value outright made these fields fail,
and since nothing could then produce 'irrelevant', should_clarify_halo fired on every query.
"""

FIELD_EXTRACTION_PROMPT = """<context>
You extract structured filter values from a marketing analytics question, for one questionnaire
dimension at a time: "{dimension}". {context}
</context>

<task_instruction>
Given the question, answer every requested field. Only use values implied by the question itself
-- do not guess. Answer "irrelevant" if the question does not mention a field at all, or "all" if
it refers to every value in general.

Values that exist for these fields:
{predefined_list}
</task_instruction>"""

DYNAMIC_TIME_PROMPT = """<context>
You are a natural language processing assistant specialized in marketing data analysis.
You will receive a user question and are expected to reason through any referenced time period.
Then, generate the correct start and end timestamps and return them in the specified format.
</context>

<task_instruction>
You are given a natural language question that may include a time reference. Based on the
question, identify the relevant time period and return the start and end timestamps of the
periods mentioned in the question.

- Today is {date}. Use this to determine what counts as recent time periods.
- The time units may refer to months, quarters (3 months), half-years (6 months), or full years
  (12 months).
- Quarters: Quarter 1: Jan to March, Quarter 2: April to June, Quarter 3: July to September,
  Quarter 4: October to December
- Half years: Half 1: Jan to June, Half 2: July to December

Format:
- Use "YYYYMM" for all dates (for example, January 2025 is "202501").

Rules:
- start date must always be on or before the end date.
- If a time period has not yet occurred, replace it with the same period from the most recent
  past year.
- Do not include any dates beyond today's date.
- If the question contains no time reference, answer "irrelevant" for both start and end.

Answer Process:
1. Identify whether the question includes a time reference.
2. Determine the time type: month, quarter, half-year, or year.
3. Use {date} as today's date when calculating past periods.
4. If any period goes beyond today's date, replace it with the latest available full past period.
5. Format the timestamps using "YYYYMM".
</task_instruction>

<example>
Examples:
{example}
</example>"""
"""Ported from ask-genome-core's DYNAMIC_TIME_PROMPT (config/prompt.yaml).

construct_prompt uses this instead of the questionnaire's own prompt whenever dimension == "time",
so it is ported as a constant. The client CSV holds an older copy without the {example}
placeholder, so reading it would silently lose the worked examples.

Two deviations, both because this package gets structured output through a tool schema: the
source's JSON answer-format block and its "reasoning inside <think>" instruction are dropped. The
schema defines the shape, and inviting prose alongside a tool call fights it.
"""


def _vendored_prompt(name: str) -> str:
    """One of upstream's prompts from the vendored config/prompt.yaml, as stored."""

    import yaml

    from ask_genome_agent.vendor import ask_genome_core

    path = Path(ask_genome_core.__file__).parent / "config" / "prompt.yaml"
    return str(yaml.safe_load(path.read_text())[name])


def readout_prompt_template() -> str:
    """Upstream's EXTERNAL_READOUT_PROMPT as stored, before rendering.

    generate_response renders it through the vendored PromptTemplates, so the text lives in the
    vendored config/prompt.yaml and changes with each sync. This is only for the graph's metadata.
    """

    return _vendored_prompt("EXTERNAL_READOUT_PROMPT")


def coverage_spec_template() -> str:
    """Upstream's COVERAGE_SPEC_PROMPT as stored, for the graph's metadata."""

    return _vendored_prompt("COVERAGE_SPEC_PROMPT")


def coverage_spec_prompt(query: str, hints: str) -> str:
    """Upstream's COVERAGE_SPEC_PROMPT, filled as src/model/coverage.py fills it."""

    return (
        coverage_spec_template()
        .replace("HINTS_PLACEHOLDER", hints)
        .replace("QUERY_PLACEHOLDER", query)
    )


def coverage_phrasing_prompt(query: str, facts: str) -> str:
    """Upstream's COVERAGE_PHRASING_PROMPT, filled as src/model/coverage.py fills it."""

    return (
        _vendored_prompt("COVERAGE_PHRASING_PROMPT")
        .replace("QUERY_PLACEHOLDER", query)
        .replace("FACTS_PLACEHOLDER", facts)
    )


_READOUT_REQUIREMENTS = """\
The figures in the context carry no currency. Do not add currency symbols or units the context \
does not give, whatever the examples above show.
When figures cover only some business units, countries or KPIs, name that scope with them (for \
example "Premium, UK") rather than presenting them as the whole market.
Before saying a pattern holds consistently, generally or across the portfolio, check it against \
every relevant row. Where some rows contradict it, say where it holds and name the exceptions.
State a ranking (highest, lowest, next, top N) only after comparing all the relevant rows."""


def readout_requirements(defaults: str, period: str = "") -> str:
    """Upstream's analytics_generator requirements, then ours. Each of ours fixes something the
    sweep caught. The period is needed because the context still says "last year"."""

    lines = [defaults.strip(), _READOUT_REQUIREMENTS]
    if period:
        lines.append(
            f"The question's time reference resolves to {period}. Answer for {period}; earlier "
            "periods are comparisons only, never the answer."
        )
    return "\n".join(lines)
