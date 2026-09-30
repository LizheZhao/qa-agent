"""Deterministic Decompose and Classify Step:
Makes one structured LLM call to process the raw question, where it:
    1. Rephrases the question if needed.
    2. Splits it into multiple subqueries when necessary.
    3. Classifies each subquery as coverage or analytical.
    4. Assigns the appropriate coarse_intent.
    5. Determines depends_on relationships between subqueries.
    6. Mirrors the existing planner_llm_call behavior from ask-genome-core.
    7. Uses with_structured_output(..., method="function_calling") because the enterprise
       gateway supports structured output through tool/function calling.
"""

from typing import Any

from orchestration_core import AgentDependencies
from pydantic import BaseModel

from ask_genome_agent.config import load_intent_catalog
from ask_genome_agent.contracts import build_planner_output_model
from ask_genome_agent.prompts import PLANNER_PROMPT
from ask_genome_agent.state import AskGenomeState


async def decompose_and_classify(
    state: AskGenomeState, dependencies: AgentDependencies
) -> dict[str, Any]:
    intent_catalog = load_intent_catalog()
    schema = build_planner_output_model(intent_catalog)
    structured_model = dependencies.model.with_structured_output(schema, method="function_calling")
    prompt = PLANNER_PROMPT.replace("INTENT_CATALOG_PLACEHOLDER", ", ".join(intent_catalog))
    result = await structured_model.ainvoke([("system", prompt), ("user", state["query"])])
    if not isinstance(result, BaseModel):
        raise TypeError("decompose_and_classify expected a structured pydantic result")
    return {"plan_output": result.model_dump()}
