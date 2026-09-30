"""Deterministic: calls the external BERT multi-task classifier service for one subquery.

Ported from ask-genome-core/src/integrations/classifier.py and
filter_generator_refactor.py:process_bert_predictions, made async because the source's
`requests.post` is synchronous and would block the event loop inside this graph node. spaCy's
synchronous load is the same issue, still open. Follows the injected-client / explicit-timeout /
close-only-if-owned pattern from enterprise_llm.adapter.
"""

import os
from typing import Any

import httpx
import pandas as pd

from ask_genome_agent.config import load_ner_data
from ask_genome_agent.state import AskGenomeState

_TIMEOUT_SECONDS = 30.0


class ClassifierServiceError(RuntimeError):
    """The BERT classifier service call failed."""


async def classify_query(
    query: str, client: httpx.AsyncClient | None = None
) -> dict[str, dict[str, float]]:
    """Returns {field: {class_label: probability, ...}, ...} for one query."""

    endpoint = f"{os.environ['CLASSIFIER_SERVICE_URL']}/predict"
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)
    try:
        response = await http_client.post(endpoint, json={"texts": [query]})
        if response.status_code != 200:
            raise ClassifierServiceError(
                f"Classifier request failed with status {response.status_code}: {response.text}"
            )
        response_data = response.json()
    except httpx.HTTPError as exc:
        raise ClassifierServiceError(f"Classifier request transport failure: {exc}") from exc
    finally:
        if owns_client:
            await http_client.aclose()

    row = response_data["predictions"][0]
    return {field: pred["probabilities"] for field, pred in row.items()}


def process_bert_predictions(
    probabilities: dict[str, Any], df_bi: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ported from filter_generator_refactor.py:process_bert_predictions."""

    probabilities = dict(probabilities)
    pred_bd = probabilities.pop("business_driver_class", None)
    if pred_bd:
        probabilities["business_driver"] = pred_bd

    pred_period = probabilities.get("period_type")
    is_fiscal_calendar = bool(len(df_bi)) and "fiscal" in str(df_bi["time"].iloc[0])
    if pred_period and not is_fiscal_calendar:
        probabilities["period_type"] = {
            ("half year" if (stripped := k.replace("fiscal ", "")) == "half" else stripped): v
            for k, v in pred_period.items()
        }

    predictions = {field: max(vals, key=vals.get) for field, vals in probabilities.items()}
    return predictions, probabilities


async def classify_fields(state: AskGenomeState) -> dict[str, Any]:
    subquery = state["plan"].subqueries[state["active_index"]]
    ner_state = dict(state.get("ner_state", {}))
    entry = dict(ner_state[subquery.id])

    raw_probabilities = await classify_query(entry["extended_query"])
    predictions, probabilities = process_bert_predictions(raw_probabilities, load_ner_data().df_bi)

    entry["predictions"] = predictions
    entry["probabilities"] = probabilities
    ner_state[subquery.id] = entry
    return {"ner_state": ner_state}
