"""Per-node tests for the postprocess stage.

What is pinned here is the node's own job: rebuild the table from the recipe rather than from
state, hand upstream's process_data the filter it expects, keep frames out of the checkpoint, and
record a stale recipe instead of answering around it. The adapter has its own tests.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import pytest
from ask_genome_agent.contracts import AskGenomePlan, Subquery, SubqueryKind
from ask_genome_agent.nodes.postprocess import run_postprocess as module
from ask_genome_agent.nodes.postprocess.run_postprocess import run_postprocess
from ask_genome_agent.nodes.postprocess.support.contract import (
    PostprocessResult,
    PostprocessWarning,
    SelectionMetadata,
)
from ask_genome_agent.nodes.question_understanding.support.table import StaleTableRecipeError

pytestmark = pytest.mark.unit


class _NarrowedStub:
    """What rehydrate_result returns: the frame plus the filter the rebuild resolved.

    The enriched filter is the point: the recipe's own predates the spaCy merge.
    """

    def __init__(self, frame: pd.DataFrame, ner_filter: dict[str, Any]) -> None:
        self.frame = pd.DataFrame({"value": [1.0], "time": ["year 2025"], "period_type": ["year"]})
        self.ner_filter = {**ner_filter, "core_dimension": {"activity_group": ["paid search"]}}
        self.ner_results = {"intention": ["margin roi"], "rank": "na"}
        self.ignore_fields: list[str] = []
        self.data_levels = ("activity_group", "measure_group", "measure")


def _recipe() -> dict[str, Any]:
    return {
        "filter_version": 4,
        "client_code": "LINKEDIN",
        "model_group_id": 12,
        "source": "bi",
        "ner_filter": {"intention": ["margin roi"], "main_metric": ["gross roi"]},
        "ner_results": {"intention": ["margin roi"], "rank": "na"},
        "ignore_fields": [],
        "spacy_filter": {},
        "row_count": 12,
        "source_mtime_ns": 1,
        "source_size": 2,
        "config_fingerprint": "abc",
    }


def _state(**entry: Any) -> dict[str, Any]:
    subquery = Subquery(id=0, query="q", kind=SubqueryKind.analytical, coarse_intent="margin roi")
    plan = AskGenomePlan(
        original_query="q",
        rephrased_query="q",
        is_multi=False,
        is_sequential=False,
        subqueries=(subquery,),
    )
    entry.setdefault("table_recipe", _recipe())
    entry.setdefault("data_levels", ["activity_group", "measure_group", "measure"])
    entry.setdefault("row_count", 12)
    return {"plan": plan, "active_index": 0, "ner_state": {0: entry}}


def _result(**overrides: Any) -> PostprocessResult:
    base: dict[str, Any] = {
        "response_context": "the answer.",
        "aggregate_table_text": "agg rows",
        "detail_table_text": "detail rows",
        "selection_metadata": SelectionMetadata(
            detail_level="m", aggregate_level="ag", search_key="N~Y~N~Y~N~Y", matched={"x": 1}
        ),
        "principle_pretext": None,
        "warnings": (
            PostprocessWarning(
                stage="selection", code="no_lookup_match", message="no lookup row matched"
            ),
        ),
    }
    base.update(overrides)
    return PostprocessResult(**base)


@pytest.fixture
def _stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the loaders and the pipeline; the node's own logic is what is under test."""

    captured: dict[str, Any] = {}

    def fake_postprocess(*args: Any, **kwargs: Any) -> PostprocessResult:
        captured["request"], captured["client_code"], captured["model_group_id"] = args[:3]
        result: PostprocessResult = captured.get("result", _result())
        return result

    monkeypatch.setattr(
        module,
        "rehydrate_result",
        lambda recipe: _NarrowedStub(pd.DataFrame({"value": [1.0]}), dict(recipe.ner_filter)),
    )
    monkeypatch.setattr(module, "postprocess", fake_postprocess)
    monkeypatch.setattr(module, "load_narrowing_config", lambda: _FakeNarrowing())
    return captured


class _FakeNarrowing:
    """The client's hierarchy and tagging columns, which the enriched filter carries."""

    level_type: ClassVar[dict[str, Any]] = {
        "general_level": ["country"],
        "halo_level": [],
        "halo_tagging": [],
    }


async def test_the_table_comes_from_the_recipe_not_from_state(_stub: dict) -> None:
    """The filtered rows are never in state, only a recipe that rebuilds them."""

    await run_postprocess(_state())
    request = _stub["request"]
    assert isinstance(request.table, pd.DataFrame)
    assert request.source == "bi"
    # The enriched filter, not the recipe's. The recipe stores the filter as it stood before the
    # spaCy merge, so core_dimension is absent from it, and core_dimension decides whether the
    # overall-figure branch runs at all. Reading the recipe's directly is the live bug this pins.
    assert request.ner_filter["core_dimension"] == {"activity_group": ["paid search"]}
    assert "core_dimension" not in _recipe()["ner_filter"]
    # Carried from the entry: computed before narrowing, so not part of the filter.
    assert "latest_time" in request.ner_filter


async def test_the_time_tag_reaches_the_filter_and_the_results(_stub: dict) -> None:
    """The source keeps it in both. The GPT readout picks its change and trend tables by the
    filter's copy, and br growth reads the results' copy, so without it both treat every
    question as a snapshot."""

    await run_postprocess(_state(time_tag="multiple_periods"))
    request = _stub["request"]
    assert request.ner_filter["time_tag"] == "multiple_periods"
    assert request.ner_results["time_tag"] == "multiple_periods"


async def test_a_missing_time_tag_reads_as_a_snapshot(_stub: dict) -> None:
    """What the source's own no-data path uses."""

    await run_postprocess(_state())
    assert _stub["request"].ner_filter["time_tag"] == "snapshot"


async def test_applied_filter_is_not_read(_stub: dict) -> None:
    """It is a display subset and omits fields the pipeline needs, so a wrong one must not
    change the result."""

    state = _state(applied_filter={"country": ["nonsense"]})
    await run_postprocess(state)
    assert "nonsense" not in str(_stub["request"].ner_filter)


async def test_no_recipe_is_not_an_error(_stub: dict) -> None:
    """An out-of-scope or rejected subquery never filtered anything; there is nothing to do."""

    state = _state()
    state["ner_state"][0].pop("table_recipe")
    assert await run_postprocess(state) == {}


async def test_state_gets_text_and_metadata_only(_stub: dict) -> None:
    update = await run_postprocess(_state())
    entry = update["ner_state"][0]
    assert entry["response_context"] == "the answer."
    assert entry["selection"]["detail_level"] == "m"
    assert entry["selection"]["search_key"] == "N~Y~N~Y~N~Y"
    for value in entry.values():
        assert not isinstance(value, pd.DataFrame)


async def test_principle_pretext_stays_none_in_state(_stub: dict) -> None:
    """Converting it to "" here would lose the distinction the contract exists to keep."""

    update = await run_postprocess(_state())
    assert update["ner_state"][0]["principle_pretext"] is None


async def test_warnings_are_carried_into_state(_stub: dict) -> None:
    update = await run_postprocess(_state())
    warnings = update["ner_state"][0]["postprocess_warnings"]
    assert warnings == [
        {
            "stage": "selection",
            "code": "no_lookup_match",
            "message": "no lookup row matched",
            "level": None,
        }
    ]


async def test_long_context_is_truncated_rather_than_failing_the_write(_stub: dict) -> None:
    """State is checkpointed into one capped document, as the sampled records already are."""

    _stub["result"] = _result(response_context="x" * 50_000, detail_table_text="y" * 50_000)
    update = await run_postprocess(_state())
    entry = update["ner_state"][0]
    assert len(entry["response_context"]) == module._MAX_CONTEXT
    assert len(entry["detail_table_text"]) == module._MAX_TABLE_TEXT


async def test_a_degraded_selection_reaches_the_dependent_subquery(_stub: dict) -> None:
    _stub["result"] = _result(
        selection_metadata=SelectionMetadata(detail_level="mg", degraded_levels=("m",))
    )
    update = await run_postprocess(_state())
    assert "degraded" in update["last_subquery_result"]
    assert "mg" in update["last_subquery_result"]


async def test_an_empty_result_says_so(_stub: dict) -> None:
    _stub["result"] = _result(response_context="")
    update = await run_postprocess(_state())
    assert "no level produced a readout" in update["last_subquery_result"]


async def test_a_stale_recipe_is_recorded_not_answered_around(
    monkeypatch: pytest.MonkeyPatch, _stub: dict
) -> None:
    """The recipe no longer describes the table its id promises, so there is nothing to report."""

    def raise_stale(recipe: Any) -> Any:
        raise StaleTableRecipeError("source file changed")

    monkeypatch.setattr(module, "rehydrate_result", raise_stale)
    update = await run_postprocess(_state())
    entry = update["ner_state"][0]
    assert "stale table recipe" in entry["postprocess_error"]
    assert "response_context" not in entry


async def test_the_client_follows_the_recipe(_stub: dict) -> None:
    """Which client's config process_data reads is a property of the table, not of the process."""

    await run_postprocess(_state())
    assert (_stub["client_code"], _stub["model_group_id"]) == ("LINKEDIN", 12)


# --------------------------------------------------- a failure must not read as a success


def _after_postprocess(update: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """The state write_context sees, as the graph hands it over."""

    merged = dict(state)
    merged.update(update)
    merged.setdefault("context", {})
    merged.setdefault("resolution", None)
    return merged


async def test_a_stale_recipe_does_not_leave_the_row_count_standing(
    monkeypatch: pytest.MonkeyPatch, _stub: dict
) -> None:
    """The worse half of the same failure.

    On a stale recipe the node writes no last_subquery_result, so the filtering step's row-count
    line would survive into context as a successful-looking analytical outcome for a subquery
    that produced nothing.
    """

    from ask_genome_agent.nodes.write_context import write_context

    def raise_stale(recipe: Any) -> Any:
        raise StaleTableRecipeError("source file changed")

    monkeypatch.setattr(module, "rehydrate_result", raise_stale)
    state = _state()
    state["last_subquery_result"] = "12 rows matched for intention=margin roi"
    update = await run_postprocess(state)
    written = write_context(_after_postprocess(update, state))

    assert written["context"][0] == ""
    assert "stale table recipe" in written["unanswered"][0]


async def test_a_successful_subquery_is_still_written_normally(_stub: dict) -> None:
    """The guard must not swallow real results."""

    from ask_genome_agent.nodes.write_context import write_context

    state = _state()
    update = await run_postprocess(state)
    written = write_context(_after_postprocess(update, state))

    assert written["context"][0] == update["last_subquery_result"]
    assert 0 not in written["unanswered"]


# ------------------------------------- what a dependent subquery is actually given


async def test_the_dependent_subquery_receives_the_analytical_statement(_stub: dict) -> None:
    """resolve_dependency hands this to an LLM to work out what 'its' refers to in a follow-up.

    Metadata alone cannot bind the pronoun: 'answered at level mg from 123 rows' names no channel,
    so the resolution either refuses or invents one. The readout has to travel.
    """

    _stub["result"] = _result(
        readout="Paid Search had the highest ROI at 1.24 in Q1 2025, ahead of Email at 0.98.",
        response_context="Paid Search had the highest ROI...\nFirst, analyze this high level...",
    )
    update = await run_postprocess(_state())
    carried = update["last_subquery_result"]

    assert "Paid Search" in carried, "the follow-up needs the driver named"
    assert "1.24" in carried
    # Provenance is still there, just not instead of the content.
    assert "answered at level m" in carried


async def test_the_prompt_scaffolding_does_not_travel(_stub: dict) -> None:
    """response_context is prompt-shaped; feeding its instructions into the resolution prompt
    would be noise, so the readout is kept apart from it."""

    _stub["result"] = _result(
        readout="Paid Search led.",
        response_context="Paid Search led.\nFirst, analyze this high level aggregated market data:",
    )
    carried = (await run_postprocess(_state()))["last_subquery_result"]
    assert "analyze this high level" not in carried


async def test_the_carried_readout_is_bounded(_stub: dict) -> None:
    """It goes into a prompt, so an unbounded readout would crowd out the question itself."""

    _stub["result"] = _result(readout="x" * 40_000)
    carried = (await run_postprocess(_state()))["last_subquery_result"]
    assert len(carried) < module._MAX_CARRIED_READOUT + 200


async def test_a_degraded_selection_is_still_flagged_alongside_the_content(_stub: dict) -> None:
    _stub["result"] = _result(
        readout="Paid Search led.",
        selection_metadata=SelectionMetadata(detail_level="mg", degraded_levels=("m",)),
    )
    carried = (await run_postprocess(_state()))["last_subquery_result"]
    assert "Paid Search led." in carried
    assert "degraded" in carried


async def test_context_reaching_dependency_resolution_can_bind_a_pronoun(_stub: dict) -> None:
    """The whole chain for case 4: postprocess -> write_context -> what resolve_dependency reads."""

    from ask_genome_agent.nodes.write_context import write_context

    _stub["result"] = _result(readout="Paid Search had the highest ROI at 1.24 in Q1 2025.")
    state = _state()
    update = await run_postprocess(state)
    written = write_context(_after_postprocess(update, state))

    predecessor_context = written["context"][0]
    assert "Paid Search" in predecessor_context


async def test_an_empty_result_is_not_reported_as_an_answer(_stub: dict) -> None:
    """The live run's second finding: postprocess produced nothing and the turn still reported
    'Found 2205 matching rows'.

    An empty result is not an exception, so the postprocess_error guard did not fire and the
    filtering step's row count stood as the subquery's outcome.
    """

    from ask_genome_agent.nodes.write_context import write_context

    _stub["result"] = _result(
        response_context="",
        warnings=(
            PostprocessWarning(
                stage="level_pipeline", code="empty_result", message="no level produced a readout"
            ),
        ),
    )
    state = _state()
    state["last_subquery_result"] = "2205 rows matched for intention=margin roi"
    update = await run_postprocess(state)
    written = write_context(_after_postprocess(update, state))

    assert written["context"][0] == "", "the row count must not stand in as the answer"
    assert 0 in written["unanswered"]
    assert "no level produced a readout" in written["unanswered"][0]
