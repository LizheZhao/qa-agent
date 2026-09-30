"""Rebuilding a filtered table from the recipe that produced it.

The invariant the whole design rests on: rehydrating a recipe returns the table the live path
returned. If that stops holding, a session's stored answer and the table the frontend shows
diverge, which is worse than storing the rows would have been.

Since the recipe boundary sits before combine_filters, that invariant now has to hold across the
spaCy narrowing too, not only the filter application.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from ask_genome_agent.config import NarrowingConfig
from ask_genome_agent.nodes.question_understanding.support import table as table_module
from ask_genome_agent.nodes.question_understanding.support.narrowing import narrow_and_apply
from ask_genome_agent.nodes.question_understanding.support.table import (
    StaleTableRecipeError,
    TableRecipe,
    recipe_from_result,
    rehydrate_table,
    table_id,
)

_DF_BI = pd.DataFrame(
    {
        "metric": ["gross roi", "activity", "gross roi", "gross roi", "gross roi"],
        "country": ["us", "us", "uk", None, "us"],
        "activity_group": ["paid search", "paid search", "print", "paid search", "digital display"],
        "business_driver": ["media", "media", "non-media", "media", "media"],
        "business_driver_detail": ["paid search"] * 2 + ["print", "paid search", "digital display"],
        "measure_group": ["sem", "sem", "print", "sem", "display"],
        "measure": ["bing", "bing", "insert", "google", "gdn"],
        "time": ["quarter 1 2024"] * 5,
        "start": [202401, 202401, 202401, 202404, 202401],
        "end": [202403, 202403, 202403, 202406, 202403],
        "period_type": ["quarter"] * 5,
        "value": [1.1, 2.2, 3.3, 4.4, 5.5],
    }
)
_DF_BR = pd.DataFrame(
    {
        "metric": ["sovc", "contribution"],
        "country": ["us", "us"],
        "activity_group": ["paid search", "print"],
        "business_driver": ["media", "base"],
        "business_driver_detail": ["paid search", "print"],
        "measure_group": ["sem", "print"],
        "measure": ["bing", "insert"],
        "time": ["quarter 1 2024"] * 2,
        "start": [202401, 202401],
        "end": [202403, 202403],
        "period_type": ["quarter", "quarter"],
        "value": [5.5, 6.6],
    }
)

_CORE_FILTERS: dict[str, Any] = {
    "paid search": [{"source": "bibr", "activity_group": ["paid search"]}],
    "digital display": [{"source": "bibr", "activity_group": ["digital display"]}],
    "united states": [{"source": "bibr", "country": ["us"]}],
}
_LEVEL_TYPE = {"general_level": ["country"], "halo_level": [], "halo_tagging": []}


def _config(**overrides: Any) -> NarrowingConfig:
    values: dict[str, Any] = {
        "client_code": "TESTCO",
        "metric_info": {},
        "core_filters": _CORE_FILTERS,
        "level_type": _LEVEL_TYPE,
        "sub_cols": ("activity_group",),
        "bi_levels": ("activity_group", "measure_group", "measure"),
        "br_levels": ("business_driver", "business_driver_detail", "activity_group"),
        "fingerprint": "cfg0",
    }
    values.update(overrides)
    return NarrowingConfig(**values)


class _Frames:
    df_bi = _DF_BI
    df_br = _DF_BR


@pytest.fixture
def _stub_source(monkeypatch) -> None:
    monkeypatch.setenv("ASK_GENOME_CLIENT_CODE", "TESTCO")
    monkeypatch.setenv("ASK_GENOME_MODEL_GROUP_ID", "1")
    monkeypatch.setattr(table_module, "load_ner_data", _Frames)
    monkeypatch.setattr(table_module, "source_stamp", lambda source: (111, 222))
    monkeypatch.setattr(table_module, "load_narrowing_config", _config)


def _boundary(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "source": "bi",
        "ner_filter": {
            "intention": ["margin roi"],
            "metric": ["gross roi"],
            "main_metric": ["gross roi"],
            "country": ["us"],
        },
        "ner_results": {"country": "us"},
        "ignore_fields": [],
        "spacy_filter": {},
    }
    values.update(overrides)
    return values


def _live(boundary: dict[str, Any], config: NarrowingConfig | None = None) -> pd.DataFrame:
    """Run the live path the node runs, from the boundary values."""

    frame = _DF_BI if boundary["source"] == "bi" else _DF_BR
    return narrow_and_apply(
        frame,
        boundary["ner_filter"],
        boundary["ner_results"],
        boundary["ignore_fields"],
        boundary["spacy_filter"],
        config or _config(),
    ).frame


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "boundary"),
    [
        ("ordinary filter", _boundary()),
        # 'all' widens to rows that never carried the dimension, so the None row comes back too.
        (
            "widened by all",
            _boundary(
                ner_filter={
                    "intention": ["margin roi"],
                    "metric": ["gross roi"],
                    "country": ["us", "uk"],
                },
                ner_results={"country": "all"},
            ),
        ),
        ("ignored field", _boundary(ignore_fields=["country"])),
        (
            "time filter",
            _boundary(
                ner_filter={
                    "intention": ["margin roi"],
                    "metric": ["gross roi"],
                    "start": [202401],
                    "end": [202403],
                },
                ner_results={},
            ),
        ),
        ("br source", _boundary(source="br")),
        # The cases the boundary move exists for: the frame is narrowed by a captured term, not
        # by anything in ner_filter.
        (
            "narrowed by one capture",
            _boundary(
                ner_filter={
                    "intention": ["margin roi"],
                    "metric": ["gross roi"],
                    "main_metric": ["gross roi"],
                },
                ner_results={},
                spacy_filter={"core_dimension": ["digital display"]},
            ),
        ),
        (
            "narrowed by two captures",
            _boundary(
                ner_filter={
                    "intention": ["margin roi"],
                    "metric": ["gross roi"],
                    "main_metric": ["gross roi"],
                },
                ner_results={},
                spacy_filter={"core_dimension": ["paid search", "digital display"]},
            ),
        ),
        (
            "a capture and a filter",
            _boundary(spacy_filter={"core_dimension": ["paid search"]}),
        ),
        # The capture pins country, which is what get_specific_values does with the narrowed frame.
        (
            "a capture that pins a level",
            _boundary(
                ner_filter={
                    "intention": ["margin roi"],
                    "metric": ["gross roi"],
                    "main_metric": ["gross roi"],
                },
                ner_results={"country": "specific"},
                spacy_filter={"core_dimension": ["united states"]},
            ),
        ),
        (
            "a capture that selects nothing",
            _boundary(spacy_filter={"core_dimension": ["print"]}),
        ),
    ],
)
def test_rehydrating_returns_the_table_the_live_path_returned(
    _stub_source, label, boundary
) -> None:
    del label
    live = _live(boundary)

    recipe = recipe_from_result(**boundary, row_count=len(live))

    pd.testing.assert_frame_equal(rehydrate_table(recipe), live)


@pytest.mark.unit
def test_the_narrowing_is_really_doing_something(_stub_source) -> None:
    """Guards the cases above against becoming vacuous. If a captured term stopped narrowing, the
    parametrised cases would still pass while proving nothing."""

    plain = _boundary(
        ner_filter={
            "intention": ["margin roi"],
            "metric": ["gross roi"],
            "main_metric": ["gross roi"],
        },
        ner_results={},
    )
    captured = {**plain, "spacy_filter": {"core_dimension": ["digital display"]}}

    assert len(_live(plain)) == 4
    assert len(_live(captured)) == 1


@pytest.mark.unit
def test_how_a_field_was_answered_is_what_makes_replay_faithful(_stub_source) -> None:
    """The one thing apply_answer_dict reads out of ner_results. Dropping it looks harmless and
    silently returns fewer rows, because 'all' also matches rows that never carried the field."""

    widened = _boundary(
        ner_filter={"intention": ["margin roi"], "metric": ["gross roi"], "country": ["us", "uk"]},
        ner_results={"country": "all"},
    )
    narrow = {**widened, "ner_results": {}}
    assert len(_live(widened)) > len(_live(narrow)), "the fixture needs a row without a country"

    recipe = recipe_from_result(**widened, row_count=len(_live(widened)))
    pd.testing.assert_frame_equal(rehydrate_table(recipe), _live(widened))

    # Dropping it would replay the narrow result, which the row-count guard now catches rather
    # than returning a smaller table under the same id.
    without = recipe.model_copy(update={"ner_results": {}})
    with pytest.raises(StaleTableRecipeError, match="current filtering produces"):
        rehydrate_table(without)


@pytest.mark.unit
def test_the_same_filter_is_the_same_table_whatever_order_it_is_written_in(_stub_source) -> None:
    """apply_answer_dict matches with isin, so value order selects the same rows. Two ids for one
    table would have the frontend caching it twice."""

    one = recipe_from_result(
        **_boundary(ner_filter={"intention": ["margin roi"], "country": ["us", "uk"]}), row_count=1
    )
    other = recipe_from_result(
        **_boundary(ner_filter={"intention": ["margin roi"], "country": ["uk", "us"]}), row_count=1
    )

    assert table_id(one) == table_id(other)


@pytest.mark.unit
def test_the_same_captures_in_another_order_are_the_same_table(_stub_source) -> None:
    """Captured terms OR together, so their order does not select different rows either."""

    one = recipe_from_result(
        **_boundary(spacy_filter={"core_dimension": ["paid search", "digital display"]}),
        row_count=1,
    )
    other = recipe_from_result(
        **_boundary(spacy_filter={"core_dimension": ["digital display", "paid search"]}),
        row_count=1,
    )

    assert table_id(one) == table_id(other)


@pytest.mark.unit
def test_a_different_filter_is_a_different_table(_stub_source) -> None:
    one = recipe_from_result(
        **_boundary(ner_filter={"intention": ["margin roi"], "country": ["us"]}), row_count=1
    )
    other = recipe_from_result(
        **_boundary(ner_filter={"intention": ["margin roi"], "country": ["uk"]}), row_count=1
    )

    assert table_id(one) != table_id(other)


@pytest.mark.unit
def test_different_captures_are_a_different_table(_stub_source) -> None:
    """The bug the narrowing was ported to fix: four questions used to collapse to two ids
    because the captured terms never reached the recipe."""

    one = recipe_from_result(
        **_boundary(spacy_filter={"core_dimension": ["paid search"]}), row_count=1
    )
    other = recipe_from_result(
        **_boundary(spacy_filter={"core_dimension": ["digital display"]}), row_count=1
    )

    assert table_id(one) != table_id(other)


@pytest.mark.unit
def test_a_recipe_survives_the_json_it_is_stored_as(_stub_source) -> None:
    """It is checkpointed as plain JSON, so it has to come back as the same recipe."""

    recipe = recipe_from_result(
        **_boundary(spacy_filter={"core_dimension": ["paid search"]}), row_count=2
    )

    restored = TableRecipe.model_validate(recipe.model_dump(mode="json"))

    assert restored == recipe
    assert table_id(restored) == table_id(recipe)


@pytest.mark.unit
def test_an_unordered_field_list_is_the_same_table(_stub_source) -> None:
    """ignore_fields is membership-tested, so order must not split one table into two ids."""

    one = recipe_from_result(**_boundary(ignore_fields=["country", "kpi"]), row_count=1)
    other = recipe_from_result(**_boundary(ignore_fields=["kpi", "country"]), row_count=1)

    assert table_id(one) == table_id(other)


@pytest.mark.unit
def test_filtering_that_moved_refuses_to_replay(_stub_source) -> None:
    """The backstop for a forgotten FILTER_VERSION bump. Same client, same file, same version, a
    different answer: the filtering itself changed, so the recorded table cannot be rebuilt."""

    recipe = recipe_from_result(**_boundary(), row_count=999)

    with pytest.raises(StaleTableRecipeError, match="current filtering produces"):
        rehydrate_table(recipe)


@pytest.mark.unit
def test_a_recipe_from_older_filtering_refuses_to_replay(_stub_source) -> None:
    """The declared signal, for when the semantics change deliberately."""

    recipe = recipe_from_result(**_boundary(), row_count=2)
    older = recipe.model_copy(update={"filter_version": recipe.filter_version - 1})

    with pytest.raises(StaleTableRecipeError, match="filter version"):
        rehydrate_table(older)


@pytest.mark.unit
def test_changed_source_data_refuses_to_replay(_stub_source, monkeypatch) -> None:
    """Returning a different table under an id the caller already holds is worse than failing."""

    recipe = recipe_from_result(**_boundary(), row_count=2)

    monkeypatch.setattr(table_module, "source_stamp", lambda source: (999, 222))
    with pytest.raises(StaleTableRecipeError, match="changed since"):
        rehydrate_table(recipe)


@pytest.mark.unit
def test_changed_narrowing_config_refuses_to_replay(_stub_source, monkeypatch) -> None:
    """A retagged term selects different rows for the same captured word, and the file's
    timestamp would not have to move for that to happen."""

    recipe = recipe_from_result(
        **_boundary(spacy_filter={"core_dimension": ["paid search"]}), row_count=3
    )

    monkeypatch.setattr(
        table_module,
        "load_narrowing_config",
        lambda: _config(
            core_filters={"paid search": [{"source": "bibr", "activity_group": ["print"]}]},
            fingerprint="cfg1",
        ),
    )
    with pytest.raises(StaleTableRecipeError, match="narrowing config changed"):
        rehydrate_table(recipe)


def _real_fingerprint(monkeypatch, core_filters: dict[str, Any], sub_cols: list[str]) -> str:
    """load_narrowing_config's own fingerprint, over config it was handed."""

    from ask_genome_agent import config as config_module

    monkeypatch.setenv("ASK_GENOME_CLIENT_CODE", "TESTCO")
    monkeypatch.setattr(config_module, "load_ner_data", lambda: _Frames)
    monkeypatch.setattr(
        config_module, "load_spacy_config", lambda: type("C", (), {"core_filters": core_filters})
    )
    monkeypatch.setattr(
        config_module,
        "load_ner_data_config",
        lambda: type(
            "C",
            (),
            {
                "data_levels": {"spacyOverrideColumns": sub_cols},
                "metric_info": {},
                "biLevels": [],
                "brLevels": [],
            },
        ),
    )
    config_module.load_narrowing_config.cache_clear()
    try:
        return config_module.load_narrowing_config().fingerprint
    finally:
        config_module.load_narrowing_config.cache_clear()


@pytest.mark.unit
def test_the_fingerprint_follows_the_config_content(monkeypatch) -> None:
    """A content hash, not a file stamp: config rewritten unchanged still replays, and a change
    that reaches these values invalidates recipes however it arrived."""

    _Frames.level_type = _LEVEL_TYPE
    same = _real_fingerprint(monkeypatch, dict(_CORE_FILTERS), ["activity_group"])
    again = _real_fingerprint(monkeypatch, dict(_CORE_FILTERS), ["activity_group"])
    retagged = _real_fingerprint(
        monkeypatch,
        {**_CORE_FILTERS, "paid search": [{"source": "bibr", "activity_group": ["print"]}]},
        ["activity_group"],
    )
    other_sub_cols = _real_fingerprint(monkeypatch, dict(_CORE_FILTERS), ["country"])

    assert same == again
    assert len({same, retagged, other_sub_cols}) == 3


@pytest.mark.unit
def test_another_client_refuses_to_replay(_stub_source, monkeypatch) -> None:
    recipe = recipe_from_result(**_boundary(), row_count=2)

    monkeypatch.setenv("ASK_GENOME_CLIENT_CODE", "SOMEONEELSE")
    with pytest.raises(StaleTableRecipeError, match="this deployment serves"):
        rehydrate_table(recipe)
