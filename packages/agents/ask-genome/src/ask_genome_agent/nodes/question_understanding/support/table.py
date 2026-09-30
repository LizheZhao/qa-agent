"""Rebuilds a subquery's filtered table from the filter that produced it.

A LINKEDIN spend question matches ~96k rows and the whole result has nowhere to live, while the
resolved filter is a few hundred bytes and regenerates those rows on demand.

The recipe is taken at the point the narrowing chain stops being about text and starts being
about rows. Everything before that -- resolving the classifier's answers against the columns,
defaulting the time range, the trend and 'all' rules, the keyword pass -- has already run, and
its whole output is four plain values. Everything after depends on the data, and replay re-runs
exactly that part through the same `narrow_and_apply` the live path used. So the artifact stays a
resolved table recipe: it does not store the question and does not replay Question Understanding.

This module defines and produces the representation. Where a reference to it is stored, and how
the rebuilt table reaches the browser, belong to the framework. What crosses to the frontend is
the rebuilt table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from ask_genome_agent.config import load_narrowing_config, load_ner_data
from ask_genome_agent.local import get_client_data_dir_path
from ask_genome_agent.nodes.question_understanding.support.narrowing import (
    NarrowedResult,
    narrow_and_apply,
)

logger = logging.getLogger(__name__)

_SOURCE_FILES = {"bi": "bi.csv", "br": "br.csv"}

# Bump when a change would make an already-stored recipe select different rows, which means
# anything narrow_and_apply does. Steps before the recipe boundary produce better new recipes and
# leave stored ones replaying exactly as before, so they do not need a bump.
#
# 2: the spaCy narrowing was wired in, so the boundary moved from the filter application back to
#    just before combine_filters.
# 3: reconciliation and the owned-media exclusion were wired in, both after the boundary.
# 4: the hierarchy dedupe was wired in, which drops rows outside the reporting levels.
FILTER_VERSION = 4


class StaleTableRecipeError(RuntimeError):
    """The data a recipe was built from has changed, so replaying it would answer differently."""


class TableRecipe(BaseModel):
    """Everything needed to rebuild one filtered table, and nothing of its contents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filter_version: int = FILTER_VERSION
    client_code: str
    model_group_id: int
    source: Literal["bi", "br"]

    # The four values the chain has resolved down to at the boundary. ner_filter carries the
    # column values to match, ner_results how each field was answered ('all' matches rows that
    # never carried the field), ignore_fields the columns already handled, spacy_filter the terms
    # the question mentioned.
    ner_filter: dict[str, Any]
    ner_results: dict[str, Any]
    ignore_fields: tuple[str, ...]
    spacy_filter: dict[str, list[str]]

    row_count: int
    source_mtime_ns: int
    source_size: int
    config_fingerprint: str


def source_stamp(source: str) -> tuple[int, int]:
    """The source CSV's mtime in nanoseconds and its size. One stat call, microseconds."""

    path = os.path.join(get_client_data_dir_path(), _SOURCE_FILES[source])
    info = os.stat(path)
    return info.st_mtime_ns, int(info.st_size)


def recipe_from_result(
    *,
    source: str,
    ner_filter: dict[str, Any],
    ner_results: dict[str, Any],
    ignore_fields: list[str],
    spacy_filter: dict[str, list[str]],
    row_count: int,
) -> TableRecipe:
    """Build a recipe from the four values the chain holds at the boundary."""

    mtime, size = source_stamp(source)
    return TableRecipe(
        client_code=os.environ["ASK_GENOME_CLIENT_CODE"],
        model_group_id=int(os.environ["ASK_GENOME_MODEL_GROUP_ID"]),
        source=source,  # type: ignore[arg-type]
        ner_filter=ner_filter,
        ner_results=ner_results,
        ignore_fields=tuple(ignore_fields),
        spacy_filter={k: list(v) for k, v in spacy_filter.items()},
        row_count=row_count,
        source_mtime_ns=mtime,
        source_size=size,
        config_fingerprint=load_narrowing_config().fingerprint,
    )


def _canonical(value: Any) -> Any:
    """Order-insensitive form of a recipe value, for the digest only.

    Every list here is matched with `isin` or tested for membership, so ['us', 'uk'] and
    ['uk', 'us'] select the same rows and must not produce two ids. The stored recipe keeps the
    original order, which is what the filter actually was.
    """

    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list | tuple):
        return sorted((_canonical(v) for v in value), key=repr)
    return value


def table_id(recipe: TableRecipe) -> str:
    """A stable id for this filter against this data.

    The same resolved recipe is the same logical table, so two differently worded questions that
    resolve to one filter share an id. That is the point: the frontend caches one table, not two.
    """

    canonical = json.dumps(
        _canonical(recipe.model_dump(mode="json")), sort_keys=True, separators=(",", ":")
    )
    return f"tbl-{hashlib.sha256(canonical.encode()).hexdigest()[:32]}"


def rehydrate_result(recipe: TableRecipe) -> NarrowedResult:
    """Rebuild the table the recipe describes, and the filter the rebuild resolved.

    The recipe stores the filter as it stood *before* the spaCy merge, because that is the point
    replay starts from. The merge then adds the fields the later stages branch on --
    core_dimension, core_dimension_composite, diag_tagging -- so anything downstream of the table
    wants this enriched filter, not the recipe's.

    Raises StaleTableRecipeError rather than returning a different table under an id the caller
    already holds. A wrong number presented as the right one is worse than a failure, and four
    things can make it wrong: another client's data, a rewritten source file, narrowing config
    that has moved, or filtering code that has moved since the recipe was written.
    """

    if recipe.filter_version != FILTER_VERSION:
        raise StaleTableRecipeError(
            f"recipe was written under filter version {recipe.filter_version}, "
            f"this deployment runs {FILTER_VERSION}"
        )

    if recipe.client_code != os.environ["ASK_GENOME_CLIENT_CODE"] or recipe.model_group_id != int(
        os.environ["ASK_GENOME_MODEL_GROUP_ID"]
    ):
        raise StaleTableRecipeError(
            f"recipe is for {recipe.client_code}/{recipe.model_group_id}, "
            f"this deployment serves {os.environ['ASK_GENOME_CLIENT_CODE']}/"
            f"{os.environ['ASK_GENOME_MODEL_GROUP_ID']}"
        )

    mtime, size = source_stamp(recipe.source)
    if (mtime, size) != (recipe.source_mtime_ns, recipe.source_size):
        raise StaleTableRecipeError(
            f"{recipe.source}.csv changed since this table was built "
            f"(mtime {recipe.source_mtime_ns} -> {mtime}, size {recipe.source_size} -> {size})"
        )

    config = load_narrowing_config()
    if recipe.config_fingerprint != config.fingerprint:
        raise StaleTableRecipeError(
            f"the narrowing config changed since this table was built "
            f"({recipe.config_fingerprint} -> {config.fingerprint})"
        )

    ner_data = load_ner_data()
    frame = ner_data.df_bi if recipe.source == "bi" else ner_data.df_br
    rebuilt = narrow_and_apply(
        frame,
        dict(recipe.ner_filter),
        dict(recipe.ner_results),
        list(recipe.ignore_fields),
        recipe.spacy_filter,
        config,
    )

    if len(rebuilt.frame) != recipe.row_count:
        raise StaleTableRecipeError(
            f"this table recorded {recipe.row_count} rows, current filtering produces "
            f"{len(rebuilt.frame)}. FILTER_VERSION is {FILTER_VERSION} and may need bumping."
        )
    return rebuilt


def rehydrate_table(recipe: TableRecipe) -> pd.DataFrame:
    """The table alone. `table_id` -> this -> DataFrame, which is the frontend's contract."""

    return rehydrate_result(recipe).frame
