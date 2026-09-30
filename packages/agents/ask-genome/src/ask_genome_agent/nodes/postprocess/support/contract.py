"""What postprocessing is given and what it returns.

No DataFrame crosses this boundary: a LINKEDIN spend question matches ~96k rows, and putting
those in graph state blew the checkpoint limit.

Absence and emptiness stay apart throughout: None means nothing produced it, "" means something
did and had nothing to say.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

WarningCode = Literal["no_lookup_match", "empty_result", "insight_fallback"]

WarningStage = Literal["level_pipeline", "selection", "insight"]


@dataclass(frozen=True)
class PostprocessWarning:
    """Something the answer should be read in light of, not an error.

    Records what process_data does silently, so a reader can tell a thin answer from a broken one.
    """

    stage: WarningStage
    code: WarningCode
    message: str
    level: str | None = None
    """'ag' | 'mg' | 'm' where it belongs to one level."""


@dataclass(frozen=True)
class SelectionMetadata:
    """Which level answered, and how that was decided."""

    detail_level: str = ""
    aggregate_level: str = ""
    overall_level: str = ""
    readout_levels: tuple[str, ...] = ()
    search_key: str = ""
    """The Y/N key the lookup was searched by, so a selection can be explained."""

    degraded_levels: tuple[str, ...] = ()
    """Levels that failed and so read as empty when the key was built. Always empty today:
    process_data only ever skips a level with no valid data, and re-raises anything else."""

    matched: dict[str, Any] = field(default_factory=dict)
    """The lookup row that applied. Empty when none did."""


@dataclass(frozen=True)
class PostprocessResult:
    """Everything postprocessing produces, for one subquery."""

    response_context: str
    """What response generation is given: the readout plus instructions and tables."""

    readout: str = ""
    """The analytical statement alone, without scaffolding or tables.

    Kept apart because a dependent subquery resolving "how did *its* spend change" needs the
    sentence naming the channel, not the analysis instructions.
    """

    aggregate_table_text: str = ""
    detail_table_text: str = ""
    selection_metadata: SelectionMetadata = field(default_factory=SelectionMetadata)
    principle_pretext: str | None = None
    """None when nothing generated it; "" when generation ran and produced nothing."""

    benchmark_text: str = ""
    fixtext: str = ""
    """The overall figure, reported alongside the answer rather than inside it."""

    is_planner_answer: bool = False
    """Built from published scenarios. The answer leads and the notice follows, not the reverse."""

    insight_query: str = ""
    """What the model is asked. Not always the user's question: upstream swaps in a fixed request
    when the readout carries no table, and prefixes a planner question with its instructions."""

    insight_context: str = ""
    """What the model reads: the GPT readout built from the display tables, or the readout above
    where that could not be built."""

    answerable: bool = False
    """Some table or scenario came back, so the model is asked at all. Upstream's own test."""

    warnings: tuple[PostprocessWarning, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Nothing to report at all. A pretext alone counts: the source shows it when there is no
        readout, and for a planner question it is the message saying the tool cannot answer."""

        return not self.response_context.strip() and not self.fixtext.strip()


@dataclass(frozen=True)
class PostprocessRequest:
    """One subquery's input: the rebuilt table and the filter that produced it.

    Never `applied_filter`, which is a display subset missing fields the pipeline branches on.
    """

    table: pd.DataFrame
    ner_filter: dict[str, Any]
    ner_results: dict[str, Any]
    data_levels: tuple[str, ...]
    source: str = "bi"
    """'bi' or 'br': which of the client's two tables the filter resolved against."""

    query: str = ""
    """The subquery as asked. The GPT readout puts it in front of its tables."""
