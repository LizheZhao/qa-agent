"""MongoDB persistence for normalized execution observability records."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

from orchestration_core import ExecutionSpan, ExecutionTrace, GraphDefinition
from pydantic import ValidationError
from pymongo import ASCENDING, DESCENDING, UpdateOne
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import CollectionInvalid, PyMongoError

from agentic_orchestration.observability.store import ExecutionSnapshot

EXECUTION_TRACES_COLLECTION = "execution_traces"
EXECUTION_SPANS_COLLECTION = "execution_spans"
GRAPH_DEFINITIONS_COLLECTION = "graph_definitions"


class ExecutionTraceStorageError(RuntimeError):
    """An execution record could not be persisted without losing ordering guarantees."""


class MongoExecutionTraceStore:
    """Persist monotonic trace/span snapshots and immutable graph definitions."""

    def __init__(self, database: AsyncDatabase[dict[str, Any]]) -> None:
        self._database = database
        self._traces = database[EXECUTION_TRACES_COLLECTION]
        self._spans = database[EXECUTION_SPANS_COLLECTION]
        self._definitions = database[GRAPH_DEFINITIONS_COLLECTION]

    async def initialize(self) -> None:
        try:
            existing = set(await self._database.list_collection_names())
            for collection_name in (
                EXECUTION_TRACES_COLLECTION,
                EXECUTION_SPANS_COLLECTION,
                GRAPH_DEFINITIONS_COLLECTION,
            ):
                if collection_name not in existing:
                    with suppress(CollectionInvalid):
                        await self._database.create_collection(collection_name)
            await self._traces.create_index(
                [
                    ("session_id", ASCENDING),
                    ("attempted_turn_number", DESCENDING),
                    ("started_at", DESCENDING),
                    ("trace_id", DESCENDING),
                ],
                name="execution_trace_session_attempts",
            )
            await self._traces.create_index(
                [("started_at", DESCENDING), ("trace_id", DESCENDING)],
                name="execution_trace_recent_attempts",
            )
            await self._spans.create_index(
                [("trace_id", ASCENDING), ("sequence_started", ASCENDING)],
                name="execution_span_trace_sequence",
            )
            await self._spans.create_index(
                [("trace_id", ASCENDING), ("parent_span_id", ASCENDING)],
                name="execution_span_trace_parent",
            )
            await self._definitions.create_index(
                [
                    ("agent_id", ASCENDING),
                    ("agent_version", ASCENDING),
                    ("definition_hash", ASCENDING),
                ],
                name="graph_definition_identity",
                unique=True,
            )
        except PyMongoError as exc:
            raise ExecutionTraceStorageError("Could not initialize execution storage") from exc

    async def save_snapshots(self, snapshots: Sequence[ExecutionSnapshot]) -> None:
        trace_operations: list[Any] = []
        span_operations: list[Any] = []
        for snapshot in snapshots:
            if isinstance(snapshot, ExecutionTrace):
                document_id = snapshot.trace_id
                target = trace_operations
            else:
                document_id = f"{snapshot.trace_id}:{snapshot.span_id}"
                target = span_operations
            document = {
                "_id": document_id,
                **snapshot.model_dump(mode="python", exclude_none=True),
            }
            target.append(_monotonic_snapshot_upsert(document, snapshot.snapshot_sequence))
        try:
            writes = []
            if trace_operations:
                writes.append(self._traces.bulk_write(trace_operations, ordered=True))
            if span_operations:
                writes.append(self._spans.bulk_write(span_operations, ordered=True))
            if writes:
                await asyncio.gather(*writes)
        except PyMongoError as exc:
            raise ExecutionTraceStorageError("Could not save execution snapshots") from exc

    async def save_graph_definition(self, definition: GraphDefinition) -> None:
        document = {
            "_id": definition.definition_id,
            **definition.model_dump(mode="python", exclude_none=True),
        }
        try:
            await self._definitions.update_one(
                {"_id": definition.definition_id},
                {"$setOnInsert": document},
                upsert=True,
            )
        except PyMongoError as exc:
            raise ExecutionTraceStorageError("Could not save graph definition") from exc

    async def get_trace(self, trace_id: str) -> ExecutionTrace | None:
        try:
            document = await self._traces.find_one({"_id": trace_id})
            return (
                ExecutionTrace.model_validate(promote_observability_document(document))
                if document is not None
                else None
            )
        except (PyMongoError, ValidationError) as exc:
            raise ExecutionTraceStorageError("Could not read execution trace") from exc

    async def spans(self, trace_id: str) -> Sequence[ExecutionSpan]:
        try:
            cursor = self._spans.find({"trace_id": trace_id}).sort("sequence_started", ASCENDING)
            return tuple(
                [
                    ExecutionSpan.model_validate(promote_observability_document(document))
                    async for document in cursor
                ]
            )
        except (PyMongoError, ValidationError) as exc:
            raise ExecutionTraceStorageError("Could not read execution spans") from exc

    async def span(self, trace_id: str, span_id: str) -> ExecutionSpan | None:
        try:
            document = await self._spans.find_one({"trace_id": trace_id, "span_id": span_id})
            return (
                ExecutionSpan.model_validate(promote_observability_document(document))
                if document is not None
                else None
            )
        except (PyMongoError, ValidationError) as exc:
            raise ExecutionTraceStorageError("Could not read execution span") from exc

    async def graph_definition(self, definition_id: str) -> GraphDefinition | None:
        try:
            document = await self._definitions.find_one({"_id": definition_id})
            return (
                GraphDefinition.model_validate(promote_observability_document(document))
                if document is not None
                else None
            )
        except (PyMongoError, ValidationError) as exc:
            raise ExecutionTraceStorageError("Could not read graph definition") from exc


def promote_observability_document(document: dict[str, Any]) -> dict[str, Any]:
    promoted = {key: value for key, value in document.items() if key != "_id"}
    schema_version = promoted.get("schema_version")
    if schema_version not in {2, 3}:
        return promoted
    promoted["schema_version"] = 4
    if schema_version == 2 and ("session_id" in promoted or "prospective_session_id" in promoted):
        session_id = promoted.get("session_id") or promoted.get("prospective_session_id")
        promoted.pop("prospective_session_id", None)
        promoted["session_id"] = session_id
    if schema_version == 2 and "root_span_id" in promoted:
        promoted["request_origin"] = "unknown"
    return promoted


def _monotonic_snapshot_upsert(document: dict[str, Any], sequence: int) -> UpdateOne:
    """Replace a snapshot only when it is not older than the stored snapshot."""

    return UpdateOne(
        {"_id": document["_id"]},
        [
            {
                "$replaceWith": {
                    "$cond": [
                        {
                            "$lte": [
                                {"$ifNull": ["$snapshot_sequence", -1]},
                                sequence,
                            ]
                        },
                        {"$literal": document},
                        "$$ROOT",
                    ]
                }
            }
        ],
        upsert=True,
    )
