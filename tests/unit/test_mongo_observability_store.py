"""Mongo execution-store command shapes without a database connection."""

from datetime import UTC, datetime
from typing import Any

from orchestration_core import (
    ExecutionStatus,
    ExecutionTrace,
    ObservabilityStatus,
)

from agentic_orchestration.observability.mongodb import (
    EXECUTION_SPANS_COLLECTION,
    EXECUTION_TRACES_COLLECTION,
    GRAPH_DEFINITIONS_COLLECTION,
    MongoExecutionTraceStore,
    promote_observability_document,
)


class FakeCollection:
    def __init__(self) -> None:
        self.indexes: list[tuple[Any, dict[str, Any]]] = []
        self.bulk_writes: list[tuple[list[Any], dict[str, Any]]] = []
        self.updates: list[tuple[Any, Any, dict[str, Any]]] = []

    async def create_index(self, keys: Any, **kwargs: Any) -> None:
        self.indexes.append((keys, kwargs))

    async def bulk_write(self, operations: list[Any], **kwargs: Any) -> None:
        self.bulk_writes.append((operations, kwargs))

    async def update_one(self, query: Any, update: Any, **kwargs: Any) -> None:
        self.updates.append((query, update, kwargs))

    async def find_one(self, query: Any) -> dict[str, Any] | None:
        del query
        return None


class FakeDatabase:
    def __init__(self) -> None:
        self.collections = {
            name: FakeCollection()
            for name in (
                EXECUTION_TRACES_COLLECTION,
                EXECUTION_SPANS_COLLECTION,
                GRAPH_DEFINITIONS_COLLECTION,
            )
        }
        self.created: list[str] = []

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections[name]

    async def list_collection_names(self) -> list[str]:
        return []

    async def create_collection(self, name: str) -> None:
        self.created.append(name)


async def test_mongo_store_initializes_collections_and_writes_monotonic_snapshot() -> None:
    database = FakeDatabase()
    store = MongoExecutionTraceStore(database)  # type: ignore[arg-type]
    await store.initialize()
    now = datetime.now(UTC)
    trace = ExecutionTrace(
        trace_id="trace",
        root_span_id="root",
        session_id="session",
        attempted_turn_number=1,
        status=ExecutionStatus.RUNNING,
        sequence_started=1,
        snapshot_sequence=1,
        started_at=now,
        deployment_version="test",
        application_version="1",
        core_version="1",
    )
    completed = ExecutionTrace(
        trace_id="trace",
        root_span_id="root",
        session_id="session",
        attempted_turn_number=1,
        committed_turn_number=1,
        status=ExecutionStatus.COMPLETED,
        observability_status=ObservabilityStatus.COMPLETE,
        sequence_started=1,
        sequence_completed=2,
        snapshot_sequence=2,
        started_at=now,
        completed_at=now,
        deployment_version="test",
        application_version="1",
        core_version="1",
    )

    await store.save_snapshots((trace, completed))

    assert set(database.created) == set(database.collections)
    operations, kwargs = database.collections[EXECUTION_TRACES_COLLECTION].bulk_writes[0]
    assert len(operations) == 2
    operation = operations[0]
    assert operation._filter == {"_id": "trace"}
    assert operation._upsert is True
    assert operation._doc[0]["$replaceWith"]["$cond"][0] == {
        "$lte": [{"$ifNull": ["$snapshot_sequence", -1]}, 1]
    }
    assert operations[1]._doc[0]["$replaceWith"]["$cond"][0] == {
        "$lte": [{"$ifNull": ["$snapshot_sequence", -1]}, 2]
    }
    assert kwargs == {"ordered": True}


def test_v2_trace_identity_promotes_to_current_contract_without_rewriting_storage() -> None:
    now = datetime.now(UTC)
    promoted = promote_observability_document(
        {
            "_id": "trace",
            "schema_version": 2,
            "trace_id": "trace",
            "root_span_id": "root",
            "session_id": None,
            "prospective_session_id": "session",
            "attempted_turn_number": 1,
            "status": "running",
            "observability_status": "recording",
            "sequence_started": 1,
            "snapshot_sequence": 1,
            "started_at": now,
            "deployment_version": "test",
            "application_version": "1",
            "core_version": "1",
            "agent_versions": {},
            "attributes": {},
        }
    )

    trace = ExecutionTrace.model_validate(promoted)
    assert trace.schema_version == 4
    assert trace.session_id == "session"
    assert trace.request_origin == "unknown"
    assert "prospective_session_id" not in promoted


def test_v3_document_promotes_to_current_contract() -> None:
    promoted = promote_observability_document(
        {
            "_id": "definition",
            "schema_version": 3,
            "definition_id": "router:1:hash",
            "definition_hash": "a" * 64,
            "agent_id": "router",
            "agent_version": "1",
            "entrypoint": "router:create",
            "nodes": [
                {"node_id": "start", "display_name": "START", "kind": "start"},
                {"node_id": "end", "display_name": "END", "kind": "end"},
            ],
            "edges": [],
            "created_at": datetime.now(UTC),
        }
    )

    assert promoted["schema_version"] == 4
