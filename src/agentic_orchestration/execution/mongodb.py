"""Repository-owned durable LangGraph checkpointer over the pinned pymongo client.

Why this exists rather than a third-party saver: the only release of
`langgraph-checkpoint-mongodb` that resolves against this workspace's
`langchain-core>=0.3.72,<1` pin is 0.2.2, which downgrades the application's MongoDB
driver to `pymongo<4.16`, pulls `langchain`, `sqlalchemy`, `lark` and more into the runtime
closure for a dependency it never imports, and reaches durability through a class already
deprecated for removal. Upstream's surviving saver wraps blocking `MongoClient` calls in a
thread pool. This adapter uses `pymongo.AsyncMongoClient` directly -- the same driver and
version the session store already runs on -- so checkpoint writes are natively async and
the dependency closure does not move. `docs/graph-interrupt-resume.md` records the
measurements behind that decision.

The stored layout deliberately follows the shape upstream proved: one document per
checkpoint holding the serialized checkpoint inline, and one document per pending write
keyed by task and index. Channel values are not split into per-version blobs because the
pause-capable state contract already bounds them to receipts, compact results, and
artifact references, and separate blob documents would add failure modes without saving
space worth having.

Two semantics are load-bearing and easy to get wrong, so they live in named helpers below
and are covered by deterministic tests: which writes a retry may replace, and the fact
that a checkpoint's "latest" is decided by its monotonic `checkpoint_id`.

Scope: checkpoint writes are not part of the session-turn transaction. This adapter opens
no MongoDB session and joins none, exactly as the third-party savers do not. Checkpoint 4
publishes the pending run in its own short session-store transaction.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, Final

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pymongo import ASCENDING, DESCENDING, UpdateOne
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import CollectionInvalid

from agentic_orchestration.execution.checkpointing import (
    GraphCheckpointerError,
)

CHECKPOINTS_COLLECTION: Final = "graph_checkpoints"
CHECKPOINT_WRITES_COLLECTION: Final = "graph_checkpoint_writes"
CHECKPOINT_INDEX: Final = "graph_checkpoint_unique"
CHECKPOINT_WRITE_INDEX: Final = "graph_checkpoint_write_unique"

SUPPORTED_PAYLOAD_TYPES: Final[frozenset[str]] = frozenset(
    {"msgpack", "json", "null", "bytes", "bytearray"}
)
"""Serialization types this adapter is willing to load back.

`pickle` is absent on purpose. The stock serializer only produces it when
`pickle_fallback` is enabled, which it is not below, but refusing it here as well means a
payload written directly into the collection cannot reach `pickle.loads` even if the
serializer is later reconfigured.
"""

ALLOWED_JSON_MODULES: Final[tuple[tuple[str, ...], ...]] = (
    ("orchestration_core", "checkpoint_state", "ActionReceipt"),
    ("orchestration_core", "checkpoint_state", "ArtifactReference"),
    ("orchestration_core", "checkpoint_state", "CheckpointedTaskState"),
    ("orchestration_core", "checkpoint_state", "CompactResult"),
    ("orchestration_core", "checkpoint_state", "TaskContext"),
    ("orchestration_core", "clarification", "ClarificationOption"),
    ("orchestration_core", "clarification", "ClarificationRequest"),
)
"""The exact application constructors the JSON path may rebuild, and no others.

The parameter this feeds also accepts `True`, which would open the path to any importable
module. It is given a closed tuple instead, so widening it is a visible edit here rather
than a default that drifts.
"""


class CheckpointPayloadError(GraphCheckpointerError):
    """A stored checkpoint payload is not in a form this adapter will deserialize."""


def build_checkpoint_serializer() -> SerializerProtocol:
    """The stock serializer, configured for restricted deserialization.

    Two separate things are going on, and it is worth not confusing them.

    `pickle_fallback=False` is the restriction that matters: it keeps `pickle.loads` out
    of the load path, so a payload stored under the `pickle` type is refused rather than
    executed. `validated_payload` refuses that type again at the document boundary, so the
    guarantee does not rest on this configuration alone.

    `allowed_json_modules` is an allowlist, not a filter over an otherwise-open path. The
    JSON path already refuses unknown constructors by way of LangChain's default `core`
    reviver; naming the bounded state contract here is what permits those specific types
    to be rebuilt, and nothing else is added.

    What this does not close: the msgpack extension hook upstream installs still rebuilds
    objects by importing whatever module a payload names, and it is not configurable
    without replacing the serializer. Only the application's own graphs write to these
    collections, and the state contract bounds what they write.
    """

    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=ALLOWED_JSON_MODULES,
    )


def checkpoint_write_index(channel: str, position: int) -> int:
    """The stable index a pending write is stored under.

    Special channels map to reserved negative indices so they cannot collide with the
    ordinary writes of the same task; everything else keeps its position in the batch.
    """

    return WRITES_IDX_MAP.get(channel, position)


def write_may_replace_existing(index: int) -> bool:
    """Whether re-writing this index overwrites the stored value or leaves it alone.

    Reserved negative indices carry a task's error, interrupt, resume, and scheduling
    state, which legitimately change when a task is retried, so a later write replaces
    them. An ordinary write is immutable once stored: a retried task re-sends the writes
    it already persisted, and the first value must win so a duplicate delivery cannot
    rewrite committed work.
    """

    return index < 0


def bson_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return checkpoint metadata as a plain BSON subdocument.

    Stored unserialized so an operator can read a paused run's step and source directly.
    Keys that BSON cannot round-trip are refused rather than silently mangled, because a
    dropped metadata key would make `list` filtering quietly wrong.
    """

    for key in metadata:
        if key.startswith("$") or "." in key:
            raise CheckpointPayloadError(f"Checkpoint metadata key {key!r} is not BSON safe")
    return dict(metadata)


def validated_payload(type_: str, data: bytes) -> tuple[str, bytes]:
    """Refuse a stored payload whose serialization type this adapter does not write."""

    if type_ not in SUPPORTED_PAYLOAD_TYPES:
        raise CheckpointPayloadError(f"Unsupported checkpoint payload type {type_!r}")
    return type_, data


class MongoCheckpointSaver(BaseCheckpointSaver[int]):
    """Durable checkpoint storage for pause-capable graphs, async only.

    The synchronous half of the saver contract is deliberately not implemented. Every
    caller in this application reaches graphs through the async executor, and a
    thread-wrapped blocking implementation is the property that disqualified the
    third-party saver; providing one here would reintroduce it. The methods raise instead
    of blocking, and a test walks a full pause, restart, and resume to show the async path
    never needs them.
    """

    def __init__(
        self,
        database: AsyncDatabase[dict[str, Any]],
        *,
        checkpoints_collection: str = CHECKPOINTS_COLLECTION,
        writes_collection: str = CHECKPOINT_WRITES_COLLECTION,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde or build_checkpoint_serializer())
        self._database = database
        self._checkpoints = database[checkpoints_collection]
        self._writes = database[writes_collection]
        self._checkpoints_collection = checkpoints_collection
        self._writes_collection = writes_collection

    async def initialize(self) -> None:
        """Create the collections and the uniqueness they depend on."""

        existing = set(await self._database.list_collection_names())
        for name in (self._checkpoints_collection, self._writes_collection):
            if name not in existing:
                # Another replica may create it after our initial listing.
                with suppress(CollectionInvalid):
                    await self._database.create_collection(name)
        await self._checkpoints.create_index(
            [
                ("thread_id", ASCENDING),
                ("checkpoint_ns", ASCENDING),
                ("checkpoint_id", DESCENDING),
            ],
            name=CHECKPOINT_INDEX,
            unique=True,
        )
        # Keyed by task and index only. `task_path` is payload, not identity: a retried
        # task may report a different path for the same write, and including it would let
        # a duplicate delivery insert a second copy of work already stored.
        await self._writes.create_index(
            [
                ("thread_id", ASCENDING),
                ("checkpoint_ns", ASCENDING),
                ("checkpoint_id", DESCENDING),
                ("task_id", ASCENDING),
                ("idx", ASCENDING),
            ],
            name=CHECKPOINT_WRITE_INDEX,
            unique=True,
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        query: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        if checkpoint_id := get_checkpoint_id(config):
            query["checkpoint_id"] = checkpoint_id
        # `checkpoint_id` is monotonically increasing, so the highest one is the latest.
        document = await self._checkpoints.find_one(query, sort=[("checkpoint_id", DESCENDING)])
        if document is None:
            return None
        return await self._tuple(document)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        query: dict[str, Any] = {}
        # Both bounds are expressed as operators on one field so that a config naming an
        # exact checkpoint and a `before` bound compose instead of overwriting each other.
        checkpoint_bounds: dict[str, Any] = {}
        if config is not None:
            query["thread_id"] = config["configurable"]["thread_id"]
            if (checkpoint_ns := config["configurable"].get("checkpoint_ns")) is not None:
                query["checkpoint_ns"] = checkpoint_ns
            if checkpoint_id := get_checkpoint_id(config):
                checkpoint_bounds["$eq"] = checkpoint_id
        if before is not None and (before_id := get_checkpoint_id(before)):
            checkpoint_bounds["$lt"] = before_id
        if checkpoint_bounds:
            query["checkpoint_id"] = checkpoint_bounds
        cursor = self._checkpoints.find(query).sort("checkpoint_id", DESCENDING)
        yielded = 0
        async for document in cursor:
            # Metadata is matched here rather than in the query: the stored subdocument
            # holds the primitives LangGraph put there, and an equality query over a
            # nested document would depend on stored key order.
            if filter and not all(
                document.get("metadata", {}).get(key) == value for key, value in filter.items()
            ):
                continue
            if limit is not None and yielded >= limit:
                return
            yielded += 1
            yield await self._tuple(document)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        type_, payload = validated_payload(*self.serde.dumps_typed(checkpoint))
        written_at = datetime.now(UTC)
        await self._checkpoints.update_one(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            },
            {
                "$set": {
                    "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
                    "type": type_,
                    "checkpoint": payload,
                    "metadata": bson_safe_metadata(get_checkpoint_metadata(config, metadata)),
                    "updated_at": written_at,
                },
                "$setOnInsert": {"created_at": written_at},
            },
            upsert=True,
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        created_at = datetime.now(UTC)
        operations: list[UpdateOne] = []
        for position, (channel, value) in enumerate(writes):
            index = checkpoint_write_index(channel, position)
            type_, payload = validated_payload(*self.serde.dumps_typed(value))
            stored = {
                "channel": channel,
                "type": type_,
                "value": payload,
                "task_path": task_path,
                "created_at": created_at,
            }
            operations.append(
                UpdateOne(
                    filter={
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "idx": index,
                    },
                    update=(
                        {"$set": stored}
                        if write_may_replace_existing(index)
                        else {"$setOnInsert": stored}
                    ),
                    upsert=True,
                )
            )
        await self._writes.bulk_write(operations, ordered=False)

    async def adelete_thread(self, thread_id: str) -> None:
        """Remove every checkpoint and pending write for one run.

        Cleanup of an obsolete checkpoint, never a correctness mechanism: expiry and
        duplicate-resume protection are decided by the pending-run record, so a thread
        that has not been cleaned up yet cannot be resumed a second time.
        """

        await self._checkpoints.delete_many({"thread_id": thread_id})
        await self._writes.delete_many({"thread_id": thread_id})

    async def _tuple(self, document: dict[str, Any]) -> CheckpointTuple:
        thread_id = document["thread_id"]
        checkpoint_ns = document["checkpoint_ns"]
        checkpoint_id = document["checkpoint_id"]
        identity = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
        writes = self._writes.find(identity).sort([("task_id", ASCENDING), ("idx", ASCENDING)])
        pending_writes = [
            (
                write["task_id"],
                write["channel"],
                self.serde.loads_typed(validated_payload(write["type"], write["value"])),
            )
            async for write in writes
        ]
        parent_checkpoint_id = document.get("parent_checkpoint_id")
        return CheckpointTuple(
            config={"configurable": identity},
            checkpoint=self.serde.loads_typed(
                validated_payload(document["type"], document["checkpoint"])
            ),
            metadata=document.get("metadata", {}),
            parent_config=(
                {"configurable": {**identity, "checkpoint_id": parent_checkpoint_id}}
                if parent_checkpoint_id
                else None
            ),
            pending_writes=pending_writes,
        )

    def _async_only(self) -> GraphCheckpointerError:
        return GraphCheckpointerError(
            f"{type(self).__name__} is async only; use the aget/alist/aput methods"
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        raise self._async_only()

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        raise self._async_only()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise self._async_only()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise self._async_only()

    def delete_thread(self, thread_id: str) -> None:
        raise self._async_only()


class MongoGraphCheckpointer:
    """Startup-owned durable checkpointer over a connection this object does not own."""

    def __init__(
        self,
        database: AsyncDatabase[dict[str, Any]],
        *,
        checkpoints_collection: str = CHECKPOINTS_COLLECTION,
        writes_collection: str = CHECKPOINT_WRITES_COLLECTION,
    ) -> None:
        self._saver = MongoCheckpointSaver(
            database,
            checkpoints_collection=checkpoints_collection,
            writes_collection=writes_collection,
        )
        self._initialized = False
        self._closed = False

    @property
    def storage_name(self) -> str:
        return "mongodb"

    @property
    def durable(self) -> bool:
        return True

    @property
    def saver(self) -> MongoCheckpointSaver:
        if not self._initialized:
            raise GraphCheckpointerError("Graph checkpointer was not initialized")
        if self._closed:
            raise GraphCheckpointerError("Graph checkpointer is closed")
        return self._saver

    async def initialize(self) -> None:
        await self._saver.initialize()
        self._initialized = True

    async def close(self) -> None:
        """Stop handing out the saver and leave the connection to its owner.

        The MongoDB client is constructed and closed by the lifespan resource stack, which
        also hands it to the session store. Closing it here would tear down conversation
        persistence along with the checkpointer.
        """

        self._closed = True
