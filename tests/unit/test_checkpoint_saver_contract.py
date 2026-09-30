"""Deterministic rules the durable checkpoint saver depends on, without a database."""

import json
import pickle
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import ERROR, INTERRUPT, RESUME, SCHEDULED
from orchestration_core import ClarificationOption

from agentic_orchestration.execution.checkpointing import GraphCheckpointerError
from agentic_orchestration.execution.mongodb import (
    ALLOWED_JSON_MODULES,
    SUPPORTED_PAYLOAD_TYPES,
    CheckpointPayloadError,
    MongoCheckpointSaver,
    bson_safe_metadata,
    build_checkpoint_serializer,
    checkpoint_write_index,
    validated_payload,
    write_may_replace_existing,
)


def saver() -> MongoCheckpointSaver:
    """A saver over a database handle. pymongo connects lazily, so nothing is contacted."""

    from pymongo import AsyncMongoClient

    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(
        "mongodb://localhost:27017", connect=False
    )
    return MongoCheckpointSaver(client["offline"])


def json_payload(constructor: list[str], **kwargs: Any) -> tuple[str, bytes]:
    body = {"lc": 2, "type": "constructor", "id": constructor, "kwargs": kwargs}
    return "json", json.dumps(body).encode("utf-8")


@pytest.mark.unit
def test_ordinary_writes_keep_their_position_in_the_batch() -> None:
    assert checkpoint_write_index("messages", 0) == 0
    assert checkpoint_write_index("receipts", 3) == 3


@pytest.mark.unit
@pytest.mark.parametrize("channel", [ERROR, SCHEDULED, INTERRUPT, RESUME])
def test_special_channels_take_reserved_negative_indices(channel: str) -> None:
    index = checkpoint_write_index(channel, 0)

    assert index < 0
    # A reserved index must not collide with the ordinary write at the same position.
    assert index != checkpoint_write_index("messages", 0)


@pytest.mark.unit
def test_reserved_indices_are_distinct_from_each_other() -> None:
    indices = [
        checkpoint_write_index(channel, 0) for channel in (ERROR, SCHEDULED, INTERRUPT, RESUME)
    ]

    assert len(set(indices)) == len(indices)


@pytest.mark.unit
def test_a_retried_ordinary_write_does_not_overwrite_the_stored_value() -> None:
    assert write_may_replace_existing(checkpoint_write_index("messages", 0)) is False
    assert write_may_replace_existing(checkpoint_write_index("messages", 7)) is False


@pytest.mark.unit
@pytest.mark.parametrize("channel", [ERROR, INTERRUPT, RESUME, SCHEDULED])
def test_a_retried_task_may_replace_its_error_and_resume_state(channel: str) -> None:
    assert write_may_replace_existing(checkpoint_write_index(channel, 0)) is True


@pytest.mark.unit
def test_pickle_is_not_a_serialization_type_this_adapter_will_load() -> None:
    assert "pickle" not in SUPPORTED_PAYLOAD_TYPES

    with pytest.raises(CheckpointPayloadError, match="Unsupported checkpoint payload type"):
        validated_payload("pickle", pickle.dumps({"trusted": False}))


@pytest.mark.unit
def test_the_serializer_refuses_to_unpickle_even_if_a_payload_claims_pickle() -> None:
    serializer = build_checkpoint_serializer()

    with pytest.raises(NotImplementedError, match="Unknown serialization type"):
        serializer.loads_typed(("pickle", pickle.dumps({"trusted": False})))


@pytest.mark.unit
def test_the_serializer_round_trips_the_bounded_state_contract() -> None:
    serializer = build_checkpoint_serializer()
    option = ClarificationOption(option_id="us", label="United States")

    restored = serializer.loads_typed(serializer.dumps_typed(option))

    assert restored == option


@pytest.mark.unit
def test_the_allowlist_is_what_admits_the_bounded_state_contract() -> None:
    """The stock default leaves these as raw dicts; naming them is what rebuilds them."""

    allowed = build_checkpoint_serializer().loads_typed(
        json_payload(
            ["orchestration_core", "clarification", "ClarificationOption"],
            option_id="us",
            label="United States",
        )
    )
    unconfigured = JsonPlusSerializer().loads_typed(
        json_payload(
            ["orchestration_core", "clarification", "ClarificationOption"],
            option_id="us",
            label="United States",
        )
    )

    assert isinstance(allowed, ClarificationOption)
    assert allowed.option_id == "us"
    assert isinstance(unconfigured, dict)


@pytest.mark.unit
def test_the_json_path_rebuilds_nothing_beyond_the_allowlist() -> None:
    refused = build_checkpoint_serializer().loads_typed(json_payload(["pathlib", "Path"]))

    assert not isinstance(refused, Path)


@pytest.mark.unit
def test_the_allowlist_is_closed_rather_than_open() -> None:
    """`True` would admit any importable module; the configuration must stay enumerated."""

    assert ALLOWED_JSON_MODULES is not True
    assert isinstance(ALLOWED_JSON_MODULES, tuple)
    assert all(
        entry[0] == "orchestration_core" and len(entry) == 3 for entry in ALLOWED_JSON_MODULES
    )


@pytest.mark.unit
def test_metadata_is_stored_as_a_readable_subdocument() -> None:
    assert bson_safe_metadata({"source": "loop", "step": 2}) == {"source": "loop", "step": 2}


@pytest.mark.unit
@pytest.mark.parametrize("key", ["$set", "langgraph.step"])
def test_metadata_keys_bson_cannot_round_trip_are_refused(key: str) -> None:
    with pytest.raises(CheckpointPayloadError, match="not BSON safe"):
        bson_safe_metadata({key: "value"})


@pytest.mark.unit
def test_the_saver_refuses_its_blocking_interface() -> None:
    """A synchronous call would block the event loop, which is why there is no sync path."""

    blocking = saver()
    config: Any = {"configurable": {"thread_id": "run-1", "checkpoint_id": "checkpoint-1"}}

    with pytest.raises(GraphCheckpointerError, match="async only"):
        blocking.get_tuple(config)
    with pytest.raises(GraphCheckpointerError, match="async only"):
        next(blocking.list(config))
    with pytest.raises(GraphCheckpointerError, match="async only"):
        blocking.put(config, {}, {}, {})  # type: ignore[arg-type]
    with pytest.raises(GraphCheckpointerError, match="async only"):
        blocking.put_writes(config, [("messages", "value")], "task-1")
    with pytest.raises(GraphCheckpointerError, match="async only"):
        blocking.delete_thread("run-1")


@pytest.mark.unit
def test_channel_versions_are_monotonic_integers() -> None:
    versions = saver()

    first = versions.get_next_version(None, None)
    second = versions.get_next_version(first, None)

    assert first == 1
    assert second > first
