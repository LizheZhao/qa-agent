"""MongoDB client configuration behavior without a network connection."""

from datetime import UTC
from typing import Any

import integrations.mongodb as mongodb


class FakeMongoClient:
    def __init__(self, uri: str, **kwargs: Any) -> None:
        self.uri = uri
        self.kwargs = kwargs

    def __getitem__(self, database: str) -> str:
        return database


def test_mongodb_decodes_bson_datetimes_as_aware_utc(monkeypatch: Any) -> None:
    created: list[FakeMongoClient] = []

    def client_factory(uri: str, **kwargs: Any) -> FakeMongoClient:
        client = FakeMongoClient(uri, **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(mongodb, "AsyncMongoClient", client_factory)

    _, database = mongodb.create_mongodb(
        host="mongo.internal",
        port=27017,
        user="user",
        password="password",
        database="orchestration",
    )

    assert database == "orchestration"
    assert created[0].kwargs == {
        "uuidRepresentation": "standard",
        "tz_aware": True,
        "tzinfo": UTC,
    }
