"""Typed asynchronous MongoDB connection construction."""

from datetime import UTC
from typing import Any
from urllib.parse import quote_plus

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase


def create_mongodb(
    *,
    host: str | None,
    port: int | None,
    user: str | None,
    password: str | None,
    database: str | None,
) -> tuple[AsyncMongoClient[dict[str, Any]], AsyncDatabase[dict[str, Any]]]:
    missing = [
        name
        for name, value in (
            ("MONGODB_HOST", host),
            ("MONGODB_PORT", port),
            ("MONGODB_USER", user),
            ("MONGODB_PWD", password),
            ("MONGODB_DB", database),
        )
        if value is None or value == ""
    ]
    if missing:
        raise RuntimeError(f"Missing required MongoDB settings: {', '.join(missing)}")

    assert host is not None
    assert port is not None
    assert user is not None
    assert password is not None
    assert database is not None
    encoded_user = quote_plus(user)
    encoded_password = quote_plus(password)
    encoded_database = quote_plus(database)
    mongo_uri = (
        f"mongodb://{encoded_user}:{encoded_password}@{host}:{port}/{encoded_database}"
        "?directConnection=true&authMechanism=SCRAM-SHA-256"
    )
    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(
        mongo_uri,
        uuidRepresentation="standard",
        tz_aware=True,
        tzinfo=UTC,
    )
    return client, client[database]
