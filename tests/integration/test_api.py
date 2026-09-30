import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from agentic_orchestration.main import create_app
from agentic_orchestration.sessions import MemorySessionStore
from tests.fakes import ScriptedChatModel

ROOT = Path(__file__).parents[2]


@pytest.mark.integration
async def test_health_and_safe_build_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PATH", str(ROOT / "deployments/local.yaml"))
    monkeypatch.setenv("DEPLOYMENT_SCHEMA_PATH", str(ROOT / "schemas/deployment.schema.json"))
    monkeypatch.setenv("SOURCE_COMMIT", "test-commit")
    monkeypatch.setenv("GATEWAY_TOKEN", "must-not-appear")
    app = create_app(
        model_factory=lambda settings: ScriptedChatModel(),
        session_store_factory=lambda settings: MemorySessionStore(),
    )
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        health = await client.get("/health")
        build = await client.get("/build")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert build.status_code == 200
    body = build.json()
    assert body["application_version"] == "0.9.0"
    assert body["deployment"] == {"name": "local-development", "revision": "0.10.0"}
    assert body["source_commit"] == "test-commit"
    assert body["uv_lock_sha256"] is None
    assert body["immutable"] is False
    assert body["architectural_packages"]["router-agent"] == "0.5.0"
    assert body["architectural_packages"]["marketing-science-agent"] == "0.3.0"
    assert "must-not-appear" not in build.text


@pytest.mark.integration
async def test_build_reads_generated_immutable_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DEPLOYMENT_PATH", str(ROOT / "deployments/local.yaml"))
    monkeypatch.setenv("DEPLOYMENT_SCHEMA_PATH", str(ROOT / "schemas/deployment.schema.json"))
    path = tmp_path / "build-metadata.json"
    path.write_text(
        json.dumps(
            {
                "repository": "agentic-orchestration",
                "application_version": "0.2.0",
                "source_commit": "immutable-commit",
                "deployment": {"name": "local-development", "revision": "0.2.0"},
                "architectural_packages": {
                    "agentic-orchestration": "0.2.0",
                    "orchestration-core": "0.2.0",
                    "enterprise-llm": "0.1.0",
                    "router-agent": "0.1.0",
                    "marketing-science-agent": "0.1.0",
                },
                "uv_lock_sha256": "lock-hash",
                "immutable": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BUILD_METADATA_PATH", str(path))
    app = create_app(
        model_factory=lambda settings: ScriptedChatModel(),
        session_store_factory=lambda settings: MemorySessionStore(),
    )
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.get("/build")
    assert response.json()["source_commit"] == "immutable-commit"
    assert response.json()["uv_lock_sha256"] == "lock-hash"
    assert response.json()["immutable"] is True


@pytest.mark.integration
async def test_build_metadata_file_wins_over_runtime_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The image's baked composition must not be overridable by runtime environment."""
    monkeypatch.setenv("DEPLOYMENT_PATH", str(ROOT / "deployments/local.yaml"))
    monkeypatch.setenv("DEPLOYMENT_SCHEMA_PATH", str(ROOT / "schemas/deployment.schema.json"))
    path = tmp_path / "build-metadata.json"
    path.write_text(
        json.dumps(
            {
                "repository": "agentic-orchestration",
                "application_version": "0.2.0",
                "source_commit": "baked-into-image",
                "deployment": {"name": "local-development", "revision": "0.2.0"},
                "architectural_packages": {
                    "agentic-orchestration": "0.2.0",
                    "orchestration-core": "0.2.0",
                    "enterprise-llm": "0.1.0",
                    "router-agent": "0.1.0",
                    "marketing-science-agent": "0.1.0",
                },
                "uv_lock_sha256": "baked-lock-hash",
                "immutable": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BUILD_METADATA_PATH", str(path))
    monkeypatch.setenv("SOURCE_COMMIT", "wrong-runtime-value")
    app = create_app(
        model_factory=lambda settings: ScriptedChatModel(),
        session_store_factory=lambda settings: MemorySessionStore(),
    )
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        body = (await client.get("/build")).json()
    assert body["source_commit"] == "baked-into-image"
    assert body["uv_lock_sha256"] == "baked-lock-hash"
    assert body["immutable"] is True
