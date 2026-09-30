from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentic_orchestration.execution.deployment_loader import (
    DeploymentValidationError,
    load_and_validate_deployment,
)

ROOT = Path(__file__).parents[2]


@pytest.mark.unit
def test_local_deployment_declares_router_and_routable_specialist() -> None:
    loaded = load_and_validate_deployment(
        ROOT / "deployments/local.yaml", ROOT / "schemas/deployment.schema.json"
    )
    assert loaded.manifest.deployment.name == "local-development"
    declaration = loaded.manifest.agents["marketing_science"]
    assert declaration.package == "marketing-science-agent"
    assert declaration.version == "0.3.0"
    assert declaration.manifest_entrypoint == "marketing_science_agent:AGENT_MANIFEST"
    assert declaration.routable is True
    assert loaded.manifest.entry_agent == "router"
    assert loaded.manifest.agents["router"].routable is False
    assert loaded.entrypoints["router"].__name__ == "create_graph"
    assert loaded.agent_manifests["marketing_science"].agent_id == "marketing_science"


@pytest.mark.unit
def test_version_mismatch_is_actionable(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "deployments/local.yaml").read_text())
    raw["platform"]["version"] = "99.0.0"
    manifest = tmp_path / "mismatch.yaml"
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(DeploymentValidationError, match=r"orchestration-core.*99\.0\.0"):
        load_and_validate_deployment(manifest, ROOT / "schemas/deployment.schema.json")


@pytest.mark.unit
def test_invalid_entrypoint_syntax_fails_schema_validation(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "deployments/local.yaml").read_text())
    raw["agents"] = {
        "fixture": {
            "package": "fixture",
            "version": "0.1.0",
            "entrypoint": "not valid",
            "manifest_entrypoint": "tests.fixtures.graph:FIXTURE_AGENT_MANIFEST",
            "routable": False,
        }
    }
    manifest = tmp_path / "invalid.yaml"
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(DeploymentValidationError, match="Invalid deployment manifest"):
        load_and_validate_deployment(
            manifest,
            ROOT / "schemas/deployment.schema.json",
            check_versions=False,
        )


@pytest.mark.unit
def test_declared_entrypoint_is_loaded_without_discovery(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "deployments/local.yaml").read_text())
    raw["agents"] = {
        "fixture": {
            "package": "fixture",
            "version": "0.1.0",
            "entrypoint": "tests.fixtures.graph:create_test_graph",
            "manifest_entrypoint": "tests.fixtures.graph:FIXTURE_AGENT_MANIFEST",
            "routable": False,
        }
    }
    raw["entry_agent"] = "fixture"
    manifest = tmp_path / "valid.yaml"
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    loaded = load_and_validate_deployment(
        manifest,
        ROOT / "schemas/deployment.schema.json",
        check_versions=False,
    )
    assert loaded.entrypoints["fixture"].__name__ == "create_test_graph"
    assert loaded.agent_manifests["fixture"].agent_id == "fixture"


@pytest.mark.unit
def test_entry_agent_must_be_declared(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "deployments/local.yaml").read_text())
    raw["entry_agent"] = "missing"
    manifest = tmp_path / "invalid-entry.yaml"
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(DeploymentValidationError, match="not present"):
        load_and_validate_deployment(
            manifest,
            ROOT / "schemas/deployment.schema.json",
            check_versions=False,
        )


@pytest.mark.unit
def test_entry_agent_cannot_be_routable(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "deployments/local.yaml").read_text())
    raw["agents"]["router"]["routable"] = True
    manifest = tmp_path / "routable-entry.yaml"
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(
        DeploymentValidationError, match="entry_agent must not be declared routable"
    ):
        load_and_validate_deployment(manifest, ROOT / "schemas/deployment.schema.json")
