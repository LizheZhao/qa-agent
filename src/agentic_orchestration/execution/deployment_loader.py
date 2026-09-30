"""Schema, installed-version, and declared-entrypoint validation."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from orchestration_core import AgentManifest
from pydantic import BaseModel, ConfigDict


class DeploymentValidationError(RuntimeError):
    """Raised with actionable deployment validation details."""


class PackageVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    package: str
    version: str


class DeploymentIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    revision: str


class AgentDeclaration(PackageVersion):
    entrypoint: str
    manifest_entrypoint: str
    routable: bool


class Integrations(BaseModel):
    model_config = ConfigDict(frozen=True)

    enterprise_llm: PackageVersion


class DeploymentManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    deployment: DeploymentIdentity
    application: PackageVersion
    platform: PackageVersion
    integrations: Integrations
    agents: dict[str, AgentDeclaration]
    entry_agent: str | None


@dataclass(frozen=True, slots=True)
class LoadedDeployment:
    manifest: DeploymentManifest
    entrypoints: dict[str, Any]
    agent_manifests: dict[str, AgentManifest]


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DeploymentValidationError(f"Cannot read deployment file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DeploymentValidationError(f"Deployment file {path} must contain a YAML object")
    return raw


def _read_schema(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentValidationError(f"Cannot read deployment schema {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DeploymentValidationError(f"Deployment schema {path} must be a JSON object")
    return raw


def load_entrypoint(value: str) -> Any:
    """Load one explicitly declared ``module:attribute`` entrypoint."""

    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        raise DeploymentValidationError(
            f"Invalid agent entrypoint {value!r}; expected 'module.path:attribute'"
        )
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise DeploymentValidationError(f"Cannot load agent entrypoint {value!r}: {exc}") from exc


def _architectural_packages(manifest: DeploymentManifest) -> list[PackageVersion]:
    return [
        manifest.application,
        manifest.platform,
        manifest.integrations.enterprise_llm,
        *manifest.agents.values(),
    ]


def validate_installed_versions(manifest: DeploymentManifest) -> None:
    errors: list[str] = []
    for declared in _architectural_packages(manifest):
        try:
            installed = importlib.metadata.version(declared.package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{declared.package}: declared {declared.version}, but is not installed")
            continue
        if installed != declared.version:
            errors.append(
                f"{declared.package}: manifest declares {declared.version}, installed {installed}"
            )
    if errors:
        raise DeploymentValidationError(
            "Architectural package version mismatch:\n- " + "\n- ".join(errors)
        )


def load_and_validate_deployment(
    deployment_path: Path,
    schema_path: Path,
    *,
    check_versions: bool = True,
    load_agents: bool = True,
) -> LoadedDeployment:
    raw = _read_mapping(deployment_path)
    schema = _read_schema(schema_path)
    try:
        jsonschema.Draft202012Validator(schema).validate(raw)
        manifest = DeploymentManifest.model_validate(raw)
    except (jsonschema.ValidationError, ValueError) as exc:
        path = ".".join(str(part) for part in getattr(exc, "absolute_path", []))
        location = f" at {path}" if path else ""
        detail = getattr(exc, "message", str(exc))
        raise DeploymentValidationError(f"Invalid deployment manifest{location}: {detail}") from exc

    if manifest.entry_agent is not None and manifest.entry_agent not in manifest.agents:
        raise DeploymentValidationError(
            f"entry_agent {manifest.entry_agent!r} is not present in the agents mapping"
        )
    if manifest.entry_agent is not None and manifest.agents[manifest.entry_agent].routable:
        raise DeploymentValidationError("entry_agent must not be declared routable")
    if check_versions:
        validate_installed_versions(manifest)
    entrypoints: dict[str, Any] = {}
    agent_manifests: dict[str, AgentManifest] = {}
    if load_agents:
        entrypoints = {
            name: load_entrypoint(agent.entrypoint) for name, agent in manifest.agents.items()
        }
        for name, declaration in manifest.agents.items():
            value = load_entrypoint(declaration.manifest_entrypoint)
            if not isinstance(value, AgentManifest):
                raise DeploymentValidationError(
                    f"Agent manifest entrypoint {declaration.manifest_entrypoint!r} "
                    "did not resolve to AgentManifest"
                )
            if value.agent_id != name:
                raise DeploymentValidationError(
                    f"Agent {name!r} exports mismatched agent_id {value.agent_id!r}"
                )
            if value.version != declaration.version:
                raise DeploymentValidationError(
                    f"Agent {name!r} exports version {value.version}, "
                    f"deployment declares {declaration.version}"
                )
            if declaration.routable and not value.capabilities:
                raise DeploymentValidationError(
                    f"Routable agent {name!r} must declare at least one capability"
                )
            agent_manifests[name] = value
    return LoadedDeployment(
        manifest=manifest,
        entrypoints=entrypoints,
        agent_manifests=agent_manifests,
    )
