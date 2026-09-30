"""Safe build and architectural composition metadata."""

import json
from importlib.metadata import version

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

router = APIRouter()


class BuildDeployment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    revision: str


class BuildResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository: str
    application_version: str
    source_commit: str | None
    deployment: BuildDeployment
    architectural_packages: dict[str, str]
    uv_lock_sha256: str | None
    immutable: bool


@router.get("/build", response_model=BuildResponse)
async def build(request: Request) -> BuildResponse:
    metadata_path = request.app.state.settings.build_metadata_path
    if metadata_path.is_file():
        try:
            immutable = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read immutable build metadata: {metadata_path}") from exc
        try:
            return BuildResponse.model_validate(immutable)
        except ValueError as exc:
            raise RuntimeError(
                "Immutable build metadata does not match the build contract"
            ) from exc

    manifest = request.app.state.deployment
    settings = request.app.state.settings
    components = [
        manifest.application,
        manifest.platform,
        manifest.integrations.enterprise_llm,
        *manifest.agents.values(),
    ]
    return BuildResponse(
        repository="agentic-orchestration",
        application_version=version("agentic-orchestration"),
        deployment=BuildDeployment.model_validate(manifest.deployment.model_dump()),
        source_commit=settings.source_commit,
        architectural_packages={component.package: component.version for component in components},
        uv_lock_sha256=None,
        immutable=False,
    )
