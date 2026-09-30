#!/usr/bin/env python3
"""Generate safe image metadata from the deployment manifest and lockfile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from agentic_orchestration.execution.deployment_loader import load_and_validate_deployment


def create_build_metadata(
    deployment_path: Path,
    schema_path: Path,
    lockfile_path: Path,
    source_commit: str,
) -> dict[str, Any]:
    manifest = load_and_validate_deployment(
        deployment_path,
        schema_path,
        load_agents=False,
    ).manifest
    components = [
        manifest.application,
        manifest.platform,
        manifest.integrations.enterprise_llm,
        *manifest.agents.values(),
    ]
    return {
        "repository": "agentic-orchestration",
        "application_version": manifest.application.version,
        "source_commit": source_commit,
        "deployment": manifest.deployment.model_dump(),
        "architectural_packages": {
            component.package: component.version for component in components
        },
        "uv_lock_sha256": hashlib.sha256(lockfile_path.read_bytes()).hexdigest(),
        "immutable": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", type=Path, default=Path("deployments/local.yaml"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/deployment.schema.json"))
    parser.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
    parser.add_argument("--output", type=Path, default=Path("build-metadata.json"))
    parser.add_argument("--source-commit", default="unknown")
    args = parser.parse_args()

    metadata = create_build_metadata(
        args.deployment,
        args.schema,
        args.lockfile,
        args.source_commit,
    )
    args.output.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
