#!/usr/bin/env python3
"""Validate deployment schema, installed versions, and declared entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_orchestration.execution.deployment_loader import (
    DeploymentValidationError,
    load_and_validate_deployment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("deployment", nargs="?", type=Path, default=Path("deployments/local.yaml"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/deployment.schema.json"))
    args = parser.parse_args()
    try:
        loaded = load_and_validate_deployment(args.deployment, args.schema)
    except DeploymentValidationError as exc:
        print(f"INVALID: {exc}")
        return 1
    manifest = loaded.manifest
    print(
        f"VALID: deployment={manifest.deployment.name} revision={manifest.deployment.revision} "
        f"agents={len(manifest.agents)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
