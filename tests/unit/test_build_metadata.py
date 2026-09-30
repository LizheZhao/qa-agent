import hashlib
from pathlib import Path

import pytest

from scripts.generate_build_metadata import create_build_metadata

ROOT = Path(__file__).parents[2]


@pytest.mark.unit
def test_build_metadata_comes_from_deployment_manifest_and_lockfile() -> None:
    lockfile = ROOT / "uv.lock"
    metadata = create_build_metadata(
        ROOT / "deployments/local.yaml",
        ROOT / "schemas/deployment.schema.json",
        lockfile,
        "source-test",
    )
    assert metadata["source_commit"] == "source-test"
    assert metadata["application_version"] == "0.9.0"
    assert metadata["deployment"] == {"name": "local-development", "revision": "0.10.0"}
    assert metadata["architectural_packages"] == {
        "agentic-orchestration": "0.9.0",
        "orchestration-core": "0.7.0",
        "enterprise-llm": "0.1.0",
        "router-agent": "0.5.0",
        "marketing-science-agent": "0.3.0",
        "ask-genome-agent": "0.7.0",
    }
    assert metadata["uv_lock_sha256"] == hashlib.sha256(lockfile.read_bytes()).hexdigest()
    assert metadata["immutable"] is True
