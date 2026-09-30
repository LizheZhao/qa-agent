import subprocess
import sys

import pytest


@pytest.mark.integration
def test_runtime_packages_import_outside_repository_root(tmp_path) -> None:  # type: ignore[no-untyped-def]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import agentic_orchestration.main; "
                "import integrations.mongodb; "
                "import orchestration_tools; "
                "import marketing_science_agent; "
                "import router_agent"
            ),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.integration
def test_enterprise_distribution_does_not_ship_scripted_test_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; assert importlib.util.find_spec('enterprise_llm.fake') is None",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
