"""The committed projections must stay mechanically derived from uv.lock.

The Docker dependency layer is keyed by requirements-runtime.txt and the image builds
wheels with --no-build-isolation against requirements-build.txt. If either drifts from
the lockfile, or either stops being hash-pinned, the image stops being reproducible.
"""

import re
from pathlib import Path

import pytest

from scripts.generate_dependency_projections import BANNER, PROJECTIONS

ROOT = Path(__file__).parents[2]
REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(==|\s@\s)")


def _requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = REQUIREMENT.match(line)
        if match:
            names.add(match.group("name").lower())
    return names


@pytest.mark.unit
def test_both_projections_are_committed() -> None:
    assert {p.path.name for p in PROJECTIONS} == {
        "requirements-runtime.txt",
        "requirements-build.txt",
    }
    for projection in PROJECTIONS:
        assert (ROOT / projection.path).exists(), projection.path


@pytest.mark.unit
def test_projections_use_locked_not_frozen() -> None:
    # --frozen would happily export a stale lockfile after a dependency edit.
    for projection in PROJECTIONS:
        assert "--locked" in projection.args, projection.path
        assert "--frozen" not in projection.args, projection.path


@pytest.mark.unit
def test_projections_exclude_the_workspace_so_internal_bumps_cannot_change_them() -> None:
    for projection in PROJECTIONS:
        assert "--no-emit-workspace" in projection.args, projection.path
        assert "--no-emit-project" in projection.args, projection.path
        # Annotations name the workspace packages that pulled each dependency.
        assert "--no-annotate" in projection.args, projection.path


@pytest.mark.unit
@pytest.mark.parametrize("projection", PROJECTIONS, ids=lambda p: p.path.name)
def test_every_requirement_is_hash_pinned(projection) -> None:  # type: ignore[no-untyped-def]
    text = (ROOT / projection.path).read_text(encoding="utf-8")
    assert text.startswith(BANNER)
    assert "--no-hashes" not in projection.args
    body = [line for line in text.splitlines() if REQUIREMENT.match(line)]
    assert body, f"{projection.path} projected no requirements"
    assert "--hash=sha256:" in text
    # Each requirement line opens a continuation carrying its hashes.
    for line in body:
        assert line.rstrip().endswith("\\"), line


@pytest.mark.unit
def test_no_workspace_distribution_appears_in_either_projection() -> None:
    workspace = {
        "agentic-orchestration",
        "orchestration-core",
        "orchestration-tools",
        "enterprise-llm",
        "marketing-science-agent",
        "router-agent",
        "ask-genome-agent",
    }
    for projection in PROJECTIONS:
        leaked = _requirement_names(ROOT / projection.path) & workspace
        assert not leaked, f"{projection.path} leaks {leaked}"


@pytest.mark.unit
def test_the_build_backend_is_pinned_and_kept_out_of_the_runtime_closure() -> None:
    build = _requirement_names(ROOT / "requirements-build.txt")
    runtime = _requirement_names(ROOT / "requirements-runtime.txt")
    assert "hatchling" in build, "the build backend must be projected, not fetched at build time"
    assert "hatchling" not in runtime, "the build backend must not ship in the runtime image"


@pytest.mark.unit
def test_the_runtime_closure_still_carries_the_ask_genome_dependencies() -> None:
    runtime = _requirement_names(ROOT / "requirements-runtime.txt")
    assert {"spacy", "nltk", "pandas", "en-core-web-sm"} <= runtime


@pytest.mark.unit
def test_every_build_system_declares_the_pinned_backend_range() -> None:
    pyprojects = [ROOT / "pyproject.toml", *(ROOT / "packages").rglob("pyproject.toml")]
    checked = 0
    for path in pyprojects:
        text = path.read_text(encoding="utf-8")
        if "[build-system]" not in text:
            continue
        checked += 1
        assert 'requires = ["hatchling"]' not in text, f"{path} leaves the backend unpinned"
        assert "hatchling>=1.27,<2" in text, path
    assert checked == 7
