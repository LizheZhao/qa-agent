"""The footprint reporter must be incapable of changing Docker state.

These tests are the safeguard. The reporter runs against a shared host under a build
suspension after a Sev-1, and under an authorization scoped to exact
`agentic-orchestration` resources only. An accidental mutation, or an out-of-scope
resource being counted as in-scope, are the failure modes that matter.
"""

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from scripts import report_docker_footprint as reporter

MODULE_SOURCE = Path(inspect.getsourcefile(reporter) or "").read_text(encoding="utf-8")
MODULE_TREE = ast.parse(MODULE_SOURCE)

MUTATING_VECTORS = [
    ("docker", "system", "prune"),
    ("docker", "image", "prune", "--force"),
    ("docker", "container", "prune"),
    ("docker", "builder", "prune", "--all"),
    ("docker", "volume", "prune"),
    ("docker", "rm", "some-container"),
    ("docker", "rmi", "agentic-orchestration:local"),
    ("docker", "stop", "worker"),
    ("docker", "kill", "worker"),
    ("docker", "build", "-t", "agentic-orchestration:local", "."),
    ("docker", "push", "localhost:5000/agentic-orchestration:poc"),
    ("docker", "pull", "agentic-orchestration:local"),
    ("docker", "tag", "a", "b"),
    ("docker", "exec", "worker", "sh"),
    ("docker", "run", "agentic-orchestration:local"),
    ("docker", "commit", "worker"),
    ("docker", "images", "--format", "{{json .}}", "-f"),
]


# --------------------------------------------------------------------------- mutation


@pytest.mark.unit
@pytest.mark.parametrize("argv", MUTATING_VECTORS)
def test_mutating_vectors_are_refused(argv: tuple[str, ...]) -> None:
    with pytest.raises(reporter.ForbiddenDockerCommand):
        reporter.assert_read_only(argv)


@pytest.mark.unit
def test_every_allowlisted_command_is_read_only() -> None:
    read_only_subcommands = {"system", "images", "ps", "buildx", "container"}
    vectors = [*reporter._READ_ONLY_COMMANDS.values(), reporter._INSPECT_CONTAINER]
    for argv in vectors:
        reporter.assert_read_only(argv)
        assert argv[0] == "docker", argv
        assert argv[1] in read_only_subcommands, argv
    for argv in vectors:
        if argv[1] == "buildx":
            assert argv[2] == "du", "buildx is only ever used for du, never build"
        if argv[1] == "container":
            assert argv[2] == "inspect", "container is only ever used for inspect"


@pytest.mark.unit
def test_run_docker_refuses_a_command_outside_the_allowlist() -> None:
    with pytest.raises(reporter.ForbiddenDockerCommand):
        reporter.run_docker("system_prune")
    with pytest.raises(reporter.ForbiddenDockerCommand):
        reporter.run_docker("images; docker rmi x")


@pytest.mark.unit
@pytest.mark.parametrize(
    "injected",
    [
        "abc123; docker rmi x",
        "--force",
        "$(docker system prune)",
        "../etc/passwd",
        "worker",
        "",
        "ZZZZZZZZZZZZ",
        "abc",
    ],
)
def test_container_inspection_refuses_anything_but_a_bare_hex_id(injected: str) -> None:
    with pytest.raises(reporter.ForbiddenDockerCommand):
        reporter.inspect_container_image(injected)


@pytest.mark.unit
def test_no_subprocess_call_uses_a_shell() -> None:
    for call in [n for n in ast.walk(MODULE_TREE) if isinstance(n, ast.Call)]:
        for keyword in call.keywords:
            if keyword.arg == "shell":
                assert isinstance(keyword.value, ast.Constant)
                assert keyword.value.value is False, "shell=True is never permitted"
    banned = {"system", "popen", "spawn", "spawnl", "spawnv", "execv", "execve"}
    for node in ast.walk(MODULE_TREE):
        if isinstance(node, ast.Attribute) and node.attr in banned:
            assert getattr(node.value, "id", "") != "os", f"os.{node.attr} is never permitted"


@pytest.mark.unit
def test_module_never_names_a_mutating_docker_subcommand_as_a_command() -> None:
    literals: set[Any] = set()
    for node in ast.walk(MODULE_TREE):
        if isinstance(node, ast.Tuple):
            values = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            if values and values[0] == "docker":
                literals.update(values)
    forbidden = literals.intersection({"prune", "rm", "rmi", "build", "push", "stop", "exec"})
    assert not forbidden, f"a docker vector literal contains {forbidden}"


@pytest.mark.unit
def test_registry_access_is_get_only_and_never_deletes() -> None:
    assert "urlopen" in MODULE_SOURCE
    methods = [
        keyword.value.value
        for node in ast.walk(MODULE_TREE)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "method" and isinstance(keyword.value, ast.Constant)
    ]
    assert methods == ["GET"], methods

    docstring_nodes = set()
    for node in ast.walk(MODULE_TREE):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstring_nodes.add(id(first.value))
    offending = {
        node.value
        for node in ast.walk(MODULE_TREE)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
        and "DELETE" in node.value.upper()
    }
    assert not offending, offending


# ------------------------------------------------------------------------ scope


@pytest.mark.unit
def test_only_the_exact_repository_is_in_scope() -> None:
    assert reporter.classify_repository("agentic-orchestration") == reporter.PROJECT
    assert reporter.classify_repository("agentic-orchestration:ckpt11c") == reporter.PROJECT


@pytest.mark.unit
@pytest.mark.parametrize(
    "reference",
    [
        "agentic-orchestration-experiment",
        "agentic-orchestration-experiment:v1",
        "old-agentic-orchestration",
        "agentic-orchestration2",
    ],
)
def test_near_match_names_are_never_in_scope(reference: str) -> None:
    assert reporter.classify_repository(reference) == reporter.NEAR_MATCH


@pytest.mark.unit
@pytest.mark.parametrize(
    "reference",
    [
        "localhost:5000/ask-genome-classifier:3453b9c",
        "localhost:5000/ask-genome-core:ab-test",
        "localhost:5000/ask-genome-app",
        "apc",
        "apfe",
        "registry:2",
        "jaegertracing/all-in-one:latest",
        "ghcr.io/huggingface/text-generation-inference:3.1.1",
    ],
)
def test_excluded_resources_are_never_in_scope(reference: str) -> None:
    assert reporter.classify_repository(reference) == reporter.UNRELATED


@pytest.mark.unit
@pytest.mark.parametrize(
    "reference",
    [
        "agentic-orchestration@sha256:" + "a" * 64,
        "agentic-orchestration@sha256:abc123def4567890",
    ],
)
def test_digest_references_to_the_project_are_in_scope(reference: str) -> None:
    # A deployment pins by digest, so the digest form is the one a release actually
    # runs under. Treating it as a near-match would hide the active release from the
    # policy and leave it out of any deletion manifest.
    assert reporter.classify_repository(reference) == reporter.PROJECT


@pytest.mark.unit
def test_registry_qualified_digest_references_stay_in_their_own_class() -> None:
    reference = "localhost:5000/agentic-orchestration@sha256:" + "b" * 64
    assert reporter.classify_repository(reference) == reporter.PROJECT_REGISTRY_TAGGED


@pytest.mark.unit
def test_a_near_match_digest_reference_is_still_out_of_scope() -> None:
    reference = "agentic-orchestration-experiment@sha256:" + "c" * 64
    assert reporter.classify_repository(reference) == reporter.NEAR_MATCH


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("agentic-orchestration", "agentic-orchestration"),
        ("agentic-orchestration:0.8.0", "agentic-orchestration"),
        ("agentic-orchestration@sha256:" + "a" * 64, "agentic-orchestration"),
        ("localhost:5000/agentic-orchestration", "localhost:5000/agentic-orchestration"),
        ("localhost:5000/agentic-orchestration:poc", "localhost:5000/agentic-orchestration"),
        ("registry:2", "registry"),
        ("ghcr.io/a/b:1.2.3", "ghcr.io/a/b"),
    ],
)
def test_repository_path_strips_tags_and_digests_but_not_registry_ports(
    reference: str, expected: str
) -> None:
    assert reporter.repository_path(reference) == expected


@pytest.mark.unit
@pytest.mark.parametrize("reference", ["sha256:" + "d" * 64, "bec4c8785967", "a" * 64])
def test_a_bare_identifier_names_no_repository_and_is_unidentified(reference: str) -> None:
    # Fail closed: an unresolved identifier must never be assumed to be ours.
    assert reporter.classify_repository(reference) == reporter.UNIDENTIFIED


@pytest.mark.unit
def test_a_container_pinned_by_project_digest_is_counted() -> None:
    containers = reporter.collect_containers(
        [
            {
                "Names": "worker",
                "Image": "agentic-orchestration@sha256:" + "a" * 64,
                "State": "running",
                "ID": "f" * 64,
            }
        ]
    )
    assert containers[reporter.PROJECT].count == 1


@pytest.mark.unit
def test_registry_qualified_project_names_are_held_separately() -> None:
    # Same repository name, but reached through a registry: visible, never counted.
    for reference in (
        "localhost:5000/agentic-orchestration",
        "localhost:5000/agentic-orchestration:poc",
    ):
        assert reporter.classify_repository(reference) == reporter.PROJECT_REGISTRY_TAGGED


@pytest.mark.unit
@pytest.mark.parametrize("reference", ["<none>", "<none>:<none>", "", "N/A"])
def test_unidentified_references_are_never_in_scope(reference: str) -> None:
    assert reporter.classify_repository(reference) == reporter.UNIDENTIFIED


@pytest.mark.unit
def test_only_the_project_class_is_measured_against_the_policy() -> None:
    images = reporter.collect_images(
        [
            {
                "Repository": "agentic-orchestration",
                "Tag": "0.7.0",
                "ID": "sha256:aa",
                "Size": "1GB",
            },
            {
                "Repository": "agentic-orchestration-x",
                "Tag": "v1",
                "ID": "sha256:bb",
                "Size": "1GB",
            },
            {
                "Repository": "localhost:5000/ask-genome-core",
                "Tag": "a",
                "ID": "sha256:cc",
                "Size": "9GB",
            },
            {"Repository": "<none>", "Tag": "<none>", "ID": "sha256:dd", "Size": "1GB"},
        ]
    )
    assert images[reporter.PROJECT].count == 1
    assert images[reporter.NEAR_MATCH].count == 1
    assert images[reporter.UNRELATED].count == 1
    assert images[reporter.UNIDENTIFIED].count == 1

    findings = reporter.evaluate_project(images, reporter.collect_containers([]), None, False)
    image_finding = next(f for f in findings if f.boundary == "project images")
    assert "1 distinct" in image_finding.observed
    assert image_finding.status == reporter.WITHIN


# ------------------------------------------------------------------ counting


@pytest.mark.unit
def test_images_are_counted_by_distinct_id_not_by_tag_row() -> None:
    # The policy permits several release tags on one digest. Counting tag rows would
    # report a compliant active+rollback pair as a breach.
    images = reporter.collect_images(
        [
            {
                "Repository": "agentic-orchestration",
                "Tag": "0.8.0",
                "ID": "sha256:aa",
                "Size": "1GB",
            },
            {
                "Repository": "agentic-orchestration",
                "Tag": "git-5deb340",
                "ID": "sha256:aa",
                "Size": "1GB",
            },
            {
                "Repository": "agentic-orchestration",
                "Tag": "production",
                "ID": "sha256:aa",
                "Size": "1GB",
            },
            {
                "Repository": "agentic-orchestration",
                "Tag": "0.7.0",
                "ID": "sha256:bb",
                "Size": "1GB",
            },
        ]
    )
    assert images[reporter.PROJECT].count == 2
    assert images[reporter.PROJECT].bytes == 2_000_000_000
    findings = reporter.evaluate_project(images, reporter.collect_containers([]), None, False)
    assert next(f for f in findings if f.boundary == "project images").status == reporter.WITHIN


# ------------------------------------------------------------------ registry


@pytest.mark.unit
def test_unreachable_registry_is_unknown_not_zero() -> None:
    reading = reporter.RegistryReading(reachable=False, error="URLError")
    findings = reporter.evaluate_project(
        reporter.collect_images([]), reporter.collect_containers([]), reading, False
    )
    registry_finding = next(f for f in findings if f.boundary == "registry project artifacts")
    assert registry_finding.status == reporter.UNKNOWN
    assert "unreachable" in registry_finding.observed


@pytest.mark.unit
def test_registry_not_inspected_is_unknown() -> None:
    findings = reporter.evaluate_project(
        reporter.collect_images([]), reporter.collect_containers([]), None, False
    )
    registry_finding = next(f for f in findings if f.boundary == "registry project artifacts")
    assert registry_finding.status == reporter.UNKNOWN


@pytest.mark.unit
def test_reachable_registry_with_no_tags_is_within() -> None:
    reading = reporter.RegistryReading(reachable=True, tags=[])
    findings = reporter.evaluate_project(
        reporter.collect_images([]), reporter.collect_containers([]), reading, False
    )
    registry_finding = next(f for f in findings if f.boundary == "registry project artifacts")
    assert registry_finding.status == reporter.WITHIN


@pytest.mark.unit
def test_live_unreachable_registry_reports_unreachable() -> None:
    reading = reporter.read_registry_project_artifacts("http://127.0.0.1:59999")
    assert reading.reachable is False
    assert reading.tags == []


# ------------------------------------------------------------------ host scope


@pytest.mark.unit
def test_build_cache_is_a_host_observation_not_a_project_boundary() -> None:
    project = reporter.evaluate_project(
        reporter.collect_images([]), reporter.collect_containers([]), None, False
    )
    assert not [f for f in project if "cache" in f.boundary]
    host = reporter.evaluate_host(20 * 1024**3, 587, 512)
    assert [f.boundary for f in host] == ["host build cache size", "host build cache age"]
    for finding in host:
        assert finding.status == reporter.BREACH
        assert "not attributable" in finding.detail


# ------------------------------------------------------------------ policy


@pytest.mark.unit
def test_policy_constants_match_the_confirmed_retention_contract() -> None:
    assert reporter.MAX_PROJECT_IMAGES == 2
    assert reporter.MAX_PROJECT_IMAGES_DURING_REPLACEMENT == 3
    assert reporter.MAX_PROJECT_CONTAINERS == 1
    assert reporter.MAX_PROJECT_CONTAINERS_DURING_ROLLOUT == 2
    assert reporter.MAX_BUILD_CACHE_BYTES == 5 * 1024**3
    assert reporter.MAX_BUILD_CACHE_AGE_DAYS == 14
    assert reporter.MAX_REGISTRY_PROJECT_ARTIFACTS == 0


@pytest.mark.unit
def test_counts_have_no_warning_tier_because_steady_state_equals_the_limit() -> None:
    # Two images (active + rollback) is the intended steady state, not a warning.
    assert reporter.status_for(1, 2) == reporter.WITHIN
    assert reporter.status_for(2, 2) == reporter.WITHIN
    assert reporter.status_for(3, 2) == reporter.BREACH


@pytest.mark.unit
def test_consumable_budgets_warn_before_the_limit() -> None:
    fraction = reporter.WARN_FRACTION
    limit = reporter.MAX_BUILD_CACHE_BYTES
    assert reporter.status_for(1 * 1024**3, limit, fraction) == reporter.WITHIN
    assert reporter.status_for(4.5 * 1024**3, limit, fraction) == reporter.WARN
    assert reporter.status_for(6 * 1024**3, limit, fraction) == reporter.BREACH
    # 14 days * 0.8 = 11.2, so 11 is still within and 12 warns.
    assert reporter.status_for(11, reporter.MAX_BUILD_CACHE_AGE_DAYS, fraction) == reporter.WITHIN
    assert reporter.status_for(12, reporter.MAX_BUILD_CACHE_AGE_DAYS, fraction) == reporter.WARN


@pytest.mark.unit
def test_size_parsing_handles_docker_units() -> None:
    assert reporter.parse_size("1.06GB") == 1_060_000_000
    assert reporter.parse_size("358MB") == 358_000_000
    assert reporter.parse_size("53.34GB (73%)") == 53_340_000_000
    assert reporter.parse_size("0B") == 0
    assert reporter.parse_size("nonsense") == 0
