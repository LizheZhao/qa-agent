#!/usr/bin/env python3
"""Report the agentic-orchestration Docker footprint against the retention policy.

Read-only by construction. Every Docker invocation comes from _READ_ONLY_COMMANDS,
every argument vector is checked against _FORBIDDEN_TOKENS before it runs, the one
parameterised command accepts only a validated hex identifier, and subprocesses run as
argument vectors with shell=False. The script cannot remove, prune, stop, tag, build,
push, or execute anything, and it never issues a registry DELETE.

Scope follows the temporary authorization held by `mlu`: exact `agentic-orchestration`
resources only. Everything else is surfaced so it is visible, and is never counted
against the policy or offered for deletion. Registry-qualified references and
near-match repository names are reported in their own classes precisely so they cannot
be mistaken for in-scope resources.

Host-wide build cache is observation only. Its ownership is not attributable, so the
reporter measures it and reports a breach but never implies it may be removed.

Policy source: docs/docker-retention-policy.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

PROJECT_REPOSITORY: Final = "agentic-orchestration"

# Confirmed policy. See docs/docker-retention-policy.md.
MAX_PROJECT_IMAGES: Final = 2
MAX_PROJECT_IMAGES_DURING_REPLACEMENT: Final = 3
MAX_PROJECT_CONTAINERS: Final = 1
MAX_PROJECT_CONTAINERS_DURING_ROLLOUT: Final = 2
MAX_BUILD_CACHE_BYTES: Final = 5 * 1024**3
MAX_BUILD_CACHE_AGE_DAYS: Final = 14
MAX_REGISTRY_PROJECT_ARTIFACTS: Final = 0

# Alert before the limit rather than after disk pressure has begun.
WARN_FRACTION: Final = 0.8

WITHIN: Final = "WITHIN"
WARN: Final = "WARN"
BREACH: Final = "BREACH"
UNKNOWN: Final = "UNKNOWN"

# Classifications. Only PROJECT is measured against the policy or eligible for a
# deletion proposal. The rest exist so nothing is silently swept into scope.
PROJECT: Final = "project"
PROJECT_REGISTRY_TAGGED: Final = "project_registry_tagged"
NEAR_MATCH: Final = "near_match"
UNRELATED: Final = "unrelated"
UNIDENTIFIED: Final = "unidentified"
CLASSES: Final = (PROJECT, PROJECT_REGISTRY_TAGGED, NEAR_MATCH, UNRELATED, UNIDENTIFIED)

# The complete set of Docker invocations this script may make. Every vector is an
# inventory or inspection read.
_READ_ONLY_COMMANDS: Final[dict[str, tuple[str, ...]]] = {
    "system_df": ("docker", "system", "df", "--format", "{{json .}}"),
    "images": ("docker", "images", "--all", "--no-trunc", "--format", "{{json .}}"),
    "containers": ("docker", "ps", "--all", "--no-trunc", "--format", "{{json .}}"),
    "build_cache": ("docker", "buildx", "du", "--verbose"),
}

# The only command taking a caller-supplied value. The value must match _HEX_ID, so no
# flag, path, or second command can be smuggled in.
_INSPECT_CONTAINER: Final[tuple[str, ...]] = (
    "docker",
    "container",
    "inspect",
    "--format",
    "{{.Image}}",
)
_HEX_ID: Final = re.compile(r"\A[0-9a-f]{12,64}\Z")
# A reference that is only an identifier names no repository, so it cannot be
# attributed to this project without resolution.
_BARE_IDENTIFIER: Final = re.compile(r"\A(sha256:)?[0-9a-f]{12,64}\Z")

# Any of these appearing anywhere in an argument vector aborts the run, even if a
# future edit adds it to _READ_ONLY_COMMANDS.
_FORBIDDEN_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "rm",
        "rmi",
        "remove",
        "prune",
        "stop",
        "kill",
        "restart",
        "start",
        "pause",
        "unpause",
        "exec",
        "run",
        "create",
        "build",
        "push",
        "pull",
        "tag",
        "untag",
        "commit",
        "cp",
        "import",
        "load",
        "save",
        "export",
        "update",
        "rename",
        "login",
        "logout",
        "attach",
        "wait",
        "--force",
        "-f",
        "--rm",
    }
)


class ForbiddenDockerCommand(RuntimeError):
    """Raised when an invocation is not a permitted read-only command."""


def assert_read_only(argv: tuple[str, ...]) -> None:
    """Reject any vector containing a mutating token."""
    offending = sorted(_FORBIDDEN_TOKENS.intersection(argv))
    if offending:
        raise ForbiddenDockerCommand(
            f"refusing to run a command containing {offending}: {' '.join(argv)}"
        )


def _execute(argv: tuple[str, ...]) -> str:
    assert_read_only(argv)
    # Fixed, validated argument vector; shell=False.
    completed = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        encoding="utf-8",
        shell=False,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {completed.returncode}: {completed.stderr}")
    return completed.stdout


def run_docker(name: str) -> str:
    """Run one allowlisted read-only Docker command and return its stdout."""
    if name not in _READ_ONLY_COMMANDS:
        raise ForbiddenDockerCommand(f"{name!r} is not an allowlisted read-only command")
    return _execute(_READ_ONLY_COMMANDS[name])


def inspect_container_image(container_id: str) -> str:
    """Resolve one container to its image ID. Read-only; the id must be a hex digest."""
    if not _HEX_ID.match(container_id):
        raise ForbiddenDockerCommand(f"{container_id!r} is not a bare hexadecimal container id")
    return _execute((*_INSPECT_CONTAINER, container_id)).strip()


def repository_path(reference: str) -> str:
    """Strip an @digest or :tag suffix, leaving the repository path.

    A digest is stripped first: `repo@sha256:...` ends in a colon-separated hex string
    that is not a tag. A colon is a tag separator only when what follows contains no
    path separator; otherwise it is a registry port, as in `localhost:5000/repo`.
    """
    if "@" in reference:
        return reference.split("@", 1)[0]
    head, separator, tail = reference.rpartition(":")
    if separator and "/" not in tail:
        return head
    return reference


def classify_repository(repository: str) -> str:
    """Classify a repository reference.

    Only an exact `agentic-orchestration` repository is in scope. A registry-qualified
    reference to the same name and any near-match name are given their own classes so
    they stay visible without ever being counted as in-scope. A bare identifier carries
    no repository at all, so it is unidentified and must be resolved before it can be
    acted on.
    """
    reference = repository.strip()
    if reference in {"<none>", "", "N/A", "<none>:<none>"}:
        return UNIDENTIFIED
    if _BARE_IDENTIFIER.match(reference):
        return UNIDENTIFIED
    path = repository_path(reference)
    bare = path.split("/")[-1]
    if bare == PROJECT_REPOSITORY:
        return PROJECT if "/" not in path else PROJECT_REGISTRY_TAGGED
    if PROJECT_REPOSITORY in bare:
        return NEAR_MATCH
    return UNRELATED


def parse_size(text: str) -> int:
    """Parse a Docker human size such as '1.06GB' or '19.33GB' into bytes."""
    units = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}
    cleaned = text.strip().split("(")[0].strip()
    for suffix, factor in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if cleaned.upper().endswith(suffix):
            try:
                return int(float(cleaned[: -len(suffix)].strip()) * factor)
            except ValueError:
                return 0
    return 0


def _json_lines(payload: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


@dataclass
class ImageGroup:
    """Images of one classification, deduplicated by image ID."""

    ids: dict[str, int] = field(default_factory=dict)
    tags: dict[str, list[str]] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Distinct images, not tag rows: one digest may carry several release tags."""
        return len(self.ids)

    @property
    def bytes(self) -> int:
        return sum(self.ids.values())


@dataclass
class ContainerGroup:
    count: int = 0
    items: list[str] = field(default_factory=list)


@dataclass
class Finding:
    boundary: str
    observed: str
    limit: str
    status: str
    detail: str = ""


def collect_images(rows: list[dict[str, Any]] | None = None) -> dict[str, ImageGroup]:
    """Group images by classification, deduplicating tag rows by image ID."""
    if rows is None:
        rows = _json_lines(run_docker("images"))
    groups = {name: ImageGroup() for name in CLASSES}
    for row in rows:
        repository = str(row.get("Repository", ""))
        kind = classify_repository(repository)
        group = groups[kind]
        image_id = str(row.get("ID", ""))
        group.ids.setdefault(image_id, parse_size(str(row.get("Size", "0B"))))
        group.tags.setdefault(image_id, []).append(f"{repository}:{row.get('Tag')}")
    return groups


def _resolved_image_id(container_id: str) -> str:
    try:
        return inspect_container_image(container_id)
    except (ForbiddenDockerCommand, RuntimeError):
        return "unresolved"


def collect_containers(
    rows: list[dict[str, Any]] | None = None,
    resolve_image_ids: bool = False,
) -> dict[str, ContainerGroup]:
    """Group containers by the classification of the image they reference."""
    if rows is None:
        rows = _json_lines(run_docker("containers"))
    groups = {name: ContainerGroup() for name in CLASSES}
    for row in rows:
        image = str(row.get("Image", ""))
        kind = classify_repository(image)
        group = groups[kind]
        group.count += 1
        resolve = resolve_image_ids and kind in {PROJECT, PROJECT_REGISTRY_TAGGED}
        image_id = _resolved_image_id(str(row.get("ID", ""))) if resolve else ""
        suffix = f" image={image_id}" if image_id else ""
        group.items.append(f"{row.get('Names')} <- {image} ({row.get('State')}){suffix}")
    return groups


def collect_build_cache() -> tuple[int, int, int]:
    """Return (bytes actually on disk, record count, oldest record age in days).

    Size comes from `docker system df`, not from summing `buildx du` records: cache
    records share layers, so summing them reports far more than the disk holds.
    """
    total = 0
    records = 0
    for row in _json_lines(run_docker("system_df")):
        if str(row.get("Type", "")).strip().lower() == "build cache":
            total = parse_size(str(row.get("Size", "0B")))
            records = int(str(row.get("TotalCount", "0")) or 0)
    oldest: datetime | None = None
    for line in run_docker("build_cache").splitlines():
        key, _, value = line.partition(":")
        if key.strip() != "Created at":
            continue
        try:
            created = datetime.strptime(value.strip()[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        except ValueError:
            continue
        if oldest is None or created < oldest:
            oldest = created
    age_days = 0 if oldest is None else (datetime.now(UTC) - oldest).days
    return total, records, age_days


@dataclass
class RegistryReading:
    """A registry read that distinguishes 'no artifacts' from 'could not tell'."""

    reachable: bool
    tags: list[str] = field(default_factory=list)
    error: str = ""


def read_registry_project_artifacts(base_url: str) -> RegistryReading:
    """List this project's tags in a registry. Read-only GET; never DELETE.

    An unreachable registry is reported as unreachable rather than as zero artifacts:
    reporting zero would silently turn an unknown into a pass.
    """
    url = f"{base_url.rstrip('/')}/v2/{PROJECT_REPOSITORY}/tags/list"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return RegistryReading(reachable=True, tags=[])
        return RegistryReading(reachable=False, error=f"HTTP {exc.code}")
    except Exception as exc:  # any failure means "could not tell", not "zero"
        return RegistryReading(reachable=False, error=type(exc).__name__)
    return RegistryReading(reachable=True, tags=[str(tag) for tag in (payload.get("tags") or [])])


def status_for(observed: float, limit: float, warn_fraction: float | None = None) -> str:
    """Grade an observation against a limit.

    Small integer counts get no warning tier: the policy's steady state for images is
    exactly the limit, so an 80% warning would fire permanently and mean nothing.
    Consumable budgets (bytes, days) do warn early, which is what "alert before
    reaching the limit" is for.
    """
    if observed > limit:
        return BREACH
    if warn_fraction is not None and limit > 0 and observed >= limit * warn_fraction:
        return WARN
    return WITHIN


def evaluate_project(
    images: dict[str, ImageGroup],
    containers: dict[str, ContainerGroup],
    registry: RegistryReading | None,
    during_replacement: bool,
) -> list[Finding]:
    """Boundaries the policy actually enforces. Exact-match project resources only."""
    image_limit = (
        MAX_PROJECT_IMAGES_DURING_REPLACEMENT if during_replacement else MAX_PROJECT_IMAGES
    )
    container_limit = (
        MAX_PROJECT_CONTAINERS_DURING_ROLLOUT if during_replacement else MAX_PROJECT_CONTAINERS
    )
    gb = 1024**3
    findings = [
        Finding(
            "project images",
            f"{images[PROJECT].count} distinct ({images[PROJECT].bytes / gb:.2f} GiB summed)",
            str(image_limit),
            status_for(images[PROJECT].count, image_limit),
            "active + one rollback" + (" + candidate" if during_replacement else ""),
        ),
        Finding(
            "project containers",
            str(containers[PROJECT].count),
            str(container_limit),
            status_for(containers[PROJECT].count, container_limit),
            "one active" + (" + candidate" if during_replacement else ""),
        ),
    ]
    if registry is None:
        findings.append(
            Finding(
                "registry project artifacts",
                "not inspected",
                str(MAX_REGISTRY_PROJECT_ARTIFACTS),
                UNKNOWN,
                "pass --registry to inspect",
            )
        )
    elif not registry.reachable:
        findings.append(
            Finding(
                "registry project artifacts",
                f"unreachable ({registry.error})",
                str(MAX_REGISTRY_PROJECT_ARTIFACTS),
                UNKNOWN,
                "an unreachable registry is not evidence of zero artifacts",
            )
        )
    else:
        listed = f" ({', '.join(registry.tags)})" if registry.tags else ""
        findings.append(
            Finding(
                "registry project artifacts",
                f"{len(registry.tags)}{listed}",
                str(MAX_REGISTRY_PROJECT_ARTIFACTS),
                status_for(len(registry.tags), MAX_REGISTRY_PROJECT_ARTIFACTS),
                "none unless deployment proves a registry is required",
            )
        )
    return findings


def evaluate_host(cache_bytes: int, cache_records: int, cache_age_days: int) -> list[Finding]:
    """Host-wide observations. Not attributable to this project, not actionable here."""
    gb = 1024**3
    return [
        Finding(
            "host build cache size",
            f"{cache_bytes / gb:.2f} GiB across {cache_records} records",
            f"{MAX_BUILD_CACHE_BYTES / gb:.0f} GiB",
            status_for(cache_bytes, MAX_BUILD_CACHE_BYTES, WARN_FRACTION),
            "observation only; ownership not attributable",
        ),
        Finding(
            "host build cache age",
            f"{cache_age_days} days (oldest record)",
            f"{MAX_BUILD_CACHE_AGE_DAYS} days",
            status_for(cache_age_days, MAX_BUILD_CACHE_AGE_DAYS, WARN_FRACTION),
            "observation only; ownership not attributable",
        ),
    ]


def _print_table(title: str, findings: list[Finding]) -> None:
    print(f"\n{title}")
    print(f"{'BOUNDARY':<30}{'OBSERVED':<36}{'LIMIT':<10}STATUS")
    for finding in findings:
        print(f"{finding.boundary:<30}{finding.observed:<36}{finding.limit:<10}{finding.status}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report the project's Docker footprint.")
    parser.add_argument("--registry", default="", help="Registry base URL to inspect read-only.")
    parser.add_argument(
        "--during-replacement",
        action="store_true",
        help="Allow the temporary candidate image/container permitted during a rollout.",
    )
    parser.add_argument(
        "--resolve-image-ids",
        action="store_true",
        help="Resolve in-scope containers to image IDs (required for a deletion manifest).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Exit non-zero on a breach, or on an unknown that could be hiding one.",
    )
    args = parser.parse_args()

    images = collect_images()
    containers = collect_containers(resolve_image_ids=args.resolve_image_ids)
    cache_bytes, cache_records, cache_age_days = collect_build_cache()
    registry = read_registry_project_artifacts(args.registry) if args.registry else None

    project_findings = evaluate_project(images, containers, registry, args.during_replacement)
    host_findings = evaluate_host(cache_bytes, cache_records, cache_age_days)

    if args.json:
        print(
            json.dumps(
                {
                    "project_scope": PROJECT_REPOSITORY,
                    "images": {
                        name: {
                            "count": group.count,
                            "bytes": group.bytes,
                            "tags": group.tags,
                        }
                        for name, group in images.items()
                    },
                    "containers": {name: vars(group) for name, group in containers.items()},
                    "host_build_cache": {
                        "bytes": cache_bytes,
                        "records": cache_records,
                        "oldest_age_days": cache_age_days,
                        "attributable": False,
                    },
                    "registry": None if registry is None else vars(registry),
                    "project_findings": [vars(f) for f in project_findings],
                    "host_findings": [vars(f) for f in host_findings],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            f"Scope: exact `{PROJECT_REPOSITORY}` resources only. "
            "Read-only; this script removes nothing."
        )
        _print_table("PROJECT BOUNDARIES (policy-enforced)", project_findings)
        _print_table(
            "HOST-WIDE OBSERVATIONS (not attributable, not actionable here)", host_findings
        )
        print("\nOUT OF SCOPE - surfaced for visibility, never counted, never actioned:")
        gb = 1024**3
        for name in (PROJECT_REGISTRY_TAGGED, NEAR_MATCH, UNRELATED, UNIDENTIFIED):
            print(
                f"  {name:<24} images={images[name].count:>4}"
                f" ({images[name].bytes / gb:>7.2f} GiB)"
                f"  containers={containers[name].count:>3}"
            )
        if images[PROJECT].tags:
            print("\nIn-scope project images:")
            for image_id, tags in images[PROJECT].tags.items():
                size = images[PROJECT].ids[image_id] / gb
                print(f"  {image_id[:19]}  {size:>6.2f} GiB  {', '.join(tags)}")
        if containers[PROJECT].items:
            print("\nIn-scope project containers:")
            for item in containers[PROJECT].items:
                print(f"  {item}")
        print("\nCleanup requires operator approval and is not performed here.")

    if args.fail_on_breach:
        blocking = {BREACH, UNKNOWN}
        if any(f.status in blocking for f in project_findings + host_findings):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
