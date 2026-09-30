#!/usr/bin/env python3
"""Project the pinned dependency sets out of uv.lock.

Two projections are emitted, both derived mechanically from uv.lock so it remains the
only lockfile in the repository:

requirements-runtime.txt
    The production third-party closure. The Docker dependency layer is keyed by this
    file, so bumping an internal package version must not change its bytes.

requirements-build.txt
    The PEP 517 build backend. The image builds workspace wheels with
    --no-build-isolation, so the backend must be installed from a hash-pinned set
    rather than fetched from the network unhashed at build time.

Run with --check in CI to fail when a committed projection has drifted from uv.lock.
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BANNER = (
    "# Generated from uv.lock by scripts/generate_dependency_projections.py.\n"
    "# Do not edit by hand and do not resolve dependencies here; edit the\n"
    "# workspace pyproject.toml files and relock instead.\n"
)

# --locked (not --frozen) fails when uv.lock is stale against the pyproject.toml files,
# so a dependency edit without a relock is reported here instead of being silently
# projected from an outdated lockfile. Annotations are dropped because they name the
# workspace packages that required each dependency, and the uv header is dropped
# because it embeds the invoking command.
_COMMON_ARGS = (
    "--format",
    "requirements.txt",
    "--locked",
    "--no-annotate",
    "--no-header",
    "--no-emit-project",
    "--no-emit-workspace",
)


@dataclass(frozen=True)
class Projection:
    """One generated requirements file and the export that produces it."""

    path: Path
    args: tuple[str, ...]
    purpose: str


PROJECTIONS: tuple[Projection, ...] = (
    Projection(
        Path("requirements-runtime.txt"),
        (*_COMMON_ARGS, "--no-dev", "--all-packages"),
        "production third-party closure",
    ),
    Projection(
        Path("requirements-build.txt"),
        (*_COMMON_ARGS, "--only-group", "build"),
        "PEP 517 build backend",
    ),
)


def project(projection: Projection, project_root: Path) -> str:
    result = subprocess.run(
        ("uv", "export", *projection.args),
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return BANNER + result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or verify dependency projections.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if a committed projection differs from uv.lock.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    failed = False

    for projection in PROJECTIONS:
        output_path = project_root / projection.path
        try:
            projected = project(projection, project_root)
        except subprocess.CalledProcessError as exc:
            print(f"FAILED: uv export exited {exc.returncode}\n{exc.stderr}", file=sys.stderr)
            return 1

        if not args.check:
            output_path.write_text(projected, encoding="utf-8")
            print(f"WROTE: {projection.path} ({len(projected.splitlines())} lines)")
            continue

        if not output_path.exists():
            print(f"MISSING: {projection.path} has not been generated")
            failed = True
            continue

        committed = output_path.read_text(encoding="utf-8")
        if committed == projected:
            print(f"CURRENT: {projection.path} matches uv.lock")
            continue

        print(f"STALE: {projection.path} does not match uv.lock")
        sys.stdout.writelines(
            difflib.unified_diff(
                committed.splitlines(keepends=True),
                projected.splitlines(keepends=True),
                fromfile=f"{projection.path} (committed)",
                tofile=f"{projection.path} (from uv.lock)",
            )
        )
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
