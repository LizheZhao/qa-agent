#!/usr/bin/env python3
"""Vendor ask-genome-core's process_data and GPT readout into the Ask Genome agent.

That logic still changes with client requests, so it is copied rather than re-ported: a sync is a
re-run of this script at a newer commit, not a translation. The copy is the module closure that
readout.py and insight_generation need at import, read from a recorded commit with `git show` so
the result does not depend on the checkout's working tree.

Upstream files are copied byte for byte apart from the edits listed in EDITS. Each edit states
how many times it must match, so an upstream change underneath one fails the sync instead of
silently skipping it. The two LLM client modules are replaced by stubs: nothing vendored calls
them on the orchestrator's path, and model calls go through its own gateway. Two prompt helpers
live in modules that import streamlit, so those functions alone are extracted (EXTRACTS).

    uv run python scripts/sync_ask_genome_core.py --source /path/to/ask-genome-core
    uv run python scripts/sync_ask_genome_core.py --source /path/to/ask-genome-core --check

--check regenerates from the commit recorded in SOURCE.json and fails if a vendored file has
drifted from it, for instance after a hand edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "packages/agents/ask-genome/src/ask_genome_agent/vendor/ask_genome_core"
PACKAGE = "ask_genome_agent.vendor.ask_genome_core"

# Upstream path -> path under VENDOR.
FILES = {
    "src/model/readout.py": "model/readout.py",
    "src/model/readout_utils.py": "model/readout_utils.py",
    "src/model/common.py": "model/common.py",
    "src/data/data_interface.py": "data/data_interface.py",
    "src/data/local.py": "data/local.py",
    "src/utils.py": "utils.py",
    "src/insight_generation/utils.py": "insight_generation/utils.py",
    "src/insight_generation/roi.py": "insight_generation/roi.py",
    "src/insight_generation/spend.py": "insight_generation/spend.py",
    "src/insight_generation/contribution.py": "insight_generation/contribution.py",
    "src/insight_generation/standard_metric.py": "insight_generation/standard_metric.py",
    "src/model/coverage.py": "model/coverage.py",
    "config/function_mapping.yaml": "config/function_mapping.yaml",
    "config/prompt.yaml": "config/prompt.yaml",
}

# Path under VENDOR -> functions copied out of upstream files whose other contents are UI code
# (streamlit, charts). The GPT readout uses these alone; the rest of each module stays behind.
EXTRACTS = {
    "model/visual_config_utils.py": [
        ("chart/common.py", "get_chart_set_inst"),
        ("src/model/visual_config_utils.py", "_generate_insights_prompt"),
    ],
}

_LLM_STUB = '''"""Stands in for ask-genome-core's {name}.

Nothing vendored calls these; they are imported by functions the agent does not reach. Model calls
in the orchestrator go through its own gateway, so reaching one of these is a bug to report.
"""

from typing import Any


def _unavailable(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("{name} is not vendored; use the orchestrator's model instead")


{names}
'''

STUBS = {
    "integrations/llm.py": _LLM_STUB.format(
        name="src/integrations/llm.py",
        names="generate_text = _unavailable\nstream_text = _unavailable",
    ),
    "integrations/llm_external.py": _LLM_STUB.format(
        name="src/integrations/llm_external.py",
        names=(
            "generate_analysis_response = _unavailable\ngenerate_denial_response = _unavailable"
        ),
    ),
    # Coverage's model calls. The answer_coverage node makes these itself.
    "integrations/sdk_utils.py": _LLM_STUB.format(
        name="src/integrations/sdk_utils.py",
        names="coverage_spec_call = _unavailable\ntext_llm_call = _unavailable",
    ),
}

PACKAGES = ["", "model", "data", "integrations", "insight_generation"]


@dataclass(frozen=True)
class Edit:
    """One local change to an upstream file, applied after the import rewrite."""

    file: str
    why: str
    old: str
    new: str
    count: int = 1


EDITS = [
    Edit(
        "model/common.py",
        "Read relative to the working directory; resolve it beside the vendored copy.",
        'load_yaml_file("./config/prompt.yaml")',
        "load_yaml_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"
        ' "config", "prompt.yaml"))',
    ),
    Edit(
        "model/readout_utils.py",
        "Imported but used only in commented-out code; not a dependency of the orchestrator.",
        "from sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import r2_score\n",
        "# from sklearn.linear_model import LinearRegression  (vendored: used only in dead code)\n"
        "# from sklearn.metrics import r2_score  (vendored: used only in dead code)\n",
    ),
    Edit(
        "model/readout.py",
        "Read at import relative to the working directory; resolve it beside the vendored copy.",
        "def set_environment_variables(path='config/function_mapping.yaml') -> None:",
        "def set_environment_variables(\n"
        "        path=str(pathlib.Path(__file__).resolve().parents[1]\n"
        "                 / 'config' / 'function_mapping.yaml'),\n"
        ") -> None:",
    ),
]

_IMPORT_LINE = re.compile(r"^(\s*(?:from|import)\s+)src(?=[.\s])", re.MULTILINE)
_YAML_FUNCTION = re.compile(r'"src\.')
# The client and its data directory, read under the agent's names wherever upstream reads them:
# bare CLIENT_CODE belongs to the enterprise LLM gateway in the orchestrator. A rule rather than
# per-occurrence edits, so a read upstream adds later is renamed too.
_CLIENT_ENV = re.compile(r"""(os\.getenv\(\s*)(["'])(DATA_DIR|CLIENT_CODE|MODEL_GROUP_ID)\2""")


def git_show(source: Path, commit: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def resolve_commit(source: Path, commit: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), "rev-parse", commit],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def banner(upstream: str, commit: str, comment: str = "#") -> str:
    return (
        f"{comment} Vendored from ask-genome-core {commit[:12]}:{upstream}\n"
        f"{comment} by scripts/sync_ask_genome_core.py. Do not edit by hand; change EDITS there.\n"
    )


def extract(text: str, name: str) -> str:
    """One top-level function's source, exactly as upstream wrote it."""

    import ast

    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = text.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise SystemExit(f"upstream no longer defines {name}; update EXTRACTS")


def render(source: Path, commit: str) -> tuple[dict[str, str], dict[str, str]]:
    """Every vendored file's contents, and the sha256 of each upstream original."""

    files: dict[str, str] = {}
    originals: dict[str, str] = {}
    env_renames: dict[str, int] = {}
    for upstream, target in FILES.items():
        text = git_show(source, commit, upstream)
        originals[upstream] = hashlib.sha256(text.encode()).hexdigest()
        if target.endswith(".py"):
            text = _IMPORT_LINE.sub(lambda m: m.group(1) + PACKAGE, text)
            text, renamed = _CLIENT_ENV.subn(r"\1\2ASK_GENOME_\3\2", text)
            env_renames[target] = renamed
        else:
            text = _YAML_FUNCTION.sub(f'"{PACKAGE}.', text)
        for edit in (e for e in EDITS if e.file == target):
            found = text.count(edit.old)
            if found != edit.count:
                raise SystemExit(
                    f"edit on {target} matched {found} time(s), expected {edit.count}: "
                    f"{edit.old!r}\nupstream changed underneath it; update EDITS"
                )
            text = text.replace(edit.old, edit.new)
        comment = "#"
        files[target] = banner(upstream, commit, comment) + text
    for target, wanted in EXTRACTS.items():
        parts = []
        for upstream, name in wanted:
            text = git_show(source, commit, upstream)
            originals[f"{upstream}:{name}"] = hashlib.sha256(
                extract(text, name).encode()
            ).hexdigest()
            parts.append(f"# from {upstream}\n" + extract(text, name))
        files[target] = (
            banner("(extracted functions)", commit) + "\n\n" + "\n\n\n".join(parts) + "\n"
        )
    for target, text in STUBS.items():
        files[target] = banner("(stub)", commit) + text
    for package in PACKAGES:
        target = f"{package}/__init__.py" if package else "__init__.py"
        files[target] = banner("(package marker)", commit)
    files["SOURCE.json"] = (
        json.dumps(
            {
                "source": "ask-genome-core",
                "commit": commit,
                "files": originals,
                "stubs": sorted(STUBS),
                "extracts": {t: [f"{u}:{n}" for u, n in w] for t, w in EXTRACTS.items()},
                "client_env_renames": {t: n for t, n in env_renames.items() if n},
                "edits": [
                    {"file": e.file, "why": e.why, "old": e.old, "new": e.new, "count": e.count}
                    for e in EDITS
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return files, originals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="ask-genome-core checkout")
    parser.add_argument("--commit", default="HEAD", help="commit to vendor (default: HEAD)")
    parser.add_argument("--check", action="store_true", help="fail if the vendored copy drifted")
    args = parser.parse_args()

    if args.check:
        manifest = json.loads((VENDOR / "SOURCE.json").read_text())
        commit = manifest["commit"]
    else:
        commit = resolve_commit(args.source, args.commit)
    files, _ = render(args.source, commit)

    if args.check:
        stale = [t for t, text in files.items() if (VENDOR / t).read_text() != text]
        extra = sorted(
            str(p.relative_to(VENDOR))
            for p in VENDOR.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and str(p.relative_to(VENDOR)) not in files
        )
        for target in stale:
            print(f"STALE: {target}")
        for target in extra:
            print(f"UNEXPECTED: {target}")
        if stale or extra:
            return 1
        print(f"CURRENT: vendored ask-genome-core matches {commit[:12]}")
        return 0

    for target, text in files.items():
        path = VENDOR / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    target = VENDOR.relative_to(ROOT)
    print(f"WROTE: {len(files)} files from ask-genome-core {commit[:12]} into {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
