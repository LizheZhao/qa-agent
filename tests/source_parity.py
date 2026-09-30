"""Loading ask-genome-core functions to compare a port against.

Question understanding's ports are compared here. Some source modules import cleanly;
filter_generator does not, since it pulls in openai and a config file resolved against the working
directory. `source_functions` lifts the real source text of the functions under test out by name
and compiles it against the few globals they need, so parity runs against what the source actually
says rather than a transcription of it.

Postprocessing is not compared: it runs ask-genome-core's own code, vendored, and
scripts/sync_ask_genome_core.py --check is what keeps that honest.

Selecting by name survives the file being rearranged, and a function that grows a dependency on
something not provided fails loudly instead of quietly diverging.
"""

from __future__ import annotations

import ast
import logging
import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# The baseline the port was built against. Set ASK_GENOME_CORE_SRC to compare against another
# checkout, such as /home/yvyas/new-ask-genome-core/src for Lizhe's multi-turn branch.
SOURCE_SRC = Path(os.environ.get("ASK_GENOME_CORE_SRC", "/home/yvyas/ask-genome-core/src"))


def source_available() -> bool:
    return SOURCE_SRC.is_dir()


def source_module(name: str = "model.data_filtering") -> Any:
    """Import one of the modules that can be imported as-is."""

    import importlib

    if str(SOURCE_SRC) not in sys.path:
        sys.path.insert(0, str(SOURCE_SRC))
    return importlib.import_module(name)


def source_functions(
    module_path: Path, names: Sequence[str], extra_globals: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compile named functions straight out of an ask-genome-core file."""

    import numpy as np
    import pandas as pd

    with warnings.catch_warnings():
        # readout_utils has unescaped regex literals. Not our file, and not worth four warnings
        # on every run of every suite that reads it.
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(module_path.read_text())
    found = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = [n for n in names if n not in found]
    if missing:
        raise AssertionError(f"{module_path.name} no longer defines {', '.join(missing)}")

    import re
    from collections import defaultdict

    namespace: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "re": re,
        "defaultdict": defaultdict,
        "logger": logging.getLogger("ask_genome_core_source"),
        **(extra_globals or {}),
    }
    module = ast.Module(body=[found[n] for n in names], type_ignores=[])
    exec(compile(module, str(module_path), "exec"), namespace)
    return namespace
