"""Explicit runtime context passed to orchestration components."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Non-secret trace and dependency context for one execution."""

    trace_id: str
    dependencies: dict[str, Any] = field(default_factory=dict)
