"""Conservative sanitization for logs and diagnostics."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SENSITIVE_KEYS = re.compile(r"token|authorization|auth|secret|password|credential", re.I)
_IDENTIFIER_KEYS = re.compile(r"client(?:_code)?|user|username", re.I)


def redact_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) <= 4:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


def sanitize(value: Any, *, show_content: bool = False) -> Any:
    """Return a copy safe for diagnostics; credentials are never revealable."""

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            label = str(key)
            if _SENSITIVE_KEYS.search(label):
                cleaned[label] = "[REDACTED]"
            elif _IDENTIFIER_KEYS.fullmatch(label):
                cleaned[label] = redact_identifier(str(item))
            elif label in {"contents", "content", "prompt", "response"} and not show_content:
                cleaned[label] = "[CONTENT HIDDEN]"
            else:
                cleaned[label] = sanitize(item, show_content=show_content)
        return cleaned
    if isinstance(value, list):
        return [sanitize(item, show_content=show_content) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize(item, show_content=show_content) for item in value)
    return value
