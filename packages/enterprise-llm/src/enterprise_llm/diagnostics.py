"""Sanitized diagnostic summaries for the enterprise gateway."""

from __future__ import annotations

from typing import Any

from enterprise_llm.adapter import ChatEnterpriseGateway
from enterprise_llm.sanitization import redact_identifier, sanitize


def connection_summary(model: ChatEnterpriseGateway) -> dict[str, Any]:
    return {
        "endpoint": model.endpoint.replace(
            model.client_code, redact_identifier(model.client_code) or "***"
        ),
        "vendor": model.vendor,
        "model": model.model_name,
        "client": redact_identifier(model.client_code),
        "user": redact_identifier(model.user),
    }


def safe_result(result: dict[str, Any], *, show_content: bool = False) -> dict[str, Any]:
    cleaned = sanitize(result, show_content=show_content)
    if not isinstance(cleaned, dict):
        raise TypeError("Diagnostic result must be a mapping")
    return cleaned
