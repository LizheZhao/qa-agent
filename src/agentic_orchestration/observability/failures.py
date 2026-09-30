"""Bounded, secret-redacting exception capture for privileged diagnostics."""

from __future__ import annotations

import re
import traceback
from hashlib import sha256
from pathlib import PurePath

from orchestration_core import ExceptionObservation, FailureDetail, TracebackFrame

MAX_EXCEPTION_CHAIN = 5
MAX_EXCEPTION_FRAMES = 32
MAX_EXCEPTION_MESSAGE = 2048
MAX_PRE_REDACTION_MESSAGE = MAX_EXCEPTION_MESSAGE * 4
MAX_EXCEPTION_NAME = 256
MAX_FRAME_FILENAME = 512
MAX_FRAME_FUNCTION = 256
MAX_SENSITIVITY_SCAN_CHAIN = 32

_SECRET_ASSIGNMENT = re.compile(
    r"(?ix)"
    r"(?<![A-Z0-9_])"
    r"(?P<quote>[\"']?)"
    r"(?P<key>(?:[A-Z0-9]+[_-])*(?:api[_-]?key|access[_-]?key|authorization|auth|cookie|"
    r"credentials?|password|secret|token)(?:[_-][A-Z0-9]+)*)"
    r"(?P=quote)"
    r"(?P<separator>\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}\]]+)"
)
_AUTHORIZATION_VALUE = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+")
_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@", re.I)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?"
    r"(?:-----END [^-\r\n]*PRIVATE KEY-----|$)",
    re.DOTALL,
)
_UNTRUSTED_MESSAGE_MODULE_PREFIXES = (
    "anthropic",
    "enterprise_llm",
    "httpcore",
    "httpx",
    "langchain_anthropic",
    "langchain_openai",
    "openai",
)


def capture_failure_detail(error: BaseException) -> FailureDetail:
    """Capture an exception chain without locals, source text, or unbounded values."""

    try:
        return _capture_failure_detail(error)
    except Exception:
        exception_type, _ = _bounded(str(type(error).__name__), MAX_EXCEPTION_NAME)
        exception_module, _ = _bounded(str(type(error).__module__), MAX_EXCEPTION_NAME)
        fingerprint = sha256(f"{exception_module}:{exception_type}".encode()).hexdigest()
        return FailureDetail(
            fingerprint=fingerprint,
            exception_chain=(
                ExceptionObservation(
                    exception_type=exception_type or "Exception",
                    exception_module=exception_module or "builtins",
                    message="<failure detail unavailable>",
                ),
            ),
            redacted=True,
            truncated=True,
        )


def _capture_failure_detail(error: BaseException) -> FailureDetail:
    observations: list[ExceptionObservation] = []
    fingerprint_parts: list[str] = []
    redacted = False
    truncated = False
    seen: set[int] = set()
    current: BaseException | None = error
    external_chain = _has_untrusted_message(error)

    while current is not None and len(observations) < MAX_EXCEPTION_CHAIN:
        if id(current) in seen:
            truncated = True
            break
        seen.add(id(current))

        exception_type, type_truncated = _bounded(type(current).__name__, MAX_EXCEPTION_NAME)
        exception_module, module_truncated = _bounded(type(current).__module__, MAX_EXCEPTION_NAME)
        raw_message = _exception_message(current)
        if external_chain:
            safe_message = "[REDACTED EXTERNAL ERROR]"
            message_redacted = True
            pre_redaction_truncated = False
        else:
            safe_message, message_redacted, pre_redaction_truncated = _redact_message(raw_message)
        safe_message, message_truncated = _bounded(safe_message, MAX_EXCEPTION_MESSAGE)
        extracted_frames = list(traceback.extract_tb(current.__traceback__))
        if len(extracted_frames) > MAX_EXCEPTION_FRAMES:
            extracted_frames = extracted_frames[-MAX_EXCEPTION_FRAMES:]
            truncated = True

        frames: list[TracebackFrame] = []
        for frame in extracted_frames:
            filename, filename_truncated = _bounded(frame.filename, MAX_FRAME_FILENAME)
            function, function_truncated = _bounded(frame.name, MAX_FRAME_FUNCTION)
            line_number = frame.lineno or 1
            frames.append(
                TracebackFrame(
                    filename=filename or "<unknown>",
                    function=function or "<unknown>",
                    line_number=max(line_number, 1),
                )
            )
            truncated = truncated or filename_truncated or function_truncated
            fingerprint_parts.append(f"{PurePath(frame.filename).name}:{function}")

        observations.append(
            ExceptionObservation(
                exception_type=exception_type or "Exception",
                exception_module=exception_module or "builtins",
                message=safe_message,
                frames=tuple(frames),
            )
        )
        fingerprint_parts.append(f"{exception_module}:{exception_type}")
        redacted = redacted or message_redacted
        truncated = (
            truncated
            or type_truncated
            or module_truncated
            or pre_redaction_truncated
            or message_truncated
        )
        current = _next_exception(current)

    if current is not None:
        truncated = True

    fingerprint = sha256("\n".join(fingerprint_parts).encode("utf-8")).hexdigest()
    return FailureDetail(
        fingerprint=fingerprint,
        exception_chain=tuple(observations),
        redacted=redacted,
        truncated=truncated,
    )


def _exception_message(error: BaseException) -> str:
    try:
        return str(error)
    except Exception:
        return "<exception message unavailable>"


def _next_exception(error: BaseException) -> BaseException | None:
    if error.__cause__ is not None:
        return error.__cause__
    if not error.__suppress_context__:
        return error.__context__
    return None


def _has_untrusted_message(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    for _ in range(MAX_SENSITIVITY_SCAN_CHAIN):
        if current is None or id(current) in seen:
            return False
        seen.add(id(current))
        if type(current).__module__.startswith(_UNTRUSTED_MESSAGE_MODULE_PREFIXES):
            return True
        current = _next_exception(current)
    return False


def _bounded(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def _redact_message(value: str) -> tuple[str, bool, bool]:
    cleaned, truncated = _bounded(value, MAX_PRE_REDACTION_MESSAGE)
    bounded_value = cleaned
    cleaned = _PRIVATE_KEY.sub("[REDACTED PRIVATE KEY]", cleaned)
    cleaned = _URL_CREDENTIALS.sub(r"\g<scheme>[REDACTED]@", cleaned)
    cleaned = _AUTHORIZATION_VALUE.sub(lambda match: f"{match.group(1)} [REDACTED]", cleaned)
    cleaned = _SECRET_ASSIGNMENT.sub(
        lambda match: (
            f"{match.group('quote')}{match.group('key')}{match.group('quote')}"
            f"{match.group('separator')}[REDACTED]"
        ),
        cleaned,
    )
    return cleaned, cleaned != bounded_value, truncated
