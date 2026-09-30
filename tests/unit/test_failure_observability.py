"""Safe exception detail capture for durable diagnostics."""

import pytest

from agentic_orchestration.observability.failures import (
    MAX_EXCEPTION_CHAIN,
    MAX_EXCEPTION_MESSAGE,
    capture_failure_detail,
)


def _capture(error: BaseException):  # type: ignore[no-untyped-def]
    try:
        raise error
    except BaseException as caught:
        return capture_failure_detail(caught)


@pytest.mark.unit
def test_failure_detail_captures_type_message_frame_and_stable_fingerprint() -> None:
    first = _capture(KeyError("ASK_GENOME_CLIENT_CODE"))
    second = _capture(KeyError("ASK_GENOME_CLIENT_CODE"))

    observation = first.exception_chain[0]
    assert observation.exception_type == "KeyError"
    assert observation.exception_module == "builtins"
    assert observation.message == "'ASK_GENOME_CLIENT_CODE'"
    assert observation.frames[-1].function == "_capture"
    assert first.fingerprint == second.fingerprint
    assert first.redacted is False
    assert first.truncated is False


@pytest.mark.unit
def test_failure_fingerprint_ignores_traceback_line_shifts() -> None:
    def capture_at_offset(offset: int):  # type: ignore[no-untyped-def]
        namespace: dict[str, object] = {}
        source = "\n" * offset + "def fail():\n    raise ValueError('same failure')\n"
        exec(compile(source, "worker.py", "exec"), namespace)
        try:
            namespace["fail"]()  # type: ignore[operator]
        except ValueError as error:
            return capture_failure_detail(error)
        raise AssertionError("generated failure did not raise")

    assert capture_at_offset(0).fingerprint == capture_at_offset(20).fingerprint


@pytest.mark.unit
def test_failure_detail_redacts_credentials_without_capturing_locals() -> None:
    detail = _capture(
        RuntimeError(
            "authorization=Bearer abc123 password=hunter2 at https://user:pass@example.invalid/path"
        )
    )

    serialized = detail.model_dump_json()
    assert "abc123" not in serialized
    assert "hunter2" not in serialized
    assert "user:pass" not in serialized
    assert "[REDACTED]" in serialized
    assert detail.redacted is True
    assert "locals" not in serialized


@pytest.mark.unit
@pytest.mark.parametrize(
    ("message", "sensitive_value"),
    [
        ("OPENAI_API_KEY=sk-ant-abc123DEADBEEF", "sk-ant-abc123DEADBEEF"),
        ("access_token=ya29.SECRETVALUE", "ya29.SECRETVALUE"),
        ("client_secret=shhh-super-secret-99", "shhh-super-secret-99"),
        ("refresh_token=1//0gSECRET", "1//0gSECRET"),
        ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI", "wJalrXUtnFEMI"),
        ("credentials=database-password", "database-password"),
        ('{"token": "abc123SECRET"}', "abc123SECRET"),
        ("{'api_key': 'sk-ant-XYZ'}", "sk-ant-XYZ"),
    ],
)
def test_failure_detail_redacts_environment_and_serialized_secret_keys(
    message: str, sensitive_value: str
) -> None:
    detail = _capture(RuntimeError(message))

    assert sensitive_value not in detail.model_dump_json()
    assert detail.redacted is True


@pytest.mark.unit
def test_failure_detail_bounds_messages_and_exception_chains() -> None:
    error: BaseException = ValueError("x" * (MAX_EXCEPTION_MESSAGE + 10))
    for index in range(MAX_EXCEPTION_CHAIN + 2):
        try:
            raise RuntimeError(f"wrapper-{index}") from error
        except RuntimeError as wrapped:
            error = wrapped

    detail = capture_failure_detail(error)

    assert len(detail.exception_chain) == MAX_EXCEPTION_CHAIN
    assert len(detail.exception_chain[-1].message) <= MAX_EXCEPTION_MESSAGE
    assert detail.truncated is True


@pytest.mark.unit
def test_failure_detail_bounds_input_before_adversarial_private_key_redaction() -> None:
    repeated_markers = "-----BEGIN PRIVATE KEY-----" * 20_000

    detail = _capture(RuntimeError(repeated_markers))

    assert len(detail.exception_chain[0].message) <= MAX_EXCEPTION_MESSAGE
    assert "BEGIN PRIVATE KEY" not in detail.exception_chain[0].message
    assert detail.redacted is True
    assert detail.truncated is True


@pytest.mark.unit
def test_external_exception_messages_are_never_persisted() -> None:
    external_error_type = type("ProviderError", (RuntimeError,), {"__module__": "httpx"})
    external = external_error_type("response contained private model content")
    try:
        raise RuntimeError("wrapper repeated private model content") from external
    except RuntimeError as wrapped:
        detail = capture_failure_detail(wrapped)

    assert detail.exception_chain[0].message == "[REDACTED EXTERNAL ERROR]"
    assert detail.exception_chain[1].message == "[REDACTED EXTERNAL ERROR]"
    assert "private model content" not in detail.model_dump_json()
    assert detail.redacted is True


@pytest.mark.unit
def test_failure_capture_falls_back_without_masking_pathological_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agentic_orchestration.observability.failures.traceback.extract_tb",
        lambda value: (_ for _ in ()).throw(RuntimeError("capture failed")),
    )

    detail = _capture(KeyError("original failure"))

    assert detail.exception_chain[0].exception_type == "KeyError"
    assert detail.exception_chain[0].message == "<failure detail unavailable>"
    assert detail.redacted is True
    assert detail.truncated is True
