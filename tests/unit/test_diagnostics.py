import json

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from scripts import diagnose_enterprise_llm as diagnostics
from scripts.diagnose_enterprise_llm import validate_diagnostic_tool_call
from tests.fakes import ScriptedChatModel


@pytest.mark.unit
def test_diagnostic_tool_call_requires_expected_name_arguments() -> None:
    validate_diagnostic_tool_call(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate_sum",
                    "args": {"a": 17, "b": 25},
                    "id": "call-diagnostic",
                    "type": "tool_call",
                }
            ],
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "message",
    [
        AIMessage(content="plain text"),
        AIMessage(
            content="",
            tool_calls=[{"name": "wrong_tool", "args": {}, "id": "call-wrong"}],
        ),
    ],
)
def test_diagnostic_tool_call_rejects_missing_or_wrong_calls(message: AIMessage) -> None:
    with pytest.raises(ValueError):
        validate_diagnostic_tool_call(message)


@pytest.mark.unit
def test_diagnostic_tool_call_validates_arguments() -> None:
    message = AIMessage(
        content="",
        tool_calls=[{"name": "calculate_sum", "args": {}, "id": "call-invalid"}],
    )
    with pytest.raises(ValidationError):
        validate_diagnostic_tool_call(message)


@pytest.mark.unit
def test_native_tool_diagnostic_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = ScriptedChatModel().queue_tool_call(
        "calculate_sum", {"a": 17, "b": 25}, call_id="call-diagnostic"
    )
    monkeypatch.setattr(diagnostics, "_model", lambda: model)
    monkeypatch.setattr("sys.argv", ["diagnose_enterprise_llm.py", "--test-tool"])
    monkeypatch.setattr(
        diagnostics,
        "connection_summary",
        lambda configured: {"endpoint": "https://gateway.invalid/client/***"},
    )

    assert diagnostics.main() == 0
    lines = capsys.readouterr().out.splitlines()
    result = json.loads(lines[-1])
    assert model.calls[0]["tool_choice"] == "calculate_sum"
    assert result["status"] == "ok"
    assert result["tool_validation"] == "passed"
    assert result["content"] == "[CONTENT HIDDEN]"
