import pytest
from enterprise_llm import GatewayTimeoutError
from enterprise_llm.sanitization import redact_identifier, sanitize
from langchain_core.messages import HumanMessage

from tests.fakes import ScriptedChatModel
from tests.fixtures.tools import add_tool


@pytest.mark.unit
async def test_scripted_model_queues_records_and_supports_sync_async() -> None:
    model = (
        ScriptedChatModel()
        .queue_text("first")
        .queue_tool_call("test_add_numbers", {"left": 1, "right": 2})
    )
    assert model.invoke("hello").content == "first"
    result = await model.bind_tools([add_tool], tool_choice="required").ainvoke(
        [HumanMessage("add")]
    )
    assert result.tool_calls[0]["args"] == {"left": 1, "right": 2}
    assert model.calls[1]["tools"][0]["name"] == "test_add_numbers"
    assert model.calls[1]["tool_choice"] == "required"


@pytest.mark.unit
def test_scripted_model_queues_errors() -> None:
    model = ScriptedChatModel().queue_error(GatewayTimeoutError("timeout"))
    with pytest.raises(GatewayTimeoutError):
        model.invoke("hello")


@pytest.mark.unit
def test_sanitizer_never_reveals_credentials_and_hides_content_by_default() -> None:
    raw = {
        "token": "secret-token",
        "Authorization": "Bearer secret",
        "client_code": "customer-1234",
        "user": "employee-5678",
        "contents": [{"text": "private prompt"}],
        "nested": {"password": "secret-password", "response": "private response"},
    }
    safe = sanitize(raw)
    rendered = repr(safe)
    for secret in (
        "secret-token",
        "Bearer secret",
        "secret-password",
        "private prompt",
        "private response",
    ):
        assert secret not in rendered
    assert safe["client_code"] == redact_identifier("customer-1234")
    shown = sanitize(raw, show_content=True)
    assert shown["contents"][0]["text"] == "private prompt"
    assert shown["token"] == "[REDACTED]"
