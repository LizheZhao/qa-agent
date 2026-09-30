"""Explicitly opt-in enterprise gateway smoke tests using application configuration."""

import os

import pytest
from enterprise_llm import ChatEnterpriseGateway, GatewayConfigurationError
from orchestration_tools.arithmetic import CalculateSumInput, calculate_sum_tool

from agentic_orchestration.config import Settings
from agentic_orchestration.llm import build_gateway

LIVE_ENABLED = os.getenv("RUN_LIVE_GATEWAY_TESTS") == "1"


def live_model() -> ChatEnterpriseGateway:
    if not LIVE_ENABLED:
        pytest.skip("set RUN_LIVE_GATEWAY_TESTS=1 to enable enterprise gateway smoke tests")
    try:
        return build_gateway(Settings())
    except GatewayConfigurationError as exc:
        pytest.fail(f"Enterprise gateway configuration is invalid: {exc}")


@pytest.mark.live
def test_live_plain_request() -> None:
    model = live_model()
    response = model.invoke("Reply with the single word OK.")
    assert response.content


@pytest.mark.live
def test_live_calculate_sum_tool_call() -> None:
    model = live_model()
    response = model.bind_tools([calculate_sum_tool], tool_choice="calculate_sum").invoke(
        "Use calculate_sum to add 17 and 25."
    )

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call["name"] == "calculate_sum"
    arguments = CalculateSumInput.model_validate(call["args"])
    assert arguments.a == 17
    assert arguments.b == 25
