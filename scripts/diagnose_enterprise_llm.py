#!/usr/bin/env python3
"""Opt-in live gateway diagnostic with credential-safe output."""

from __future__ import annotations

import argparse
import json
import time

from enterprise_llm import ChatEnterpriseGateway, GatewayError
from enterprise_llm.diagnostics import connection_summary, safe_result
from langchain_core.messages import AIMessage, HumanMessage
from orchestration_tools.arithmetic import CalculateSumInput, calculate_sum_tool

from agentic_orchestration.config import Settings
from agentic_orchestration.llm import build_gateway


def validate_diagnostic_tool_call(response: AIMessage) -> None:
    """Require exactly one schema-valid native calculate_sum call."""

    if len(response.tool_calls) != 1:
        raise ValueError(
            f"Expected exactly one diagnostic tool call, got {len(response.tool_calls)}"
        )
    call = response.tool_calls[0]
    if call["name"] != "calculate_sum":
        raise ValueError("Gateway returned an unexpected diagnostic tool name")
    arguments = CalculateSumInput.model_validate(call["args"])
    if arguments.a != 17 or arguments.b != 25:
        raise ValueError("Gateway returned unexpected diagnostic tool arguments")


def _model() -> ChatEnterpriseGateway:
    return build_gateway(Settings())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-tool", action="store_true")
    parser.add_argument("--show-content", action="store_true")
    args = parser.parse_args()
    try:
        model = _model()
    except (ValueError, GatewayError) as exc:
        print(json.dumps({"configuration": "invalid", "error_type": type(exc).__name__}))
        return 2

    print(json.dumps({"connection": connection_summary(model)}, sort_keys=True))
    started = time.perf_counter()
    try:
        runnable = (
            model.bind_tools([calculate_sum_tool], tool_choice="calculate_sum")
            if args.test_tool
            else model
        )
        prompt = (
            "Use calculate_sum to add 17 and 25."
            if args.test_tool
            else "Reply briefly that the diagnostic succeeded."
        )
        response = runnable.invoke([HumanMessage(prompt)])
        if not isinstance(response, AIMessage):
            raise ValueError("Gateway returned a non-assistant message")
        if args.test_tool:
            validate_diagnostic_tool_call(response)
    except (GatewayError, ValueError) as exc:
        elapsed = (time.perf_counter() - started) * 1000
        print(
            json.dumps(
                {
                    "status": "failed",
                    "latency_ms": round(elapsed, 1),
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 1

    elapsed = (time.perf_counter() - started) * 1000
    result = {
        "status": "ok",
        "latency_ms": round(elapsed, 1),
        "envelope": "valid",
        "tool_calls": len(response.tool_calls),
        "tool_validation": "passed" if args.test_tool else "not_requested",
        "content": response.content,
    }
    print(json.dumps(safe_result(result, show_content=args.show_content), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
