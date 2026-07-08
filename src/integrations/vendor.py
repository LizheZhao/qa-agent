"""LLM vendor resolution and tool selection for the agent layer.

The agent graph makes its decisions through these helpers. If ANTHROPIC_API_KEY is set,
every call routes direct to Anthropic (ChatAnthropic), overriding the vendor picked on
the page sidebar. Otherwise the call goes through the external endpoints via call_api,
passing the tools list in the request body.

Tools are plain dicts in anthropic format: {"name", "description", "input_schema"}.
"""
import os
import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from src.integrations.llm_external import call_api

logger = logging.getLogger(__name__)


def anthropic_override() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def resolve_vendor(sidebar_vendor: str) -> str:
    """The vendor that will actually serve agent calls; used for display on the page."""
    return "anthropic-direct" if anthropic_override() else sidebar_vendor


def agent_tool_call(system_prompt: str, user_prompt: str, tools: list,
                    sidebar_vendor: str, verbose: bool = False) -> dict:
    """Ask the LLM to pick exactly one tool. Returns {"tool": name, "args": dict}.

    Falls back to the first tool with a warning if the model returns no tool call,
    so callers always get a usable decision."""
    if verbose:
        logger.info(f"tool selection candidates: {[t['name'] for t in tools]}")

    if anthropic_override():
        llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                            api_key=os.getenv("ANTHROPIC_API_KEY"), max_tokens=4096)
        message = llm.bind_tools(tools).invoke([SystemMessage(content=system_prompt),
                                                HumanMessage(content=user_prompt)])
        calls = message.tool_calls or []
        selected = {"tool": calls[0]["name"], "args": calls[0]["args"] or {}} if calls else None
    else:
        data = call_api(system_prompt, user_prompt, llm_vendor=sidebar_vendor,
                        tools=tools, verbose=verbose)
        selected = _extract_tool_call(data)

    if selected is None:
        logger.warning(f"no tool call in LLM response, falling back to '{tools[0]['name']}'")
        selected = {"tool": tools[0]["name"], "args": {}}
    if verbose:
        logger.info(f"tool selected: {selected['tool']} "
                    f"args={json.dumps(selected['args'], default=str)[:300]}")
    return selected


def _extract_tool_call(data) -> dict:
    """Find the first tool call in an endpoint response, wherever the vendor put it.

    Handles the OpenAI shape ({"function": {"name", "arguments"}}) and the anthropic
    shape ({"name", "input"}); searches nested containers because the enterprise
    endpoint wraps provider responses. Returns None if nothing is found."""
    if isinstance(data, dict):
        for call in data.get("tool_calls") or []:
            function = call.get("function", call)
            name = function.get("name")
            args = function.get("arguments", function.get("input", function.get("args", {})))
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                return {"tool": name, "args": args or {}}
        if data.get("type") == "tool_use" and data.get("name"):
            return {"tool": data["name"], "args": data.get("input") or {}}
        for value in data.values():
            found = _extract_tool_call(value)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _extract_tool_call(item)
            if found:
                return found
    return None
