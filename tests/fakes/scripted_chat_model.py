"""Deterministic scripted model for graph and API tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

from enterprise_llm.adapter import normalize_tools
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, PrivateAttr


class ScriptedChatModel(BaseChatModel):
    """Queue responses/errors and record calls without simulating gateway transport."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _responses: deque[AIMessage | Exception] = PrivateAttr(default_factory=deque)
    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-test-chat-model"

    @property
    def calls(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._calls)

    def queue_text(self, text: str) -> ScriptedChatModel:
        self._responses.append(AIMessage(content=text))
        return self

    def queue_tool_call(
        self, name: str, args: dict[str, Any], *, call_id: str = "call_scripted"
    ) -> ScriptedChatModel:
        self._responses.append(
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
            )
        )
        return self

    def queue_error(self, error: Exception) -> ScriptedChatModel:
        self._responses.append(error)
        return self

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable[Any, BaseMessage]:
        return self.bind(tools=list(tools), tool_choice=tool_choice, **kwargs)

    def _next(self, messages: list[BaseMessage], kwargs: dict[str, Any]) -> ChatResult:
        if not self._responses:
            raise RuntimeError("ScriptedChatModel response queue is empty")
        tools = normalize_tools(kwargs.pop("tools", []))
        tool_choice = kwargs.pop("tool_choice", None)
        self._calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "tool_choice": tool_choice,
                "kwargs": kwargs,
            }
        )
        value = self._responses.popleft()
        if isinstance(value, Exception):
            raise value
        return ChatResult(generations=[ChatGeneration(message=value)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        return self._next(messages, kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        return self._next(messages, kwargs)
