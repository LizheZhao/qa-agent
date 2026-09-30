"""Non-streaming chat endpoint for the deployed entry agent."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, cast
from uuid import uuid4

from enterprise_llm import GatewayError, GatewayTimeoutError
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from orchestration_core import (
    AgentExecutionError,
    AgentInvocationError,
    AgentInvocationTimeoutError,
    ExecutionContext,
    PausedConversation,
    RequestOrigin,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_orchestration.api.dependencies import (
    get_entry_agent,
    get_executor,
    get_graph_definition_ids,
    get_session_store,
)
from agentic_orchestration.api.presentation import PendingClarification, clarification_view
from agentic_orchestration.execution.executor import Executor
from agentic_orchestration.sessions.contracts import (
    CommitTurn,
    PendingRunConflictError,
    PublishPendingRun,
    RoutingOutcome,
    SessionConflictError,
    SessionStorageError,
    SessionStore,
)
from agentic_orchestration.sessions.expiry import release_if_expired

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(default=None, min_length=1)
    message: str

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class CompletedToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    status: Literal["completed"] = "completed"


class ChatResponse(BaseModel):
    session_id: str
    trace_id: str
    message: AssistantMessage
    tool_calls: list[CompletedToolCall]


class ChatError(BaseModel):
    code: str
    message: str


class ChatErrorResponse(BaseModel):
    error: ChatError
    session_id: str
    trace_id: str


class TracedChatError(Exception):
    """A sanitized HTTP failure for a request whose trace already exists."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        session_id: str,
        trace_id: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = ChatErrorResponse(
            error=ChatError(code=code, message=message),
            session_id=session_id,
            trace_id=trace_id,
        )


async def traced_chat_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, TracedChatError)
    return JSONResponse(status_code=exc.status_code, content=exc.response.model_dump(mode="json"))


def _traced_error(
    *,
    status_code: int,
    code: str,
    message: str,
    context: ExecutionContext,
) -> TracedChatError:
    return TracedChatError(
        status_code=status_code,
        code=code,
        message=message,
        session_id=context.session_id,
        trace_id=context.trace_id,
    )


def _content_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block if isinstance(block, str) else str(block.get("text", ""))
        for block in message.content
        if isinstance(block, (str, dict))
    )


def _completed_tool_calls(messages: list[BaseMessage]) -> list[CompletedToolCall]:
    completed_ids = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    return [
        CompletedToolCall(name=call["name"], arguments=call.get("args", {}))
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["id"] in completed_ids
    ]


async def _finalize_failed_trace(executor: Executor, context: ExecutionContext) -> None:
    """Close an aborting request's trace so its writer task cannot outlive the response."""

    try:
        await executor.fail(context)
    except Exception as cleanup_error:
        logger.warning(
            "trace failure cleanup failed trace_id=%s type=%s",
            context.trace_id,
            type(cleanup_error).__name__,
        )


def _report_cleanup_result(task: asyncio.Task[None]) -> None:
    """Observe a shielded cleanup that outlives a repeatedly cancelled request task."""

    try:
        task.result()
    except asyncio.CancelledError:
        logger.warning("background trace cleanup was cancelled")
    except Exception as exc:
        logger.warning("background trace cleanup failed type=%s", type(exc).__name__)


def _request_origin(value: str) -> RequestOrigin:
    try:
        return RequestOrigin(value)
    except ValueError:
        return RequestOrigin.UNKNOWN


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        202: {"model": PendingClarification},
        **{status: {"model": ChatErrorResponse} for status in (409, 500, 502, 503, 504)},
    },
)
async def chat(
    request: ChatRequest,
    executor: Annotated[Executor, Depends(get_executor)],
    session_store: Annotated[SessionStore, Depends(get_session_store)],
    entry_agent: Annotated[str, Depends(get_entry_agent)],
    graph_definition_ids: Annotated[dict[str, str], Depends(get_graph_definition_ids)],
    request_origin_header: Annotated[
        str, Header(alias="X-Orchestration-Origin")
    ] = RequestOrigin.API.value,
) -> Any:
    request_origin = _request_origin(request_origin_header)
    session_id = request.session_id or str(uuid4())
    async with session_store.lock(session_id):
        try:
            snapshot = await session_store.load(session_id)
            # A pause that lapsed is closed here, before the session is judged busy: an
            # expired run otherwise holds the slot forever and every later message meets a
            # conflict nothing would clear.
            pending = await release_if_expired(
                session_store,
                await session_store.active_pending_run(session_id),
                at=datetime.now(UTC),
            )
            if pending is None:
                snapshot = await session_store.load(session_id)
        except SessionStorageError as exc:
            logger.exception("session load failed")
            raise HTTPException(status_code=503, detail="Session storage is unavailable") from exc
        if pending is not None:
            # An ordinary message while a pause is open is a conflict, never an implicit
            # answer to the question on screen. Checked before the not-found case: a
            # session whose first request paused holds a run but has no committed turn
            # yet, and reporting it as unknown would hide the question already on screen.
            raise HTTPException(
                status_code=409,
                detail="This session is waiting on a clarification response",
            )
        if request.session_id is not None and snapshot is None:
            raise HTTPException(status_code=404, detail="Session not found")

        previous = snapshot.messages if snapshot is not None else ()
        active_agent_id = snapshot.active_agent_id if snapshot is not None else None
        trace_id = str(uuid4())
        execution_context = ExecutionContext(
            trace_id=trace_id,
            root_span_id=str(uuid4()),
            session_id=session_id,
            attempted_turn_number=1 if snapshot is None else snapshot.revision + 1,
            request_origin=request_origin,
        )
        await executor.begin(entry_agent, execution_context)
        finalized = False
        committed = False
        completion: asyncio.Task[None] | None = None
        try:
            input_message = HumanMessage(content=request.message, id=str(uuid4()))
            invocation_messages = [*previous, input_message]
            try:
                result = await executor.invoke(
                    entry_agent,
                    {
                        "messages": invocation_messages,
                        "active_agent_id": active_agent_id,
                    },
                    context=execution_context,
                )
            except (GatewayTimeoutError, AgentInvocationTimeoutError) as exc:
                logger.warning("chat invocation failed: gateway_timeout")
                raise _traced_error(
                    status_code=504,
                    code="gateway_timeout",
                    message="Enterprise gateway timed out",
                    context=execution_context,
                ) from exc
            except GatewayError as exc:
                logger.warning("chat invocation failed: gateway_error type=%s", type(exc).__name__)
                raise _traced_error(
                    status_code=502,
                    code="gateway_error",
                    message="Enterprise gateway request failed",
                    context=execution_context,
                ) from exc
            except AgentExecutionError as exc:
                if isinstance(exc.__cause__, GatewayTimeoutError):
                    logger.warning("chat child invocation failed: gateway_timeout")
                    raise _traced_error(
                        status_code=504,
                        code="gateway_timeout",
                        message="Enterprise gateway timed out",
                        context=execution_context,
                    ) from exc
                if isinstance(exc.__cause__, GatewayError):
                    logger.warning("chat child invocation failed: gateway_error")
                    raise _traced_error(
                        status_code=502,
                        code="gateway_error",
                        message="Enterprise gateway request failed",
                        context=execution_context,
                    ) from exc
                logger.exception("chat child invocation failed: execution_error")
                raise _traced_error(
                    status_code=500,
                    code="agent_execution_error",
                    message="Agent execution failed",
                    context=execution_context,
                ) from exc
            except AgentInvocationError as exc:
                logger.exception(
                    "chat child invocation failed: invocation_error type=%s", type(exc).__name__
                )
                raise _traced_error(
                    status_code=500,
                    code="agent_execution_error",
                    message="Agent execution failed",
                    context=execution_context,
                ) from exc
            except Exception as exc:
                logger.exception(
                    "chat invocation failed: execution_error type=%s", type(exc).__name__
                )
                raise _traced_error(
                    status_code=500,
                    code="agent_execution_error",
                    message="Agent execution failed",
                    context=execution_context,
                ) from exc

            paused = result.get("paused_run")
            if isinstance(paused, PausedConversation):
                published = await _publish_pause(
                    session_store,
                    paused=paused,
                    session_id=session_id,
                    snapshot=snapshot,
                    entry_agent=entry_agent,
                    input_message=input_message,
                    trace_id=trace_id,
                    graph_definition_ids=graph_definition_ids,
                )
                finalized = True
                # The trace closes without a committed turn: the turn is not finished, and
                # the pause is only visible because publication already succeeded.
                await _finalize_paused_trace(executor, execution_context)
                return JSONResponse(
                    status_code=202,
                    content=clarification_view(published, trace_id).model_dump(mode="json"),
                )

            messages = list(result.get("messages", []))
            generated = messages[len(invocation_messages) :]
            for message in generated:
                if message.id is None:
                    message.id = str(uuid4())
            final = next(
                (message for message in reversed(generated) if isinstance(message, AIMessage)), None
            )
            if final is None or final.tool_calls:
                logger.error("chat invocation returned no final assistant message")
                raise _traced_error(
                    status_code=500,
                    code="invalid_agent_result",
                    message="Agent returned an invalid result",
                    context=execution_context,
                )
            routing_outcome = result.get("routing_outcome")
            selected_agent_id = result.get("selected_agent_id")
            if routing_outcome not in {"answer", "delegate", "clarify", "reject"}:
                logger.error("entry agent returned an invalid routing outcome")
                raise _traced_error(
                    status_code=500,
                    code="invalid_agent_result",
                    message="Agent returned an invalid result",
                    context=execution_context,
                )
            if not isinstance(selected_agent_id, (str, type(None))) or (
                (routing_outcome == "delegate") != (selected_agent_id is not None)
            ):
                logger.error("entry agent returned invalid route affinity")
                raise _traced_error(
                    status_code=500,
                    code="invalid_agent_result",
                    message="Agent returned an invalid result",
                    context=execution_context,
                )
            controlled_outcome = cast(RoutingOutcome, routing_outcome)
            assert final.id is not None
            command = CommitTurn(
                session_id=session_id,
                expected_revision=snapshot.revision if snapshot is not None else None,
                tenant_id=snapshot.tenant_id if snapshot is not None else None,
                user_id=snapshot.user_id if snapshot is not None else None,
                agent_id=entry_agent,
                input_message=input_message,
                generated_messages=tuple(generated),
                final_message_id=final.id,
                routing_outcome=controlled_outcome,
                selected_agent_id=selected_agent_id,
                trace_id=trace_id,
            )
            commit_task = asyncio.create_task(
                session_store.commit_turn(command),
                name=f"session-commit:{session_id}:{execution_context.attempted_turn_number}",
            )
            try:
                await asyncio.shield(commit_task)
                committed = True
            except asyncio.CancelledError:
                try:
                    await commit_task
                except Exception:
                    pass
                else:
                    committed = True
                raise
            finalized = True
            completion = asyncio.create_task(
                executor.complete(
                    execution_context,
                    committed_turn_number=execution_context.attempted_turn_number,
                ),
                name=f"trace-complete:{trace_id}",
            )
            await asyncio.shield(completion)
            return ChatResponse(
                session_id=session_id,
                trace_id=trace_id,
                message=AssistantMessage(content=_content_text(final)),
                tool_calls=_completed_tool_calls(generated),
            )
        except SessionConflictError as exc:
            finalized = True
            await _finalize_failed_trace(executor, execution_context)
            logger.warning("session commit conflict")
            raise _traced_error(
                status_code=409,
                code="session_conflict",
                message="Session changed; retry the request",
                context=execution_context,
            ) from exc
        except SessionStorageError as exc:
            finalized = True
            await _finalize_failed_trace(executor, execution_context)
            logger.exception("session commit failed")
            raise _traced_error(
                status_code=503,
                code="session_storage_unavailable",
                message="Session storage is unavailable",
                context=execution_context,
            ) from exc
        except asyncio.CancelledError:
            if finalized:
                if completion is not None and not completion.done():
                    completion.add_done_callback(_report_cleanup_result)
            else:
                finalized = True
                cleanup = asyncio.create_task(
                    executor.complete(
                        execution_context,
                        committed_turn_number=execution_context.attempted_turn_number,
                    )
                    if committed
                    else executor.cancel(execution_context),
                    name=f"trace-cancel:{trace_id}",
                )
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cleanup.add_done_callback(_report_cleanup_result)
                    logger.warning("trace cancellation cleanup continuing trace_id=%s", trace_id)
            raise
        except Exception:
            if not finalized:
                finalized = True
                await _finalize_failed_trace(executor, execution_context)
            raise
        finally:
            # Net for any exit that skipped the handlers above; cancellation already
            # finalizes itself, so this never awaits while a CancelledError propagates.
            if not finalized:
                finalized = True
                await _finalize_failed_trace(executor, execution_context)


async def _publish_pause(
    session_store: SessionStore,
    *,
    paused: PausedConversation,
    session_id: str,
    snapshot: Any,
    entry_agent: str,
    input_message: BaseMessage,
    trace_id: str,
    graph_definition_ids: dict[str, str],
) -> Any:
    """Make the pause durable before it is returned to anyone.

    The checkpoint is already written by the time the child reports a pause; this is the
    publication that makes it answerable. Returning the question before this succeeded
    would show a user a choice the system could not honour, so a failure here is reported
    as a failed request and the orphan checkpoint is simply never referenced.
    """

    try:
        return await session_store.publish_pending_run(
            PublishPendingRun(
                session_id=session_id,
                expected_revision=snapshot.revision if snapshot is not None else None,
                tenant_id=snapshot.tenant_id if snapshot is not None else None,
                user_id=snapshot.user_id if snapshot is not None else None,
                agent_id=entry_agent,
                selected_agent_id=paused.agent_id,
                run_id=paused.run_id,
                clarification=paused.clarification,
                input_message=input_message,
                route_reason_code=paused.clarification.reason_code,
                trace_id=trace_id,
                graph_definition_id=(
                    paused.graph_definition_id or graph_definition_ids.get(paused.agent_id)
                ),
                continuation_contract_version=paused.continuation_contract_version,
            )
        )
    except PendingRunConflictError as exc:
        raise HTTPException(
            status_code=409, detail="This session is waiting on a clarification response"
        ) from exc
    except SessionConflictError as exc:
        raise HTTPException(status_code=409, detail="Session changed; retry the request") from exc
    except SessionStorageError as exc:
        logger.exception("publishing the pending run failed")
        raise HTTPException(status_code=503, detail="Session storage is unavailable") from exc


async def _finalize_paused_trace(executor: Executor, context: ExecutionContext) -> None:
    try:
        await executor.cancel(context)
    except Exception as cleanup_error:
        logger.warning(
            "paused trace finalization failed trace_id=%s type=%s",
            context.trace_id,
            type(cleanup_error).__name__,
        )
