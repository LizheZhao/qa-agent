"""Read-only APIs for the internal orchestration diagnostic UI."""

from collections.abc import Awaitable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from orchestration_core import GraphDefinition

from agentic_orchestration.api.dependencies import (
    get_deployment_manifest,
    get_diagnostic_read_store,
    get_graph_definition_ids,
)
from agentic_orchestration.diagnostics.contracts import (
    AgentDeployment,
    AttemptSummary,
    DeploymentDiagnostic,
    Page,
    SessionSummary,
    SpanDetail,
    SpanSummary,
    TraceDetail,
    TurnSummary,
)
from agentic_orchestration.diagnostics.cursors import InvalidCursorError
from agentic_orchestration.diagnostics.store import DiagnosticReadError, DiagnosticReadStore
from agentic_orchestration.execution.deployment_loader import DeploymentManifest

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])
PageLimit = Annotated[int, Query(ge=1, le=50)]


async def _read[T](operation: Awaitable[T]) -> T:
    try:
        return await operation
    except InvalidCursorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DiagnosticReadError as exc:
        raise HTTPException(status_code=503, detail="Diagnostic storage is unavailable") from exc


@router.get("/sessions", response_model=Page[SessionSummary])
async def sessions(
    store: Annotated[DiagnosticReadStore, Depends(get_diagnostic_read_store)],
    limit: PageLimit = 10,
    cursor: str | None = None,
) -> Page[SessionSummary]:
    return await _read(store.sessions(limit=limit, cursor=cursor))


@router.get("/sessions/{session_id}", response_model=SessionSummary)
async def session(
    session_id: str,
    store: Annotated[DiagnosticReadStore, Depends(get_diagnostic_read_store)],
) -> SessionSummary:
    item = await _read(store.session(session_id))
    if item is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return item


@router.get("/sessions/{session_id}/turns", response_model=Page[TurnSummary])
async def turns(
    session_id: str,
    store: Annotated[DiagnosticReadStore, Depends(get_diagnostic_read_store)],
    limit: PageLimit = 25,
    cursor: str | None = None,
) -> Page[TurnSummary]:
    return await _read(store.turns(session_id, limit=limit, cursor=cursor))


@router.get("/sessions/{session_id}/traces", response_model=Page[AttemptSummary])
async def session_attempts(
    session_id: str,
    store: Annotated[DiagnosticReadStore, Depends(get_diagnostic_read_store)],
    limit: PageLimit = 25,
    cursor: str | None = None,
) -> Page[AttemptSummary]:
    return await _read(store.session_attempts(session_id, limit=limit, cursor=cursor))


@router.get("/traces", response_model=Page[AttemptSummary])
async def recent_attempts(
    store: Annotated[DiagnosticReadStore, Depends(get_diagnostic_read_store)],
    limit: PageLimit = 25,
    cursor: str | None = None,
) -> Page[AttemptSummary]:
    return await _read(store.recent_attempts(limit=limit, cursor=cursor))


@router.get("/traces/{trace_id}", response_model=TraceDetail)
async def trace(
    trace_id: str,
    store: Annotated[DiagnosticReadStore, Depends(get_diagnostic_read_store)],
) -> TraceDetail:
    item = await _read(store.trace(trace_id))
    if item is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return item


@router.get("/traces/{trace_id}/spans", response_model=list[SpanSummary])
async def spans(
    trace_id: str,
    store: Annotated[DiagnosticReadStore, Depends(get_diagnostic_read_store)],
) -> list[SpanSummary]:
    trace_item = await _read(store.trace(trace_id))
    if trace_item is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return await _read(store.spans(trace_id))


@router.get("/traces/{trace_id}/spans/{span_id}", response_model=SpanDetail)
async def span(
    trace_id: str,
    span_id: str,
    store: Annotated[DiagnosticReadStore, Depends(get_diagnostic_read_store)],
) -> SpanDetail:
    item = await _read(store.span(trace_id, span_id))
    if item is None:
        raise HTTPException(status_code=404, detail="Span not found")
    return item


@router.get("/graph-definitions/{definition_id}", response_model=GraphDefinition)
async def graph_definition(
    definition_id: str,
    store: Annotated[DiagnosticReadStore, Depends(get_diagnostic_read_store)],
) -> GraphDefinition:
    item = await _read(store.graph_definition(definition_id))
    if item is None:
        raise HTTPException(status_code=404, detail="Graph definition not found")
    return item


@router.get("/deployment", response_model=DeploymentDiagnostic)
async def deployment(
    manifest: Annotated[DeploymentManifest, Depends(get_deployment_manifest)],
    definition_ids: Annotated[dict[str, str], Depends(get_graph_definition_ids)],
) -> DeploymentDiagnostic:
    assert manifest.entry_agent is not None
    return DeploymentDiagnostic(
        name=manifest.deployment.name,
        revision=manifest.deployment.revision,
        entry_agent=manifest.entry_agent,
        agents=[
            AgentDeployment(
                agent_id=agent_id,
                version=declaration.version,
                routable=declaration.routable,
                graph_definition_id=definition_ids.get(agent_id),
            )
            for agent_id, declaration in sorted(manifest.agents.items())
        ],
    )
