"""Typed FastAPI dependency providers for assembled runtime services."""

from typing import cast

from fastapi import Request

from agentic_orchestration.diagnostics.store import DiagnosticReadStore
from agentic_orchestration.execution.agent_invoker import RegistryAgentInvoker
from agentic_orchestration.execution.deployment_loader import DeploymentManifest
from agentic_orchestration.execution.executor import Executor
from agentic_orchestration.sessions.contracts import SessionStore


async def get_executor(request: Request) -> Executor:
    return cast(Executor, request.app.state.executor)


async def get_session_store(request: Request) -> SessionStore:
    return cast(SessionStore, request.app.state.session_store)


async def get_agent_invoker(request: Request) -> RegistryAgentInvoker:
    """The restricted child-invocation seam a continuation resumes through."""

    return cast(RegistryAgentInvoker, request.app.state.agent_invoker)


async def get_entry_agent(request: Request) -> str:
    return cast(str, request.app.state.entry_agent)


async def get_diagnostic_read_store(request: Request) -> DiagnosticReadStore:
    return cast(DiagnosticReadStore, request.app.state.diagnostic_read_store)


async def get_deployment_manifest(request: Request) -> DeploymentManifest:
    return cast(DeploymentManifest, request.app.state.deployment)


async def get_graph_definition_ids(request: Request) -> dict[str, str]:
    return cast(dict[str, str], request.app.state.graph_definition_ids)
