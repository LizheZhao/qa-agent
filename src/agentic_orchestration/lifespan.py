"""Application startup validation and immutable runtime assembly."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager, suppress

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from orchestration_core import AgentDependencies, AgentDescriptor
from orchestration_tools.arithmetic import calculate_sum_tool

from agentic_orchestration.config import Settings
from agentic_orchestration.diagnostics.memory import MemoryDiagnosticReadStore
from agentic_orchestration.diagnostics.mongodb import MongoDiagnosticReadStore
from agentic_orchestration.diagnostics.store import DiagnosticReadStore
from agentic_orchestration.execution.agent_invoker import RegistryAgentInvoker
from agentic_orchestration.execution.agent_registry import AgentRegistry
from agentic_orchestration.execution.checkpointing import (
    GraphCheckpointer,
    MemoryGraphCheckpointer,
    graph_checkpointer_scope,
    require_durable_checkpointer,
)
from agentic_orchestration.execution.deployment_loader import (
    LoadedDeployment,
    load_and_validate_deployment,
)
from agentic_orchestration.execution.executor import Executor
from agentic_orchestration.execution.mongodb import MongoGraphCheckpointer
from agentic_orchestration.execution.tool_registry import ToolRegistry
from agentic_orchestration.llm import build_gateway
from agentic_orchestration.observability import (
    ExecutionTraceStorageError,
    MemoryExecutionTraceStore,
    MongoExecutionTraceStore,
    TraceRecorder,
)
from agentic_orchestration.observability.definitions import (
    build_graph_definition,
    graph_predecessors,
)
from agentic_orchestration.observability.store import ExecutionTraceStore
from agentic_orchestration.sessions import MemorySessionStore, MongoSessionStore
from agentic_orchestration.sessions.contracts import SessionStore
from agentic_orchestration.sessions.expiry import run_expiry_sweeper
from integrations.mongodb import create_mongodb

logger = logging.getLogger(__name__)

# Agents compiled with a checkpointer, and so able to pause mid-run and resume in place.
# Pause capability is per agent, not per deployment: ask_genome interrupts to ask the user about
# an ambiguous filter field, while the router and marketing_science stay stateless. Named here
# rather than in the deployment manifest because it is a property of the graph, not of the
# deployment; a manifest flag is the natural home if a third agent ever needs one.
PAUSE_CAPABLE_AGENTS = frozenset({"ask_genome"})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with AsyncExitStack() as resources:
        settings = Settings()
        loaded = load_and_validate_deployment(
            settings.deployment_path,
            settings.deployment_schema_path,
        )
        session_store_factory = getattr(app.state, "session_store_factory", None)
        execution_store: ExecutionTraceStore
        diagnostic_read_store: DiagnosticReadStore
        checkpointer: GraphCheckpointer
        if session_store_factory is not None:
            session_store = session_store_factory(settings)
            if not isinstance(session_store, MemorySessionStore):
                raise TypeError(
                    "Injected session stores must provide a compatible diagnostic read adapter"
                )
            memory_execution_store = MemoryExecutionTraceStore()
            execution_store = memory_execution_store
            diagnostic_read_store = MemoryDiagnosticReadStore(session_store, memory_execution_store)
            # This branch is test injection only, and it has no MongoDB to checkpoint into, so it
            # gets the same process-local saver the session store just got.
            checkpointer = MemoryGraphCheckpointer()
        else:
            mongodb_client, database = create_mongodb(
                host=settings.mongodb_host,
                port=settings.mongodb_port,
                user=settings.mongodb_user,
                password=(
                    settings.mongodb_pwd.get_secret_value()
                    if settings.mongodb_pwd is not None
                    else None
                ),
                database=settings.mongodb_db,
            )
            resources.push_async_callback(mongodb_client.close)
            await mongodb_client.admin.command("ping")
            session_store = MongoSessionStore(database)
            mongo_execution_store = MongoExecutionTraceStore(database)
            await session_store.initialize()
            try:
                await mongo_execution_store.initialize()
            except ExecutionTraceStorageError as exc:
                logger.warning(
                    "execution observability initialization degraded type=%s",
                    type(exc).__name__,
                )
            execution_store = mongo_execution_store
            diagnostic_read_store = MongoDiagnosticReadStore(database, mongo_execution_store)
            # A pause that evaporates on restart is worse than no pause: the user is left holding
            # a question whose run no longer exists. require_durable_checkpointer makes that a
            # startup failure rather than a runtime surprise.
            checkpointer = require_durable_checkpointer(MongoGraphCheckpointer(database))

        # On the resource stack, so a later startup failure still closes the saver.
        await resources.enter_async_context(graph_checkpointer_scope(checkpointer))
        await _assemble_runtime(app, settings, loaded, session_store, execution_store, checkpointer)
        app.state.diagnostic_read_store = diagnostic_read_store
        if settings.pending_run_sweep_interval_seconds > 0:
            sweeper = asyncio.create_task(
                run_expiry_sweeper(
                    session_store,
                    interval_seconds=settings.pending_run_sweep_interval_seconds,
                    limit=settings.pending_run_sweep_limit,
                ),
                name="pending-run-expiry-sweep",
            )
            # Registered immediately after starting it, so a later startup failure cannot
            # leave the task running against a half-assembled runtime.
            resources.push_async_callback(_stop_sweeper, sweeper)
        yield


async def _stop_sweeper(sweeper: asyncio.Task[None]) -> None:
    sweeper.cancel()
    with suppress(asyncio.CancelledError):
        await sweeper


async def _assemble_runtime(
    app: FastAPI,
    settings: Settings,
    loaded: LoadedDeployment,
    session_store: SessionStore,
    execution_store: ExecutionTraceStore,
    checkpointer: GraphCheckpointer,
) -> None:
    """Assemble validated runtime state while lifespan owns external resources."""

    # Kept as one startup phase so any failure unwinds the lifespan resource stack.
    manifest = loaded.manifest
    entrypoints = loaded.entrypoints
    agent_manifests = loaded.agent_manifests
    model_factory = getattr(app.state, "model_factory", build_gateway)
    model: BaseChatModel = model_factory(settings)
    entry_agent = manifest.entry_agent
    if entry_agent is None:
        raise RuntimeError("Deployment must declare an entry_agent")
    tool_registry = ToolRegistry((calculate_sum_tool,))
    trace_recorder = TraceRecorder(
        execution_store,
        deployment_version=manifest.deployment.revision,
        application_version=manifest.application.version,
        core_version=manifest.platform.version,
        agent_versions={name: declaration.version for name, declaration in manifest.agents.items()},
        flush_timeout_seconds=settings.observability_flush_timeout_seconds,
        max_pending_snapshots=settings.observability_max_pending_snapshots,
    )
    child_names = tuple(name for name in entrypoints if name != entry_agent)
    # The entry agent is deliberately excluded: the router is one stateless invocation per turn,
    # and it is the child it delegates to that pauses.
    checkpointed = tuple(name for name in child_names if name in PAUSE_CAPABLE_AGENTS)
    child_registry = AgentRegistry.compile(
        {name: entrypoints[name] for name in child_names},
        {
            name: AgentDependencies(
                model=model,
                tools=tool_registry.resolve(agent_manifests[name].tool_ids),
            )
            for name in child_names
        },
        savers={name: checkpointer.saver for name in checkpointed},
    )
    child_definitions = {
        name: build_graph_definition(
            child_registry.get(name),
            agent_id=name,
            agent_version=agent_manifests[name].version,
            entrypoint=manifest.agents[name].entrypoint,
        )
        for name in child_names
    }
    eligible_children = tuple(name for name in child_names if manifest.agents[name].routable)
    invoker = RegistryAgentInvoker(
        child_registry,
        caller_agent_id=entry_agent,
        allowed_agent_ids=eligible_children,
        timeout_seconds=settings.gateway_timeout_seconds,
        recorder=trace_recorder,
        graph_definition_ids={
            name: definition.definition_id for name, definition in child_definitions.items()
        },
        graph_predecessors={
            name: dict(graph_predecessors(definition))
            for name, definition in child_definitions.items()
        },
    )
    catalog = tuple(
        AgentDescriptor(
            agent_id=name,
            description=agent_manifests[name].description,
            capabilities=agent_manifests[name].capabilities,
        )
        for name in eligible_children
    )
    entry_dependencies = AgentDependencies(
        model=model,
        tools=tool_registry.resolve(agent_manifests[entry_agent].tool_ids),
        agent_invoker=invoker,
        agent_catalog=catalog,
    )
    entry_registry = AgentRegistry.compile(
        {entry_agent: entrypoints[entry_agent]},
        {entry_agent: entry_dependencies},
    )
    registry = AgentRegistry({**child_registry.graphs, **entry_registry.graphs})
    entry_definition = build_graph_definition(
        entry_registry.get(entry_agent),
        agent_id=entry_agent,
        agent_version=agent_manifests[entry_agent].version,
        entrypoint=manifest.agents[entry_agent].entrypoint,
    )
    graph_definitions = {**child_definitions, entry_agent: entry_definition}
    for definition in graph_definitions.values():
        try:
            await execution_store.save_graph_definition(definition)
        except ExecutionTraceStorageError as exc:
            logger.warning(
                "graph definition persistence degraded definition_id=%s type=%s",
                definition.definition_id,
                type(exc).__name__,
            )
    app.state.settings = settings
    app.state.deployment = manifest
    app.state.registry = registry
    app.state.executor = Executor(
        registry,
        trace_recorder,
        {name: definition.definition_id for name, definition in graph_definitions.items()},
        {
            name: dict(graph_predecessors(definition))
            for name, definition in graph_definitions.items()
        },
    )
    app.state.execution_store = execution_store
    # The continuation endpoint resumes a paused child through the same restricted seam
    # the router delegates through, so eligibility is enforced identically on both paths.
    app.state.agent_invoker = invoker
    app.state.graph_definition_ids = {
        name: definition.definition_id for name, definition in graph_definitions.items()
    }
    app.state.entry_agent = entry_agent
    app.state.session_store = session_store
    logger.info(
        "session_store=%s durable=%s",
        session_store.storage_name,
        str(session_store.durable).lower(),
    )
    logger.info(
        "graph_checkpointer=%s durable=%s pause_capable=%s",
        checkpointer.storage_name,
        str(checkpointer.durable).lower(),
        ",".join(sorted(checkpointed)) or "none",
    )
