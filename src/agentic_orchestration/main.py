"""FastAPI application factory and importable ASGI app."""

from collections.abc import Callable

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel

from agentic_orchestration.api.build import router as build_router
from agentic_orchestration.api.chat import (
    TracedChatError,
    traced_chat_error_handler,
)
from agentic_orchestration.api.chat import (
    router as chat_router,
)
from agentic_orchestration.api.clarifications import router as clarifications_router
from agentic_orchestration.api.diagnostics import router as diagnostics_router
from agentic_orchestration.api.health import router as health_router
from agentic_orchestration.api.sessions import router as sessions_router
from agentic_orchestration.config import Settings
from agentic_orchestration.diagnostics.frontend import configure_diagnostic_frontend
from agentic_orchestration.lifespan import lifespan
from agentic_orchestration.sessions.contracts import SessionStore


def create_app(
    *,
    model_factory: Callable[[Settings], BaseChatModel] | None = None,
    session_store_factory: Callable[[Settings], SessionStore] | None = None,
) -> FastAPI:
    application = FastAPI(title="Agentic Orchestration", version="0.6.0", lifespan=lifespan)
    if model_factory is not None:
        application.state.model_factory = model_factory
    if session_store_factory is not None:
        application.state.session_store_factory = session_store_factory
    application.include_router(health_router)
    application.include_router(build_router)
    application.include_router(chat_router)
    application.include_router(sessions_router)
    application.include_router(clarifications_router)
    application.include_router(diagnostics_router)
    application.add_exception_handler(TracedChatError, traced_chat_error_handler)
    configure_diagnostic_frontend(application)
    return application


app = create_app()
