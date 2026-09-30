"""Concurrency-safe process-local implementation used by deterministic tests."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from langchain_core.messages import AIMessage
from orchestration_core import ClarificationRequest, ClarificationResponse

from agentic_orchestration.sessions.contracts import (
    ClaimedResponse,
    ClaimResponse,
    ClarificationAuditEntry,
    ClosePendingRun,
    CommitContinuedTurn,
    CommitTurn,
    PendingRun,
    PendingRunConflictError,
    PublishPendingRun,
    SessionConflictError,
    SessionSnapshot,
    SessionTurn,
    TurnContinuation,
)
from agentic_orchestration.sessions.pending import (
    decide_claim,
    decide_close,
    decide_commit,
    decide_republish,
    response_text,
)


class MemorySessionStore:
    """Implement revisioned successful-turn commits without durable storage."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionSnapshot] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._pending: dict[str, PendingRun] = {}
        self._audit: dict[str, list[ClarificationAuditEntry]] = {}
        # Lineage for the turn the next commit writes; the Mongo adapter puts this on the
        # turn document directly, inside the same transaction.
        self._continuation: dict[str, TurnContinuation] = {}

    @property
    def storage_name(self) -> str:
        return "process_memory"

    @property
    def durable(self) -> bool:
        return False

    @asynccontextmanager
    async def lock(self, session_id: str) -> AsyncIterator[None]:
        async with self._locks_guard:
            session_lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with session_lock:
            yield

    async def load(self, session_id: str) -> SessionSnapshot | None:
        snapshot = self._sessions.get(session_id)
        if snapshot is None:
            return None
        return SessionSnapshot(
            session_id=snapshot.session_id,
            revision=snapshot.revision,
            tenant_id=snapshot.tenant_id,
            user_id=snapshot.user_id,
            agent_id=snapshot.agent_id,
            turns=tuple(snapshot.turns),
            status=snapshot.status,
            created_at=snapshot.created_at,
            updated_at=snapshot.updated_at,
        )

    async def commit_turn(self, command: CommitTurn) -> SessionSnapshot:
        current = self._sessions.get(command.session_id)
        if command.expected_revision is None:
            if current is not None:
                raise SessionConflictError("Session already exists")
            revision = 1
            previous: tuple[SessionTurn, ...] = ()
        else:
            if current is None or current.revision != command.expected_revision:
                raise SessionConflictError("Session revision changed")
            if current.agent_id != command.agent_id:
                raise SessionConflictError("Session agent changed")
            revision = current.revision + 1
            previous = current.turns

        final = next(
            (
                message
                for message in command.generated_messages
                if message.id == command.final_message_id
            ),
            None,
        )
        if not isinstance(final, AIMessage) or final.tool_calls:
            raise ValueError("final_message_id must identify a final generated AI message")
        turn = SessionTurn(
            turn_number=revision,
            input_message=command.input_message,
            generated_messages=command.generated_messages,
            final_message_id=command.final_message_id,
            routing_outcome=command.routing_outcome,
            selected_agent_id=command.selected_agent_id,
            trace_id=command.trace_id,
            committed_at=datetime.now(UTC),
            continuation=self._continuation.pop(command.session_id, None),
        )

        committed_at = turn.committed_at

        snapshot = SessionSnapshot(
            session_id=command.session_id,
            revision=revision,
            tenant_id=current.tenant_id if current is not None else command.tenant_id,
            user_id=current.user_id if current is not None else command.user_id,
            agent_id=command.agent_id,
            turns=(*previous, turn),
            status=current.status if current is not None else "active",
            created_at=current.created_at if current is not None else committed_at,
            updated_at=committed_at,
        )
        self._sessions[command.session_id] = snapshot
        return await self.load(command.session_id) or snapshot

    async def publish_pending_run(self, command: PublishPendingRun) -> PendingRun:
        existing = self._pending.get(command.session_id)
        if existing is not None and existing.status in {"awaiting_response", "claimed"}:
            raise PendingRunConflictError("The session already holds an in-flight run")
        current = self._sessions.get(command.session_id)
        current_revision = current.revision if current is not None else None
        if current_revision != command.expected_revision:
            raise SessionConflictError("Session revision changed")
        published_at = datetime.now(UTC)
        run = PendingRun(
            session_id=command.session_id,
            run_id=command.run_id,
            clarification=command.clarification,
            status="awaiting_response",
            agent_id=command.agent_id,
            selected_agent_id=command.selected_agent_id,
            expected_session_revision=command.expected_revision,
            tenant_id=command.tenant_id,
            user_id=command.user_id,
            input_message=command.input_message,
            route_reason_code=command.route_reason_code,
            trace_id=command.trace_id,
            graph_definition_id=command.graph_definition_id,
            continuation_contract_version=command.continuation_contract_version,
            submission_id=None,
            claimed_response=None,
            created_at=published_at,
            expires_at=command.clarification.expires_at(published_at),
            attempt_trace_ids=(command.trace_id,),
        )
        self._pending[command.session_id] = run
        self._record_published(run, command.trace_id)
        return run

    async def due_pending_runs(self, *, at: datetime, limit: int) -> tuple[PendingRun, ...]:
        due = sorted(
            (
                run
                for run in self._pending.values()
                if run.status == "awaiting_response" and run.expires_at <= at
            ),
            key=lambda run: run.expires_at,
        )
        return tuple(due[:limit])

    async def active_pending_run(self, session_id: str) -> PendingRun | None:
        run = self._pending.get(session_id)
        if run is None or run.status not in {"awaiting_response", "claimed"}:
            return None
        return run

    async def claim_response(self, command: ClaimResponse, *, at: datetime) -> ClaimedResponse:
        claimed = decide_claim(self._pending.get(command.session_id), command, at=at)
        if claimed.replayed:
            return claimed
        self._pending[command.session_id] = claimed.pending_run
        if command.response.form != "cancel":
            # A cancellation is audited once, as the closure it causes.
            self._append_audit(
                claimed.pending_run,
                event="answered",
                response=command.response,
                submission_id=command.submission_id,
                attempt_trace_id=command.attempt_trace_id,
            )
        return claimed

    async def republish_pending_run(
        self,
        session_id: str,
        *,
        run_id: str,
        submission_id: str,
        clarification: ClarificationRequest,
        attempt_trace_id: str,
    ) -> PendingRun:
        run = decide_republish(
            self._pending.get(session_id),
            run_id=run_id,
            submission_id=submission_id,
            clarification=clarification,
            published_at=datetime.now(UTC),
        )
        self._pending[session_id] = run
        self._record_published(run, attempt_trace_id)
        return run

    async def commit_continued_turn(self, command: CommitContinuedTurn) -> SessionSnapshot:
        consumed = decide_commit(self._pending.get(command.session_id), command)
        self._continuation[command.session_id] = TurnContinuation(
            run_id=command.run_id,
            clarification_ids=command.clarification_ids,
            closed_by="answered",
        )
        snapshot = await self.commit_turn(
            CommitTurn(
                session_id=command.session_id,
                expected_revision=command.expected_revision,
                tenant_id=command.tenant_id,
                user_id=command.user_id,
                agent_id=command.agent_id,
                input_message=command.input_message,
                generated_messages=command.generated_messages,
                final_message_id=command.final_message_id,
                routing_outcome="delegate",
                selected_agent_id=command.selected_agent_id,
                trace_id=command.trace_id,
            )
        )
        self._pending[command.session_id] = consumed
        return snapshot

    async def close_pending_run(self, command: ClosePendingRun) -> SessionSnapshot:
        run = self._pending.get(command.session_id)
        closed = decide_close(run, command)
        already_closed = run is not None and run.status == closed.status
        snapshot = await self.load(command.session_id)
        if not already_closed:
            self._continuation[command.session_id] = TurnContinuation(
                run_id=closed.run_id,
                clarification_ids=(closed.clarification_id,),
                closed_by=command.reason,
            )
            snapshot = await self.commit_turn(
                CommitTurn(
                    session_id=command.session_id,
                    expected_revision=closed.expected_session_revision,
                    tenant_id=closed.tenant_id,
                    user_id=closed.user_id,
                    agent_id=closed.agent_id,
                    input_message=closed.input_message,
                    generated_messages=(command.final_message,),
                    final_message_id=str(command.final_message.id),
                    routing_outcome="delegate",
                    selected_agent_id=closed.selected_agent_id,
                    trace_id=command.trace_id,
                )
            )
            self._pending[command.session_id] = closed
            self._append_audit(
                closed,
                event=command.reason,
                response=None,
                submission_id=command.submission_id,
                attempt_trace_id=command.attempt_trace_id,
            )
        assert snapshot is not None
        return snapshot

    async def clarification_audit(self, session_id: str) -> tuple[ClarificationAuditEntry, ...]:
        return tuple(self._audit.get(session_id, ()))

    def _record_published(self, run: PendingRun, attempt_trace_id: str) -> None:
        self._append_audit(
            run,
            event="published",
            response=None,
            submission_id=None,
            attempt_trace_id=attempt_trace_id,
        )

    def _append_audit(
        self,
        run: PendingRun,
        *,
        event: str,
        response: ClarificationResponse | None,
        submission_id: str | None,
        attempt_trace_id: str | None,
    ) -> None:
        entries = self._audit.setdefault(run.session_id, [])
        entries.append(
            ClarificationAuditEntry(
                session_id=run.session_id,
                run_id=run.run_id,
                clarification_id=run.clarification_id,
                sequence=len(entries) + 1,
                event=event,  # type: ignore[arg-type]
                recorded_at=datetime.now(UTC),
                selection_mode=(run.clarification.selection_mode if event == "published" else None),
                question=(run.clarification.question if event == "published" else None),
                option_ids=(run.clarification.option_ids if event == "published" else ()),
                response=response,
                response_text=(
                    response_text(response, run.clarification) if response is not None else None
                ),
                submission_id=submission_id,
                attempt_trace_id=attempt_trace_id,
            )
        )

    async def snapshots(self) -> tuple[SessionSnapshot, ...]:
        """Return immutable snapshots for the deterministic diagnostic adapter."""

        snapshots = []
        for session_id in sorted(self._sessions):
            snapshot = await self.load(session_id)
            if snapshot is not None:
                snapshots.append(snapshot)
        return tuple(snapshots)
