"""Transactional MongoDB implementation of durable conversation history."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any

from orchestration_core import CONTINUATION_CONTRACT_VERSION, ClarificationRequest
from pydantic import ValidationError
from pymongo import ASCENDING, DESCENDING
from pymongo.asynchronous.client_session import AsyncClientSession
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import CollectionInvalid, DuplicateKeyError, PyMongoError
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern

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
    SessionStorageError,
    SessionTurn,
    TurnContinuation,
)
from agentic_orchestration.sessions.messages import decode_message, encode_message
from agentic_orchestration.sessions.pending import (
    decide_claim,
    decide_close,
    decide_commit,
    decide_republish,
    response_text,
)
from agentic_orchestration.sessions.schema import (
    ACTIVE_PENDING_RUN_STATUSES,
    CLARIFICATION_AUDIT_COLLECTION,
    CLARIFICATION_AUDIT_INDEX,
    PENDING_RUN_ACTIVE_INDEX,
    PENDING_RUN_DUE_INDEX,
    PENDING_RUNS_COLLECTION,
    SESSION_TURN_INDEX,
    SESSION_TURNS_COLLECTION,
    SESSIONS_COLLECTION,
    ClarificationAuditDocument,
    PendingRunDocument,
    SessionDocument,
    StoredContinuation,
    TurnDocument,
    validate_turn_document,
)


class MongoSessionStore:
    """Persist immutable successful turns and revisioned session metadata."""

    def __init__(self, database: AsyncDatabase[dict[str, Any]]) -> None:
        self._database = database
        self._sessions = database[SESSIONS_COLLECTION]
        self._turns = database[SESSION_TURNS_COLLECTION]
        self._pending = database[PENDING_RUNS_COLLECTION]
        self._audit = database[CLARIFICATION_AUDIT_COLLECTION]
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    @property
    def storage_name(self) -> str:
        return "mongodb"

    @property
    def durable(self) -> bool:
        return True

    async def initialize(self) -> None:
        existing = set(await self._database.list_collection_names())
        for collection_name in (
            SESSIONS_COLLECTION,
            SESSION_TURNS_COLLECTION,
            PENDING_RUNS_COLLECTION,
            CLARIFICATION_AUDIT_COLLECTION,
        ):
            if collection_name not in existing:
                # Another replica may create it after our initial listing.
                with suppress(CollectionInvalid):
                    await self._database.create_collection(collection_name)
        await self._turns.create_index(
            [("session_id", ASCENDING), ("turn_number", ASCENDING)],
            name=SESSION_TURN_INDEX,
            unique=True,
        )
        await self._sessions.create_index(
            [("updated_at", DESCENDING), ("_id", DESCENDING)],
            name="session_recent_activity",
        )
        # `active_session_key` carries the session ID only while the run holds the session's
        # single run slot, so one sparse unique index enforces "at most one in-flight run per
        # session" without a partial filter expression. Sparse means absent, not null, so a
        # closed run omits the key rather than setting it null. See _pending_document.
        await self._pending.create_index(
            [("active_session_key", ASCENDING)],
            name=PENDING_RUN_ACTIVE_INDEX,
            unique=True,
            sparse=True,
        )
        await self._pending.create_index(
            [("session_id", ASCENDING), ("clarification_id", ASCENDING)],
            name="pending_run_clarification",
        )
        # Serves the expiry sweep. Status leads the key so the scan only ever reaches runs
        # nobody has answered; a claimed run is never a candidate.
        await self._pending.create_index(
            [("status", ASCENDING), ("expires_at", ASCENDING)],
            name=PENDING_RUN_DUE_INDEX,
        )
        await self._audit.create_index(
            [("run_id", ASCENDING), ("sequence", ASCENDING)],
            name=CLARIFICATION_AUDIT_INDEX,
            unique=True,
        )
        await self._audit.create_index(
            [("session_id", ASCENDING), ("recorded_at", ASCENDING)],
            name="clarification_audit_session",
        )

    @asynccontextmanager
    async def lock(self, session_id: str) -> AsyncIterator[None]:
        async with self._locks_guard:
            session_lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with session_lock:
            yield

    async def load(self, session_id: str) -> SessionSnapshot | None:
        try:
            raw_session = await self._sessions.find_one({"_id": session_id})
            if raw_session is None:
                return None
            session = SessionDocument.model_validate(raw_session)
            cursor = self._turns.find(
                {"session_id": session_id, "turn_number": {"$lte": session.revision}}
            ).sort("turn_number", ASCENDING)
            turns = []
            expected_turn = 1
            async for raw_turn in cursor:
                turn = validate_turn_document(raw_turn)
                if turn.turn_number != expected_turn:
                    raise SessionStorageError("Stored session turns are not contiguous")
                turns.append(
                    SessionTurn(
                        turn_number=turn.turn_number,
                        input_message=decode_message(turn.input_message),
                        generated_messages=tuple(
                            decode_message(message) for message in turn.generated_messages
                        ),
                        final_message_id=turn.final_message_id,
                        routing_outcome=turn.routing_outcome,
                        selected_agent_id=turn.selected_agent_id,
                        trace_id=turn.trace_id,
                        committed_at=turn.committed_at,
                        continuation=(
                            TurnContinuation(
                                run_id=turn.continuation.run_id,
                                clarification_ids=tuple(turn.continuation.clarification_ids),
                                closed_by=turn.continuation.closed_by,
                            )
                            if turn.continuation is not None
                            else None
                        ),
                    )
                )
                expected_turn += 1
            if expected_turn - 1 != session.revision:
                raise SessionStorageError("Stored session revision does not match its turns")
            return SessionSnapshot(
                session_id=session.id,
                revision=session.revision,
                tenant_id=session.tenant_id,
                user_id=session.user_id,
                agent_id=session.agent_id,
                turns=tuple(turns),
                status=session.status,
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
        except SessionStorageError:
            raise
        except (PyMongoError, ValidationError, ValueError) as exc:
            raise SessionStorageError("Could not load session history") from exc

    async def _in_transaction(self, body: Any) -> None:
        """Run one short session-store transaction at snapshot isolation."""

        async with self._database.client.start_session() as mongo_session:
            await mongo_session.with_transaction(
                body,
                read_concern=ReadConcern("snapshot"),
                write_concern=WriteConcern("majority"),
            )

    async def _apply_turn(
        self,
        mongo_session: AsyncClientSession,
        *,
        session_id: str,
        expected_revision: int | None,
        agent_id: str,
        tenant_id: str | None,
        user_id: str | None,
        turn_data: dict[str, Any],
        committed_at: datetime,
    ) -> None:
        """Write one turn and advance its session, inside the caller's transaction.

        Shared so that an ordinary turn and a continued one commit under exactly the same
        revision guard, and so a continued turn can be committed together with consuming
        the pending run that produced it.
        """

        if expected_revision is None:
            session = SessionDocument(
                _id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                agent_id=agent_id,
                revision=1,
                created_at=committed_at,
                updated_at=committed_at,
            )
            await self._sessions.insert_one(
                session.model_dump(by_alias=True, mode="python"),
                session=mongo_session,
            )
            await self._turns.insert_one(turn_data, session=mongo_session)
            return
        await self._turns.insert_one(turn_data, session=mongo_session)
        result = await self._sessions.update_one(
            {"_id": session_id, "agent_id": agent_id, "revision": expected_revision},
            {"$set": {"updated_at": committed_at}, "$inc": {"revision": 1}},
            session=mongo_session,
        )
        if result.matched_count != 1 or result.modified_count != 1:
            raise SessionConflictError("Session revision changed")

    async def commit_turn(self, command: CommitTurn) -> SessionSnapshot:
        turn_number = 1 if command.expected_revision is None else command.expected_revision + 1
        committed_at = datetime.now(UTC)
        try:
            turn = TurnDocument(
                _id=f"{command.session_id}:{turn_number}",
                session_id=command.session_id,
                turn_number=turn_number,
                input_message=encode_message(command.input_message),
                generated_messages=[
                    encode_message(message) for message in command.generated_messages
                ],
                final_message_id=command.final_message_id,
                routing_outcome=command.routing_outcome,
                selected_agent_id=command.selected_agent_id,
                trace_id=command.trace_id,
                committed_at=committed_at,
            )
            turn_data = turn.model_dump(by_alias=True, mode="python", exclude_none=True)

            async def transaction(mongo_session: AsyncClientSession) -> None:
                await self._apply_turn(
                    mongo_session,
                    session_id=command.session_id,
                    expected_revision=command.expected_revision,
                    agent_id=command.agent_id,
                    tenant_id=command.tenant_id,
                    user_id=command.user_id,
                    turn_data=turn_data,
                    committed_at=committed_at,
                )

            await self._in_transaction(transaction)
        except SessionConflictError:
            raise
        except DuplicateKeyError as exc:
            raise SessionConflictError("Session turn already exists") from exc
        except (PyMongoError, ValidationError, ValueError, TypeError) as exc:
            raise SessionStorageError("Could not commit session turn") from exc

        snapshot = await self.load(command.session_id)
        if snapshot is None:
            raise SessionStorageError("Committed session could not be loaded")
        return snapshot

    async def publish_pending_run(self, command: PublishPendingRun) -> PendingRun:
        """Publish the pause and its question before any of it is shown to a client.

        The session revision is compared here as well as at completion: a pause published
        against a session that moved on would be answered into a conversation that no
        longer matches the request it came from.
        """

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

        async def transaction(mongo_session: AsyncClientSession) -> None:
            await self._require_revision(
                mongo_session, command.session_id, command.expected_revision
            )
            await self._pending.insert_one(self._pending_document(run), session=mongo_session)
            await self._append_audit(
                mongo_session,
                run,
                event="published",
                response=None,
                submission_id=None,
                attempt_trace_id=command.trace_id,
            )

        try:
            await self._in_transaction(transaction)
        except SessionStorageError:
            raise
        except DuplicateKeyError as exc:
            raise PendingRunConflictError("The session already holds an in-flight run") from exc
        except (PyMongoError, ValidationError, ValueError, TypeError) as exc:
            raise SessionStorageError("Could not publish the pending run") from exc
        return run

    async def active_pending_run(self, session_id: str) -> PendingRun | None:
        try:
            document = await self._pending.find_one({"active_session_key": session_id})
        except PyMongoError as exc:
            raise SessionStorageError("Could not load the pending run") from exc
        if document is None:
            return None
        return self._pending_run(document)

    async def due_pending_runs(self, *, at: datetime, limit: int) -> tuple[PendingRun, ...]:
        """Runs whose pause has elapsed and that nobody has answered.

        Only `awaiting_response` is a candidate: a claimed run is in the middle of a
        continuation, and closing it would discard work already under way.
        """

        try:
            cursor = (
                self._pending.find({"status": "awaiting_response", "expires_at": {"$lte": at}})
                .sort("expires_at", ASCENDING)
                .limit(limit)
            )
            documents = [document async for document in cursor]
        except PyMongoError as exc:
            raise SessionStorageError("Could not list expired pending runs") from exc
        return tuple(self._pending_run(document) for document in documents)

    async def claim_response(self, command: ClaimResponse, *, at: datetime) -> ClaimedResponse:
        claimed: ClaimedResponse | None = None

        async def transaction(mongo_session: AsyncClientSession) -> None:
            nonlocal claimed
            current = await self._read_pending(
                mongo_session, command.session_id, clarification_id=command.clarification_id
            )
            decided = decide_claim(current, command, at=at)
            if decided.replayed:
                claimed = decided
                return
            await self._replace_pending(mongo_session, decided.pending_run, previous=current)
            if command.response.form != "cancel":
                # A cancellation is audited once, as the closure it causes.
                await self._append_audit(
                    mongo_session,
                    decided.pending_run,
                    event="answered",
                    response=command.response,
                    submission_id=command.submission_id,
                    attempt_trace_id=command.attempt_trace_id,
                )
            claimed = decided

        await self._guarded(transaction, "Could not claim the submitted response")
        assert claimed is not None
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
        republished: PendingRun | None = None

        async def transaction(mongo_session: AsyncClientSession) -> None:
            nonlocal republished
            current = await self._read_pending(mongo_session, session_id)
            decided = decide_republish(
                current,
                run_id=run_id,
                submission_id=submission_id,
                clarification=clarification,
                published_at=datetime.now(UTC),
            )
            await self._replace_pending(mongo_session, decided, previous=current)
            await self._append_audit(
                mongo_session,
                decided,
                event="published",
                response=None,
                submission_id=None,
                attempt_trace_id=attempt_trace_id,
            )
            republished = decided

        await self._guarded(transaction, "Could not publish the next clarification")
        assert republished is not None
        return republished

    async def commit_continued_turn(self, command: CommitContinuedTurn) -> SessionSnapshot:
        """Commit the turn a paused run produced and consume the run in one transaction."""

        committed_at = datetime.now(UTC)
        turn_number = 1 if command.expected_revision is None else command.expected_revision + 1
        turn = TurnDocument(
            _id=f"{command.session_id}:{turn_number}",
            session_id=command.session_id,
            turn_number=turn_number,
            input_message=encode_message(command.input_message),
            generated_messages=[encode_message(message) for message in command.generated_messages],
            final_message_id=command.final_message_id,
            routing_outcome="delegate",
            selected_agent_id=command.selected_agent_id,
            trace_id=command.trace_id,
            continuation=StoredContinuation(
                run_id=command.run_id,
                clarification_ids=list(command.clarification_ids),
                continuation_contract_version=CONTINUATION_CONTRACT_VERSION,
                attempt_trace_ids=list(command.attempt_trace_ids),
                closed_by="answered",
            ),
            committed_at=committed_at,
        )
        turn_data = turn.model_dump(by_alias=True, mode="python", exclude_none=True)

        async def transaction(mongo_session: AsyncClientSession) -> None:
            current = await self._read_pending(mongo_session, command.session_id)
            consumed = decide_commit(current, command)
            await self._apply_turn(
                mongo_session,
                session_id=command.session_id,
                expected_revision=command.expected_revision,
                agent_id=command.agent_id,
                tenant_id=command.tenant_id,
                user_id=command.user_id,
                turn_data=turn_data,
                committed_at=committed_at,
            )
            await self._replace_pending(mongo_session, consumed, previous=current)

        await self._guarded(transaction, "Could not commit the continued turn")
        snapshot = await self.load(command.session_id)
        if snapshot is None:
            raise SessionStorageError("Committed session could not be loaded")
        return snapshot

    async def close_pending_run(self, command: ClosePendingRun) -> SessionSnapshot:
        """Close a run at its last displayed question, committing that turn and consuming it."""

        committed_at = datetime.now(UTC)
        already_closed = False

        async def transaction(mongo_session: AsyncClientSession) -> None:
            nonlocal already_closed
            current = await self._read_pending(mongo_session, command.session_id)
            closed = decide_close(current, command)
            if current is not None and current.status == closed.status:
                already_closed = True
                return
            turn_number = (
                1
                if closed.expected_session_revision is None
                else closed.expected_session_revision + 1
            )
            turn = TurnDocument(
                _id=f"{command.session_id}:{turn_number}",
                session_id=command.session_id,
                turn_number=turn_number,
                input_message=encode_message(closed.input_message),
                generated_messages=[encode_message(command.final_message)],
                final_message_id=str(command.final_message.id),
                routing_outcome="delegate",
                selected_agent_id=closed.selected_agent_id,
                trace_id=command.trace_id,
                continuation=StoredContinuation(
                    run_id=closed.run_id,
                    clarification_ids=[closed.clarification_id],
                    continuation_contract_version=closed.continuation_contract_version,
                    attempt_trace_ids=list(closed.attempt_trace_ids),
                    closed_by=command.reason,
                ),
                committed_at=committed_at,
            )
            await self._apply_turn(
                mongo_session,
                session_id=command.session_id,
                expected_revision=closed.expected_session_revision,
                agent_id=closed.agent_id,
                tenant_id=closed.tenant_id,
                user_id=closed.user_id,
                turn_data=turn.model_dump(by_alias=True, mode="python", exclude_none=True),
                committed_at=committed_at,
            )
            await self._replace_pending(mongo_session, closed, previous=current)
            await self._append_audit(
                mongo_session,
                closed,
                event=command.reason,
                response=None,
                submission_id=command.submission_id,
                attempt_trace_id=command.attempt_trace_id,
            )

        await self._guarded(transaction, "Could not close the pending run")
        snapshot = await self.load(command.session_id)
        if snapshot is None:
            raise SessionStorageError("Closed session could not be loaded")
        return snapshot

    async def clarification_audit(self, session_id: str) -> tuple[ClarificationAuditEntry, ...]:
        try:
            cursor = self._audit.find({"session_id": session_id}).sort(
                [("run_id", ASCENDING), ("sequence", ASCENDING)]
            )
            entries = [
                self._audit_entry(ClarificationAuditDocument.model_validate(document))
                async for document in cursor
            ]
            return tuple(entries)
        except (PyMongoError, ValidationError, ValueError) as exc:
            raise SessionStorageError("Could not load the clarification audit") from exc

    async def _guarded(self, body: Any, message: str) -> None:
        try:
            await self._in_transaction(body)
        except SessionStorageError:
            raise
        except DuplicateKeyError as exc:
            raise PendingRunConflictError("The pending run changed concurrently") from exc
        except (PyMongoError, ValidationError, ValueError, TypeError) as exc:
            raise SessionStorageError(message) from exc

    async def _require_revision(
        self,
        mongo_session: AsyncClientSession,
        session_id: str,
        expected_revision: int | None,
    ) -> None:
        document = await self._sessions.find_one({"_id": session_id}, session=mongo_session)
        current = SessionDocument.model_validate(document).revision if document else None
        if current != expected_revision:
            raise SessionConflictError("Session revision changed")

    async def _read_pending(
        self,
        mongo_session: AsyncClientSession,
        session_id: str,
        *,
        clarification_id: str | None = None,
    ) -> PendingRun | None:
        """Read the run this session most recently paused on, open or closed.

        Closed runs are read too: a replayed submission has to be recognised as the one
        that already produced an outcome rather than treated as addressing nothing.
        """

        query: dict[str, Any] = {"session_id": session_id}
        if clarification_id is not None:
            query["clarification_id"] = clarification_id
        document = await self._pending.find_one(
            query, sort=[("created_at", DESCENDING)], session=mongo_session
        )
        return None if document is None else self._pending_run(document)

    async def _replace_pending(
        self,
        mongo_session: AsyncClientSession,
        run: PendingRun,
        *,
        previous: PendingRun | None,
    ) -> None:
        """Compare-and-set the pending record on the status the decision was made from."""

        result = await self._pending.replace_one(
            {
                "_id": run.run_id,
                "status": previous.status if previous is not None else "awaiting_response",
                "submission_id": previous.submission_id if previous is not None else None,
            },
            self._pending_document(run),
            session=mongo_session,
        )
        if result.matched_count != 1:
            raise PendingRunConflictError("The pending run changed concurrently")

    async def _append_audit(
        self,
        mongo_session: AsyncClientSession,
        run: PendingRun,
        *,
        event: str,
        response: Any,
        submission_id: str | None,
        attempt_trace_id: str | None,
    ) -> None:
        sequence = (
            await self._audit.count_documents({"run_id": run.run_id}, session=mongo_session) + 1
        )
        entry = ClarificationAuditDocument(
            _id=f"{run.run_id}:{sequence}",
            session_id=run.session_id,
            run_id=run.run_id,
            clarification_id=run.clarification_id,
            sequence=sequence,
            event=event,  # type: ignore[arg-type]
            selection_mode=(run.clarification.selection_mode if event == "published" else None),
            question=(run.clarification.question if event == "published" else None),
            options=(list(run.clarification.options) if event == "published" else None),
            response=response,
            response_text=(
                response_text(response, run.clarification) if response is not None else None
            ),
            submission_id=submission_id,
            attempt_trace_id=attempt_trace_id,
            recorded_at=datetime.now(UTC),
        )
        await self._audit.insert_one(
            entry.model_dump(by_alias=True, mode="python"), session=mongo_session
        )

    def _pending_document(self, run: PendingRun) -> dict[str, Any]:
        document = PendingRunDocument(
            _id=run.run_id,
            session_id=run.session_id,
            run_id=run.run_id,
            clarification_id=run.clarification_id,
            active_session_key=(
                run.session_id if run.status in ACTIVE_PENDING_RUN_STATUSES else None
            ),
            tenant_id=run.tenant_id,
            user_id=run.user_id,
            agent_id=run.agent_id,
            selected_agent_id=run.selected_agent_id,
            expected_session_revision=run.expected_session_revision,
            graph_definition_id=run.graph_definition_id,
            original_input_message=encode_message(run.input_message),
            route_reason_code=run.route_reason_code,
            clarification=run.clarification,
            status=run.status,
            submission_id=run.submission_id,
            claimed_response=run.claimed_response,
            trace_id=run.trace_id,
            attempt_trace_ids=list(run.attempt_trace_ids),
            created_at=run.created_at,
            expires_at=run.expires_at,
        )
        data = document.model_dump(by_alias=True, mode="python")
        # A sparse index skips a document only when the field is absent, so an explicit null is
        # still indexed and every closed run would share the key `null`. Dropping the key is what
        # actually makes the slot held by nobody.
        if data.get("active_session_key") is None:
            data.pop("active_session_key", None)
        return data

    def _pending_run(self, document: dict[str, Any]) -> PendingRun:
        stored = PendingRunDocument.model_validate(document)
        return PendingRun(
            session_id=stored.session_id,
            run_id=stored.run_id,
            clarification=stored.clarification,
            status=stored.status,
            agent_id=stored.agent_id,
            selected_agent_id=stored.selected_agent_id,
            expected_session_revision=stored.expected_session_revision,
            tenant_id=stored.tenant_id,
            user_id=stored.user_id,
            input_message=decode_message(stored.original_input_message),
            route_reason_code=stored.route_reason_code,
            trace_id=stored.trace_id,
            graph_definition_id=stored.graph_definition_id,
            continuation_contract_version=stored.continuation_contract_version,
            submission_id=stored.submission_id,
            claimed_response=stored.claimed_response,
            created_at=stored.created_at,
            expires_at=stored.expires_at,
            attempt_trace_ids=tuple(stored.attempt_trace_ids),
        )

    def _audit_entry(self, stored: ClarificationAuditDocument) -> ClarificationAuditEntry:
        return ClarificationAuditEntry(
            session_id=stored.session_id,
            run_id=stored.run_id,
            clarification_id=stored.clarification_id,
            sequence=stored.sequence,
            event=stored.event,
            recorded_at=stored.recorded_at,
            selection_mode=stored.selection_mode,
            question=stored.question,
            option_ids=tuple(option.option_id for option in stored.options or ()),
            response=stored.response,
            response_text=stored.response_text,
            submission_id=stored.submission_id,
            attempt_trace_id=stored.attempt_trace_id,
        )
