"""Bounded, ordered, best-effort persistence for execution snapshots."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import suppress

from agentic_orchestration.observability.store import ExecutionSnapshot, ExecutionTraceStore

logger = logging.getLogger(__name__)


class BufferedExecutionWriter:
    """Batch snapshots and persist them without delaying graph callbacks."""

    def __init__(
        self,
        store: ExecutionTraceStore,
        *,
        trace_id: str,
        max_pending_snapshots: int = 256,
        batch_delay_seconds: float = 0.01,
        retry_delay_seconds: float = 0.25,
    ) -> None:
        if max_pending_snapshots < 1:
            raise ValueError("max_pending_snapshots must be positive")
        self._store = store
        self._trace_id = trace_id
        self._max_pending_snapshots = max_pending_snapshots
        self._batch_delay_seconds = batch_delay_seconds
        self._retry_delay_seconds = retry_delay_seconds
        self._pending: deque[ExecutionSnapshot] = deque()
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False
        self._degraded = False
        self._overflow_reported = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def enqueue(self, snapshot: ExecutionSnapshot) -> bool:
        """Queue the newest snapshot for a record without awaiting storage I/O."""

        if self._closing or self._closed:
            self._mark_dropped()
            return False
        if len(self._pending) >= self._max_pending_snapshots:
            self._mark_dropped()
            return False
        self._pending.append(snapshot)
        self._idle.clear()
        self._wake.set()
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"execution-writer:{self._trace_id}")
        return True

    async def close(self, *, timeout_seconds: float) -> bool:
        """Flush pending snapshots within a bound and release the writer task."""

        if self._closed:
            return not self._degraded
        self._closing = True
        self._wake.set()
        try:
            async with asyncio.timeout(timeout_seconds):
                await self._idle.wait()
                self._wake.set()
                if self._task is not None:
                    await self._task
        except asyncio.CancelledError:
            self._degraded = True
            if self._task is not None:
                self._task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._task
            raise
        except TimeoutError:
            self._degraded = True
            logger.error("execution observability flush timed out trace_id=%s", self._trace_id)
            if self._task is not None:
                self._task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._task
        finally:
            self._closed = True
        return not self._degraded

    async def _run(self) -> None:
        storage_failures = 0
        while True:
            await self._wake.wait()
            if not self._closing and self._batch_delay_seconds:
                await asyncio.sleep(self._batch_delay_seconds)
            batch = tuple(self._pending)
            self._pending.clear()
            self._wake.clear()
            if batch:
                try:
                    await self._store.save_snapshots(batch)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    storage_failures += 1
                    if storage_failures == 1:
                        logger.warning(
                            "execution observability write failed; retrying trace_id=%s type=%s",
                            self._trace_id,
                            type(exc).__name__,
                        )
                    self._restore_failed_batch(batch)
                    await asyncio.sleep(self._retry_delay_seconds)
                else:
                    storage_failures = 0
            if self._pending:
                self._wake.set()
                continue
            self._idle.set()
            if self._closing:
                return

    def _restore_failed_batch(self, batch: tuple[ExecutionSnapshot, ...]) -> None:
        available = self._max_pending_snapshots - len(self._pending)
        restored = batch[-available:] if available else ()
        if len(restored) != len(batch):
            for _ in range(len(batch) - len(restored)):
                self._mark_dropped()
        self._pending.extendleft(reversed(restored))
        self._wake.set()

    def _mark_dropped(self) -> None:
        self._degraded = True
        if not self._overflow_reported:
            self._overflow_reported = True
            logger.error("execution observability queue overflow trace_id=%s", self._trace_id)
