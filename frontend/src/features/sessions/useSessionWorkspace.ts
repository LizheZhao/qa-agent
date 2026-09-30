import { useCallback, useMemo, useRef, useState } from "react";

import { ApiError } from "../../api/client";
import type { SessionSummary, TurnSummary } from "../../api/types";
import type { WorkspaceView } from "../../app/navigation";
import { updateWorkspaceUrl } from "../../app/navigation";
import { errorMessage } from "../../shared/presentation";
import { sessionsApi } from "./api";

type OpenTrace = (
  traceId: string,
  sessionId: string | null,
  view: WorkspaceView,
  preferredSpan?: string | null,
) => Promise<void>;

export function useSessionWorkspace({
  onError,
  onResetTrace,
  openTrace,
}: {
  onError: (message: string) => void;
  onResetTrace: () => void;
  openTrace: OpenTrace;
}) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionCursor, setSessionCursor] = useState<string | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [selectedSession, setSelectedSession] = useState<SessionSummary | null>(null);
  const [turns, setTurns] = useState<TurnSummary[]>([]);
  const [turnCursor, setTurnCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const sessionRequest = useRef(0);
  const sessionPageRequest = useRef(0);
  const turnPageRequest = useRef(0);
  const activeSessionId = useRef<string | null>(null);
  const catalogRequest = useRef(0);

  const loadSessions = useCallback(async () => {
    const requestId = ++catalogRequest.current;
    sessionPageRequest.current += 1;
    const page = await sessionsApi.sessions();
    if (requestId !== catalogRequest.current) return page.items;
    setSessions(page.items);
    setSessionCursor(page.next_cursor ?? null);
    return page.items;
  }, []);

  const openSession = useCallback(
    async (
      sessionId: string,
      view: WorkspaceView,
      preferredTrace?: string | null,
      preferredSpan?: string | null,
    ) => {
      const requestId = ++sessionRequest.current;
      turnPageRequest.current += 1;
      setLoading(true);
      onError("");
      activeSessionId.current = sessionId;
      setSelectedSessionId(sessionId);
      setSelectedSession(null);
      setTurns([]);
      setTurnCursor(null);
      onResetTrace();
      try {
        const [sessionDetail, turnPage] = await Promise.all([
          sessionsApi.session(sessionId),
          sessionsApi.turns(sessionId),
        ]);
        if (requestId !== sessionRequest.current) return;
        setSelectedSession(sessionDetail);
        setTurns(turnPage.items);
        setTurnCursor(turnPage.next_cursor ?? null);
        const targetTrace = preferredTrace ?? turnPage.items[0]?.trace_id;
        if (targetTrace) await openTrace(targetTrace, sessionId, view, preferredSpan);
        else updateWorkspaceUrl(view, sessionId);
      } catch (caught) {
        if (requestId !== sessionRequest.current) return;
        if (caught instanceof ApiError && caught.status === 404 && preferredTrace) {
          setSelectedSession(null);
          setTurns([]);
          await openTrace(preferredTrace, sessionId, view, preferredSpan);
        } else {
          onError(errorMessage(caught));
        }
      } finally {
        if (requestId === sessionRequest.current) setLoading(false);
      }
    },
    [onError, onResetTrace, openTrace],
  );

  const openLabSession = useCallback(
    async (sessionId: string) => {
      const requestId = ++sessionRequest.current;
      turnPageRequest.current += 1;
      setLoading(true);
      onError("");
      activeSessionId.current = sessionId;
      setSelectedSessionId(sessionId);
      setSelectedSession(null);
      setTurns([]);
      setTurnCursor(null);
      onResetTrace();
      try {
        const [sessionDetail, turnPage] = await Promise.all([
          sessionsApi.session(sessionId),
          sessionsApi.turns(sessionId),
        ]);
        if (requestId !== sessionRequest.current) return false;
        setSelectedSession(sessionDetail);
        setTurns(turnPage.items);
        setTurnCursor(turnPage.next_cursor ?? null);
        updateWorkspaceUrl("lab", sessionId);
        return true;
      } catch (caught) {
        if (requestId !== sessionRequest.current) return false;
        activeSessionId.current = null;
        setSelectedSessionId(null);
        onError(errorMessage(caught));
        updateWorkspaceUrl("lab", null);
        return false;
      } finally {
        if (requestId === sessionRequest.current) setLoading(false);
      }
    },
    [onError, onResetTrace],
  );

  const loadOlderSessions = useCallback(async () => {
    if (!sessionCursor) return;
    const requestId = ++sessionPageRequest.current;
    const page = await sessionsApi.sessions(sessionCursor);
    if (requestId !== sessionPageRequest.current) return;
    setSessions((current) => [...current, ...page.items]);
    setSessionCursor(page.next_cursor ?? null);
  }, [sessionCursor]);

  const loadOlderTurns = useCallback(async () => {
    if (!selectedSessionId || !turnCursor) return;
    const requestId = ++turnPageRequest.current;
    const targetSessionId = selectedSessionId;
    const page = await sessionsApi.turns(selectedSessionId, turnCursor);
    if (
      requestId !== turnPageRequest.current ||
      targetSessionId !== activeSessionId.current
    )
      return;
    setTurns((current) => [...current, ...page.items]);
    setTurnCursor(page.next_cursor ?? null);
  }, [selectedSessionId, turnCursor]);

  const clearSelection = useCallback((sessionId: string | null = null) => {
    sessionRequest.current += 1;
    turnPageRequest.current += 1;
    activeSessionId.current = sessionId;
    setSelectedSessionId(sessionId);
    setSelectedSession(null);
    setTurns([]);
    setTurnCursor(null);
  }, []);

  const sortedTurns = useMemo(
    () => [...turns].sort((left, right) => left.turn_number - right.turn_number),
    [turns],
  );

  return {
    sessions,
    sessionCursor,
    selectedSessionId,
    selectedSession,
    sortedTurns,
    turnCursor,
    loading,
    loadSessions,
    openSession,
    openLabSession,
    loadOlderSessions,
    loadOlderTurns,
    clearSelection,
  };
}
