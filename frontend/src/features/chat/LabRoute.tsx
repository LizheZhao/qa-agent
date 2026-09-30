import { useCallback, useEffect, useRef, useState } from "react";

import type { ContinuationBody, PendingClarification } from "../../api/types";
import { updateWorkspaceUrl } from "../../app/navigation";
import { errorMessage } from "../../shared/presentation";
import { useSessionWorkspace } from "../sessions/useSessionWorkspace";
import { LabView } from "./LabView";
import { type ChatResult, chatApi } from "./api";
import { useChatComposer } from "./useChatComposer";

const resetTrace = () => undefined;
const ignoreTrace = async () => undefined;

export function LabRoute() {
  const initialized = useRef(false);
  const [labSessionId, setLabSessionId] = useState<string | null>(null);
  const [initializing, setInitializing] = useState(true);
  const [pendingClarification, setPendingClarification] =
    useState<PendingClarification | null>(null);
  const [clarificationSubmitting, setClarificationSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reportError = useCallback((message: string) => setError(message || null), []);
  const sessions = useSessionWorkspace({
    onError: reportError,
    onResetTrace: resetTrace,
    openTrace: ignoreTrace,
  });

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    void (async () => {
      try {
        await sessions.loadSessions();
        const sessionId = new URLSearchParams(window.location.search).get("session");
        if (sessionId) {
          const history = await chatApi.history(sessionId);
          setLabSessionId(sessionId);
          setPendingClarification(history.pending_clarification ?? null);
          if (history.messages.length > 0) await sessions.openLabSession(sessionId);
        }
      } catch (caught) {
        reportError(errorMessage(caught));
      } finally {
        setInitializing(false);
      }
    })();
  }, [reportError, sessions.loadSessions, sessions.openLabSession]);

  function startNewSession() {
    setLabSessionId(null);
    setPendingClarification(null);
    composer.clearOptimisticMessage();
    setError(null);
    sessions.clearSelection(null);
    updateWorkspaceUrl("lab", null);
  }

  const chatSucceeded = useCallback(
    async (response: ChatResult) => {
      setLabSessionId(response.value.session_id);
      if (response.kind === "clarification") {
        setPendingClarification(response.value);
        updateWorkspaceUrl("lab", response.value.session_id);
      } else {
        setPendingClarification(null);
      }
      try {
        await sessions.loadSessions();
        if (response.kind === "completed") {
          await sessions.openLabSession(response.value.session_id);
        }
      } catch (caught) {
        reportError(errorMessage(caught));
      }
    },
    [reportError, sessions.loadSessions, sessions.openLabSession],
  );
  const chatFailed = useCallback(async () => {
    try {
      await sessions.loadSessions();
    } catch (caught) {
      reportError(errorMessage(caught));
    }
  }, [reportError, sessions.loadSessions]);
  const composer = useChatComposer({
    sessionId: labSessionId,
    onSuccess: chatSucceeded,
    onTracedFailure: chatFailed,
    onError: reportError,
  });

  const respondToClarification = useCallback(
    async (response: ContinuationBody["response"]) => {
      if (!pendingClarification || clarificationSubmitting) return;
      setClarificationSubmitting(true);
      reportError("");
      try {
        const result = await chatApi.respond(pendingClarification, response);
        if (result.kind === "clarification") {
          setPendingClarification(result.value);
        } else {
          setPendingClarification(null);
          composer.clearOptimisticMessage();
          await sessions.loadSessions();
          await sessions.openLabSession(result.value.session_id);
        }
      } catch (caught) {
        try {
          const history = await chatApi.history(pendingClarification.session_id);
          setPendingClarification(history.pending_clarification ?? null);
        } catch {
          // Preserve the original response failure; history refresh is best effort.
        }
        reportError(errorMessage(caught));
      } finally {
        setClarificationSubmitting(false);
      }
    }, [
      clarificationSubmitting,
      composer.clearOptimisticMessage,
      pendingClarification,
      reportError,
      sessions.loadSessions,
      sessions.openLabSession,
    ]);

  return (
    <LabView
      session={sessions.selectedSession}
      turns={sessions.sortedTurns}
      turnCursor={sessions.turnCursor}
      labSessionId={labSessionId}
      loading={initializing || sessions.loading}
      error={error}
      onNewSession={startNewSession}
      message={composer.message}
      submitting={composer.submitting}
      optimisticMessage={composer.optimisticMessage}
      pendingClarification={pendingClarification}
      clarificationSubmitting={clarificationSubmitting}
      onMessageChange={composer.setMessage}
      onSubmit={(event) => void composer.submit(event)}
      onClarificationResponse={(response) => void respondToClarification(response)}
      onLoadOlder={() => void sessions.loadOlderTurns()}
    />
  );
}
