import type { FormEvent } from "react";
import { useEffect, useRef } from "react";

import type {
  ContinuationBody,
  PendingClarification,
  SessionSummary,
  TurnSummary,
} from "../../api/types";
import { EmptyState } from "../../components/EmptyState";
import { WorkspacePanel } from "../../components/WorkspacePanel";
import { shortId } from "../../shared/presentation";
import { ConversationTranscript } from "../sessions/ConversationTranscript";
import { ChatComposer } from "./ChatComposer";
import { ClarificationForm } from "./ClarificationForm";

export function LabView({
  session,
  turns,
  turnCursor,
  labSessionId,
  loading,
  error,
  onNewSession,
  message,
  submitting,
  optimisticMessage,
  pendingClarification,
  clarificationSubmitting,
  onMessageChange,
  onSubmit,
  onClarificationResponse,
  onLoadOlder,
}: {
  session: SessionSummary | null;
  turns: TurnSummary[];
  turnCursor: string | null;
  labSessionId: string | null;
  loading: boolean;
  error: string | null;
  onNewSession: () => void;
  message: string;
  submitting: boolean;
  optimisticMessage: string | null;
  pendingClarification: PendingClarification | null;
  clarificationSubmitting: boolean;
  onMessageChange: (message: string) => void;
  onSubmit: (event: FormEvent) => void;
  onClarificationResponse: (response: ContinuationBody["response"]) => void;
  onLoadOlder: () => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = scrollRef.current;
    if (element) element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
  }, [turns, submitting, optimisticMessage, pendingClarification]);

  const actions = (
    <>
      <button className="header-action" onClick={onNewSession}>New chat</button>
      {session && (
        <span className="muted">
          {session.revision} committed turns · {shortId(session.session_id)}
        </span>
      )}
    </>
  );

  return (
    <WorkspacePanel
      eyebrow={labSessionId ? "Lab conversation" : "New diagnostic session"}
      title={
        session?.display_title ?? (pendingClarification ? "Clarification needed" : "Ask the router")
      }
      actions={actions}
      error={error}
      scrollRef={scrollRef}
      footer={
        pendingClarification ? (
          <ClarificationForm
            key={pendingClarification.clarification_id}
            clarification={pendingClarification}
            submitting={clarificationSubmitting}
            onRespond={onClarificationResponse}
          />
        ) : (
          <ChatComposer
            sessionId={labSessionId}
            message={message}
            submitting={submitting}
            onMessageChange={onMessageChange}
            onSubmit={onSubmit}
          />
        )
      }
    >
      {loading && <EmptyState>Loading diagnostic state…</EmptyState>}
      {!loading && turns.length === 0 && !optimisticMessage && !pendingClarification && (
        <EmptyState>
          <strong>Start a diagnostic conversation.</strong>
          <span>Messages run through the live router and can continue across multiple turns.</span>
        </EmptyState>
      )}
      <ConversationTranscript
        turns={turns}
        turnCursor={turnCursor}
        view="lab"
        onOpenTrace={() => undefined}
        onLoadOlder={onLoadOlder}
        showTraceLinks={false}
      />
      {optimisticMessage && (
        <article className="turn turn--optimistic">
          <div className="turn-label">Pending turn</div>
          <div className="message message--user">{optimisticMessage}</div>
        </article>
      )}
      {submitting && (
        <div className="message message--assistant message--pending">Router is responding…</div>
      )}
    </WorkspacePanel>
  );
}
