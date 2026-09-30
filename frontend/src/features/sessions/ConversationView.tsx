import type {
  SessionSummary,
  SpanDetail,
  SpanSummary,
  TraceDetail,
  TurnSummary,
} from "../../api/types";
import { EmptyState } from "../../components/EmptyState";
import { WorkspacePanel } from "../../components/WorkspacePanel";
import { shortId } from "../../shared/presentation";
import { TraceExplorer } from "../traces/TraceExplorer";
import type { ProjectedCompiledNode, ProjectedTrace } from "../traces/projection";
import { ConversationTranscript } from "./ConversationTranscript";

export function ConversationView({
  session,
  turns,
  turnCursor,
  selectedTraceId,
  trace,
  projection,
  selectedSpan,
  spans,
  traceLoading,
  loading,
  error,
  onOpenTrace,
  onLoadOlder,
  onOpenSpan,
  onSelectCompiled,
}: {
  session: SessionSummary | null;
  turns: TurnSummary[];
  turnCursor: string | null;
  selectedTraceId: string | null;
  trace: TraceDetail | null;
  projection: ProjectedTrace | null;
  selectedSpan: SpanDetail | null;
  spans: SpanSummary[];
  traceLoading: boolean;
  loading: boolean;
  error: string | null;
  onOpenTrace: (traceId: string) => void;
  onLoadOlder: () => void;
  onOpenSpan: (spanId: string) => void;
  onSelectCompiled: (node: ProjectedCompiledNode) => void;
}) {
  const actions = session ? (
    <span className="muted">
      {session.revision} committed turns · {shortId(session.session_id)}
    </span>
  ) : undefined;

  return (
    <WorkspacePanel
      eyebrow="Conversation"
      title={session?.display_title ?? "Ask the router"}
      actions={actions}
      error={error}
      contentClassName="conversation-split"
    >
      <section className="conversation-split__transcript" aria-label="Conversation transcript">
        {loading && <EmptyState>Loading diagnostic state…</EmptyState>}
        {!loading && !session && (
          <EmptyState>Select a conversation from the left.</EmptyState>
        )}
        <ConversationTranscript
          turns={turns}
          turnCursor={turnCursor}
          view="conversations"
          selectedTraceId={selectedTraceId}
          onOpenTrace={onOpenTrace}
          onLoadOlder={onLoadOlder}
        />
      </section>
      <section className="conversation-split__trace" aria-label="Selected turn execution">
        {!trace && !traceLoading && (
          <EmptyState compact>Select an assistant response to inspect its execution.</EmptyState>
        )}
        <TraceExplorer
          trace={trace}
          projection={projection}
          span={selectedSpan}
          spans={spans}
          loading={traceLoading}
          onOpenSpan={onOpenSpan}
          onSelectCompiled={onSelectCompiled}
          variant="split"
        />
      </section>
    </WorkspacePanel>
  );
}
