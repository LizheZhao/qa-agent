import type { TurnSummary } from "../../api/types";
import type { WorkspaceView } from "../../app/navigation";
import { LoadMoreButton } from "../../components/LoadMoreButton";
import "./sessions.css";

export function ConversationTranscript({
  turns,
  turnCursor,
  view,
  onOpenTrace,
  onLoadOlder,
  showTraceLinks = true,
  selectedTraceId = null,
}: {
  turns: TurnSummary[];
  turnCursor: string | null;
  view: WorkspaceView;
  onOpenTrace: (traceId: string, view: WorkspaceView) => void;
  onLoadOlder: () => void;
  showTraceLinks?: boolean;
  selectedTraceId?: string | null;
}) {
  return (
    <>
      {turns.map((turn) => (
        <article className="turn" key={turn.turn_number}>
          <div className="turn-label">Turn {turn.turn_number}</div>
          <div className="message message--user">{turn.input.content}</div>
          {showTraceLinks && turn.trace_id ? (
            <button
              className={`message message--assistant message--selectable ${selectedTraceId === turn.trace_id ? "is-selected" : ""}`}
              type="button"
              aria-pressed={selectedTraceId === turn.trace_id}
              onClick={() => onOpenTrace(turn.trace_id!, view)}
            >
              <span>{turn.response.content}</span>
              <small>
                {turn.routing_outcome}
                {turn.selected_agent_id ? ` → ${turn.selected_agent_id}` : ""} · inspect execution
              </small>
            </button>
          ) : (
            <div className="message message--assistant">
              {turn.response.content}
              {showTraceLinks && <small className="message-trace-missing">No trace recorded</small>}
            </div>
          )}
        </article>
      ))}
      {turnCursor && (
        <LoadMoreButton inline onClick={onLoadOlder}>
          Load older turns
        </LoadMoreButton>
      )}
    </>
  );
}
