import type { AttemptSummary } from "../../api/types";
import { LoadMoreButton } from "../../components/LoadMoreButton";
import { StatusBadge } from "../../components/StatusBadge";
import { formatTime, shortId } from "../../shared/presentation";
import "./runs.css";

export function RunsBrowser({
  attempts,
  cursor,
  sessionTitles,
  selectedTraceId,
  onSelect,
  onLoadOlder,
}: {
  attempts: AttemptSummary[];
  cursor: string | null;
  sessionTitles: Map<string, string>;
  selectedTraceId: string | null;
  onSelect: (attempt: AttemptSummary) => void;
  onLoadOlder: () => void;
}) {
  return (
    <div className="runs-browser">
      <p className="browser-explainer">
        Every router invocation, including failures that never became a conversation turn.
      </p>
      <div className="attempt-list">
        {attempts.map((attempt) => {
          const conversationTitle = sessionTitles.get(attempt.session_id);
          const title =
            attempt.input_preview ??
            attempt.error?.message ??
            conversationTitle ??
            (attempt.committed_turn_number === null
              ? "Uncommitted router run"
              : `Router run for turn ${attempt.attempted_turn_number}`);
          return (
            <button
              className={`attempt-row ${selectedTraceId === attempt.trace_id ? "is-selected" : ""}`}
              key={attempt.trace_id}
              onClick={() => onSelect(attempt)}
              title={`Trace ${attempt.trace_id}\nSession ${attempt.session_id}`}
            >
              <StatusBadge status={attempt.status} />
              <span className="attempt-copy">
                <strong>{title}</strong>
                <small>Turn {attempt.attempted_turn_number} · trace {shortId(attempt.trace_id)}</small>
              </span>
              <time>{formatTime(attempt.started_at)}</time>
            </button>
          );
        })}
      </div>
      {cursor && (
        <LoadMoreButton onClick={onLoadOlder}>
          Load older runs
        </LoadMoreButton>
      )}
    </div>
  );
}
