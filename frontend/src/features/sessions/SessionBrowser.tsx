import type { SessionSummary } from "../../api/types";
import { BrowserList, BrowserRow } from "../../components/BrowserList";
import { LoadMoreButton } from "../../components/LoadMoreButton";
import { formatTime, shortId } from "../../shared/presentation";

export function SessionBrowser({
  sessions,
  cursor,
  selectedSessionId,
  onSelect,
  onLoadOlder,
}: {
  sessions: SessionSummary[];
  cursor: string | null;
  selectedSessionId: string | null;
  onSelect: (sessionId: string) => void;
  onLoadOlder: () => void;
}) {
  return (
    <BrowserList>
      {sessions.map((item) => (
        <BrowserRow
          selected={selectedSessionId === item.session_id}
          key={item.session_id}
          onClick={() => onSelect(item.session_id)}
        >
          <span className="row-title" title={item.display_title}>
            {item.display_title}
          </span>
          <span className="row-meta">
            {item.revision} turn{item.revision === 1 ? "" : "s"} · {shortId(item.session_id)}
          </span>
          <span className="row-time">{formatTime(item.updated_at)}</span>
        </BrowserRow>
      ))}
      {cursor && (
        <LoadMoreButton onClick={onLoadOlder}>
          Load older sessions
        </LoadMoreButton>
      )}
    </BrowserList>
  );
}
