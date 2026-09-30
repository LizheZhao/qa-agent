import { useCallback, useEffect, useRef, useState } from "react";

import { errorMessage } from "../../shared/presentation";
import { updateWorkspaceUrl } from "../../app/navigation";
import { Inspector } from "../inspector/Inspector";
import { useTraceExplorer } from "../traces/useTraceExplorer";
import { ConversationView } from "./ConversationView";
import { SessionBrowser } from "./SessionBrowser";
import { useSessionWorkspace } from "./useSessionWorkspace";

export function ConversationsRoute() {
  const initialized = useRef(false);
  const [initializing, setInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const reportError = useCallback((message: string) => setError(message || null), []);
  const traces = useTraceExplorer(reportError);
  const sessions = useSessionWorkspace({
    onError: reportError,
    onResetTrace: traces.clear,
    openTrace: traces.openTrace,
  });

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    void (async () => {
      try {
        const items = await sessions.loadSessions();
        const match = window.location.pathname.match(/^\/diagnostics\/sessions\/([^/]+)$/);
        const sessionId = match ? decodeURIComponent(match[1] ?? "") : items[0]?.session_id;
        const search = new URLSearchParams(window.location.search);
        if (sessionId) {
          await sessions.openSession(
            sessionId,
            "conversations",
            search.get("trace"),
            search.get("span"),
          );
        }
      } catch (caught) {
        reportError(errorMessage(caught));
      } finally {
        setInitializing(false);
      }
    })();
  }, [reportError, sessions.loadSessions, sessions.openSession]);

  return (
    <>
      <aside className="browser-panel">
        <SessionBrowser
          sessions={sessions.sessions}
          cursor={sessions.sessionCursor}
          selectedSessionId={sessions.selectedSessionId}
          onSelect={(sessionId) => void sessions.openSession(sessionId, "conversations")}
          onLoadOlder={() => void sessions.loadOlderSessions()}
        />
      </aside>
      <ConversationView
        session={sessions.selectedSession}
        turns={sessions.sortedTurns}
        turnCursor={sessions.turnCursor}
        selectedTraceId={traces.selectedTraceId}
        trace={traces.trace}
        projection={traces.projection}
        selectedSpan={traces.span}
        spans={traces.spans}
        traceLoading={traces.loading}
        loading={initializing || sessions.loading}
        error={error}
        onOpenTrace={(traceId) =>
          void traces.openTrace(traceId, sessions.selectedSessionId, "conversations")
        }
        onLoadOlder={() => void sessions.loadOlderTurns()}
        onOpenSpan={(spanId) =>
          traces.trace &&
          void traces.openSpan(
            traces.trace.trace_id,
            spanId,
            sessions.selectedSessionId,
            "conversations",
          )
        }
        onSelectCompiled={(node) => {
          traces.selectCompiledNode(node);
          if (traces.trace) {
            updateWorkspaceUrl(
              "conversations",
              sessions.selectedSessionId,
              traces.trace.trace_id,
            );
          }
        }}
      />
      <aside className="inspector-panel">
        <Inspector
          trace={traces.trace}
          span={traces.span}
          spans={traces.spans}
          compiledNode={traces.compiledNode}
          definition={null}
          onSelectSpan={(spanId) =>
            traces.trace &&
            void traces.openSpan(
              traces.trace.trace_id,
              spanId,
              sessions.selectedSessionId,
              "conversations",
            )
          }
        />
      </aside>
    </>
  );
}
