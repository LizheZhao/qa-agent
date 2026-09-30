import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { errorMessage } from "../../shared/presentation";
import { updateWorkspaceUrl } from "../../app/navigation";
import { Inspector } from "../inspector/Inspector";
import { useTraceExplorer } from "../traces/useTraceExplorer";
import { RunView } from "./RunView";
import { RunsBrowser } from "./RunsBrowser";
import { useRunCatalog } from "./useRunCatalog";

export function RunsRoute() {
  const initialized = useRef(false);
  const [initializing, setInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const reportError = useCallback((message: string) => setError(message || null), []);
  const traces = useTraceExplorer(reportError);
  const runs = useRunCatalog();
  const sessionTitles = useMemo(() => new Map<string, string>(), []);

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    void (async () => {
      try {
        const attempts = await runs.loadRuns();
        const search = new URLSearchParams(window.location.search);
        const traceId = search.get("trace") ?? attempts[0]?.trace_id;
        const target = attempts.find((item) => item.trace_id === traceId);
        if (traceId) {
          setSessionId(target?.session_id ?? null);
          await traces.openTrace(
            traceId,
            target?.session_id ?? null,
            "runs",
            search.get("span"),
          );
        }
      } catch (caught) {
        reportError(errorMessage(caught));
      } finally {
        setInitializing(false);
      }
    })();
  }, [reportError, runs.loadRuns, traces.openTrace]);

  return (
    <>
      <aside className="browser-panel">
        <RunsBrowser
          attempts={runs.attempts}
          cursor={runs.cursor}
          sessionTitles={sessionTitles}
          selectedTraceId={traces.selectedTraceId}
          onSelect={(attempt) => {
            setSessionId(attempt.session_id);
            void traces.openTrace(attempt.trace_id, attempt.session_id, "runs");
          }}
          onLoadOlder={() => void runs.loadOlderRuns()}
        />
      </aside>
      <RunView
        trace={traces.trace}
        projection={traces.projection}
        selectedSpan={traces.span}
        spans={traces.spans}
        traceLoading={traces.loading}
        loading={initializing}
        error={error}
        onOpenSpan={(spanId) =>
          traces.trace &&
          void traces.openSpan(traces.trace.trace_id, spanId, sessionId, "runs")
        }
        onSelectCompiled={(node) => {
          traces.selectCompiledNode(node);
          if (traces.trace) updateWorkspaceUrl("runs", sessionId, traces.trace.trace_id);
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
            void traces.openSpan(traces.trace.trace_id, spanId, sessionId, "runs")
          }
        />
      </aside>
    </>
  );
}
