import type { SpanDetail, SpanSummary, TraceDetail } from "../../api/types";
import { EmptyState } from "../../components/EmptyState";
import { Notice } from "../../components/Notice";
import { StatusBadge } from "../../components/StatusBadge";
import { shortId } from "../../shared/presentation";
import { TraceCanvas } from "./TraceCanvas";
import { TraceProjectionList } from "./TraceProjectionList";
import { runningTraceMessage } from "./status";
import type { ProjectedCompiledNode, ProjectedTrace } from "./projection";

export function TraceExplorer({
  trace,
  projection,
  span,
  spans,
  loading,
  onOpenSpan,
  onSelectCompiled,
  variant = "default",
}: {
  trace: TraceDetail | null;
  projection: ProjectedTrace | null;
  span: SpanDetail | null;
  spans: SpanSummary[];
  loading: boolean;
  onOpenSpan: (spanId: string) => void;
  onSelectCompiled: (node: ProjectedCompiledNode) => void;
  variant?: "default" | "split";
}) {
  if (loading) {
    return (
      <section className={`execution-list ${variant === "split" ? "execution-list--split" : ""}`}>
        <EmptyState compact>Loading execution…</EmptyState>
      </section>
    );
  }
  if (!trace) return null;
  const runningMessage = runningTraceMessage(trace, spans);

  return (
    <section className={`execution-list ${variant === "split" ? "execution-list--split" : ""}`}>
      <div className="section-heading">
        <span>Trace {shortId(trace.trace_id)}</span>
        <StatusBadge status={trace.status} />
      </div>
      {trace.observability_status === "degraded" && (
        <Notice tone="warning">Observability is degraded; this trace may be incomplete.</Notice>
      )}
      {runningMessage && <Notice tone="warning">{runningMessage}</Notice>}
      {projection ? (
        <>
          <TraceCanvas
            key={projection.traceId}
            projection={projection}
            onSelectSpan={onOpenSpan}
            onSelectCompiled={onSelectCompiled}
          />
          <details className="structured-execution">
            <summary>Structured execution list</summary>
            <TraceProjectionList
              projection={projection}
              selectedSpanId={span?.span_id ?? null}
              onSelect={onOpenSpan}
            />
          </details>
        </>
      ) : (
        <EmptyState compact>Hydrating graph definitions…</EmptyState>
      )}
    </section>
  );
}
