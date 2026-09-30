import type { SpanDetail, SpanSummary, TraceDetail } from "../../api/types";
import { EmptyState } from "../../components/EmptyState";
import { WorkspacePanel } from "../../components/WorkspacePanel";
import { TraceExplorer } from "../traces/TraceExplorer";
import type { ProjectedCompiledNode, ProjectedTrace } from "../traces/projection";

export function RunView({
  trace,
  projection,
  selectedSpan,
  spans,
  traceLoading,
  loading,
  error,
  onOpenSpan,
  onSelectCompiled,
}: {
  trace: TraceDetail | null;
  projection: ProjectedTrace | null;
  selectedSpan: SpanDetail | null;
  spans: SpanSummary[];
  traceLoading: boolean;
  loading: boolean;
  error: string | null;
  onOpenSpan: (spanId: string) => void;
  onSelectCompiled: (node: ProjectedCompiledNode) => void;
}) {
  return (
    <WorkspacePanel
      eyebrow="Execution run"
      title={
        trace
          ? `Turn ${trace.attempted_turn_number} · ${trace.status}`
          : "Select a run"
      }
      error={error}
    >
      {loading && <EmptyState>Loading diagnostic state…</EmptyState>}
      <TraceExplorer
        trace={trace}
        projection={projection}
        span={selectedSpan}
        spans={spans}
        loading={traceLoading}
        onOpenSpan={onOpenSpan}
        onSelectCompiled={onSelectCompiled}
      />
    </WorkspacePanel>
  );
}
