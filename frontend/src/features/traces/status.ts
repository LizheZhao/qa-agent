import type { SpanSummary, TraceDetail } from "../../api/types";

export const STALE_TRACE_AFTER_MS = 5 * 60 * 1000;

export function runningTraceMessage(
  trace: TraceDetail,
  spans: SpanSummary[],
  now = Date.now(),
): string | null {
  if (trace.status !== "running") return null;
  const timestamps = [
    trace.started_at,
    ...spans.flatMap((span) => [span.started_at, span.completed_at].filter(isTimestamp)),
  ].map((value) => Date.parse(value));
  const latestUpdate = Math.max(...timestamps.filter(Number.isFinite));
  return now - latestUpdate > STALE_TRACE_AFTER_MS
    ? "Running trace has no live updates and its persisted state may be stale."
    : "Running trace · no live updates. Refresh to hydrate its latest persisted state.";
}

function isTimestamp(value: string | null | undefined): value is string {
  return typeof value === "string";
}
