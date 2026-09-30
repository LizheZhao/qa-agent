import { describe, expect, it } from "vitest";

import type { SpanSummary, TraceDetail } from "../../api/types";
import { runningTraceMessage, STALE_TRACE_AFTER_MS } from "./status";

const now = Date.parse("2026-08-18T12:00:00Z");

function runningTrace(startedAt: string): TraceDetail {
  return {
    trace_id: "trace",
    session_id: "session",
    root_span_id: "root",
    attempted_turn_number: 1,
    request_origin: "diagnostic_ui",
    status: "running",
    observability_status: "recording",
    started_at: startedAt,
    sequence_started: 1,
    deployment_version: "local",
    application_version: "1",
    core_version: "1",
    agent_versions: {},
  };
}

describe("runningTraceMessage", () => {
  it("distinguishes current persisted execution from stale execution", () => {
    const current = runningTrace(new Date(now - STALE_TRACE_AFTER_MS + 1).toISOString());
    const stale = runningTrace(new Date(now - STALE_TRACE_AFTER_MS - 1).toISOString());

    expect(runningTraceMessage(current, [], now)).toContain("no live updates");
    expect(runningTraceMessage(stale, [], now)).toContain("may be stale");
  });

  it("uses the newest persisted span update and ignores terminal traces", () => {
    const trace = runningTrace(new Date(now - STALE_TRACE_AFTER_MS - 1).toISOString());
    const span = {
      started_at: new Date(now - 1_000).toISOString(),
      completed_at: null,
    } as SpanSummary;

    expect(runningTraceMessage(trace, [span], now)).not.toContain("stale");
    expect(runningTraceMessage({ ...trace, status: "completed" }, [span], now)).toBeNull();
  });
});
