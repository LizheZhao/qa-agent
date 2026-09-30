// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { SpanDetail, TraceDetail } from "../../api/types";
import { Inspector } from "./Inspector";

afterEach(cleanup);

describe("Inspector", () => {
  it("explains absent and unavailable capture content", () => {
    render(
      <Inspector
        trace={trace}
        span={span}
        spans={[span]}
        compiledNode={null}
        definition={null}
        onSelectSpan={() => undefined}
      />,
    );

    expect(screen.getAllByText("Not captured for this span.")).toHaveLength(3);
    expect(screen.getByText("Capture was recorded but content is unavailable.")).toBeTruthy();
  });

  it("renders privileged failure detail independently from the public error", () => {
    render(
      <Inspector
        trace={{ ...trace, status: "failed", committed_turn_number: null }}
        span={{
          ...span,
          status: "failed",
          error: {
            code: "agent_execution_failed",
            message: "Agent execution failed",
            retryable: false,
          },
          failure_detail: {
            fingerprint: "a".repeat(64),
            exception_chain: [
              {
                exception_type: "KeyError",
                exception_module: "builtins",
                message: "'ASK_GENOME_CLIENT_CODE'",
                frames: [
                  {
                    filename: "ask_genome_agent/config.py",
                    function: "_client_code",
                    line_number: 123,
                  },
                ],
              },
            ],
            redacted: false,
            truncated: false,
          },
        }}
        spans={[span]}
        compiledNode={null}
        definition={null}
        onSelectSpan={() => undefined}
      />,
    );

    expect(screen.getByText("Failure detail")).toBeTruthy();
    expect(screen.getByText(/ASK_GENOME_CLIENT_CODE/)).toBeTruthy();
  });
});

const trace: TraceDetail = {
  trace_id: "trace",
  session_id: "session",
  root_span_id: "span",
  attempted_turn_number: 1,
  committed_turn_number: 1,
  request_origin: "diagnostic_ui",
  status: "completed",
  observability_status: "complete",
  started_at: "2026-08-18T12:00:00Z",
  completed_at: "2026-08-18T12:00:01Z",
  sequence_started: 1,
  sequence_completed: 2,
  deployment_version: "local",
  application_version: "1",
  core_version: "1",
  agent_versions: {},
};

const span: SpanDetail = {
  span_id: "span",
  parent_span_id: null,
  caused_by_span_ids: [],
  agent_id: "router",
  agent_version: "1",
  kind: "node",
  status: "completed",
  sequence_started: 1,
  sequence_completed: 2,
  superstep: 1,
  iteration: 1,
  started_at: "2026-08-18T12:00:00Z",
  completed_at: "2026-08-18T12:00:01Z",
  duration_ms: 1_000,
  captures: { input: false, resolved_messages: true, output: false, state_delta: false },
};
