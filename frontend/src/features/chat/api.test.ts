import { afterEach, describe, expect, it, vi } from "vitest";

import type { TracedChatError } from "../../api/types";
import { chatApi, tracedFailureMessage } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("chat API", () => {
  it("keeps the failed run discoverable without leaving Lab", () => {
    const failure: TracedChatError = {
      error: { code: "gateway_error", message: "Enterprise gateway request failed" },
      session_id: "session",
      trace_id: "12345678-aaaa-bbbb-cccc-123456789abc",
    };

    expect(tracedFailureMessage(failure)).toBe(
      "Enterprise gateway request failed · run 12345678…9abc",
    );
  });

  it("distinguishes a pending clarification from a completed chat response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          session_id: "session-1",
          trace_id: "trace-1",
          clarification_id: "clarification-1",
          question: "Which market?",
          reason_code: "ambiguous_market",
          selection_mode: "single",
          options: [],
          free_text_allowed: true,
          cancel_allowed: true,
          expires_at: "2026-09-17T12:00:00Z",
        }),
      ),
    );

    const result = await chatApi.send("Show performance");

    expect(result.kind).toBe("clarification");
    expect(result.value.session_id).toBe("session-1");
  });

  it("submits a typed clarification response to the encoded resource", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      response({
        session_id: "session/1",
        trace_id: "trace-2",
        turn_number: 1,
        message: "Done",
        closed_by: "answered",
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await chatApi.respond(
      {
        session_id: "session/1",
        trace_id: "trace-1",
        clarification_id: "clarification/1",
        question: "Which market?",
        reason_code: "ambiguous_market",
        selection_mode: "single",
        options: [],
        free_text_allowed: true,
        cancel_allowed: true,
        expires_at: "2026-09-17T12:00:00Z",
      },
      { form: "options", option_ids: ["us"] },
    );

    expect(result.kind).toBe("completed");
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/sessions/session%2F1/clarifications/clarification%2F1/responses",
    );
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({
      submission_id: expect.any(String),
      response: { form: "options", option_ids: ["us"] },
    });
  });
});

function response(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
