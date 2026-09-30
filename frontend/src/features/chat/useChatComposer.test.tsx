// @vitest-environment jsdom

import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ChatResult } from "./api";
import { useChatComposer } from "./useChatComposer";

const api = vi.hoisted(() => ({ send: vi.fn() }));
vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  chatApi: api,
}));

afterEach(cleanup);
beforeEach(() => vi.clearAllMocks());

describe("useChatComposer", () => {
  it("moves the submitted message out of the composer while the request is running", async () => {
    const pending = deferred<ChatResult>();
    api.send.mockReturnValue(pending.promise);
    const onSuccess = vi.fn(async () => undefined);
    const { result } = renderHook(() =>
      useChatComposer({
        sessionId: null,
        onSuccess,
        onTracedFailure: vi.fn(async () => undefined),
        onError: vi.fn(),
      }),
    );

    act(() => result.current.setMessage("Show performance"));
    let submission!: Promise<void>;
    act(() => {
      submission = result.current.submit({ preventDefault: vi.fn() } as never);
    });

    expect(result.current.message).toBe("");
    expect(result.current.optimisticMessage).toBe("Show performance");
    expect(result.current.submitting).toBe(true);

    await act(async () => {
      pending.resolve({
        kind: "completed",
        value: {
          session_id: "session-1",
          trace_id: "trace-1",
          message: { role: "assistant", content: "Done" },
          tool_calls: [],
        },
      });
      await submission;
    });
    expect(result.current.optimisticMessage).toBeNull();
  });

  it("keeps the optimistic message while clarification is pending", async () => {
    api.send.mockResolvedValue({
      kind: "clarification",
      value: {
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
      },
    } satisfies ChatResult);
    const { result } = renderHook(() =>
      useChatComposer({
        sessionId: null,
        onSuccess: vi.fn(async () => undefined),
        onTracedFailure: vi.fn(async () => undefined),
        onError: vi.fn(),
      }),
    );

    act(() => result.current.setMessage("Show performance"));
    await act(async () => {
      await result.current.submit({ preventDefault: vi.fn() } as never);
    });

    expect(result.current.optimisticMessage).toBe("Show performance");
  });
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}
