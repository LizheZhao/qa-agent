// @vitest-environment jsdom

import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SessionSummary, TurnSummary } from "../../api/types";
import { useSessionWorkspace } from "./useSessionWorkspace";

const api = vi.hoisted(() => ({
  sessions: vi.fn(),
  session: vi.fn(),
  turns: vi.fn(),
}));

vi.mock("./api", () => ({ sessionsApi: api }));

afterEach(cleanup);
beforeEach(() => vi.clearAllMocks());

describe("useSessionWorkspace", () => {
  it("does not let an older session request replace a newer selection", async () => {
    const sessionA = deferred<SessionSummary>();
    const sessionB = deferred<SessionSummary>();
    const turnsA = deferred<{ items: TurnSummary[]; next_cursor: null }>();
    const turnsB = deferred<{ items: TurnSummary[]; next_cursor: null }>();
    api.session.mockImplementation((id: string) => (id === "a" ? sessionA.promise : sessionB.promise));
    api.turns.mockImplementation((id: string) => (id === "a" ? turnsA.promise : turnsB.promise));
    const openTrace = vi.fn(async () => undefined);
    const { result } = renderHook(() =>
      useSessionWorkspace({ onError: vi.fn(), onResetTrace: vi.fn(), openTrace }),
    );

    let openA!: Promise<void>;
    let openB!: Promise<void>;
    act(() => {
      openA = result.current.openSession("a", "conversations");
      openB = result.current.openSession("b", "conversations");
    });
    await act(async () => {
      sessionB.resolve(summary("b"));
      turnsB.resolve({ items: [turn("trace-b")], next_cursor: null });
      await openB;
    });
    await act(async () => {
      sessionA.resolve(summary("a"));
      turnsA.resolve({ items: [turn("trace-a")], next_cursor: null });
      await openA;
    });

    expect(result.current.selectedSessionId).toBe("b");
    expect(result.current.selectedSession?.session_id).toBe("b");
    expect(result.current.sortedTurns[0]?.trace_id).toBe("trace-b");
    expect(openTrace).toHaveBeenCalledTimes(1);
    expect(openTrace).toHaveBeenCalledWith("trace-b", "b", "conversations", undefined);
  });
});

function summary(sessionId: string): SessionSummary {
  return {
    session_id: sessionId,
    agent_id: "router",
    display_title: sessionId,
    revision: 1,
    status: "active",
    created_at: "2026-08-18T12:00:00Z",
    updated_at: "2026-08-18T12:00:00Z",
  };
}

function turn(traceId: string): TurnSummary {
  return {
    turn_number: 1,
    committed_at: "2026-08-18T12:00:00Z",
    input: { role: "user", content: "Question" },
    response: { role: "assistant", content: "Answer" },
    routing_outcome: "answer",
    trace_id: traceId,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}
