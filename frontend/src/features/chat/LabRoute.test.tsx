// @vitest-environment jsdom

import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LabRoute } from "./LabRoute";

const sessionWorkspace = vi.hoisted(() => ({
  loadSessions: vi.fn(async () => []),
  openLabSession: vi.fn(async () => true),
  clearSelection: vi.fn(),
  loadOlderTurns: vi.fn(async () => undefined),
  selectedSession: null,
  sortedTurns: [],
  turnCursor: null,
  loading: false,
}));
const api = vi.hoisted(() => ({
  send: vi.fn(),
  history: vi.fn(),
  respond: vi.fn(),
}));

vi.mock("../sessions/useSessionWorkspace", () => ({
  useSessionWorkspace: () => sessionWorkspace,
}));
vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return { ...original, chatApi: { ...original.chatApi, ...api } };
});

afterEach(cleanup);
beforeEach(() => {
  vi.clearAllMocks();
  HTMLElement.prototype.scrollTo = vi.fn();
  window.history.replaceState(null, "", "/diagnostics/lab?session=session-1");
  api.history.mockResolvedValue({
    session_id: "session-1",
    storage: "mongodb",
    durable: true,
    messages: [],
    pending_clarification: {
      session_id: "session-1",
      trace_id: "trace-1",
      clarification_id: "clarification-1",
      question: "Which market?",
      reason_code: "ambiguous_market",
      selection_mode: "single",
      options: [{ option_id: "us", label: "United States", detail: null }],
      free_text_allowed: true,
      cancel_allowed: true,
      expires_at: "2026-09-17T12:00:00Z",
    },
  });
});

describe("LabRoute", () => {
  it("restores a first-turn clarification without requesting a missing committed session", async () => {
    render(<LabRoute />);

    expect(await screen.findByRole("group", { name: "Which market?" })).toBeTruthy();
    await waitFor(() => expect(api.history).toHaveBeenCalledWith("session-1"));
    expect(sessionWorkspace.openLabSession).not.toHaveBeenCalled();
    expect(screen.queryByText("Session not found")).toBeNull();
  });

  it("moves a sent message into the transcript and shows work before a clarification arrives", async () => {
    window.history.replaceState(null, "", "/diagnostics/lab");
    const response = deferred<Awaited<ReturnType<typeof api.send>>>();
    api.send.mockReturnValue(response.promise);
    render(<LabRoute />);
    const input = await screen.findByRole("textbox", { name: "Diagnostic message" });

    await userEvent.type(input, "Show performance");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(screen.getByText("Show performance")).toBeTruthy();
    const running = screen.getByRole("button", { name: "Running…" });
    expect((running as HTMLButtonElement).disabled).toBe(true);
    expect(running.querySelector(".working-spinner")).toBeTruthy();

    await act(async () => {
      response.resolve({
        kind: "clarification",
        value: {
          session_id: "session-1",
          trace_id: "trace-1",
          clarification_id: "clarification-1",
          question: "Which market?",
          reason_code: "ambiguous_market",
          selection_mode: "single",
          options: [{ option_id: "us", label: "United States", detail: null }],
          free_text_allowed: true,
          cancel_allowed: true,
          expires_at: "2026-09-17T12:00:00Z",
        },
      });
      await response.promise;
    });

    expect(await screen.findByRole("group", { name: "Which market?" })).toBeTruthy();
    expect(sessionWorkspace.openLabSession).not.toHaveBeenCalled();
  });

  it("submits the selected clarification and opens the committed conversation", async () => {
    api.respond.mockResolvedValue({
      kind: "completed",
      value: {
        session_id: "session-1",
        trace_id: "trace-2",
        turn_number: 1,
        message: "Done",
        closed_by: "answered",
      },
    });
    render(<LabRoute />);
    await userEvent.click(
      await screen.findByRole("radio", { name: "United States" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => expect(sessionWorkspace.openLabSession).toHaveBeenCalledWith("session-1"));
    expect(api.respond).toHaveBeenCalledWith(
      expect.objectContaining({ clarification_id: "clarification-1" }),
      { form: "options", option_ids: ["us"] },
    );
  });
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}
