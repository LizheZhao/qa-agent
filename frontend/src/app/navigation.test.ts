// @vitest-environment jsdom

import { describe, expect, it, vi } from "vitest";

import { listenForWorkspaceHistory, updateWorkspaceUrl } from "./navigation";

describe("workspace navigation", () => {
  it("writes canonical agent and conversation deep links without duplicates", () => {
    window.history.replaceState(null, "", "/diagnostics/lab");
    const push = vi.spyOn(window.history, "pushState");

    updateWorkspaceUrl("agents", "router:def");
    expect(window.location.pathname).toBe("/diagnostics/graphs/router%3Adef");
    updateWorkspaceUrl("agents", "router:def");
    expect(push).toHaveBeenCalledTimes(1);
    updateWorkspaceUrl("conversations", "session/1", "trace-1", "span-1");
    expect(`${window.location.pathname}${window.location.search}`).toBe(
      "/diagnostics/sessions/session%2F1?trace=trace-1&span=span-1",
    );
    push.mockRestore();
  });

  it("notifies the app when browser history moves", () => {
    const navigate = vi.fn();
    const stop = listenForWorkspaceHistory(navigate);

    window.dispatchEvent(new PopStateEvent("popstate"));
    expect(navigate).toHaveBeenCalledOnce();
    stop();
    window.dispatchEvent(new PopStateEvent("popstate"));
    expect(navigate).toHaveBeenCalledOnce();
  });
});
