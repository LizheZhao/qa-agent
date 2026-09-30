// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { updateWorkspaceUrl } from "./navigation";

vi.mock("../features/agents/AgentRoute", () => ({
  AgentRoute: () => <main>Agent route</main>,
  DeploymentIndicator: () => <div>Local deployment</div>,
}));
vi.mock("../features/chat/LabRoute", () => ({ LabRoute: () => <main>Lab route</main> }));
vi.mock("../features/runs/RunsRoute", () => ({
  RunsRoute: () => (
    <main>
      <span>Runs route</span>
      <input aria-label="Runs local state" defaultValue="initial" />
    </main>
  ),
}));
vi.mock("../features/sessions/ConversationsRoute", () => ({
  ConversationsRoute: () => <main>Conversations route</main>,
}));

import { App } from "./App";

afterEach(cleanup);

describe("App", () => {
  it("owns top-level routes and restores the last deep link for each tab", () => {
    window.history.replaceState(null, "", "/diagnostics/lab");
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Runs" }));
    expect(screen.getByText("Runs route")).toBeTruthy();
    const localState = screen.getByRole("textbox", { name: "Runs local state" });
    fireEvent.change(localState, { target: { value: "preserved" } });
    updateWorkspaceUrl("runs", null, "trace-1");
    fireEvent.click(screen.getByRole("button", { name: "Lab" }));
    expect(screen.getByText("Lab route")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Runs" }));

    expect(`${window.location.pathname}${window.location.search}`).toBe(
      "/diagnostics/runs?trace=trace-1",
    );
    expect((screen.getByRole("textbox", { name: "Runs local state" }) as HTMLInputElement).value)
      .toBe("preserved");
  });
});
