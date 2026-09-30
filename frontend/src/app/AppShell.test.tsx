// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppShell } from "./AppShell";

afterEach(cleanup);

function renderShell(activeView: "lab" | "agents" | "conversations" | "runs") {
  return render(
    <AppShell
      activeView={activeView}
      deploymentStatus={<span>Local</span>}
      onSelectView={vi.fn()}
    >
      <aside>Browser</aside>
      <main>Main</main>
      <aside>Inspector</aside>
    </AppShell>,
  );
}

describe("AppShell pane resizing", () => {
  it("resizes a pane and resets widths when the active tab changes", () => {
    const { container, rerender } = renderShell("agents");
    const workspace = container.querySelector<HTMLElement>(".workspace")!;
    vi.spyOn(workspace, "getBoundingClientRect").mockReturnValue({
      bottom: 900,
      height: 900,
      left: 0,
      right: 1800,
      top: 0,
      width: 1800,
      x: 0,
      y: 0,
      toJSON: () => undefined,
    });

    const browserHandle = screen.getByRole("separator", { name: "Resize browser pane" });
    fireEvent.pointerDown(browserHandle, { pointerId: 1 });
    fireEvent.pointerMove(browserHandle, { clientX: 417, pointerId: 1 });
    fireEvent.pointerUp(browserHandle, { pointerId: 1 });
    expect(workspace.style.getPropertyValue("--browser-pane-width")).toBe("400px");

    rerender(
      <AppShell
        activeView="conversations"
        deploymentStatus={<span>Local</span>}
        onSelectView={vi.fn()}
      >
        <aside>Browser</aside>
        <main>Main</main>
        <aside>Inspector</aside>
      </AppShell>,
    );
    expect(workspace.style.getPropertyValue("--browser-pane-width")).toBe("248px");

    fireEvent.keyDown(
      screen.getByRole("separator", { name: "Resize inspector pane" }),
      { key: "ArrowLeft" },
    );
    expect(workspace.style.getPropertyValue("--inspector-pane-width")).toBe("346px");
  });

  it("does not render pane controls in Lab", () => {
    renderShell("lab");

    expect(screen.queryAllByRole("separator")).toHaveLength(0);
  });
});
