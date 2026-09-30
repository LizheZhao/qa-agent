// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppErrorBoundary } from "./AppErrorBoundary";

afterEach(cleanup);

describe("AppErrorBoundary", () => {
  it("keeps a render failure inside the diagnostic UI", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);

    render(
      <AppErrorBoundary>
        <BrokenView />
      </AppErrorBoundary>,
    );

    expect(screen.getByRole("alert").textContent).toContain("This view could not be rendered.");
    expect(screen.getByRole("button", { name: "Reload diagnostics" })).toBeTruthy();
    consoleError.mockRestore();
  });
});

function BrokenView(): never {
  throw new Error("render failure");
}
