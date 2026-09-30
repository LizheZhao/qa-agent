// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { WorkspacePanel } from "./WorkspacePanel";

afterEach(cleanup);

describe("WorkspacePanel", () => {
  it("keeps content and composer in stable regions when no error is rendered", () => {
    const { container } = render(
      <WorkspacePanel
        eyebrow="Lab"
        title="Long conversation"
        error={null}
        footer={<form aria-label="Composer">Composer</form>}
      >
        <div>{Array.from({ length: 50 }, (_, index) => <p key={index}>Turn {index}</p>)}</div>
      </WorkspacePanel>,
    );

    expect(container.querySelector(".workspace-panel__top")).toBeTruthy();
    expect(container.querySelector(".workspace-panel__content-region")).toBeTruthy();
    expect(
      container
        .querySelector(".workspace-panel__footer")
        ?.contains(screen.getByRole("form", { name: "Composer" })),
    ).toBe(true);
  });

  it("keeps the optional error inside the top region", () => {
    const { container } = render(
      <WorkspacePanel eyebrow="Lab" title="Conversation" error="Request failed">
        Transcript
      </WorkspacePanel>,
    );

    expect(container.querySelector(".workspace-panel__top")?.textContent).toContain(
      "Request failed",
    );
    expect(container.querySelector(".workspace-panel__footer")).toBeNull();
  });
});
