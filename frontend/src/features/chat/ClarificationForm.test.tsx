// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PendingClarification } from "../../api/types";
import { ClarificationForm } from "./ClarificationForm";

afterEach(cleanup);

describe("ClarificationForm", () => {
  it("submits one selected option for a single-choice question", async () => {
    const onRespond = vi.fn();
    render(
      <ClarificationForm
        clarification={clarification("single")}
        submitting={false}
        onRespond={onRespond}
      />,
    );

    await userEvent.click(screen.getByRole("radio", { name: "United States" }));
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(onRespond).toHaveBeenCalledWith({ form: "options", option_ids: ["us"] });
  });

  it("supports multiple choices and free text as mutually exclusive answers", async () => {
    const onRespond = vi.fn();
    render(
      <ClarificationForm
        clarification={clarification("multiple")}
        submitting={false}
        onRespond={onRespond}
      />,
    );

    await userEvent.click(screen.getByRole("checkbox", { name: "United States" }));
    await userEvent.click(screen.getByRole("checkbox", { name: /Europe/ }));
    await userEvent.type(screen.getByRole("textbox", { name: "Clarification answer" }), "Canada");
    expect(
      (screen.getByRole("checkbox", { name: "United States" }) as HTMLInputElement).checked,
    ).toBe(false);
    expect((screen.getByRole("checkbox", { name: /Europe/ }) as HTMLInputElement).checked).toBe(
      false,
    );

    await userEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(onRespond).toHaveBeenCalledWith({ form: "free_text", text: "Canada" });
  });

  it("offers cancellation and disables the fieldset while submitting", async () => {
    const onRespond = vi.fn();
    const { rerender } = render(
      <ClarificationForm
        clarification={clarification("single")}
        submitting={false}
        onRespond={onRespond}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "Cancel request" }));
    expect(onRespond).toHaveBeenCalledWith({ form: "cancel" });

    rerender(
      <ClarificationForm
        clarification={clarification("single")}
        submitting
        onRespond={onRespond}
      />,
    );
    expect(screen.getByRole("group", { name: "Which market?" }).hasAttribute("disabled")).toBe(
      true,
    );
    fireEvent.submit(screen.getByRole("button", { name: "Submitting…" }).closest("form")!);
    expect(onRespond).toHaveBeenCalledTimes(1);
  });
});

function clarification(selectionMode: string): PendingClarification {
  return {
    session_id: "session-1",
    trace_id: "trace-1",
    clarification_id: "clarification-1",
    question: "Which market?",
    reason_code: "ambiguous_market",
    selection_mode: selectionMode,
    options: [
      { option_id: "us", label: "United States", detail: null },
      { option_id: "eu", label: "Europe", detail: "Includes the United Kingdom" },
    ],
    free_text_allowed: true,
    cancel_allowed: true,
    expires_at: "2026-09-17T12:00:00Z",
  };
}
