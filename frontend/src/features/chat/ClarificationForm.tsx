import { type FormEvent, useState } from "react";

import type { ContinuationBody, PendingClarification } from "../../api/types";
import "./clarification.css";

type ClarificationResponse = ContinuationBody["response"];

export function ClarificationForm({
  clarification,
  submitting,
  onRespond,
}: {
  clarification: PendingClarification;
  submitting: boolean;
  onRespond: (response: ClarificationResponse) => void;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const [freeText, setFreeText] = useState("");
  const single = clarification.selection_mode === "single";

  function select(optionId: string) {
    setFreeText("");
    setSelected((current) =>
      single
        ? [optionId]
        : current.includes(optionId)
          ? current.filter((id) => id !== optionId)
          : [...current, optionId],
    );
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    const text = freeText.trim();
    if (text) onRespond({ form: "free_text", text });
    else if (selected.length) onRespond({ form: "options", option_ids: selected });
  }

  return (
    <form className="clarification" onSubmit={submit}>
      <fieldset disabled={submitting}>
        <legend>{clarification.question}</legend>
        {clarification.options.length > 0 && (
          <div className="clarification-options">
            {clarification.options.map((option) => (
              <label className="clarification-option" key={option.option_id}>
                <input
                  checked={selected.includes(option.option_id)}
                  name={`clarification-${clarification.clarification_id}`}
                  type={single ? "radio" : "checkbox"}
                  value={option.option_id}
                  onChange={() => select(option.option_id)}
                />
                <span>
                  {option.label}
                  {option.detail && <small>{option.detail}</small>}
                </span>
              </label>
            ))}
          </div>
        )}
        {clarification.free_text_allowed && (
          <label className="clarification-free-text">
            <span>Or type an answer</span>
            <textarea
              aria-label="Clarification answer"
              rows={2}
              value={freeText}
              onChange={(event) => {
                setFreeText(event.target.value);
                if (event.target.value) setSelected([]);
              }}
            />
          </label>
        )}
        <div className="clarification-actions">
          {clarification.cancel_allowed && (
            <button
              className="clarification-cancel"
              type="button"
              onClick={() => onRespond({ form: "cancel" })}
            >
              Cancel request
            </button>
          )}
          <button
            className="clarification-submit"
            disabled={!freeText.trim() && selected.length === 0}
            type="submit"
          >
            {submitting && <span className="working-spinner" aria-hidden="true" />}
            {submitting ? "Submitting…" : "Continue"}
          </button>
        </div>
      </fieldset>
    </form>
  );
}
