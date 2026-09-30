import type { FormEvent } from "react";

import { shortId } from "../../shared/presentation";
import "./chat.css";

export function ChatComposer({
  sessionId,
  message,
  submitting,
  onMessageChange,
  onSubmit,
}: {
  sessionId: string | null;
  message: string;
  submitting: boolean;
  onMessageChange: (message: string) => void;
  onSubmit: (event: FormEvent) => void;
}) {
  return (
    <form className="composer" onSubmit={onSubmit}>
      <textarea
        aria-label="Diagnostic message"
        onChange={(event) => onMessageChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            event.currentTarget.form?.requestSubmit();
          }
        }}
        placeholder={
          sessionId
            ? "Continue this diagnostic conversation…"
            : "Send a message through the router…"
        }
        rows={2}
        value={message}
      />
      <div className="composer-footer">
        <span>
          {sessionId ? `Continuing ${shortId(sessionId)}` : "Starts a new durable conversation"}
        </span>
        <button disabled={!message.trim() || submitting} type="submit">
          {submitting && <span className="working-spinner" aria-hidden="true" />}
          {submitting ? "Running…" : "Send"}
        </button>
      </div>
    </form>
  );
}
