import { type FormEvent, useCallback, useState } from "react";

import type { TracedChatError } from "../../api/types";
import { errorMessage } from "../../shared/presentation";
import { type ChatResult, chatApi, parseTracedError, tracedFailureMessage } from "./api";

export function useChatComposer({
  sessionId,
  onSuccess,
  onTracedFailure,
  onError,
}: {
  sessionId: string | null;
  onSuccess: (response: ChatResult) => Promise<void>;
  onTracedFailure: (error: TracedChatError) => Promise<void>;
  onError: (message: string) => void;
}) {
  const [message, setMessage] = useState("");
  const [optimisticMessage, setOptimisticMessage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const clearOptimisticMessage = useCallback(() => setOptimisticMessage(null), []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const value = message.trim();
    if (!value || submitting) return;
    setSubmitting(true);
    setOptimisticMessage(value);
    setMessage("");
    onError("");
    try {
      const response = await chatApi.send(value, sessionId ?? undefined);
      await onSuccess(response);
      if (response.kind === "completed") setOptimisticMessage(null);
    } catch (caught) {
      setOptimisticMessage(null);
      setMessage(value);
      const traced = parseTracedError(caught);
      if (traced) {
        await onTracedFailure(traced);
        onError(tracedFailureMessage(traced));
      } else {
        onError(errorMessage(caught));
      }
    } finally {
      setSubmitting(false);
    }
  }

  return {
    message,
    setMessage,
    optimisticMessage,
    clearOptimisticMessage,
    submitting,
    submit,
  };
}
