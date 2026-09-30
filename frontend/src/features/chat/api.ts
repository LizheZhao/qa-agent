import { ApiError, request } from "../../api/client";
import type {
  ChatResponse,
  ContinuationBody,
  ContinuationCompleted,
  PendingClarification,
  SessionHistoryResponse,
  TracedChatError,
} from "../../api/types";
import { shortId } from "../../shared/presentation";

export type ChatResult =
  | { kind: "completed"; value: ChatResponse }
  | { kind: "clarification"; value: PendingClarification };

export type ContinuationResult =
  | { kind: "completed"; value: ContinuationCompleted }
  | { kind: "clarification"; value: PendingClarification };

function isPendingClarification(
  value: ChatResponse | ContinuationCompleted | PendingClarification,
): value is PendingClarification {
  return "clarification_id" in value;
}

export const chatApi = {
  async send(message: string, sessionId?: string): Promise<ChatResult> {
    const response = await request<ChatResponse | PendingClarification>("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Orchestration-Origin": "diagnostic_ui",
      },
      body: JSON.stringify({ message, session_id: sessionId ?? null }),
    });
    return isPendingClarification(response)
      ? { kind: "clarification", value: response }
      : { kind: "completed", value: response };
  },
  history: (sessionId: string) =>
    request<SessionHistoryResponse>(
      `/api/sessions/${encodeURIComponent(sessionId)}/history`,
    ),
  async respond(
    clarification: PendingClarification,
    response: ContinuationBody["response"],
  ): Promise<ContinuationResult> {
    const result = await request<ContinuationCompleted | PendingClarification>(
      `/api/sessions/${encodeURIComponent(clarification.session_id)}/clarifications/${encodeURIComponent(clarification.clarification_id)}/responses`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ submission_id: crypto.randomUUID(), response }),
      },
    );
    return isPendingClarification(result)
      ? { kind: "clarification", value: result }
      : { kind: "completed", value: result };
  },
};

export function parseTracedError(error: unknown): TracedChatError | null {
  if (!(error instanceof ApiError) || typeof error.payload !== "object" || !error.payload) {
    return null;
  }
  const payload = error.payload as Partial<TracedChatError>;
  return payload.error && payload.session_id && payload.trace_id
    ? (payload as TracedChatError)
    : null;
}

export function tracedFailureMessage(error: TracedChatError): string {
  return `${error.error.message} · run ${shortId(error.trace_id)}`;
}
