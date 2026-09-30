import { paged, request } from "../../api/client";
import type { SessionPage, SessionSummary, TurnPage } from "../../api/types";

export const sessionsApi = {
  sessions: (cursor?: string | null) =>
    request<SessionPage>(paged("/api/diagnostics/sessions", 10, cursor)),
  session: (sessionId: string) =>
    request<SessionSummary>(`/api/diagnostics/sessions/${encodeURIComponent(sessionId)}`),
  turns: (sessionId: string, cursor?: string | null) =>
    request<TurnPage>(
      paged(`/api/diagnostics/sessions/${encodeURIComponent(sessionId)}/turns`, 25, cursor),
    ),
};
