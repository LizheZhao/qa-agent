import { request } from "../../api/client";
import type { SpanDetail, SpanSummary, TraceDetail } from "../../api/types";

export const tracesApi = {
  trace: (traceId: string) =>
    request<TraceDetail>(`/api/diagnostics/traces/${encodeURIComponent(traceId)}`),
  spans: (traceId: string) =>
    request<SpanSummary[]>(`/api/diagnostics/traces/${encodeURIComponent(traceId)}/spans`),
  span: (traceId: string, spanId: string) =>
    request<SpanDetail>(
      `/api/diagnostics/traces/${encodeURIComponent(traceId)}/spans/${encodeURIComponent(spanId)}`,
    ),
};
