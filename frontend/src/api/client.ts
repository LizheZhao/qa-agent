import type { GraphDefinition, TracedChatError } from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly payload: unknown;

  constructor(status: number, message: string, payload: unknown) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Accept", "application/json");
  const response = await fetch(path, {
    ...init,
    headers,
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, responseMessage(payload, response.status), payload);
  }
  return payload as T;
}

export function paged(path: string, limit: number, cursor?: string | null): string {
  const search = new URLSearchParams({ limit: String(limit) });
  if (cursor) search.set("cursor", cursor);
  return `${path}?${search.toString()}`;
}

// Graph definitions are read by both the agent catalog and trace hydration.
export function graphDefinition(definitionId: string): Promise<GraphDefinition> {
  return request<GraphDefinition>(
    `/api/diagnostics/graph-definitions/${encodeURIComponent(definitionId)}`,
  );
}

function responseMessage(payload: unknown, status: number): string {
  if (typeof payload !== "object" || payload === null) return `Request failed (${status})`;
  if ("detail" in payload) return String(payload.detail);
  if ("error" in payload) return String((payload as TracedChatError).error.message);
  return `Request failed (${status})`;
}
