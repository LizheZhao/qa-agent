import { ApiError, graphDefinition as fetchGraphDefinition } from "../../api/client";
import type { GraphDefinition, SpanSummary, TraceDetail } from "../../api/types";
import { projectTrace, type ProjectedTrace } from "./projection";

const definitionCache = new Map<string, Promise<GraphDefinition | null>>();

async function graphDefinition(definitionId: string): Promise<GraphDefinition | null> {
  const cached = definitionCache.get(definitionId);
  if (cached) return cached;

  const request = fetchGraphDefinition(definitionId).catch((error: unknown) => {
    if (error instanceof ApiError && error.status === 404) return null;
    definitionCache.delete(definitionId);
    throw error;
  });
  definitionCache.set(definitionId, request);
  return request;
}

export async function hydrateProjection(
  trace: TraceDetail,
  spans: SpanSummary[],
): Promise<ProjectedTrace> {
  const definitionIds = [
    ...new Set(
      spans.flatMap((span) => (span.graph_definition_id ? [span.graph_definition_id] : [])),
    ),
  ];
  const loaded = await Promise.all(
    definitionIds.map(async (definitionId) => [definitionId, await graphDefinition(definitionId)] as const),
  );
  const definitions = new Map(
    loaded.filter(
      (item): item is readonly [string, GraphDefinition] => item[1] !== null,
    ),
  );
  return projectTrace({ trace, spans, definitions });
}
