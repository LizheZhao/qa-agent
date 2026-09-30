import { describe, expect, it } from "vitest";

import type { GraphDefinition, SpanSummary, TraceDetail } from "../../api/types";
import { projectDefinition, projectTrace } from "./projection";

const timestamp = "2026-08-17T00:00:00Z";

function trace(overrides: Partial<TraceDetail> = {}): TraceDetail {
  return {
    trace_id: "trace-1",
    session_id: "session-1",
    root_span_id: "turn-1",
    attempted_turn_number: 1,
    committed_turn_number: 1,
    request_origin: "diagnostic_ui",
    status: "completed",
    observability_status: "complete",
    started_at: timestamp,
    completed_at: timestamp,
    sequence_started: 1,
    sequence_completed: 20,
    application_version: "test",
    core_version: "test",
    deployment_version: "test",
    agent_versions: {},
    ...overrides,
  };
}

function span(
  spanId: string,
  kind: SpanSummary["kind"],
  parentSpanId: string | null,
  sequence: number,
  overrides: Partial<SpanSummary> = {},
): SpanSummary {
  return {
    span_id: spanId,
    parent_span_id: parentSpanId,
    kind,
    agent_id: "router",
    agent_version: "1",
    status: "completed",
    started_at: timestamp,
    completed_at: timestamp,
    duration_ms: 1,
    sequence_started: sequence,
    sequence_completed: sequence + 1,
    superstep: 0,
    iteration: 1,
    caused_by_span_ids: [],
    captures: { input: false, output: false, resolved_messages: false, state_delta: false },
    ...overrides,
  };
}

function definition(
  definitionId: string,
  nodeIds: string[],
  edges: Array<[string, string]> = [],
): GraphDefinition {
  return {
    definition_id: definitionId,
    definition_hash: definitionId,
    agent_id: definitionId.split(":")[0] ?? "agent",
    agent_version: "1",
    entrypoint: nodeIds[0] ?? "__start__",
    schema_version: 4,
    created_at: timestamp,
    nodes: [
      {
        node_id: "__start__",
        display_name: "START",
        kind: "start",
        owner_path: [],
      },
      ...nodeIds.map((nodeId) => ({
        node_id: nodeId,
        display_name: nodeId,
        kind: "deterministic" as const,
        owner_path: [],
      })),
      {
        node_id: "__end__",
        display_name: "END",
        kind: "end",
        owner_path: [],
      },
    ],
    edges: edges.map(([source, target]) => ({
      source_node_id: source,
      target_node_id: target,
      kind: "direct",
    })),
  };
}

function project(
  spans: SpanSummary[],
  definitions: GraphDefinition[] = [],
  traceOverrides: Partial<TraceDetail> = {},
) {
  return projectTrace({
    trace: trace(traceOverrides),
    spans,
    definitions: new Map(definitions.map((item) => [item.definition_id, item])),
  });
}

describe("projectTrace", () => {
  it("preserves sequential causal order and adds one START and END", () => {
    const graphDefinition = definition("router:1:def", ["route", "answer"]);
    const result = project(
      [
        span("turn-1", "turn", null, 1),
        span("graph-1", "graph", "turn-1", 2, {
          graph_definition_id: graphDefinition.definition_id,
        }),
        span("node-route", "node", "graph-1", 3, {
          definition_node_id: "route",
          superstep: 1,
        }),
        span("node-answer", "node", "graph-1", 5, {
          definition_node_id: "answer",
          superstep: 2,
          caused_by_span_ids: ["node-route"],
        }),
      ],
      [graphDefinition],
    );

    const graph = result.graphs[0]!;
    expect(graph.runtimeNodes.map((node) => node.displayName)).toEqual([
      "START",
      "route",
      "answer",
      "END",
    ]);
    expect(graph.runtimeEdges.map((edge) => [edge.source, edge.target])).toEqual([
      ["runtime:node-route", "runtime:node-answer"],
      ["boundary:graph:graph-1:start", "runtime:node-route"],
      ["runtime:node-answer", "boundary:graph:graph-1:end"],
    ]);
  });

  it("keeps concurrent nodes in the same superstep layer", () => {
    const result = project([
      span("graph-1", "graph", "turn-1", 2),
      span("node-a", "node", "graph-1", 3, { definition_node_id: "a", superstep: 1 }),
      span("node-b", "node", "graph-1", 4, { definition_node_id: "b", superstep: 1 }),
    ]);

    expect(result.graphs[0]?.layers.find((layer) => layer.superstep === 1)?.runtimeNodeIds).toEqual([
      "runtime:node-a",
      "runtime:node-b",
    ]);
  });

  it("keeps loop iterations as distinct runtime instances of one compiled node", () => {
    const graphDefinition = definition("diagnostic:1:def", ["refine"]);
    const result = project(
      [
        span("graph-1", "graph", "turn-1", 2, {
          graph_definition_id: graphDefinition.definition_id,
        }),
        span("refine-1", "node", "graph-1", 3, {
          definition_node_id: "refine",
          superstep: 1,
          iteration: 1,
        }),
        span("refine-2", "node", "graph-1", 5, {
          definition_node_id: "refine",
          superstep: 2,
          iteration: 2,
          caused_by_span_ids: ["refine-1"],
        }),
      ],
      [graphDefinition],
    );

    const graph = result.graphs[0]!;
    expect(graph.runtimeNodes.filter((node) => node.definitionNodeId === "refine")).toHaveLength(2);
    expect(graph.compiledNodes.find((node) => node.definitionNodeId === "refine")?.runtimeNodeIds).toEqual([
      "runtime:refine-1",
      "runtime:refine-2",
    ]);
  });

  it("links delegated child graphs without merging their node layers", () => {
    const result = project([
      span("router-graph", "graph", "turn-1", 2),
      span("delegate", "node", "router-graph", 3, {
        definition_node_id: "delegate",
        superstep: 1,
      }),
      span("handoff", "child_agent", "delegate", 4),
      span("diagnostic-graph", "graph", "handoff", 5, {
        agent_id: "diagnostic",
      }),
      span("query", "node", "diagnostic-graph", 6, {
        agent_id: "diagnostic",
        definition_node_id: "query",
        superstep: 1,
      }),
    ]);

    expect(result.rootGraphIds).toEqual(["graph:router-graph"]);
    expect(result.graphs.find((graph) => graph.spanId === "diagnostic-graph")?.parentGraphId).toBe(
      "graph:router-graph",
    );
    expect(result.attachments.find((item) => item.spanId === "handoff")).toMatchObject({
      parentRuntimeNodeId: "runtime:delegate",
      childGraphIds: ["graph:diagnostic-graph"],
    });
  });

  it.each(["failed", "cancelled"] as const)("preserves %s runtime status", (status) => {
    const result = project([
      span("graph-1", "graph", "turn-1", 2, { status }),
      span("node-1", "node", "graph-1", 3, {
        definition_node_id: "work",
        status,
      }),
    ]);
    expect(result.graphs[0]?.status).toBe(status);
    expect(result.graphs[0]?.runtimeNodes.find((node) => node.spanId === "node-1")?.status).toBe(
      status,
    );
  });

  it("renders runtime evidence and warns when its immutable definition is missing", () => {
    const result = project([
      span("graph-1", "graph", "turn-1", 2, { graph_definition_id: "missing:def" }),
      span("node-1", "node", "graph-1", 3, { definition_node_id: "known-at-runtime" }),
    ]);

    expect(result.graphs[0]?.runtimeNodes.some((node) => node.spanId === "node-1")).toBe(true);
    expect(result.warnings).toContainEqual(
      expect.objectContaining({ code: "missing_definition", definitionId: "missing:def" }),
    );
  });

  it("marks runtime nodes without definition metadata as unknown", () => {
    const result = project([
      span("graph-1", "graph", "turn-1", 2),
      span("node-1", "node", "graph-1", 3, { definition_node_id: "mystery" }),
    ]);

    expect(result.graphs[0]?.runtimeNodes.find((node) => node.spanId === "node-1")?.kind).toBe(
      "unknown",
    );
  });
});

describe("projectDefinition", () => {
  it("creates a static compiled catalog projection without runtime spans", () => {
    const graphDefinition = definition("router:1:def", ["decide", "respond"], [
      ["__start__", "decide"],
      ["decide", "respond"],
      ["respond", "__end__"],
    ]);

    const result = projectDefinition(graphDefinition);

    expect(result.graphs[0]?.runtimeNodes).toEqual([]);
    expect(result.graphs[0]?.compiledNodes).toHaveLength(4);
    expect(result.kind).toBe("compiled");
    expect(result.graphs[0]?.status).toBeNull();
    expect(result.graphs[0]?.sequenceStarted).toBeNull();
    expect(result.graphs[0]?.compiledNodes.every((node) => node.visited === null)).toBe(true);
    expect(result.graphs[0]?.compiledEdges).toHaveLength(3);
  });
});
