import { describe, expect, it } from "vitest";

import { buildCanvasModel } from "./canvas-model";
import type { ProjectedGraph, ProjectedTrace } from "./projection";

function graph(id: string, parentGraphId: string | null = null): ProjectedGraph {
  return {
    id,
    spanId: id.replace("graph:", ""),
    parentGraphId,
    ownerSpanId: null,
    agentId: id,
    agentVersion: "1",
    definitionId: "definition",
    status: "completed",
    sequenceStarted: 1,
    compiledNodes: [],
    compiledEdges: [],
    runtimeNodes: [
      {
        id: `boundary:${id}:start`,
        graphId: id,
        spanId: null,
        definitionNodeId: "__start__",
        displayName: "START",
        kind: "start",
        status: "boundary",
        sequenceStarted: null,
        superstep: -1,
        iteration: 1,
        boundary: "start",
      },
      ...["a", "b"].map((name, index) => ({
        id: `runtime:${id}:${name}`,
        graphId: id,
        spanId: `${id}:${name}`,
        definitionNodeId: name,
        displayName: name,
        kind: "deterministic" as const,
        status: "completed" as const,
        sequenceStarted: index + 2,
        superstep: 1,
        iteration: 1,
        boundary: null,
      })),
      {
        id: `boundary:${id}:end`,
        graphId: id,
        spanId: null,
        definitionNodeId: "__end__",
        displayName: "END",
        kind: "end",
        status: "boundary",
        sequenceStarted: null,
        superstep: 2,
        iteration: 1,
        boundary: "end",
      },
    ],
    runtimeEdges: [],
    layers: [
      { superstep: -1, runtimeNodeIds: [`boundary:${id}:start`] },
      { superstep: 1, runtimeNodeIds: [`runtime:${id}:a`, `runtime:${id}:b`] },
      { superstep: 2, runtimeNodeIds: [`boundary:${id}:end`] },
    ],
  };
}

function projection(graphs: ProjectedGraph[]): ProjectedTrace {
  return {
    kind: "runtime",
    traceId: "trace",
    rootGraphIds: graphs.filter((item) => item.parentGraphId === null).map((item) => item.id),
    graphs,
    attachments: [],
    warnings: [],
  };
}

describe("buildCanvasModel", () => {
  it("places nodes in one superstep in the same column", () => {
    const result = buildCanvasModel(projection([graph("graph:root")]), "executed");
    const a = result.nodes.find((node) => node.id === "runtime:graph:root:a")!;
    const b = result.nodes.find((node) => node.id === "runtime:graph:root:b")!;
    expect(a.position.x).toBe(b.position.x);
    expect(a.position.y).not.toBe(b.position.y);
  });

  it("connects a delegation to the child graph START node", () => {
    const root = graph("graph:root");
    const child = graph("graph:child", root.id);
    const trace = projection([root, child]);
    trace.attachments.push({
      id: "attachment:handoff",
      spanId: "handoff",
      graphId: root.id,
      parentRuntimeNodeId: "runtime:graph:root:a",
      kind: "child_agent",
      status: "completed",
      agentId: "child",
      sequenceStarted: 3,
      childGraphIds: [child.id],
    });
    const result = buildCanvasModel(trace, "executed");
    expect(result.nodes).toContainEqual(
      expect.objectContaining({ id: "attachment:handoff", parentId: "graph:root" }),
    );
    expect(result.edges).toContainEqual(
      expect.objectContaining({
        source: "runtime:graph:root:a",
        target: "attachment:handoff",
      }),
    );
    expect(result.edges).toContainEqual(
      expect.objectContaining({
        source: "attachment:handoff",
        target: "boundary:graph:child:start",
      }),
    );
  });
});
