// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { ProjectedTrace } from "./projection";
import { TraceProjectionList } from "./TraceProjectionList";

afterEach(cleanup);

describe("TraceProjectionList", () => {
  it("labels END as a boundary instead of another numbered step", () => {
    render(
      <TraceProjectionList projection={projection} selectedSpanId={null} onSelect={() => undefined} />,
    );

    expect(screen.getByText("Start")).toBeTruthy();
    expect(screen.getByText("End")).toBeTruthy();
    expect(screen.queryByText("Step 2")).toBeNull();
  });
});

const projection: ProjectedTrace = {
  kind: "runtime",
  traceId: "trace",
  rootGraphIds: ["graph"],
  attachments: [],
  warnings: [],
  graphs: [
    {
      id: "graph",
      spanId: "graph-span",
      parentGraphId: null,
      ownerSpanId: null,
      agentId: "router",
      agentVersion: "1",
      definitionId: null,
      status: "completed",
      sequenceStarted: 1,
      compiledNodes: [],
      compiledEdges: [],
      runtimeEdges: [],
      runtimeNodes: [
        {
          id: "start",
          graphId: "graph",
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
        {
          id: "end",
          graphId: "graph",
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
      layers: [
        { superstep: -1, runtimeNodeIds: ["start"] },
        { superstep: 2, runtimeNodeIds: ["end"] },
      ],
    },
  ],
};
