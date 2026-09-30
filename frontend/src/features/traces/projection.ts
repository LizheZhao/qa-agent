import type {
  ExecutionStatus,
  GraphDefinition,
  GraphEdgeDefinition,
  GraphNodeDefinition,
  SpanSummary,
  TraceDetail,
} from "../../api/types";

export type ProjectionWarningCode =
  | "missing_definition"
  | "missing_graph_parent"
  | "unknown_definition_node";

export interface ProjectionWarning {
  code: ProjectionWarningCode;
  message: string;
  spanId?: string;
  definitionId?: string;
}

export interface ProjectedCompiledNode {
  id: string;
  graphId: string;
  definitionNodeId: string;
  displayName: string;
  kind: GraphNodeDefinition["kind"];
  visited: boolean | null;
  runtimeNodeIds: string[];
  promptTemplateId?: string;
  promptTemplateText?: string;
  destinationAgentId?: string;
}

export interface ProjectedRuntimeNode {
  id: string;
  graphId: string;
  spanId: string | null;
  definitionNodeId: string;
  displayName: string;
  kind: GraphNodeDefinition["kind"] | "unknown";
  status: ExecutionStatus | "boundary";
  sequenceStarted: number | null;
  superstep: number;
  iteration: number;
  boundary: "start" | "end" | null;
}

export interface ProjectedEdge {
  id: string;
  graphId: string;
  source: string;
  target: string;
  kind: "causal" | "boundary" | "compiled";
  label?: string;
  conditional?: boolean;
}

export interface ProjectedAttachment {
  id: string;
  spanId: string;
  graphId: string | null;
  parentRuntimeNodeId: string | null;
  kind: SpanSummary["kind"];
  status: ExecutionStatus;
  agentId: string;
  sequenceStarted: number;
  childGraphIds: string[];
}

export interface ProjectedLayer {
  superstep: number;
  runtimeNodeIds: string[];
}

export interface ProjectedGraph {
  id: string;
  spanId: string;
  parentGraphId: string | null;
  ownerSpanId: string | null;
  agentId: string;
  agentVersion: string;
  definitionId: string | null;
  status: ExecutionStatus | null;
  sequenceStarted: number | null;
  compiledNodes: ProjectedCompiledNode[];
  compiledEdges: ProjectedEdge[];
  runtimeNodes: ProjectedRuntimeNode[];
  runtimeEdges: ProjectedEdge[];
  layers: ProjectedLayer[];
}

export interface ProjectedTrace {
  kind: "runtime" | "compiled";
  traceId: string;
  rootGraphIds: string[];
  graphs: ProjectedGraph[];
  attachments: ProjectedAttachment[];
  warnings: ProjectionWarning[];
}

export interface TraceBundle {
  trace: TraceDetail;
  spans: SpanSummary[];
  definitions: Map<string, GraphDefinition>;
}

export function projectTrace(bundle: TraceBundle): ProjectedTrace {
  const { trace, spans, definitions } = bundle;
  const warnings: ProjectionWarning[] = [];
  const spansById = new Map(spans.map((span) => [span.span_id, span]));
  const graphSpans = spans
    .filter((span) => span.kind === "graph")
    .sort((left, right) => left.sequence_started - right.sequence_started);
  const graphIdBySpanId = new Map(graphSpans.map((span) => [span.span_id, graphId(span.span_id)]));

  const nearestAncestor = (span: SpanSummary, kinds: Set<SpanSummary["kind"]>) => {
    let parentId = span.parent_span_id;
    const visited = new Set<string>();
    while (parentId && !visited.has(parentId)) {
      visited.add(parentId);
      const parent = spansById.get(parentId);
      if (!parent) return null;
      if (kinds.has(parent.kind)) return parent;
      parentId = parent.parent_span_id;
    }
    return null;
  };

  const graphs = graphSpans.map((graphSpan) => {
    const definition = graphSpan.graph_definition_id
      ? definitions.get(graphSpan.graph_definition_id)
      : undefined;
    if (graphSpan.graph_definition_id && !definition) {
      warnings.push({
        code: "missing_definition",
        message: `Graph definition ${graphSpan.graph_definition_id} is unavailable.`,
        spanId: graphSpan.span_id,
        definitionId: graphSpan.graph_definition_id,
      });
    }

    const parentGraphSpan = nearestAncestor(graphSpan, new Set(["graph"]));
    const currentGraphId = graphId(graphSpan.span_id);
    const runtimeSpans = spans
      .filter(
        (span) =>
          span.kind === "node" &&
          nearestAncestor(span, new Set(["graph"]))?.span_id === graphSpan.span_id,
      )
      .sort((left, right) => left.sequence_started - right.sequence_started);
    const definitionNodes = new Map(
      (definition?.nodes ?? []).map((node) => [node.node_id, node]),
    );
    const runtimeNodes = runtimeSpans.map((span) => {
      const nodeDefinition = span.definition_node_id
        ? definitionNodes.get(span.definition_node_id)
        : undefined;
      if (span.definition_node_id && definition && !nodeDefinition) {
        warnings.push({
          code: "unknown_definition_node",
          message: `Runtime node ${span.definition_node_id} is absent from its graph definition.`,
          spanId: span.span_id,
          definitionId: definition.definition_id,
        });
      }
      return runtimeNode(currentGraphId, span, nodeDefinition);
    });
    const runtimeIdsBySpan = new Map(
      runtimeNodes
        .filter((node): node is ProjectedRuntimeNode & { spanId: string } => node.spanId !== null)
        .map((node) => [node.spanId, node.id]),
    );
    const start = boundaryNode(currentGraphId, "start", runtimeNodes);
    const end = boundaryNode(currentGraphId, "end", runtimeNodes);
    const causalEdges: ProjectedEdge[] = [];
    for (const runtimeSpan of runtimeSpans) {
      const target = runtimeIdsBySpan.get(runtimeSpan.span_id);
      if (!target) continue;
      for (const causeId of runtimeSpan.caused_by_span_ids) {
        const source = runtimeIdsBySpan.get(causeId);
        if (source) {
          causalEdges.push({
            id: `causal:${source}:${target}`,
            graphId: currentGraphId,
            source,
            target,
            kind: "causal",
          });
        }
      }
    }
    const incoming = new Set(causalEdges.map((edge) => edge.target));
    const outgoing = new Set(causalEdges.map((edge) => edge.source));
    const boundaryEdges: ProjectedEdge[] = [];
    if (runtimeNodes.length === 0) {
      boundaryEdges.push(boundaryEdge(currentGraphId, start.id, end.id));
    } else {
      for (const node of runtimeNodes) {
        if (!incoming.has(node.id)) {
          boundaryEdges.push(boundaryEdge(currentGraphId, start.id, node.id));
        }
        if (!outgoing.has(node.id)) {
          boundaryEdges.push(boundaryEdge(currentGraphId, node.id, end.id));
        }
      }
    }

    const compiledNodes = (definition?.nodes ?? []).map((node) => {
      const matching = runtimeNodes.filter(
        (runtime) => runtime.definitionNodeId === node.node_id,
      );
      return compiledNode(currentGraphId, node, matching);
    });
    const compiledEdges = (definition?.edges ?? []).map((edge, index) =>
      compiledEdge(currentGraphId, edge, index),
    );
    const withBoundaries = [start, ...runtimeNodes, end];
    const layers = [...new Set(withBoundaries.map((node) => node.superstep))]
      .sort((left, right) => left - right)
      .map((superstep) => ({
        superstep,
        runtimeNodeIds: withBoundaries
          .filter((node) => node.superstep === superstep)
          .map((node) => node.id),
      }));

    return {
      id: currentGraphId,
      spanId: graphSpan.span_id,
      parentGraphId: parentGraphSpan
        ? (graphIdBySpanId.get(parentGraphSpan.span_id) ?? null)
        : null,
      ownerSpanId: graphSpan.parent_span_id ?? null,
      agentId: graphSpan.agent_id,
      agentVersion: graphSpan.agent_version,
      definitionId: graphSpan.graph_definition_id ?? null,
      status: graphSpan.status,
      sequenceStarted: graphSpan.sequence_started,
      compiledNodes,
      compiledEdges,
      runtimeNodes: withBoundaries,
      runtimeEdges: [...causalEdges, ...boundaryEdges],
      layers,
    } satisfies ProjectedGraph;
  });

  const attachments = spans
    .filter((span) => !new Set(["turn", "graph", "node"]).has(span.kind))
    .map((attachment) => {
      const parentNode = nearestAncestor(attachment, new Set(["node"]));
      const parentGraph = nearestAncestor(attachment, new Set(["graph"]));
      const childGraphIds = graphSpans
        .filter((candidate) => candidate.parent_span_id === attachment.span_id)
        .map((candidate) => graphIdBySpanId.get(candidate.span_id))
        .filter((value): value is string => value !== undefined);
      return {
        id: `attachment:${attachment.span_id}`,
        spanId: attachment.span_id,
        graphId: parentGraph ? (graphIdBySpanId.get(parentGraph.span_id) ?? null) : null,
        parentRuntimeNodeId: parentNode ? runtimeNodeId(parentNode.span_id) : null,
        kind: attachment.kind,
        status: attachment.status,
        agentId: attachment.agent_id,
        sequenceStarted: attachment.sequence_started,
        childGraphIds,
      } satisfies ProjectedAttachment;
    });

  for (const span of spans.filter((item) => item.kind === "node")) {
    if (!nearestAncestor(span, new Set(["graph"]))) {
      warnings.push({
        code: "missing_graph_parent",
        message: `Runtime node ${span.span_id} has no graph ancestor.`,
        spanId: span.span_id,
      });
    }
  }

  return {
    kind: "runtime",
    traceId: trace.trace_id,
    rootGraphIds: graphs.filter((graph) => graph.parentGraphId === null).map((graph) => graph.id),
    graphs,
    attachments,
    warnings,
  };
}

export function projectDefinition(definition: GraphDefinition): ProjectedTrace {
  const currentGraphId = `definition:${definition.definition_id}`;
  return {
    kind: "compiled",
    traceId: currentGraphId,
    rootGraphIds: [currentGraphId],
    graphs: [
      {
        id: currentGraphId,
        spanId: currentGraphId,
        parentGraphId: null,
        ownerSpanId: null,
        agentId: definition.agent_id,
        agentVersion: definition.agent_version,
        definitionId: definition.definition_id,
        status: null,
        sequenceStarted: null,
        compiledNodes: definition.nodes.map((node) =>
          compiledNode(currentGraphId, node, null),
        ),
        compiledEdges: definition.edges.map((edge, index) =>
          compiledEdge(currentGraphId, edge, index),
        ),
        runtimeNodes: [],
        runtimeEdges: [],
        layers: [],
      },
    ],
    attachments: [],
    warnings: [],
  };
}

function graphId(spanId: string): string {
  return `graph:${spanId}`;
}

function runtimeNodeId(spanId: string): string {
  return `runtime:${spanId}`;
}

function runtimeNode(
  currentGraphId: string,
  span: SpanSummary,
  definition: GraphNodeDefinition | undefined,
): ProjectedRuntimeNode {
  return {
    id: runtimeNodeId(span.span_id),
    graphId: currentGraphId,
    spanId: span.span_id,
    definitionNodeId: span.definition_node_id ?? `unknown:${span.span_id}`,
    displayName: definition?.display_name ?? span.definition_node_id ?? "Unknown node",
    kind: definition?.kind ?? "unknown",
    status: span.status,
    sequenceStarted: span.sequence_started,
    superstep: span.superstep,
    iteration: span.iteration,
    boundary: null,
  };
}

function boundaryNode(
  currentGraphId: string,
  boundary: "start" | "end",
  runtimeNodes: ProjectedRuntimeNode[],
): ProjectedRuntimeNode {
  const steps = runtimeNodes.map((node) => node.superstep);
  const superstep =
    boundary === "start" ? Math.min(0, ...steps) - 1 : Math.max(0, ...steps) + 1;
  return {
    id: `boundary:${currentGraphId}:${boundary}`,
    graphId: currentGraphId,
    spanId: null,
    definitionNodeId: boundary === "start" ? "__start__" : "__end__",
    displayName: boundary.toUpperCase(),
    kind: boundary,
    status: "boundary",
    sequenceStarted: null,
    superstep,
    iteration: 1,
    boundary,
  };
}

function boundaryEdge(graph: string, source: string, target: string): ProjectedEdge {
  return {
    id: `boundary-edge:${source}:${target}`,
    graphId: graph,
    source,
    target,
    kind: "boundary",
  };
}

function compiledNode(
  currentGraphId: string,
  node: GraphNodeDefinition,
  runtimeNodes: ProjectedRuntimeNode[] | null,
): ProjectedCompiledNode {
  return {
    id: `compiled:${currentGraphId}:${node.node_id}`,
    graphId: currentGraphId,
    definitionNodeId: node.node_id,
    displayName: node.display_name,
    kind: node.kind,
    visited:
      runtimeNodes === null
        ? null
        : runtimeNodes.length > 0 || node.kind === "start" || node.kind === "end",
    runtimeNodeIds: runtimeNodes?.map((runtime) => runtime.id) ?? [],
    promptTemplateId: node.prompt_template?.template_id ?? undefined,
    promptTemplateText: node.prompt_template?.template_text ?? undefined,
    destinationAgentId: node.destination_agent_id ?? undefined,
  };
}

function compiledEdge(
  currentGraphId: string,
  edge: GraphEdgeDefinition,
  index: number,
): ProjectedEdge {
  return {
    id: `compiled-edge:${currentGraphId}:${index}`,
    graphId: currentGraphId,
    source: `compiled:${currentGraphId}:${edge.source_node_id}`,
    target: `compiled:${currentGraphId}:${edge.target_node_id}`,
    kind: "compiled",
    label: edge.label ?? undefined,
    conditional: edge.kind === "conditional",
  };
}
