import { StatusBadge } from "../../components/StatusBadge";
import type { ProjectedGraph, ProjectedTrace } from "./projection";

export function TraceProjectionList({
  projection,
  selectedSpanId,
  onSelect,
}: {
  projection: ProjectedTrace;
  selectedSpanId: string | null;
  onSelect: (spanId: string) => void;
}) {
  const graphById = new Map(projection.graphs.map((graph) => [graph.id, graph]));
  const childrenByGraph = new Map<string, ProjectedGraph[]>();
  for (const graph of projection.graphs) {
    if (!graph.parentGraphId) continue;
    const children = childrenByGraph.get(graph.parentGraphId) ?? [];
    children.push(graph);
    childrenByGraph.set(graph.parentGraphId, children);
  }

  const renderGraph = (graph: ProjectedGraph, depth: number) => (
    <section className="projected-graph" key={graph.id} style={{ marginLeft: depth * 12 }}>
      <div className="projected-graph__header">
        <span>{graph.agentId} · {graph.agentVersion}</span>
        {graph.status && <StatusBadge status={graph.status} />}
      </div>
      {graph.layers.map((layer) => (
        <div className="runtime-layer" key={`${graph.id}:${layer.superstep}`}>
          <span className="runtime-layer__label">
            {layerLabel(graph, layer.runtimeNodeIds, layer.superstep)}
          </span>
          <div className="runtime-layer__nodes">
            {layer.runtimeNodeIds.map((nodeId) => {
              const node = graph.runtimeNodes.find((candidate) => candidate.id === nodeId);
              if (!node) return null;
              if (!node.spanId) {
                return <span className="boundary-node" key={node.id}>{node.displayName}</span>;
              }
              return (
                <button
                  className={`runtime-node ${selectedSpanId === node.spanId ? "is-selected" : ""}`}
                  key={node.id}
                  onClick={() => onSelect(node.spanId!)}
                >
                  <span>
                    <strong>{node.displayName}</strong>
                    <small>iteration {node.iteration}</small>
                  </span>
                  <StatusBadge status={node.status} />
                </button>
              );
            })}
          </div>
        </div>
      ))}
      {(childrenByGraph.get(graph.id) ?? []).map((child) => renderGraph(child, depth + 1))}
    </section>
  );

  return (
    <div className="projection-list">
      {projection.warnings.map((warning, index) => (
        <div className="projection-warning" key={`${warning.code}:${warning.spanId ?? index}`}>
          {warning.message}
        </div>
      ))}
      {projection.rootGraphIds.map((graphId) => {
        const graph = graphById.get(graphId);
        return graph ? renderGraph(graph, 0) : null;
      })}
    </div>
  );
}

function layerLabel(
  graph: ProjectedGraph,
  runtimeNodeIds: string[],
  superstep: number,
): string {
  const boundaries = runtimeNodeIds.map(
    (nodeId) => graph.runtimeNodes.find((node) => node.id === nodeId)?.boundary,
  );
  if (boundaries.length > 0 && boundaries.every((boundary) => boundary === "start")) {
    return "Start";
  }
  if (boundaries.length > 0 && boundaries.every((boundary) => boundary === "end")) {
    return "End";
  }
  return `Step ${superstep}`;
}
