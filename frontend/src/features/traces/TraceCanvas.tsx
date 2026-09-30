import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo, useState } from "react";

import type { ProjectedCompiledNode, ProjectedTrace } from "./projection";
import {
  buildCanvasModel,
  type CanvasMode,
  type GraphCanvasNode,
  type TraceCanvasNode,
} from "./canvas-model";
import "./traces.css";

const nodeTypes = { traceNode: TraceNode, graphGroup: GraphGroup };

export function TraceCanvas({
  projection,
  onSelectSpan,
  onSelectCompiled,
  fixedMode,
}: {
  projection: ProjectedTrace;
  onSelectSpan: (spanId: string) => void;
  onSelectCompiled: (node: ProjectedCompiledNode) => void;
  fixedMode?: CanvasMode;
}) {
  const [mode, setMode] = useState<CanvasMode>("executed");
  const activeMode = fixedMode ?? mode;
  const model = useMemo(
    () => buildCanvasModel(projection, activeMode),
    [projection, activeMode],
  );

  return (
    <div className="trace-canvas-shell">
      <div className="canvas-toolbar">
        {fixedMode ? (
          <strong>Compiled topology</strong>
        ) : (
          <div className="segmented canvas-mode" aria-label="Graph view">
            <button className={mode === "executed" ? "is-active" : ""} onClick={() => setMode("executed")}>Executed</button>
            <button className={mode === "compiled" ? "is-active" : ""} onClick={() => setMode("compiled")}>Compiled</button>
          </div>
        )}
        <span>{activeMode === "executed" ? "Observed order and concurrency" : fixedMode ? "Active immutable definition" : "Possible topology · unvisited paths dimmed"}</span>
      </div>
      <div className="trace-canvas">
        <ReactFlow
          nodes={model.nodes}
          edges={model.edges}
          nodeTypes={nodeTypes}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable
          fitView
          fitViewOptions={{ padding: 0.16 }}
          minZoom={0.25}
          maxZoom={1.7}
          onNodeClick={(_, node) => {
            if (node.type !== "traceNode") return;
            const data = node.data as TraceCanvasNode["data"];
            if (data.spanId) onSelectSpan(data.spanId);
            else if (data.compiledNode) onSelectCompiled(data.compiledNode);
          }}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#e7e7e3" gap={18} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  );
}

function TraceNode({ data, selected }: NodeProps<TraceCanvasNode>) {
  return (
    <div className={`canvas-node ${data.boundary ? "is-boundary" : ""} ${data.visited === false ? "is-unvisited" : ""} ${selected ? "is-selected" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <strong>{data.label}</strong>
      {data.detail && <small>{data.detail}</small>}
      {!data.boundary && <span className={statusClass(data.status)}>{data.status}</span>}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

function GraphGroup({ data }: NodeProps<GraphCanvasNode>) {
  return (
    <div className="canvas-graph-group">
      <strong>{data.label}</strong>
      <span>{data.detail}</span>
    </div>
  );
}

function statusClass(status: string): string {
  return status === "failed" || status === "cancelled"
    ? "canvas-node__status is-error"
    : "canvas-node__status";
}
