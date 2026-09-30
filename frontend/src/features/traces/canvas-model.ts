import type { Edge, Node } from "@xyflow/react";

import type {
  ProjectedCompiledNode,
  ProjectedGraph,
  ProjectedRuntimeNode,
  ProjectedTrace,
} from "./projection";

export type CanvasMode = "executed" | "compiled";

export interface TraceNodeData extends Record<string, unknown> {
  label: string;
  detail: string;
  status: string;
  spanId: string | null;
  compiledNode: ProjectedCompiledNode | null;
  boundary: boolean;
  visited: boolean | null;
}

export interface GraphGroupData extends Record<string, unknown> {
  label: string;
  detail: string;
}

export type TraceCanvasNode = Node<TraceNodeData, "traceNode">;
export type GraphCanvasNode = Node<GraphGroupData, "graphGroup">;
export type CanvasNode = TraceCanvasNode | GraphCanvasNode;

const nodeWidth = 166;
const nodeHeight = 58;
const columnGap = 66;
const rowGap = 22;
const groupPadding = 24;
const groupHeader = 48;
const graphGap = 92;

export function buildCanvasModel(
  projection: ProjectedTrace,
  mode: CanvasMode,
): { nodes: CanvasNode[]; edges: Edge[] } {
  const nodes: CanvasNode[] = [];
  const edges: Edge[] = [];
  let graphX = 0;
  const graphPositions = new Map<string, { x: number; y: number; width: number }>();

  for (const graph of projection.graphs) {
    const layers = mode === "executed" ? executedLayers(graph) : compiledLayers(graph);
    const graphAttachments =
      mode === "executed"
        ? projection.attachments.filter((attachment) => attachment.graphId === graph.id)
        : [];
    const maxRows = Math.max(1, ...layers.map((layer) => layer.nodes.length));
    const columnCount = Math.max(1, layers.length, graphAttachments.length);
    const width = groupPadding * 2 + columnCount * nodeWidth + (columnCount - 1) * columnGap;
    const attachmentHeight = graphAttachments.length ? nodeHeight + rowGap + 12 : 0;
    const height =
      groupHeader + groupPadding + maxRows * nodeHeight + (maxRows - 1) * rowGap + attachmentHeight;
    const parent = graph.parentGraphId ? graphPositions.get(graph.parentGraphId) : undefined;
    const x = parent ? Math.max(graphX, parent.x + parent.width + graphGap) : graphX;
    const y = parent ? 34 : 0;
    graphPositions.set(graph.id, { x, y, width });
    graphX = x + width + graphGap;

    nodes.push({
      id: graph.id,
      type: "graphGroup",
      position: { x, y },
      data: {
        label: graph.agentId,
        detail:
          mode === "compiled"
            ? `${graph.agentVersion} · compiled graph`
            : `${graph.agentVersion} · ${graph.status ?? "unknown"}`,
      },
      style: { width, height },
      selectable: false,
      draggable: false,
    });

    for (const [column, layer] of layers.entries()) {
      for (const [row, item] of layer.nodes.entries()) {
        nodes.push({
          id: item.id,
          type: "traceNode",
          parentId: graph.id,
          extent: "parent",
          position: {
            x: groupPadding + column * (nodeWidth + columnGap),
            y: groupHeader + row * (nodeHeight + rowGap),
          },
          data: item.data,
          style: { width: nodeWidth, height: nodeHeight },
          draggable: false,
        });
      }
    }

    const attachmentY = groupHeader + maxRows * nodeHeight + (maxRows - 1) * rowGap + 16;
    for (const [index, attachment] of graphAttachments.entries()) {
      const label =
        attachment.kind === "child_agent"
          ? `handoff · ${attachment.agentId}`
          : attachment.kind;
      nodes.push({
        id: attachment.id,
        type: "traceNode",
        parentId: graph.id,
        extent: "parent",
        position: {
          x: groupPadding + index * (nodeWidth + columnGap),
          y: attachmentY,
        },
        data: {
          label,
          detail: `${attachment.kind} span`,
          status: attachment.status,
          spanId: attachment.spanId,
          compiledNode: null,
          boundary: false,
          visited: true,
        },
        style: { width: nodeWidth, height: nodeHeight },
        draggable: false,
      });
      if (attachment.parentRuntimeNodeId) {
        edges.push({
          id: `attachment-edge:${attachment.spanId}`,
          source: attachment.parentRuntimeNodeId,
          target: attachment.id,
          type: "smoothstep",
          style: { stroke: "#8b8b86", strokeDasharray: "3 3" },
        });
      }
    }

    const sourceEdges = mode === "executed" ? graph.runtimeEdges : graph.compiledEdges;
    edges.push(
      ...sourceEdges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.label,
        type: "smoothstep",
        animated: false,
        style: {
          stroke: mode === "compiled" && edge.conditional ? "#777773" : "#aaa9a4",
          strokeDasharray: mode === "compiled" && edge.conditional ? "4 4" : undefined,
        },
      })),
    );
  }

  if (mode === "executed") {
    for (const attachment of projection.attachments) {
      for (const childGraphId of attachment.childGraphIds) {
        const childGraph = projection.graphs.find((graph) => graph.id === childGraphId);
        const childStart = childGraph?.runtimeNodes.find((node) => node.boundary === "start");
        if (!childStart) continue;
        edges.push({
          id: `delegation:${attachment.spanId}:${childGraphId}`,
          source: attachment.id,
          target: childStart.id,
          type: "smoothstep",
          label: "delegates",
          style: { stroke: "#555551", strokeDasharray: "5 4" },
        });
      }
    }
  }

  return { nodes, edges };
}

interface LayerItem {
  id: string;
  data: TraceNodeData;
}

interface CanvasLayer {
  nodes: LayerItem[];
}

function executedLayers(graph: ProjectedGraph): CanvasLayer[] {
  return graph.layers.map((layer) => ({
    nodes: layer.runtimeNodeIds.flatMap((nodeId) => {
      const node = graph.runtimeNodes.find((candidate) => candidate.id === nodeId);
      return node ? [runtimeItem(node)] : [];
    }),
  }));
}

function runtimeItem(node: ProjectedRuntimeNode): LayerItem {
  return {
    id: node.id,
    data: {
      label: node.displayName,
      detail: node.boundary ? "" : `step ${node.superstep} · iteration ${node.iteration}`,
      status: node.status,
      spanId: node.spanId,
      compiledNode: null,
      boundary: node.boundary !== null,
      visited: true,
    },
  };
}

function compiledLayers(graph: ProjectedGraph): CanvasLayer[] {
  const assigned = new Map<string, number>();
  const startNodes = graph.compiledNodes.filter((node) => node.kind === "start");
  for (const node of startNodes) assigned.set(node.id, 0);

  for (let pass = 0; pass < graph.compiledNodes.length; pass += 1) {
    let changed = false;
    for (const edge of graph.compiledEdges) {
      const sourceLayer = assigned.get(edge.source);
      if (sourceLayer === undefined || assigned.has(edge.target)) continue;
      assigned.set(edge.target, sourceLayer + 1);
      changed = true;
    }
    if (!changed) break;
  }

  for (const node of graph.compiledNodes) {
    if (!assigned.has(node.id)) assigned.set(node.id, node.kind === "end" ? 2 : 1);
  }
  const endLayer = Math.max(1, ...assigned.values());
  for (const node of graph.compiledNodes.filter((item) => item.kind === "end")) {
    assigned.set(node.id, endLayer + 1);
  }

  const layerNumbers = [...new Set(assigned.values())].sort((left, right) => left - right);
  return layerNumbers.map((layerNumber) => ({
    nodes: graph.compiledNodes
      .filter((node) => assigned.get(node.id) === layerNumber)
      .map((node) => ({
        id: node.id,
        data: {
          label: node.displayName,
          detail: node.kind,
          status:
            node.visited === null ? node.kind : node.visited ? "visited" : "unvisited",
          spanId: null,
          compiledNode: node,
          boundary: node.kind === "start" || node.kind === "end",
          visited: node.visited,
        },
      })),
  }));
}
