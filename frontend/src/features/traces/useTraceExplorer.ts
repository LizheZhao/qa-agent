import { useCallback, useRef, useState } from "react";

import type { SpanDetail, SpanSummary, TraceDetail } from "../../api/types";
import type { WorkspaceView } from "../../app/navigation";
import { updateWorkspaceUrl } from "../../app/navigation";
import { errorMessage } from "../../shared/presentation";
import { hydrateProjection } from "./hydration";
import type { ProjectedCompiledNode, ProjectedTrace } from "./projection";
import { tracesApi } from "./api";

export function useTraceExplorer(onError: (message: string) => void) {
  const [trace, setTrace] = useState<TraceDetail | null>(null);
  const [spans, setSpans] = useState<SpanSummary[]>([]);
  const [projection, setProjection] = useState<ProjectedTrace | null>(null);
  const [span, setSpan] = useState<SpanDetail | null>(null);
  const [compiledNode, setCompiledNode] = useState<ProjectedCompiledNode | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const traceRequest = useRef(0);
  const spanRequest = useRef(0);
  const activeTraceId = useRef<string | null>(null);

  const clear = useCallback(() => {
    traceRequest.current += 1;
    spanRequest.current += 1;
    activeTraceId.current = null;
    setTrace(null);
    setSpans([]);
    setProjection(null);
    setSpan(null);
    setCompiledNode(null);
    setLoading(false);
    setSelectedTraceId(null);
  }, []);

  const openSpan = useCallback(
    async (
      traceId: string,
      spanId: string,
      sessionId: string | null,
      view: WorkspaceView,
    ) => {
      const requestId = ++spanRequest.current;
      try {
        const detail = await tracesApi.span(traceId, spanId);
        if (requestId !== spanRequest.current || activeTraceId.current !== traceId) return;
        setSpan(detail);
        setCompiledNode(null);
        updateWorkspaceUrl(view, sessionId, traceId, spanId);
      } catch (caught) {
        if (requestId === spanRequest.current && activeTraceId.current === traceId) {
          onError(errorMessage(caught));
        }
      }
    },
    [onError],
  );

  const openTrace = useCallback(
    async (
      traceId: string,
      sessionId: string | null,
      view: WorkspaceView,
      preferredSpan?: string | null,
    ) => {
      const requestId = ++traceRequest.current;
      spanRequest.current += 1;
      activeTraceId.current = traceId;
      setSelectedTraceId(traceId);
      onError("");
      setTrace(null);
      setSpans([]);
      setProjection(null);
      setSpan(null);
      setCompiledNode(null);
      setLoading(true);
      try {
        const [traceDetail, traceSpans] = await Promise.all([
          tracesApi.trace(traceId),
          tracesApi.spans(traceId),
        ]);
        const projectedTrace = await hydrateProjection(traceDetail, traceSpans);
        if (requestId !== traceRequest.current || activeTraceId.current !== traceId) return;
        setTrace(traceDetail);
        setSpans(traceSpans);
        setProjection(projectedTrace);
        const target = preferredSpan
          ? traceSpans.find((item) => item.span_id === preferredSpan)
          : (traceSpans.find((item) => item.kind === "node") ?? traceSpans[0]);
        if (target) {
          await openSpan(traceId, target.span_id, sessionId, view);
        } else {
          setSpan(null);
          updateWorkspaceUrl(view, sessionId, traceId);
        }
      } catch (caught) {
        if (requestId === traceRequest.current && activeTraceId.current === traceId) {
          onError(errorMessage(caught));
        }
      } finally {
        if (requestId === traceRequest.current && activeTraceId.current === traceId) {
          setLoading(false);
        }
      }
    },
    [onError, openSpan],
  );

  const selectCompiledNode = useCallback((node: ProjectedCompiledNode) => {
    setSpan(null);
    setCompiledNode(node);
  }, []);

  const clearCompiledNode = useCallback(() => {
    setCompiledNode(null);
  }, []);

  return {
    trace,
    spans,
    projection,
    span,
    compiledNode,
    loading,
    selectedTraceId,
    clear,
    openSpan,
    openTrace,
    selectCompiledNode,
    clearCompiledNode,
  };
}
