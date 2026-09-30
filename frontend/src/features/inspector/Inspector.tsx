import type { GraphDefinition, SpanDetail, SpanSummary, TraceDetail } from "../../api/types";
import { formatTime, shortId } from "../../shared/presentation";
import type { ProjectedCompiledNode } from "../traces/projection";
import "./inspector.css";

export function Inspector({
  trace,
  span,
  spans,
  compiledNode,
  definition,
  onSelectSpan,
}: {
  trace: TraceDetail | null;
  span: SpanDetail | null;
  spans: SpanSummary[];
  compiledNode: ProjectedCompiledNode | null;
  definition: GraphDefinition | null;
  onSelectSpan: (spanId: string) => void;
}) {
  if (compiledNode) {
    return (
      <div className="inspector-content">
        <span className="eyebrow">Compiled node</span>
        <h2>{compiledNode.displayName}</h2>
        <div className="inspector-group">
          <KeyValue label="Kind" value={compiledNode.kind} />
          {compiledNode.visited !== null && (
            <KeyValue label="Visited" value={compiledNode.visited ? "yes" : "no"} />
          )}
          <KeyValue label="Node ID" value={compiledNode.definitionNodeId} mono />
          <KeyValue label="Prompt" value={compiledNode.promptTemplateId ?? "—"} mono />
          <KeyValue label="Destination" value={compiledNode.destinationAgentId ?? "—"} />
          {compiledNode.visited !== null && (
            <KeyValue label="Instances" value={String(compiledNode.runtimeNodeIds.length)} />
          )}
        </div>
        <InspectorSection title="Prompt template" value={compiledNode.promptTemplateText} />
        {compiledNode.visited === null && (
          <p className="inspector-note">
            Static graph metadata only. It has no runtime input, output, timing, or status.
          </p>
        )}
      </div>
    );
  }

  if (definition) {
    return (
      <div className="inspector-content">
        <span className="eyebrow">Graph definition</span>
        <h2>{definition.agent_id}</h2>
        <div className="inspector-group">
          <KeyValue label="Version" value={definition.agent_version} />
          <KeyValue label="Entrypoint" value={definition.entrypoint} mono />
          <KeyValue label="Nodes" value={String(definition.nodes.length)} />
          <KeyValue label="Edges" value={String(definition.edges.length)} />
          <KeyValue label="Definition" value={definition.definition_id} mono />
          <KeyValue label="Hash" value={definition.definition_hash} mono />
          <KeyValue label="Created" value={formatTime(definition.created_at)} />
        </div>
        <p className="inspector-note">Select a compiled node to inspect its prompt and routing metadata.</p>
      </div>
    );
  }

  if (!trace) {
    return (
      <div className="inspector-empty">
        <span className="eyebrow">Inspector</span>
        <h2>Select an execution</h2>
        <p>Trace and node details will appear here.</p>
      </div>
    );
  }

  if (!span) {
    return (
      <div className="inspector-content">
        <span className="eyebrow">Trace</span>
        <h2>{shortId(trace.trace_id)}</h2>
        <KeyValue label="Status" value={trace.status} />
        <KeyValue label="Observability" value={trace.observability_status} />
        <KeyValue label="Attempted turn" value={String(trace.attempted_turn_number)} />
        <KeyValue label="Started" value={formatTime(trace.started_at)} />
      </div>
    );
  }

  return (
    <div className="inspector-content">
      <span className="eyebrow">{span.kind}</span>
      <h2>{span.definition_node_id ?? span.kind}</h2>
      <div className="inspector-group">
        <KeyValue label="Agent" value={`${span.agent_id} · ${span.agent_version}`} />
        <KeyValue label="Status" value={span.status} />
        <KeyValue label="Sequence" value={String(span.sequence_started)} />
        <KeyValue label="Superstep" value={String(span.superstep)} />
        <KeyValue label="Iteration" value={String(span.iteration)} />
        <KeyValue label="Prompt" value={span.prompt_template_id ?? "—"} mono />
        <KeyValue
          label="Duration"
          value={span.duration_ms == null ? "—" : `${span.duration_ms.toFixed(2)} ms`}
        />
      </div>
      {span.model && <InspectorSection title="Model" value={span.model} />}
      {span.tool && <InspectorSection title="Tool" value={span.tool} />}
      <CaptureSection title="Input" value={span.input} captured={span.captures.input} />
      <CaptureSection
        title="Resolved messages"
        value={span.resolved_messages}
        captured={span.captures.resolved_messages}
      />
      <CaptureSection title="Output" value={span.output} captured={span.captures.output} />
      <CaptureSection
        title="State delta"
        value={span.state_delta}
        captured={span.captures.state_delta}
      />
      {span.error && <InspectorSection title="Error" value={span.error} />}
      {span.failure_detail && (
        <InspectorSection title="Failure detail" value={span.failure_detail} />
      )}
      <SpanNavigation span={span} spans={spans} onSelect={onSelectSpan} />
      <div className="inspector-group">
        <KeyValue label="Span ID" value={span.span_id} mono />
        <KeyValue label="Parent" value={span.parent_span_id ?? "—"} mono />
        <KeyValue
          label="Caused by"
          value={span.caused_by_span_ids.length ? span.caused_by_span_ids.join(", ") : "—"}
          mono
        />
      </div>
      <div className="inspector-group">
        <KeyValue label="Deployment" value={trace.deployment_version} mono />
        <KeyValue label="Application" value={trace.application_version} mono />
        <KeyValue label="Core" value={trace.core_version} mono />
        <InspectorSection title="Agent versions" value={trace.agent_versions} />
      </div>
    </div>
  );
}

function SpanNavigation({
  span,
  spans,
  onSelect,
}: {
  span: SpanDetail;
  spans: SpanSummary[];
  onSelect: (spanId: string) => void;
}) {
  const byId = new Map(spans.map((item) => [item.span_id, item]));
  const relations = [
    ...(span.parent_span_id
      ? [{ label: "Parent", target: byId.get(span.parent_span_id) }]
      : []),
    ...spans
      .filter((item) => item.parent_span_id === span.span_id)
      .map((target) => ({ label: "Child", target })),
    ...span.caused_by_span_ids.map((id) => ({ label: "Caused by", target: byId.get(id) })),
    ...spans
      .filter((item) => item.caused_by_span_ids.includes(span.span_id))
      .map((target) => ({ label: "Causes", target })),
  ].filter((relation): relation is { label: string; target: SpanSummary } => Boolean(relation.target));

  if (!relations.length) return null;
  return (
    <div className="inspector-group">
      <span className="inspector-group__title">Related spans</span>
      <div className="span-relations">
        {relations.map(({ label, target }) => (
          <button key={`${label}:${target.span_id}`} onClick={() => onSelect(target.span_id)}>
            <span>{label}</span>
            <strong>{target.definition_node_id ?? target.kind}</strong>
            <small>{shortId(target.span_id)}</small>
          </button>
        ))}
      </div>
    </div>
  );
}

function KeyValue({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="key-value">
      <span>{label}</span>
      <strong className={mono ? "mono" : ""}>{value}</strong>
    </div>
  );
}

function InspectorSection({ title, value }: { title: string; value: unknown }) {
  if (value == null) return null;
  return (
    <details className="inspector-section" open={title === "Output" || title === "Error"}>
      <summary>{title}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

function CaptureSection({
  title,
  value,
  captured,
}: {
  title: string;
  value: unknown;
  captured: boolean;
}) {
  if (value != null) return <InspectorSection title={title} value={value} />;
  return (
    <div className="inspector-capture-absent">
      <strong>{title}</strong>
      <span>
        {captured
          ? "Capture was recorded but content is unavailable."
          : "Not captured for this span."}
      </span>
    </div>
  );
}
