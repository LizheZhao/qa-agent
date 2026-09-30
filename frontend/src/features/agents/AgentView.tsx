import type { GraphDefinition } from "../../api/types";
import { EmptyState } from "../../components/EmptyState";
import { WorkspacePanel } from "../../components/WorkspacePanel";
import { shortId } from "../../shared/presentation";
import { TraceCanvas } from "../traces/TraceCanvas";
import type { ProjectedCompiledNode, ProjectedTrace } from "../traces/projection";
import "./agents.css";

export function AgentView({
  agentId,
  definition,
  projection,
  onSelectCompiled,
  loading,
  error,
}: {
  agentId: string | null;
  definition: GraphDefinition | null;
  projection: ProjectedTrace | null;
  onSelectCompiled: (node: ProjectedCompiledNode) => void;
  loading: boolean;
  error: string | null;
}) {
  return (
    <WorkspacePanel
      eyebrow="Active agent graph"
      title={agentId ?? "Select an agent"}
      error={error}
    >
      {loading && <EmptyState>Loading diagnostic state…</EmptyState>}
      {definition && projection && (
        <section className="agent-catalog">
          <div className="agent-summary">
            <span>Entrypoint <strong>{definition.entrypoint}</strong></span>
            <span>{definition.nodes.length} nodes · {definition.edges.length} edges</span>
            <span className="mono">{shortId(definition.definition_id)}</span>
          </div>
          <TraceCanvas
            projection={projection}
            fixedMode="compiled"
            onSelectSpan={() => undefined}
            onSelectCompiled={onSelectCompiled}
          />
        </section>
      )}
    </WorkspacePanel>
  );
}
