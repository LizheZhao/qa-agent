import { useCallback, useEffect, useRef, useState } from "react";

import { errorMessage } from "../../shared/presentation";
import { Inspector } from "../inspector/Inspector";
import { useTraceExplorer } from "../traces/useTraceExplorer";
import { AgentBrowser } from "./AgentBrowser";
import { AgentView } from "./AgentView";
import { agentsApi } from "./api";
import { useAgentCatalog } from "./useAgentCatalog";

export function AgentRoute() {
  const initialized = useRef(false);
  const [initializing, setInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const reportError = useCallback((message: string) => setError(message || null), []);
  const traces = useTraceExplorer(reportError);
  const agents = useAgentCatalog({
    onError: reportError,
    onClearCompiledNode: traces.clearCompiledNode,
  });

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    void (async () => {
      try {
        const detail = await agents.loadDeployment();
        const match = window.location.pathname.match(/^\/diagnostics\/graphs\/([^/]+)$/);
        const definitionId = match ? decodeURIComponent(match[1] ?? "") : null;
        const target =
          detail.agents.find((item) => item.graph_definition_id === definitionId) ??
          detail.agents[0];
        if (target) await agents.openAgent(target, definitionId ?? undefined);
      } catch (caught) {
        reportError(errorMessage(caught));
      } finally {
        setInitializing(false);
      }
    })();
  }, [agents.loadDeployment, agents.openAgent, reportError]);

  return (
    <>
      <aside className="browser-panel">
        <AgentBrowser
          agents={agents.deployment?.agents ?? []}
          selectedAgentId={agents.selectedAgent?.agent_id ?? null}
          onSelect={(agent) => void agents.openAgent(agent)}
        />
      </aside>
      <AgentView
        agentId={agents.selectedAgent?.agent_id ?? null}
        definition={agents.selectedDefinition}
        projection={agents.projection}
        onSelectCompiled={traces.selectCompiledNode}
        loading={initializing}
        error={error}
      />
      <aside className="inspector-panel">
        <Inspector
          trace={null}
          span={null}
          spans={[]}
          compiledNode={traces.compiledNode}
          definition={agents.selectedDefinition}
          onSelectSpan={() => undefined}
        />
      </aside>
    </>
  );
}

export function DeploymentIndicator() {
  const [label, setLabel] = useState("Connecting");
  useEffect(() => {
    let active = true;
    void agentsApi.deployment()
      .then((detail) => {
        if (active) setLabel(`${detail.name} · ${detail.revision}`);
      })
      .catch(() => {
        if (active) setLabel("Deployment unavailable");
      });
    return () => {
      active = false;
    };
  }, []);
  return (
    <div className="deployment">
      <span className="live-dot" />
      {label}
    </div>
  );
}
