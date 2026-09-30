import { useCallback, useState } from "react";

import { graphDefinition } from "../../api/client";
import type { AgentDeployment, DeploymentDiagnostic, GraphDefinition } from "../../api/types";
import { updateWorkspaceUrl } from "../../app/navigation";
import { errorMessage } from "../../shared/presentation";
import { projectDefinition, type ProjectedTrace } from "../traces/projection";
import { agentsApi } from "./api";

export function useAgentCatalog({
  onError,
  onClearCompiledNode,
}: {
  onError: (message: string) => void;
  onClearCompiledNode: () => void;
}) {
  const [deployment, setDeployment] = useState<DeploymentDiagnostic | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<AgentDeployment | null>(null);
  const [selectedDefinition, setSelectedDefinition] = useState<GraphDefinition | null>(null);
  const [projection, setProjection] = useState<ProjectedTrace | null>(null);

  const loadDeployment = useCallback(async () => {
    const detail = await agentsApi.deployment();
    setDeployment(detail);
    return detail;
  }, []);

  const openAgent = useCallback(
    async (agent: AgentDeployment, definitionIdOverride?: string) => {
      const definitionId = definitionIdOverride ?? agent.graph_definition_id;
      onError("");
      setSelectedAgent(agent);
      onClearCompiledNode();
      setSelectedDefinition(null);
      setProjection(null);
      if (!definitionId) {
        updateWorkspaceUrl("agents", null);
        return;
      }
      try {
        const definition = await graphDefinition(definitionId);
        setSelectedDefinition(definition);
        setProjection(projectDefinition(definition));
        updateWorkspaceUrl("agents", definition.definition_id);
      } catch (caught) {
        onError(errorMessage(caught));
      }
    },
    [onClearCompiledNode, onError],
  );

  return {
    deployment,
    selectedAgent,
    selectedDefinition,
    projection,
    loadDeployment,
    openAgent,
  };
}
