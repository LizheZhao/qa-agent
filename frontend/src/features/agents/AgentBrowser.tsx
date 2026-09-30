import type { AgentDeployment } from "../../api/types";
import { BrowserList, BrowserRow } from "../../components/BrowserList";

export function AgentBrowser({
  agents,
  selectedAgentId,
  onSelect,
}: {
  agents: AgentDeployment[];
  selectedAgentId: string | null;
  onSelect: (agent: AgentDeployment) => void;
}) {
  return (
    <BrowserList>
      {agents.map((agent) => (
        <BrowserRow
          selected={selectedAgentId === agent.agent_id}
          key={agent.agent_id}
          onClick={() => onSelect(agent)}
        >
          <span className="row-title">{agent.agent_id}</span>
          <span className="row-meta">
            version {agent.version} · {agent.routable ? "routable" : "internal"}
          </span>
          <span className="row-time">
            {agent.graph_definition_id ? "Compiled" : "No definition"}
          </span>
        </BrowserRow>
      ))}
    </BrowserList>
  );
}
