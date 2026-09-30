import { useEffect, useRef, useState } from "react";

import { AgentRoute, DeploymentIndicator } from "../features/agents/AgentRoute";
import { LabRoute } from "../features/chat/LabRoute";
import { RunsRoute } from "../features/runs/RunsRoute";
import { ConversationsRoute } from "../features/sessions/ConversationsRoute";
import { AppErrorBoundary } from "./AppErrorBoundary";
import { AppShell } from "./AppShell";
import {
  initialView,
  listenForWorkspaceHistory,
  type WorkspaceView,
  workspaceRoot,
} from "./navigation";

export function App() {
  const [activeView, setActiveView] = useState<WorkspaceView>(initialView);
  const [visitedViews, setVisitedViews] = useState<Set<WorkspaceView>>(
    () => new Set([activeView]),
  );
  const lastLocation = useRef<Partial<Record<WorkspaceView, string>>>({
    [activeView]: `${window.location.pathname}${window.location.search}`,
  });
  useEffect(
    () => listenForWorkspaceHistory(() => window.location.reload()),
    [],
  );

  function selectView(view: WorkspaceView) {
    if (view === activeView) return;
    lastLocation.current[activeView] = `${window.location.pathname}${window.location.search}`;
    const target = lastLocation.current[view] ?? workspaceRoot(view);
    if (`${window.location.pathname}${window.location.search}` !== target) {
      window.history.pushState(null, "", target);
    }
    setVisitedViews((visited) => new Set(visited).add(view));
    setActiveView(view);
  }

  return (
    <AppErrorBoundary>
      <AppShell
        activeView={activeView}
        deploymentStatus={<DeploymentIndicator />}
        onSelectView={selectView}
      >
        <div className="workspace-route" hidden={activeView !== "lab"}>
          {visitedViews.has("lab") && <LabRoute />}
        </div>
        <div className="workspace-route" hidden={activeView !== "agents"}>
          {visitedViews.has("agents") && <AgentRoute />}
        </div>
        <div className="workspace-route" hidden={activeView !== "conversations"}>
          {visitedViews.has("conversations") && <ConversationsRoute />}
        </div>
        <div className="workspace-route" hidden={activeView !== "runs"}>
          {visitedViews.has("runs") && <RunsRoute />}
        </div>
      </AppShell>
    </AppErrorBoundary>
  );
}
