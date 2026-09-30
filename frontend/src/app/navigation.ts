export type WorkspaceView = "lab" | "agents" | "conversations" | "runs";

export function workspaceRoot(view: WorkspaceView): string {
  if (view === "agents") return "/diagnostics/graphs";
  if (view === "conversations") return "/diagnostics/conversations";
  return `/diagnostics/${view}`;
}

export function initialView(): WorkspaceView {
  const path = window.location.pathname;
  if (path.startsWith("/diagnostics/graphs")) return "agents";
  if (path.startsWith("/diagnostics/runs")) return "runs";
  if (
    path.startsWith("/diagnostics/sessions") ||
    path.startsWith("/diagnostics/conversations")
  ) {
    return "conversations";
  }
  return "lab";
}

export function updateWorkspaceUrl(
  view: WorkspaceView,
  resourceId: string | null,
  traceId?: string | null,
  spanId?: string | null,
) {
  const path =
    view === "lab"
      ? "/diagnostics/lab"
      : view === "agents"
        ? resourceId
          ? `/diagnostics/graphs/${encodeURIComponent(resourceId)}`
          : "/diagnostics/graphs"
      : view === "runs"
        ? "/diagnostics/runs"
        : resourceId
          ? `/diagnostics/sessions/${encodeURIComponent(resourceId)}`
          : "/diagnostics/conversations";
  const search = new URLSearchParams();
  if (view === "lab" && resourceId) search.set("session", resourceId);
  if (traceId) search.set("trace", traceId);
  if (spanId) search.set("span", spanId);
  const target = `${path}${search.size ? `?${search}` : ""}`;
  if (`${window.location.pathname}${window.location.search}` !== target) {
    window.history.pushState(null, "", target);
  }
}

export function listenForWorkspaceHistory(onNavigate: () => void): () => void {
  window.addEventListener("popstate", onNavigate);
  return () => window.removeEventListener("popstate", onNavigate);
}
