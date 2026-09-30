import {
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";

import type { WorkspaceView } from "./navigation";
import {
  availablePaneWidth,
  defaultPaneSizes,
  PANE_LIMITS,
  resizePane,
  type ResizablePane,
  WORKSPACE_GAP,
  WORKSPACE_PADDING,
} from "./paneSizing";
import "./AppShell.css";

const views: WorkspaceView[] = ["lab", "agents", "conversations", "runs"];

function responsiveDefaults() {
  const compact =
    typeof window.matchMedia === "function" && window.matchMedia("(max-width: 1320px)").matches;
  return defaultPaneSizes(compact);
}

export function AppShell({
  activeView,
  deploymentStatus,
  onSelectView,
  children,
}: {
  activeView: WorkspaceView;
  deploymentStatus: ReactNode;
  onSelectView: (view: WorkspaceView) => void;
  children: ReactNode;
}) {
  const workspaceRef = useRef<HTMLDivElement>(null);
  const [paneSizes, setPaneSizes] = useState(responsiveDefaults);
  const [dragging, setDragging] = useState<ResizablePane | null>(null);

  useEffect(() => {
    setPaneSizes(responsiveDefaults());
    setDragging(null);
  }, [activeView]);

  function updatePane(pane: ResizablePane, desiredWidth: number) {
    const workspaceWidth = workspaceRef.current?.getBoundingClientRect().width;
    if (!workspaceWidth) return;
    setPaneSizes((current) =>
      resizePane(current, pane, desiredWidth, availablePaneWidth(workspaceWidth)),
    );
  }

  function handlePointerMove(pane: ResizablePane, event: PointerEvent<HTMLButtonElement>) {
    if (dragging !== pane || !workspaceRef.current) return;
    const bounds = workspaceRef.current.getBoundingClientRect();
    const width =
      pane === "browser"
        ? event.clientX - bounds.left - WORKSPACE_PADDING - WORKSPACE_GAP / 2
        : bounds.right - event.clientX - WORKSPACE_PADDING - WORKSPACE_GAP / 2;
    updatePane(pane, width);
  }

  function handleKeyDown(pane: ResizablePane, event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const paneDirection = pane === "browser" ? direction : -direction;
    updatePane(pane, paneSizes[pane] + paneDirection * 16);
  }

  const workspaceStyle = {
    "--browser-pane-width": `${paneSizes.browser}px`,
    "--inspector-pane-width": `${paneSizes.inspector}px`,
  } as CSSProperties;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="eyebrow">Internal engineering</span>
          <h1>Orchestration diagnostics</h1>
        </div>
        <nav className="topnav" aria-label="Diagnostic workspace">
          {views.map((view) => (
            <button
              className={activeView === view ? "is-active" : ""}
              key={view}
              onClick={() => onSelectView(view)}
            >
              {view.charAt(0).toUpperCase() + view.slice(1)}
            </button>
          ))}
        </nav>
        {deploymentStatus}
      </header>
      <div
        className={`workspace workspace--${activeView}`}
        ref={workspaceRef}
        style={workspaceStyle}
      >
        {children}
        {activeView !== "lab" && (
          <>
            <button
              aria-label="Resize browser pane"
              aria-orientation="vertical"
              aria-valuemax={PANE_LIMITS.browser.max}
              aria-valuemin={PANE_LIMITS.browser.min}
              aria-valuenow={Math.round(paneSizes.browser)}
              className={`pane-resizer pane-resizer--browser ${dragging === "browser" ? "is-dragging" : ""}`}
              onKeyDown={(event) => handleKeyDown("browser", event)}
              onPointerDown={(event) => {
                event.currentTarget.setPointerCapture?.(event.pointerId);
                setDragging("browser");
              }}
              onPointerMove={(event) => handlePointerMove("browser", event)}
              onPointerCancel={() => setDragging(null)}
              onLostPointerCapture={() => setDragging(null)}
              onPointerUp={() => setDragging(null)}
              role="separator"
              type="button"
            />
            <button
              aria-label="Resize inspector pane"
              aria-orientation="vertical"
              aria-valuemax={PANE_LIMITS.inspector.max}
              aria-valuemin={PANE_LIMITS.inspector.min}
              aria-valuenow={Math.round(paneSizes.inspector)}
              className={`pane-resizer pane-resizer--inspector ${dragging === "inspector" ? "is-dragging" : ""}`}
              onKeyDown={(event) => handleKeyDown("inspector", event)}
              onPointerDown={(event) => {
                event.currentTarget.setPointerCapture?.(event.pointerId);
                setDragging("inspector");
              }}
              onPointerMove={(event) => handlePointerMove("inspector", event)}
              onPointerCancel={() => setDragging(null)}
              onLostPointerCapture={() => setDragging(null)}
              onPointerUp={() => setDragging(null)}
              role="separator"
              type="button"
            />
          </>
        )}
      </div>
    </div>
  );
}
