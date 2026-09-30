import type { ReactNode, RefObject } from "react";

import { Notice } from "./Notice";
import "./WorkspacePanel.css";

export function WorkspacePanel({
  eyebrow,
  title,
  actions,
  error,
  children,
  footer,
  scrollRef,
  contentClassName = "workspace-panel__content",
}: {
  eyebrow: string;
  title: string;
  actions?: ReactNode;
  error: string | null;
  children: ReactNode;
  footer?: ReactNode;
  scrollRef?: RefObject<HTMLDivElement | null>;
  contentClassName?: string;
}) {
  return (
    <main className="conversation-panel">
      <div className="workspace-panel__top">
        <section className="conversation-header">
          <div>
            <span className="eyebrow">{eyebrow}</span>
            <h2>{title}</h2>
          </div>
          {actions}
        </section>
        {error && <Notice tone="error">{error}</Notice>}
      </div>
      <div className={`workspace-panel__content-region ${contentClassName}`} ref={scrollRef}>
        {children}
      </div>
      {footer && <div className="workspace-panel__footer">{footer}</div>}
    </main>
  );
}
