import type { ReactNode } from "react";

import "./EmptyState.css";

export function EmptyState({ children, compact = false }: { children: ReactNode; compact?: boolean }) {
  return <div className={compact ? "empty-state compact" : "empty-state"}>{children}</div>;
}
