import type { ReactNode } from "react";

import "./Notice.css";

export function Notice({ tone, children }: { tone: "error" | "warning"; children: ReactNode }) {
  return <div className={tone === "error" ? "error-banner" : "warning-banner"}>{children}</div>;
}
