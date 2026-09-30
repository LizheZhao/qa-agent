import type { ButtonHTMLAttributes, ReactNode } from "react";

import "./BrowserList.css";

export function BrowserList({ children }: { children: ReactNode }) {
  return <div className="browser-list">{children}</div>;
}

export function BrowserRow({
  selected,
  children,
  className = "",
  ...buttonProps
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  selected: boolean;
  children: ReactNode;
}) {
  return (
    <button
      {...buttonProps}
      className={`browser-row ${selected ? "is-selected" : ""} ${className}`.trim()}
    >
      {children}
    </button>
  );
}
