import type { ReactNode } from "react";

import "./LoadMoreButton.css";

export function LoadMoreButton({
  children,
  inline = false,
  onClick,
}: {
  children: ReactNode;
  inline?: boolean;
  onClick: () => void;
}) {
  return (
    <button className={inline ? "load-more inline" : "load-more"} onClick={onClick}>
      {children}
    </button>
  );
}
