export type PaneSizes = {
  browser: number;
  inspector: number;
};

export type ResizablePane = keyof PaneSizes;

export const PANE_LIMITS = {
  browser: { min: 200, max: 420 },
  main: { min: 520, max: 1440 },
  inspector: { min: 280, max: 520 },
} as const;

export const WORKSPACE_PADDING = 12;
export const WORKSPACE_GAP = 10;

export function defaultPaneSizes(compact: boolean): PaneSizes {
  return compact ? { browser: 220, inspector: 300 } : { browser: 248, inspector: 330 };
}

export function availablePaneWidth(workspaceWidth: number): number {
  return workspaceWidth - WORKSPACE_PADDING * 2 - WORKSPACE_GAP * 2;
}

export function resizePane(
  sizes: PaneSizes,
  pane: ResizablePane,
  desiredWidth: number,
  availableWidth: number,
): PaneSizes {
  const otherWidth = pane === "browser" ? sizes.inspector : sizes.browser;
  const limits = PANE_LIMITS[pane];
  const lowerBound = Math.max(
    limits.min,
    availableWidth - otherWidth - PANE_LIMITS.main.max,
  );
  const upperBound = Math.min(
    limits.max,
    availableWidth - otherWidth - PANE_LIMITS.main.min,
  );

  if (lowerBound > upperBound) return sizes;
  const width = Math.min(upperBound, Math.max(lowerBound, desiredWidth));
  return { ...sizes, [pane]: width };
}
