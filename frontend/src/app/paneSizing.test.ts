import { describe, expect, it } from "vitest";

import { defaultPaneSizes, resizePane } from "./paneSizing";

describe("pane sizing", () => {
  it("uses the existing normal and compact defaults", () => {
    expect(defaultPaneSizes(false)).toEqual({ browser: 248, inspector: 330 });
    expect(defaultPaneSizes(true)).toEqual({ browser: 220, inspector: 300 });
  });

  it("clamps side panes and preserves the main pane limits", () => {
    const defaults = defaultPaneSizes(false);

    expect(resizePane(defaults, "browser", 100, 1800).browser).toBe(200);
    expect(resizePane(defaults, "browser", 600, 1800).browser).toBe(420);
    expect(resizePane(defaults, "inspector", 100, 1800).inspector).toBe(280);
    expect(resizePane(defaults, "inspector", 700, 1800).inspector).toBe(520);

    const mainLimited = resizePane(defaults, "browser", 420, 1100);
    expect(mainLimited.browser).toBe(250);
  });
});
