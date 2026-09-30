import { useCallback, useState } from "react";

import type { AttemptSummary } from "../../api/types";
import { runsApi } from "./api";

export function useRunCatalog() {
  const [attempts, setAttempts] = useState<AttemptSummary[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);

  const loadRuns = useCallback(async () => {
    const page = await runsApi.recentAttempts();
    setAttempts(page.items);
    setCursor(page.next_cursor ?? null);
    return page.items;
  }, []);

  const loadOlderRuns = useCallback(async () => {
    if (!cursor) return;
    const page = await runsApi.recentAttempts(cursor);
    setAttempts((current) => [...current, ...page.items]);
    setCursor(page.next_cursor ?? null);
  }, [cursor]);

  return { attempts, cursor, loadRuns, loadOlderRuns };
}
