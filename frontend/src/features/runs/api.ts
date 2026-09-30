import { paged, request } from "../../api/client";
import type { AttemptPage } from "../../api/types";

export const runsApi = {
  recentAttempts: (cursor?: string | null) =>
    request<AttemptPage>(paged("/api/diagnostics/traces", 25, cursor)),
};
