import { request } from "../../api/client";
import type { DeploymentDiagnostic } from "../../api/types";

export const agentsApi = {
  deployment: () => request<DeploymentDiagnostic>("/api/diagnostics/deployment"),
};
