"""Static marketing-science package metadata."""

from orchestration_core import AgentManifest

AGENT_MANIFEST = AgentManifest(
    agent_id="marketing_science",
    version="0.3.0",
    description=(
        "Explain marketing-science concepts, methods, measurement, experimentation, and modeling."
    ),
    capabilities=("marketing_science_explanation", "measurement_guidance"),
    tool_ids=("calculate_sum",),
)
