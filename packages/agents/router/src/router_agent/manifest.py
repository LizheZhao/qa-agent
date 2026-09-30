"""Static router package metadata."""

from orchestration_core import AgentManifest

AGENT_MANIFEST = AgentManifest(
    agent_id="router",
    version="0.5.0",
    description="Own the conversation, answer ordinary chat, and select one eligible child agent.",
    capabilities=("conversation_routing", "general_conversation"),
)
