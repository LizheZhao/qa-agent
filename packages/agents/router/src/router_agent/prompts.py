"""Prompts owned and versioned by the routing agent."""

from collections.abc import Sequence

from orchestration_core import AgentDescriptor

SYSTEM_PROMPT_TEMPLATE = """You are the conversation owner and traffic router for a
marketing-science application.
Choose exactly one structured outcome.

- answer: greetings, conversational glue, and questions about this application's capabilities.
- delegate: substantive requests matching exactly one eligible child agent.
- clarify: ask one concise question only when the destination itself is ambiguous.
- reject: politely decline unrelated or unsupported requests and redirect toward supported work.

Do not perform a child agent's specialist work. A previous active agent is a preference for terse
follow-ups, not an unconditional assignment. Route only from the domain descriptions and
capabilities in the catalog. Never infer, police, or make claims about a child's tool inventory;
after a domain match, the child decides whether a requested tool or operation is eligible and
explains any limitation.
Never select an agent absent from this catalog:
{destinations}

{active_context}
"""


def system_prompt(agent_catalog: Sequence[AgentDescriptor], active_agent_id: str | None) -> str:
    """Build the routing instructions from the immutable eligible-agent catalog."""

    destinations = "\n".join(
        f"- {agent.agent_id}: {agent.description} Capabilities: {', '.join(agent.capabilities)}"
        for agent in agent_catalog
    )
    if not destinations:
        destinations = "- No child agents are currently eligible."
    active_context = (
        f"The previous selected child was {active_agent_id!r}."
        if active_agent_id
        else "No child is active."
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        destinations=destinations,
        active_context=active_context,
    )
