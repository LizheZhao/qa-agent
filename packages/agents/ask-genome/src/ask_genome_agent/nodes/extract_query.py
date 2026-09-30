"""Deterministic: pull the raw query text out of the conversation state.

Takes the last human message and hands it to the planning flow as a plain string.

This used to walk backward through a clarification thread and render it as a transcript, because
a clarification ended the turn and its answer came back as a new one. The graph pauses now and
resumes in place, so a clarification answer never reaches this node at all: execution continues
from coordinate_clarification with the filter already built.

Note this does not carry conversational context generally. A follow-up after a completed answer
is planned as a brand new question, which is a real limitation and the subject of the wider
multi-turn discussion, not something this node should decide on its own.
"""

from typing import Any

from langchain_core.messages import HumanMessage

from ask_genome_agent.state import AskGenomeState


def extract_query(state: AskGenomeState) -> dict[str, Any]:
    messages = state["messages"]
    last_human = next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)), None
    )
    if last_human is None:
        raise ValueError("AskGenomeState requires at least one HumanMessage")

    content = last_human.content
    return {"query": content if isinstance(content, str) else str(content)}
