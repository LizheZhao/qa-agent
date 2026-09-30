"""Unit tests for ask_genome_agent.nodes.extract_query."""

from __future__ import annotations

import pytest
from ask_genome_agent.nodes.extract_query import extract_query
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.unit
def test_extract_query_takes_the_last_human_message() -> None:
    state = {"messages": [HumanMessage(content="what was spend last quarter")]}
    result = extract_query(state)
    assert result["query"] == "what was spend last quarter"


@pytest.mark.unit
def test_extract_query_plans_only_the_newest_question() -> None:
    """Earlier turns are not context. A clarification answer never reaches this node any more, so
    anything that does is a fresh question, and pulling the previous exchange in would plan the
    wrong thing."""

    state = {
        "messages": [
            HumanMessage(content="first question"),
            AIMessage(content="a completed, unrelated answer"),
            HumanMessage(content="an entirely new question"),
        ]
    }
    result = extract_query(state)
    assert result["query"] == "an entirely new question"


@pytest.mark.unit
def test_extract_query_rejects_a_conversation_with_no_question() -> None:
    with pytest.raises(ValueError):
        extract_query({"messages": [AIMessage(content="an answer to nothing")]})
